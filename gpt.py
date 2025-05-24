import torch 
import torch.nn as nn
import transformers
import torch.nn.functional as F
from dataclasses import dataclass
import inspect
import time
import numpy as np
import tiktoken
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from hellaswag import render_example, iterate_examples


enc = tiktoken.get_encoding('gpt2')
torch.set_float32_matmul_precision('high')


dropout = 0.2

class MultiheadScaledAttention(nn.Module):

    def __init__(self, emb_size, n_heads):
        super().__init__()
        assert emb_size % n_heads == 0
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.projection2qkv = nn.Linear(emb_size,3* emb_size)

        # i don't know it seems like some people do it
        self.final_projection = nn.Linear(emb_size, emb_size)
        self.final_projection.NANOGPT_SCALE_INIT = 1 #serves as a flag for weights init method
        

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.projection2qkv(x)

        #division of qkv
        q, k, v = qkv.split(self.emb_size, dim=2)
    
        #reshaping to heads

        #     change    just devid my heads             swap T and heads so heads are treated as batch 
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        #get back to the x.size()
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.final_projection(out)
        return out 


class GPTFeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.c_fc    = nn.Linear(emb_size, 4 * emb_size)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * emb_size, emb_size)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        

class GPTblock(nn.Module):

    def __init__(self, emb_size, n_heads):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(emb_size)
        self.MultiheadScaledAttention = MultiheadScaledAttention(emb_size, n_heads)
        self.layer_norm_2 = nn.LayerNorm(emb_size)
        self.FeedForward = GPTFeedForward(emb_size)


    def forward(self, x):

        x = x + self.MultiheadScaledAttention(self.layer_norm_1(x))
        x = x + self.FeedForward(self.layer_norm_2(x))
        return x



class GPT(nn.Module):

    def __init__(self, n_blocks, vocab_size, n_heads, context_size, emb_size):
        super().__init__()

        self.n_blocks = n_blocks

        self.emb = nn.Embedding(vocab_size, emb_size) 
        self.pe = nn.Embedding(context_size, emb_size )
        self.blocks = nn.ModuleList([GPTblock(emb_size, n_heads) for _ in range(n_blocks)])
        self.gpt_head = nn.Linear(emb_size, vocab_size)
        self.ln_f = nn.LayerNorm(emb_size) #changed

        # weight sharing scheme
        self.emb.weight = self.gpt_head.weight


        # init params
        self.apply(self._init_weights)
    
    def return_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_blocks) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def forward(self, x):
        batch_size, seq_len = x.size()
        B, T = x.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device) # shape (T)

        x = self.emb(x) + self.pe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.gpt_head(x)
        return x
    

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}

        data_root = '/home/asukhov/dt/edu_fineweb10B'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) >0, 'no shards found'
        if master_process:
            print(f'found {len(shards)} shards for split {split}')

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

        ### OLD CODE
        # with open('input.txt', 'r') as f:
        #     text = f.read()

        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)

        # self.current_position = self.B * self.T * self.process_rank
        ### !OLD CODE
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard +1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank


        return x, y
    



#setting up a ddp
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run ?
if ddp:
    assert torch.cuda.is_available() , 'cuda not available'
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank =0 
    ddp_world_size = 1
    master_process =True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

device_type = "cuda" if device.startswith("cuda") else "cpu"
print('device_type: ', device_type)
### setting lr
import math
max_lr = 6e-4
min_lr = max_lr*0.1

max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)



@dataclass
class GPTConfig:
    context_size: int = 1024
    vocab_size: int = 50304 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    emb_size: int = 768
    n_heads: int = 12
    n_blocks: int = 12

#we need this so in every gpu the model is initialised in the same way
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

gpt_cnfg = GPTConfig()
gpt = GPT(gpt_cnfg.n_blocks, gpt_cnfg.vocab_size, gpt_cnfg.n_heads, gpt_cnfg.context_size, gpt_cnfg.emb_size)
gpt.to(device)
# gpt = torch.compile(gpt)



#we need to wrap the model to a ddp container
if ddp:
    gpt = DDP(gpt, device_ids=[ddp_local_rank])

raw_gpt = gpt.module if ddp else gpt


### GRADIENT ACCUMULATION

total_batch_size = 524288 # we need 0.5 M 0.5M/1024 = >=489k, but it's ugly so we take 2**19
B = 16
T= 1024
assert total_batch_size % (B*T*ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split='train')
val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split='val')






def evaluate_model():
    gpt.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = gpt(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    gpt.train()

def get_generation():
    gpt.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = gpt(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        if master_process:
            print(f"rank {ddp_rank} sample {i}: {decoded}")
            with open(log_file, "a") as f:
                f.write(f" generation {decoded }\n")
    gpt.train()

def save_model(step):
    
    # optionally write model checkpoints
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
    checkpoint = {
        'model': raw_gpt.state_dict(),
        'config': gpt_cnfg,
        'step': step,
    }
    # you might also want to add optimizer.state_dict() and
    # rng seeds etc., if you wanted to more exactly resume training
    torch.save(checkpoint, checkpoint_path)

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

def evaluate_hellaswag():
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits= gpt(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} hella {acc_norm:.4f}\n")

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


max_steps = 19073 # if we want to see a dataset only ones we need to do make not inf steps but numtokens in ds / tokens per step
warmup_steps = 715 # if we want to do warm up for n tokens we also need some steps

# optimize!
optimizer = raw_gpt.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
criterion = nn.CrossEntropyLoss()

for step in range(max_steps):
    st = time.time()
    optimizer.zero_grad()
    last_step = (step == max_steps - 1)

    if step%250 == 0:
        evaluate_model()

    if step%250 == 0:
        get_generation()

    if step == 3000:
        save_model(step)
    
    if step%7000 == 0 or last_step:
        save_model(step)

    # if step % 10 == 0 or last_step:
    #     evaluate_hellaswag()
        
    
    #gradient accumulation
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = gpt(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        #in torch loss use 'mean' reduction 
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        # i didn't really understand it
        # author said that after it all nodes will have same averaged gradients
        if ddp:
            gpt.require_backward_grad_sync = (micro_step == grad_accum_steps-1)

        loss.backward()
    # it takes all loss_accum variables from all nodes and means them and each process now has it
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    #doing gradient clipping after backward; also it's from gpt3 paper
    norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)

    #set new lr
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()

    torch.cuda.synchronize()
    fn = time.time()


    tk_pocessed = train_loader.B * train_loader.T*ddp_world_size*grad_accum_steps

    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {(fn-st)*1000:.2f}ms | tok/sec: {tk_pocessed:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
    
if ddp:
    destroy_process_group()
    


    
