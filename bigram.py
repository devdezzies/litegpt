import torch 
import torch.nn as nn 
from torch.nn import functional as F 

# hyperparameters 
batch_size = 64 # how many independent sequences will we process in parallel 
block_size = 256 # what is the maximum content length for predictions 
max_iters = 50 
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f: 
    text = f.read() 

# all the unique chars that occur in this text 
chars = sorted(list(set(text))) 
voc_size = len(chars) 

# create mapping from chars to integers and vice versa 
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # take a string outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # take a list of integers, output a string 

# train and test splits 
data =  torch.tensor(encode(text), dtype=torch.long) 
n = int(0.8*len(data)) 
train_data = data[:n] 
val_data = data[n:]

@torch.no_grad()
def estimate_loss(): 
    out = {} 
    model.eval()
    for split in ['train', 'val']: 
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters): 
            X, Y = get_batch(split) 
            logits, loss = model(X, Y) 
            losses[k] = loss.item() 
        out[split] = losses.mean()
    model.train() 
    return out

# data loading
def get_batch(split): 
    data = train_data if split == 'train' else val_data 
    ix = torch.randint(len(data) - block_size, (batch_size, )) # random numbers in batch_size 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y 

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        B, T, C = x.shape 
        k = self.key(x) # (B,T,C) 
        q = self.query(x) # (B,T,C)
        # compute attention scores 
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values 
        v = self.value(x) # (B,T,C) 
        out = wei @ v 
        return out
    
class MultiHeadAttention(nn.Module): 
    def __init__(self, num_heads, head_size): 
        super().__init__() 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): 
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module): 
    def __init__(self, n_embed): 
        super().__init__()
        self.net = nn.Sequential( 
            nn.Linear(n_embed, 4 * n_embed), 
            nn.ReLU(), 
            nn.Linear(4 * n_embed, n_embed), 
            nn.Dropout()
        )
    
    def forward(self, x): 
        return self.net(x)
    
class Block(nn.Module): 
    """ Transformer Block """

    def __init__(self, n_embed, n_head): 
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed) 
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x): 
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x)) 
        return x

class BigramLanguageModel(nn.Module): 
    def __init__(self, vocab_size): 
        super().__init__() 
        # taking a tensor input (word ids, tokenized) into vector representations 
        # num_embeddings means how large is your vocabulary (in this case 65 unique symbols) 
        # embedding_dim means how long is the vector representation for each symbols (each symbol will be represented in 65 dimension vector)
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(in_features=n_embed, out_features=vocab_size) # fully connected layer

    def forward(self, input_tensors, target_tensors=None): 
        B, T = input_tensors.shape
                
        # forward pass without activation function
        tok_emb = self.token_embedding_table(input_tensors) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if target_tensors is None: 
            loss = None 
        else: 
            B, T, C = logits.shape 
            logits = logits.view(B*T, C) 
            target_tensors = target_tensors.view(B*T)
            loss = F.cross_entropy(logits, target_tensors)

        return logits, loss

    def generate(self, input_tensors, max_new_tokens): 
        for _ in range(max_new_tokens): 
            idx_cond = input_tensors[:, -block_size:]
            # get the predictions 
            logits, loss = self(idx_cond) 
            # focus only on the last time step 
            logits = logits[:, -1, :] # becomes (B, C) <=> batch and the logits for each sequence 
            # apply softmax to get probabilities 
            probs = F.softmax(logits, dim=-1)  # (B, C) 
            # sample from the distribution 
            next_tensor = torch.multinomial(probs, num_samples=1) # (B, 1) 
            # append sampled index to the running sequence 
            input_tensors = torch.cat((input_tensors, next_tensor), dim=1) # (B, T+1)
        return input_tensors
    
model = BigramLanguageModel(voc_size) 
m = model.to(device) 

# create a pytorch optimizer (optimizing parameters)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters): 
    if iter % eval_interval == 0: 
        losses = estimate_loss() 
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch  
    xb, yb = get_batch('train') 

    # evaluate the loss 
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model 
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))