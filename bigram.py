import torch 
import torch.nn as nn 
from torch.nn import functional as F 

# hyperparameters 
batch_size = 32 # how many independent sequences will we process in parallel 
block_size = 8 # what is the maximum content length for predictions 
max_iters = 3000 
eval_interval = 300
learning_rate = 1e-2 
device = 'cude' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 

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

class BigramLanguageModel(nn.Module): 
    def __init__(self, vocab_size): 
        super().__init__() 
        # taking a tensor input (word ids, tokenized) into vector representations 
        # num_embeddings means how large is your vocabulary (in this case 65 unique symbols) 
        # embedding_dim means how long is the vector representation for each symbols (each symbol will be represented in 65 dimension vector)
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size) 

    def forward(self, input_tensors, target_tensors=None): 
        # forward pass without activation function
        logits = self.token_embedding_table(input_tensors) # (B,T,C)
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
            # get the predictions 
            logits, loss = self(input_tensors) 
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