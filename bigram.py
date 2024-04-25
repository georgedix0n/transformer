import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
number_of_embedding_dimensions = 384
num_heads = 6
n_layer = 6
dropout = 0.2

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Device: {device}")
eval_iters = 200


torch.manual_seed(1337)

with open('input.txt','r', encoding='utf-8') as f:
    input_text = f.read()

chars = sorted(list(set(input_text)))
vocab_size = len(chars)

#very simple tokenization
char_to_integer_mapping = {char:i for i, char in enumerate(chars)}
integer_to_char_mapping = { i:char for i, char in enumerate(chars)}
#encode returns list of integers for given list of chars (string)
encode = lambda string: [char_to_integer_mapping[char] for char in string] 
#decode returns string from list of integers
decode = lambda encoded_string: ''.join([integer_to_char_mapping[i] for i in encoded_string])



data = torch.tensor(encode(input_text), dtype=torch.long)

#defining train validation split (linear 90:10)
n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

def get_batch(split):
    data = train_data if split =='train' else validation_data
    random_offsets = torch.randint(len(data)-block_size, (batch_size,))
    batches = torch.stack([data[i:i+block_size] for i in random_offsets])
    blocks = torch.stack([data[i+1:i+block_size+1] for i in random_offsets])
    return batches.to(device), blocks.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(number_of_embedding_dimensions, head_size, bias=False)
        self.query = nn.Linear(number_of_embedding_dimensions, head_size, bias=False)
        self.value = nn.Linear(number_of_embedding_dimensions, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2,-1) * C**-0.5 #(B,T,16) @ (B,16,T) --> (B,T,T) 

        weights = weights.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        weights = nn.functional.softmax(weights, dim=-1)
        v = self.value(x)
        return weights @ v
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(number_of_embedding_dimensions, number_of_embedding_dimensions) #linked to residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, number_of_embedding_dimensions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(number_of_embedding_dimensions, 4* number_of_embedding_dimensions),
            nn.ReLU(),
            nn.Linear(4 * number_of_embedding_dimensions, number_of_embedding_dimensions),#projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block  """

    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd// num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)# (B,T,C) 'thinking on the data accumulated from the attention'
        self.layer_norm = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.self_attention(self.layer_norm(x)) # + is residual connection
        x = x + self.ffwd(self.layer_norm2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, number_of_embedding_dimensions)
        self.position_embedding_table = nn.Embedding(block_size, number_of_embedding_dimensions)
        self.blocks = nn.Sequential(
            *[Block(number_of_embedding_dimensions, num_heads=num_heads) for _ in range(n_layer)]
        )
        self.layernormfinal = nn.LayerNorm(number_of_embedding_dimensions)
        self.lm_head = nn.Linear(number_of_embedding_dimensions, vocab_size)


    def forward(self, offset, targets=None):

        B, T = offset.shape
        token_embedding = self.token_embedding_table(offset) # rank 3 tensor-> Block, time, channel, channel=vocab_size elements 
        positional_embedding = self.position_embedding_table(torch.arange(T,device=device)) #(T,C)
        x = token_embedding + positional_embedding #(B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x)#(B,T, vocab_size)
        
        if targets is None:
            loss = None
        else:
                
            B, T, C = logits.shape #reshaping for cross entropy input
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, offset, max_new_tokens):
        for _ in range(max_new_tokens):
            #prevent overflow
            offset_cond = offset[:,-block_size:]
            logits, loss = self(offset_cond)
            logits = logits[:, -1, :] # this only takes the element before target? not very efficient as passes whole seq in
            probs = nn.functional.softmax(logits, dim=1)
            offset_next = torch.multinomial(probs,num_samples=1)
            offset = torch.cat((offset,offset_next), dim=1)
        return offset
    


model = BigramLanguageModel()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    batch, block = get_batch('train')

    logits, loss = model(batch, block)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))