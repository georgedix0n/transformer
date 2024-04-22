import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 100000
eval_interval = 10000
learning_rate = 1e-2

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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, offset, targets=None):
        logits = self.token_embedding_table(offset) # rank 3 tensor-> Block, time, channel, channel=vocab_size elements 
       
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
            logits, loss = self(offset)
            logits = logits[:, -1, :] # this only takes the element before target? not very efficient as passes whole seq in
            probs = nn.functional.softmax(logits, dim=1)
            offset_next = torch.multinomial(probs,num_samples=1)
            offset = torch.cat((offset,offset_next), dim=1)
        return offset
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

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