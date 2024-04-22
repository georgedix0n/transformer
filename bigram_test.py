import torch
import torch.nn as nn

with open('input.txt', 'r', encoding='utf-8') as f:
    input_text = f.read()

print(f"Length of dataset (characters): {len(input_text)}")

chars = sorted(list(set(input_text)))
vocab_size = len(chars)
print(f"chars used: {''.join(chars)}")
print(f"vocab size: {vocab_size}")

#very simple tokenization
char_to_integer_mapping = {char:i for i, char in enumerate(chars)}
integer_to_char_mapping = { i:char for i, char in enumerate(chars)}
#encode returns list of integers for given list of chars (string)
encode = lambda string: [char_to_integer_mapping[char] for char in string] 
#decode returns string from list of integers
decode = lambda encoded_string: ''.join([integer_to_char_mapping[i] for i in encoded_string])

data = torch.tensor(encode(input_text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

#defining train validation split (linear 90:10)
n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

#block passed into transformer n+1, n+1th element is target element where sequence leads to
block_size = 8
train_data[:block_size+1]

x=train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")


torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split =='train' else validation_data
    random_offsets = torch.randint(len(data)-block_size, (batch_size,))
    batches = torch.stack([data[i:i+block_size] for i in random_offsets])
    blocks = torch.stack([data[i+1:i+block_size+1] for i in random_offsets])
    return batches, blocks

batch, block = get_batch('train')

for b in range(batch_size):
    for t in range(block_size):
        context = batch[b, :t+1]
        target = block[b,t]
        print(f"When input is {context.tolist()} the target: {target}")

print(batch)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # rank 3 tensor-> Block, time, channel, channel=vocab_size elements 
       
        if targets is None:
            loss = None
        else:
                
            B, T, C = logits.shape #reshaping for cross entropy input
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # this only takes the element before target? not very efficient as passes whole seq in
            probs = nn.functional.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next), dim=1)
        return idx
    
model = BigramLanguageModel(vocab_size)
logits, loss = model(batch, block)

print(logits.shape)
print(loss)

idx = torch.zeros((1,1), dtype=torch.long) 
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32

for steps in range(100000):
    batch, block = get_batch('train')

    logits, loss = model(batch, block)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

idx = torch.zeros((1,1), dtype=torch.long) 
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))