import torch

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
