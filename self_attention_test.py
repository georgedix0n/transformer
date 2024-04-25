import torch
import torch.nn as nn
#coupling batch items, each element coupled with previous contexts

torch.manual_seed(1337)

B, T, C = 4, 8, 32

x = torch.randn(B,T,C)
x.shape

#using very lossy coupling (avrge of prev contexts)

# averaging = torch.zeros((B, T, C))
# for b in range(B):
#     for t in range(T):
#         xprev = x[b, :t+1]
#         averaging[b,t] = torch.mean(xprev, 0)



torch.manual_seed(42)

#a more efficient version of the above commented out averaging loop

# weights = torch.tril(torch.ones(T,T))
# weights = weights / weights.sum(1, keepdim=True)
# averaging = weights @ x # (B,T,T) @(B,T,C) ----> (B,T,C)

#adds data dependence on 'averaging'; single head performing self-attention
head_size = 16
key = nn.Linear(C, head_size, bias = False)
query = nn.Linear(C, head_size, bias = False)
value = nn.Linear(C, head_size, bias = False)
k=key(x)
q=query(x)

weights = q @ k.transpose(-2,-1) #(B,T,16) @ (B,16,T) --> (B,T,T) 



#a variant using softmax

tril = torch.tril(torch.ones(T,T))
# weights = torch.zeros((T,T))
weights = weights.masked_fill(tril==0, float('-inf'))
weights = nn.functional.softmax(weights, dim=-1)
v = value(x)
averaging = weights @ v

print(averaging.shape)