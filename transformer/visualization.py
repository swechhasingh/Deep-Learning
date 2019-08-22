from transformer import Transformer, positional_embedding, attention
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch

plt.figure(figsize=(15, 5))
x = Variable(torch.zeros(1, 10, 20), requires_grad=True)
y = positional_embedding(x, 20)
print(y.requires_grad)
print(x.requires_grad)

plt.plot(np.arange(10), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
plt.show()
x, a = attention(20, x, x, x, mask=None)
print(x)
print(a)
plt.figure(figsize=(5, 5))
plt.imshow(x[0].detach().numpy())
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(a[0].detach().numpy())
plt.show()

