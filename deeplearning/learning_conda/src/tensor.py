from torch import nn
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import random
'''
x=torch.zeros((2, 3, 4))
print(x)
print(x.shape)
print(x.dtype)
print(x.device)
print('\n')
y=torch.randn(3, 4)
print(y)
#####
help(torch.ones)'''
'''
def normal(x,mu,sigma):
    p=1/(math.sqrt(2*math.pi)*sigma)
    return p*np.exp(-(x-mu)**2/(2*sigma**2))
x=np.arange(-5,5,0.1)
y=normal(x,0,1)
plt.plot(x,y)
plt.show()'''
net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.1)
net[0].bias.data.fill_(0)
print(net)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
ephochs=30
for epoch in range(ephochs):
    inputs=torch.randn(4,2)
    targets=torch.tensor([0,1,2,3])
    optimizer.zero_grad()
    outputs=net(inputs)
    loss_val=loss(outputs,targets)
    loss_val.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{ephochs}, Loss: {loss_val.item():.4f}")



