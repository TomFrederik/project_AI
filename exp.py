import torch
from time import time as t
def f(): 
    a = torch.ones(100,150)
    a = a.to('cuda')
    time1=t()
    for i in range(20): 
        torch.argmax(a,dim=1)
    print((t()-time1)/20)
    time1=t()
    for i in range(20): 
        torch.argmax(a.detach(),dim=1)
    print('detach',(t()-time1)/20)
f()
