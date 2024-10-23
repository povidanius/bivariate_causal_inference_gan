import torch
import torch.nn as nn
from typing import Union
from torch import FloatTensor
from torch.autograd import Variable

NetIO = Union[FloatTensor, Variable]


class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x: NetIO) -> NetIO:
        # compute the representation for each data point
        x = self.phi.forward(x)


        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.

        x = torch.mean(x, dim=1, keepdim=True)
        #x = torch.quantile(x, 0.75, dim=1, keepdim=True)

        # compute the output
        out = self.rho.forward(x)

        return out

class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim = 2, encoder_dim = 20):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim

        self.fc1 = nn.Linear(input_dim, 15)        
        self.fc2 = nn.Linear(15, self.encoder_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x: NetIO) -> NetIO:
        #print("shape in {}".format(x.shape))

        #print(x.shape)
        x = x.permute(0,2,1)
        #print(x.shape)
        x = self.relu(self.fc1(x))  
        x = self.fc2(x)
        #print("encoder shape {}".format(x.shape))
        return x

class DeepSetDecoder(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, self.output_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x: NetIO) -> NetIO:
        #print("decoder input {}".format(x.shape))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim = 1, output_dim = 1, encoder_dim = 15):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_dim = encoder_dim
        self.relu = nn.ReLU()

        self.phi = DeepSetEncoder(input_dim = self.input_dim + self.output_dim, encoder_dim = self.encoder_dim)
        self.rho = DeepSetDecoder(input_size=self.encoder_dim, output_size=1)
        self.model = InvariantModel(phi=self.phi, rho=self.rho)


    def forward(self,x,y):  
        xy = torch.cat((x,y), dim=1)
        yx = torch.cat((y,x), dim=1)

        y_f = self.model(xy)
        y_b = self.model(yx)
        return y_f, y_b


    def normalize01(self,AA):
        AA = AA.view(AA.size(0), -1)
        AA = AA - AA.min(1, keepdim=True)[0]
        AA = AA / AA.max(1, keepdim=True)[0]
        return AA

    def loss(self,x,y):
        f,b = self.forward(x,y)      
        loss = (1 - f + b).mean()       
        return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":

    batch_size = 4
    dim_x = 1
    dim_y = 1
    num_points = 1024
    X = torch.randn(batch_size, dim_x, num_points)
    Y = torch.randn(batch_size, dim_y, num_points)

    D = Discriminator(input_dim = dim_x, output_dim = dim_y)    
    D.eval()    
    
    out_f, out_b = D.forward(X,Y)    
    print("Forward:")
    print(out_f)
    print("Backward:")
    print(out_b)    


