import torch
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt
import random
import sklearn.preprocessing as sk
import numpy as np

def calibrate1d(x, xp, yp):
    """
    x: [N, C]
    xp: [C, K]
    yp: [C, K]
    """
    x_breakpoints = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((x.shape[0], 1, 1))], dim=2)
    num_x_points = xp.shape[1]
    sorted_x_breakpoints, x_indices = torch.sort(x_breakpoints, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, num_x_points), torch.tensor(num_x_points - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_x_breakpoints, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_x_breakpoints, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, num_x_points), torch.tensor(num_x_points - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(x.shape[0], -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x + 1e-7)
    return cand

def random_sort_rows(tensor):
    """
    Randomly sorts the rows of a 2D PyTorch tensor either in ascending or descending order.

    Parameters:
    tensor (torch.Tensor): An n x k tensor.

    Returns:
    torch.Tensor: The tensor with rows randomly sorted.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D.")

    # Randomly choose ascending or descending for each row
    directions = torch.randint(0, 2, (tensor.size(0),), dtype=torch.bool).to(tensor.device)

    # Sort the tensor and get the sorted indices
    sorted_tensor, _ = torch.sort(tensor, dim=1)

    # Use the directions to either return the sorted tensor or its reverse
    sorted_rows = torch.where(directions.unsqueeze(1), sorted_tensor, sorted_tensor.flip(1))

    return sorted_rows

class Generator(nn.Module):
    def __init__(self,n_batch = 4, n_points_pwl=20, latent_dim = 8, noise_level=1.0):
        super(Generator, self).__init__()

        self.n_batch = n_batch
        self.n_points_pwl = n_points_pwl
        self.latent_dim = latent_dim
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.gen_x = self.mlp([self.latent_dim,8,1]) # 8
        self.gen_e = self.mlp([self.latent_dim,8,1]) # 8

        self.gen_fun = self.mlp([self.latent_dim,8,self.n_points_pwl]) #8


        #self.noise_level = torch.nn.Parameter(noise_level*torch.ones(1), requires_grad=True)




    def mlp(self, architecture):
        prev_neurones = architecture[0]
        net = []
        for neurones in architecture[1:]:
            net.append(nn.Linear(prev_neurones, neurones))
            net.append(nn.Tanh())
            prev_neurones = neurones

        return nn.Sequential(*net)    



    def gen_f(self, z):                   
        
        pts = random_sort_rows(self.gen_fun(z))
        self.xp = torch.range(0.0,1,1/(self.n_points_pwl-1)).repeat(self.n_batch ,1).to(z.device)
        self.yp = pts 



    def forward_f(self,x):
        return calibrate1d(x, self.xp, self.yp) 

    def forward_x(self,z):    
        x = self.gen_x.forward(z).squeeze(2).permute(1,0)
        x = self.normalize01(x.permute(1,0)).permute(1,0)  
        return  x 

    def forward_e(self,z):
        e = self.gen_e.forward(z).squeeze(2).permute(1,0) 
        e = e - e.mean()
        e = e / e.std()
        return e 

    def forward_y(self,x,ze):        
        y = self.forward_f(x) + 0.1*self.forward_e(ze)  
        return y

    def normalize01(self,AA):
        AA = AA.view(AA.size(0), -1)
        AA = AA - AA.min(1, keepdim=True)[0]
        AA = AA / AA.max(1, keepdim=True)[0]
        return AA


if __name__ == "__main__":
    print("Test")
    nb = 8
    pts = 1024 
    latent_dim = 10
    pts_pwl = 4

    G = Generator(n_batch=nb, n_points_pwl=pts_pwl, latent_dim=latent_dim)
    Zx = torch.randn(nb, pts, latent_dim)
    Ze = torch.randn(nb, pts, latent_dim) 
    Zf = torch.randn(nb, latent_dim) 
    
    G.gen_f(Zf)
    x = G.forward_x(Zx)   
    y = G.forward_y(x,Ze) 


    x = x.detach().numpy()
    y = y.detach().numpy()
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, nb+1):
        ax = fig.add_subplot(2, 4, i)
        n1 = np.random.randint(128, pts)
        #print(n1)
        ax.plot(x[:n1,i-1],y[:n1,i-1],'r.')
        xp = G.xp #G.normalize01(G.xp)
        yp = G.yp #G.normalize01(G.yp)
        ax.plot(xp[i-1,:].detach().numpy(), yp[i-1,:].detach().numpy(),'b-')    
    plt.show()


