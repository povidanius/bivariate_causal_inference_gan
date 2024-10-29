import torch
import matplotlib.pyplot as plt


def laplacian_regularization_with_multivariate_inputs(x, y, sigma=1.0, normalized=True):
    """
    Computes the Laplacian regularization term given multivariate inputs x and multivariate outputs y.
    
    Args:
        x (torch.Tensor): Tensor of shape (num_nodes, input_dim) representing the input features.
        y (torch.Tensor): Tensor of shape (num_nodes, output_dim) representing the multivariate output values (predictions).
        sigma (float): The bandwidth parameter for the Gaussian kernel used to compute the weight matrix.
        normalized (bool): Whether to use the normalized Laplacian regularization.
    
    Returns:
        torch.Tensor: The computed Laplacian regularization term (scalar).
    """
    num_nodes = x.size(0)
    input_dim = x.size(1)
    output_dim = y.size(1)

    # Compute pairwise squared Euclidean distances between input points (multivariate x)
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # Shape: (num_nodes, num_nodes, input_dim)
    dist_sq = torch.sum(diff ** 2, dim=2)   # Shape: (num_nodes, num_nodes)

    # Compute the weight matrix using a Gaussian kernel
    W = torch.exp(-dist_sq / (2 * sigma ** 2))  # Shape: (num_nodes, num_nodes)

    # Compute the degree matrix
    D = torch.diag(torch.sum(W, dim=1))  # Shape: (num_nodes, num_nodes)

    # Compute the Laplacian matrix
    L = D - W  # Shape: (num_nodes, num_nodes)

    reg_term = 0.0  # Initialize regularization term
    
    # Loop over each output dimension to compute regularization term
    for d in range(output_dim):
        y_d = y[:, d].view(-1, 1)  # Select the d-th dimension of y (Shape: (num_nodes, 1))

        if normalized:
            # Compute D^(-1/2)
            D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-6))  # Add epsilon for numerical stability

            # Compute normalized predictions for this output dimension
            y_normalized = torch.matmul(D_inv_sqrt, y_d)

            # Compute the normalized Laplacian regularization: (D^(-1/2) * y)^T L (D^(-1/2) * y)
            reg_term += torch.matmul(y_normalized.t(), torch.matmul(L, y_normalized)).squeeze()
        else:
            # Compute the unnormalized Laplacian regularization: y^T L y for this output dimension
            reg_term += torch.matmul(y_d.t(), torch.matmul(L, y_d)).squeeze()

    # Return the scalar value (sum over all output dimensions)
    return reg_term / 2.0

if __name__ == "__main__":
    x = torch.rand(512,1)
    y = x*x  + 0.1*torch.randn(512,1)

    sigma = 0.1
    delta_xy = laplacian_regularization_with_multivariate_inputs(x,y, normalized=True, sigma=sigma)
    delta_yx = laplacian_regularization_with_multivariate_inputs(y,x, normalized=True, sigma=sigma)

    print(delta_xy)
    print(delta_yx)
    
    plt.plot(x,y,'+'); plt.show()


    breakpoint()