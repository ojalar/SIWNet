import torch

def trunc_gauss_log_loss(mean, std, label):
    # implementation of the maximum likelihood loss function for truncated gaussian

    # initialise tensors for a & b, which are the boundaries of the truncated gaussian
    a = torch.zeros_like(mean)
    b = torch.ones_like(mean)
    sqrt2 = torch.sqrt(torch.mul(torch.ones_like(mean), 2))
    # clamp std for stability
    std = torch.clamp(std, min = 1e-4)
    p1 = torch.log(std)
    p2 = torch.mul(0.5, torch.square(torch.div(torch.sub(mean, label), std)))
    p3 = torch.log(torch.sub(torch.erf(torch.div(torch.sub(b, mean), torch.mul(std, sqrt2))), 
        torch.erf(torch.div(torch.sub(a, mean), torch.mul(std, sqrt2)))))
    
    return torch.sum(torch.add(torch.add(p3, p2), p1))
    
