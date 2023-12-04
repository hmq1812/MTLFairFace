import torch
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

def orthonormalize(vectors, normalize=True):
    """
    Orthonormalizes a set of vectors.
    Args:
        vectors (Tensor): The matrix of vectors to orthonormalize.
        normalize (bool): Whether to normalize the vectors.
    Returns:
        Tensor: The orthonormalized vectors.
    """
    assert vectors.size(1) <= vectors.size(0), 'Number of vectors must be smaller or equal to the dimension'

    if normalize:
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)

    for i in range(1, vectors.size(1)):
        vector = vectors[:, i]
        V = vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        if normalize:
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)

    return vectors

def project_vec(vec, proj_basis):
    """
    Projects a vector onto a basis.
    Args:
        vec (Tensor): The vector to project.
        proj_basis (Tensor): The basis to project onto.
    Returns:
        Tensor: The projected vector.
    """
    if proj_basis.shape[1] > 0:
        dots = torch.matmul(vec, proj_basis)
        out = torch.matmul(proj_basis, dots.T)
        return out
    else:
        return torch.zeros_like(vec)

def parameters_to_grad_vector(parameters):
    """
    Converts parameters to a single gradient vector.
    Args:
        parameters (Iterable): An iterable of parameters.
    Returns:
        Tensor: The concatenated gradient vector.
    """
    vec = []
    for param in parameters:
        if param.grad is not None:
            vec.append(param.grad.view(-1))
        else:
            vec.append(torch.zeros_like(param).view(-1))
    return torch.cat(vec)

def grad_vector_to_parameters(vec, parameters):
    """
    Converts a gradient vector back to parameters.
    Args:
        vec (Tensor): The gradient vector.
        parameters (Iterable): An iterable of parameters.
    """
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.grad = vec[pointer:pointer + num_param].view_as(param).clone()
        pointer += num_param

