# This loss functions makes use of a distance metric between n vectors simultaneously. This is done by forming a 
# low rank approximation of matrix (of shape d x n) consisting of the n vectors normalized by the euclidean distance. 
# Specifically, it is the sum of the K largest singular values of the matrix (a natural choice is K=1 since that can 
# be normalized to fall within [0,1]). The largest singular value squared of such a matrix is n when all vectors are linearly 
# dependent and 1 when they are orthogonal.

# Curretly, power iteration is used to compute the biggest singular values to increase speed (thus K is always 1).

# TODO:
# Problem: when two vectors correspond to the same material but the third is different, the loss still forces 
# this to be zero although it should probably be higher --> set these to 1/2 or 2/3
# This is the diagonal on each slice. 
# NOTE: Not only the diagonal element in each slice should be non-zero but also (for slice i) elements (i,i,x) and (i,x,i)

# TODO: compare differnt alignment staregies (e.g anchored clip with this), when only fitting linear head to the 
# representatoin and freezinf the rest. Then the differance will be bigger and more down to the representation
# (i.e this low rank approximation might not work at all but if all parameters are finetuned it shouldn't be wors 
# than training from scratch. Since the differance in performace is very small this would be problematic)

# TODO: When computing the singular values for a matrix made up of embedding vectors, the ordering of embedding 
# vectors doesn't matter for the singular values. Thus, it's enough to compute the largest singular value for 
# all combinations of embedding vectors and not for all permutations

import torch
import torch.nn.functional as F
from ..loss.infonce import gather_from_all

def power_iteration_singular_values(A, num_iterations=15, epsilon=1e-2):
    """Find the bigest singular value of A using power iteration on A^T A (the 
    largest eigenvalue of A^T A is equal to the largest singular value of A squared).
    We actually want the square or the largest singular value so that's what's returned
    Note: The largest singular value squared is computed"""

    # reshape A to have a single batch dimension
    *batch_dims, m, n = A.shape
    A = A.reshape(-1, m, n)
    batch_size = A.shape[0]

    # randomly initialize guesses for the eigenvector corresponding to the largest eigenvalue
    v = torch.rand(batch_size, n, device=A.device, dtype=A.dtype)
    v /= v.norm(dim=-1, keepdim=True)
    
    # initial guess for the largest singular value squared
    sigma_old = torch.zeros(batch_size, device=A.device, dtype=A.dtype)

    # power iteration
    for _ in range(num_iterations):
        # compute updated approximation of eigenvectors
        Av = torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
        v_new = torch.bmm(A.transpose(1, 2), Av.unsqueeze(-1)).squeeze(-1)
        v_new /= v_new.norm(dim=-1, keepdim=True)

        # compute updated approximation of largest singular values squared
        sigma_new = (v_new * torch.bmm(A.transpose(1, 2), Av.unsqueeze(-1)).squeeze(-1)).sum(-1)
        
        # Check for convergence based on relative change in singular values aquared
        if torch.max(torch.abs((sigma_new - sigma_old) / sigma_new)) < epsilon:
            break
            
        sigma_old = sigma_new
        v = v_new

    return sigma_new.reshape(*batch_dims)


def low_rank_approximation_loss(embeddings, args):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # K is the rank, i.e how many singular values we want to use
    # # When using power iteration, K isn't used and must be equal to one since it only computes the largest singular value
    # K = 1 

    embeddings_proj_norm = {}
    modalities_list = list(embeddings.keys())
    num_modalities = len(modalities_list)
    num_samples = embeddings[modalities_list[0]].shape[0]
    embedding_dim = embeddings[modalities_list[0]].shape[1]

    # get normalized projected embeddings
    for modality in modalities_list:
        # projection_matrix_modality = proj_matrices[modality]
        embeddings_modality = F.normalize(embeddings[modality], dim=-1)
        # embeddings_modality_proj = F.normalize(projection_matrix_modality(embeddings_modality), dim=-1)
        # embeddings_proj_norm[modality] = embeddings_modality_proj
        embeddings_proj_norm[modality] = embeddings_modality

    # extend the dimensions of each tensor for broadcasting
    for i, modality in enumerate(modalities_list):
        if args.distribute:
            embeddings_proj_norm[modality] = gather_from_all(embeddings_proj_norm[modality])
        num_samples = embeddings_proj_norm[modality].shape[0]
        shape = [1] * num_modalities
        shape[i] = num_samples
        embeddings_proj_norm[modality] = embeddings_proj_norm[modality].reshape(*shape, embedding_dim)
        
            
    # get all permutations of embedding vectors and sort into a tensor
    embedding_tensor = torch.empty(*([num_samples] * num_modalities), embedding_dim, num_modalities).cuda()
    for i, modality in enumerate(modalities_list):
        embedding_tensor[..., i] = embeddings_proj_norm[modality]

    # # calculate singular values using torch
    # singular_values = torch.linalg.svdvals(embedding_tensor)[..., :K]      # only want the K biggest singular values (K is typically 1)
    # singular_values = torch.sum(singular_values, dim=-1)                   # we get the low rank approximation by summing singular values
    # singular_values_squared = singular_values ** 2
    # TODO: If torch is used to compute the singular values, they need to be squared

    # it's must faster to compute the largest singular value squared using power iteration
    # NOTE: the largest singular value squared is computed
    singular_values_squared = power_iteration_singular_values(embedding_tensor)    

    # compute similarity scores and normalize to [0,1] (1 is when all vectors are linearly dependent and 0 when they are orthogonal)
    similarity_scores = (singular_values_squared - 1) / (num_modalities - 1)

    # create target tensor (1 on hyperdiagonal and 0 elsewhere)
    target = torch.zeros([num_samples]*num_modalities).cuda()
    # target.fill_diagonal_(1) # set hyper diagonal to 1
    for i in range(b):
        target[i, i, :] = 0.5
        target[i, :, i] = 0.5
    target.diagonal(dim1=2,dim2=1).copy_(target.diagonal(dim1=2,dim2=1).add_(0.5))

    # indices = torch.arange(num_samples, device=device)
    # target = torch.zeros([num_samples]*num_modalities, device=device)
    # target[tuple(indices for _ in range(num_modalities))] = 1

    # # where num_modalities - 1 indices are equal, set target to 1/2
    # # TODO: generalize to any number of modalities0
    # for i in range(num_samples):
    #     for j in range(num_samples):
    #         for k in range(num_samples):
    #             idx_sum = (i == j) + (i == k) + (j == k)
    #             if idx_sum == 1:
    #                 target[i, j, k] = 1/2

    # compute loss
    loss = F.mse_loss(similarity_scores, target)

    return loss


# Stuff to try -------------------------------------------

# def custom_logit_loss(output, w_diag=1.0, w_off_diag=1.0):
#     # Extracting the diagonal elements
#     diagonal = output[torch.arange(output.size(0)), torch.arange(output.size(0)), torch.arange(output.size(0))]
    
#     # Getting the off-diagonal elements using a mask
#     mask = torch.ones_like(output)
#     mask[torch.arange(output.size(0)), torch.arange(output.size(0)), torch.arange(output.size(0))] = 0
#     off_diagonal = output * mask

#     # Loss for diagonal elements: encourage them to be close to 1
#     loss_diagonal = -torch.log(diagonal)
    
#     # Loss for off-diagonal elements: encourage them to be close to 0
#     loss_off_diagonal = -torch.log(1 - off_diagonal)

#     # Combine the two loss components with weights
#     return w_diag * torch.mean(loss_diagonal) + w_off_diag * torch.mean(loss_off_diagonal)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

# # Initialize the weights for diagonal and off-diagonal losses. 
# # They're wrapped in Variable() to enable gradient computations.
# w_diag = Variable(torch.tensor(1.0), requires_grad=True)
# w_off_diag = Variable(torch.tensor(1.0), requires_grad=True)

# def custom_logit_loss_with_learned_weights(output):
#     # Extracting the diagonal elements
#     diagonal = output[torch.arange(output.size(0)), torch.arange(output.size(0)), torch.arange(output.size(0))]
    
#     # Getting the off-diagonal elements using a mask
#     mask = torch.ones_like(output)
#     mask[torch.arange(output.size(0)), torch.arange(output.size(0)), torch.arange(output.size(0))] = 0
#     off_diagonal = output * mask

#     # Loss for diagonal elements
#     loss_diagonal = -torch.log(diagonal)
    
#     # Loss for off-diagonal elements
#     loss_off_diagonal = -torch.log(1 - off_diagonal)

#     # Combine with weights
#     combined_loss = torch.exp(w_diag) * torch.mean(loss_diagonal) + torch.exp(w_off_diag) * torch.mean(loss_off_diagonal)
    
#     return combined_loss

# # Mock output tensor for demonstration
# output = torch.rand((10, 10, 10), requires_grad=True)

# # Compute loss
# loss = custom_logit_loss_with_learned_weights(output)

# # Backpropagate
# loss.backward()

# # Print gradients (just for demonstration)
# print(w_diag.grad)
# print(w_off_diag.grad)

# # Here, you'd typically use an optimizer to update both the model parameters and the loss weights.


if __name__ == '__main__':
    # # test similarities
    # import torch.nn.init as init
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # A = torch.empty(5, 128, device=device)
    # # init.orthogonal_(A)
    # A = torch.eye(2)
    # embeddings = {'crystal': A, 'dos': A}
    # proj_matrices = {'crystal': torch.nn.Identity(), 'dos': torch.nn.Identity(), 'charge': torch.nn.Identity()}
    # loss = low_rank_approximation_loss(embeddings, proj_matrices, None)

    # # test power iteration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # A = torch.rand(100, 100, 3, 128, device=device)

    # singular_value_power_iter = power_iteration_singular_values(A, num_iterations=30, epsilon=1e-9)
    # singular_value_pytorch = torch.linalg.svdvals(A, driver='gesvda')[..., 0] ** 2

    # print(torch.isclose(singular_value_power_iter, singular_value_pytorch, atol=1e-4, rtol=0).all())
    # print(singular_value_power_iter.shape)
    pass