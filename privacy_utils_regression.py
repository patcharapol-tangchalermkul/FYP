import torch.nn.functional as F
import math
import logging
from typing import Literal, List
import torch
import scipy
import numpy as np

from abstract_gradient_training import test_metrics
from abstract_gradient_training.bounded_models import BoundedModel

from typing import Literal

def noisy_test_mse(
    model: torch.nn.Sequential | BoundedModel,
    batch: torch.Tensor,
    labels: torch.Tensor,
    *,
    noise_level: float | torch.Tensor = 0.0,
    noise_type: str = "laplace",
) -> float:
    """
    Given a pytorch (or bounded) model, calculate the prediction accuracy on a batch of the test set when adding the
    specified noise to the predictions.
    NOTE: For now, this function only supports binary classification via the noise + threshold dp mechanism. This
          should be extended to support multi-class problems via the noisy-argmax mechanism in the future.

    Args:
        model (torch.nn.Sequential | BoundedModel): The model to evaluate.
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        labels (torch.Tensor): Targets for the input batch (shape [batchsize, ]).
        noise_level (float | torch.Tensor, optional): Noise level for privacy-preserving predictions using the laplace
            mechanism. Can either be a float or a torch.Tensor of shape (batchsize, ).
        noise_type (str, optional): Type of noise to add to the predictions, one of ["laplace", "cauchy"].

    Returns:
        float: The noisy accuracy of the model on the test set.
    """
    # get the test batch and send it to the correct device
    if isinstance(model, BoundedModel):
        device = torch.device(model.device) if model.device != -1 else torch.device("cpu")
    else:
        device = torch.device(next(model.parameters()).device)
    batch = batch.to(device)

    # validate the labels
    if labels.dim() > 1:
        labels = labels.squeeze()
    labels = labels.to(device).type(torch.int64)
    assert labels.dim() == 1, "Labels must be of shape (batchsize, )"

    # validate the noise parameters and set up the distribution
    assert noise_type in ["laplace", "cauchy"], f"Noise type must be one of ['laplace', 'cauchy'], got {noise_type}"
    noise_level += 1e-7  # can't set distributions scale to zero
    noise_level = torch.tensor(noise_level) if isinstance(noise_level, float) else noise_level
    noise_level = noise_level.to(device).type(batch.dtype)  # type: ignore
    noise_level = noise_level.expand(labels.size())
    if noise_type == "laplace":
        noise_distribution = torch.distributions.Laplace(0, noise_level)
    else:
        noise_distribution = torch.distributions.Cauchy(0, noise_level)

    # # nominal, lower and upper bounds for the forward pass
    # logit_n = model.forward(batch).squeeze()

    # # transform 2-logit models to a single output
    # if logit_n.shape[-1] == 2:
    #     logit_n = logit_n[:, 1] - logit_n[:, 0]
    # if logit_n.dim() > 1:
    #     raise NotImplementedError("Noisy accuracy is not supported for multi-class classification.")

    # nominal, lower and upper bounds for the forward pass
    y_n = model.forward(batch).squeeze()

    # transform 2-logit models to a single output
    if y_n.shape[-1] == 2:
        y_n = y_n[:, 1] - y_n[:, 0]
    if y_n.dim() > 1:
        raise NotImplementedError("Noisy accuracy is not supported for multi-class classification.")

    # # apply noise + threshold dp mechanisim
    # y_n = (logit_n > 0).to(torch.float32).squeeze()
    # noise = noise_distribution.sample().to(y_n.device).squeeze()
    # assert noise.shape == y_n.shape
    # y_n = (y_n + noise) > 0.5
    # accuracy = (y_n == labels).float().mean().item()

    # apply noise + threshold dp mechanisim
    noise = noise_distribution.sample().to(y_n.device).squeeze()
    assert noise.shape == y_n.shape
    y_n = y_n + noise
    accuracy = F.mse_loss(y_n, labels.squeeze()).item()
    return accuracy

def get_calibrated_noise_level(
    batch: torch.Tensor,
    bounded_model_dict: dict[int, BoundedModel],
    max_bound: float,
    min_bound: float,
    epsilon: float,
    delta: float = 0.0,
    noise_type: Literal["cauchy", "laplace"] = "cauchy",
) -> torch.Tensor:
    """
    Compute the noise level calibrated to the smooth sensitivity bounds of each prediction in the batch. There are two
    possible mechanisms:

        - Adding Lap(2 * S(x) / epsilon) gives epsilon-delta dp.
        - Adding Cauchy(6 * S(x) / epsilon) gives epsilon dp.

    Args:
        batch (torch.Tensor): Input batch of data (shape [batchsize, ...]).
        bounded_model_dict (dict[int, BoundedModel): Dictionary of k: bounded_model values obtained from AGT for varying
            values of k_private.
        max_bound (float): Maximum of range
        min_bound (float): Minimum of range
        epsilon (float): Global privacy loss parameter.
        delta (float): Global privacy failure parameter.
        noise_type (Literal["cauchy", "laplace"]): Which noise mechanism to use.

    Returns:
        torch.Tensor: Noise level calibrated to the smooth sensitivity bounds of each prediction in the batch.
    """
    k_list = np.sort(list(bounded_model_dict.keys()))
    certified_matrix = compute_all_certified_k(batch, max_bound, min_bound, k_list, bounded_model_dict)
    if noise_type == "laplace":
        assert epsilon > 0 and 1 > delta > 0, "Epsilon must be positive."
        beta = epsilon / (2 * math.log(2 / delta))
        smooth_sens = compute_smooth_sensitivity(certified_matrix, k_list, beta)
        noise_level = 2 * smooth_sens / epsilon
    elif noise_type == "cauchy":
        assert epsilon > 0, "Epsilon must be positive."
        if delta > 0:
            LOGGER.debug("Ignoring delta > 0 for the Cauchy noise mechanism.")
        beta = epsilon / 6 - 1e-7
        smooth_sens = compute_smooth_sensitivity(certified_matrix, k_list, beta)
        noise_level = 6 * smooth_sens / epsilon
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return noise_level

def compute_all_certified_k(
    batch: torch.Tensor,
    max_bound: float,
    min_bound: float,
    k_list: List[int],
    bounded_model_dict: dict[int, BoundedModel]
) -> np.ndarray:
    """
    Returns a NumPy object array of lists containing certified k_private values for each point.
    """
    param_n_check = next(iter(bounded_model_dict.values())).param_n
    check_flag = False
    batch_size = batch.size(0)
    certified_k_lists = torch.tensor([[] for _ in range(batch_size)])

    batch_tensored = False

    for k in k_list:
        bounded_model = bounded_model_dict[k]
        if not batch_tensored:
            batch_tensored = True
            batch = batch.to(bounded_model.device).type(bounded_model.dtype)
            certified_k_lists = certified_k_lists.to(bounded_model.device).type(bounded_model.dtype)

        if not all(torch.allclose(a, b) for a, b in zip(bounded_model.param_n, param_n_check)):
            check_flag = True

        worst_case, best_case = bounded_model.bound_forward(batch, batch)
        nominal = bounded_model.forward(batch)

        # worst_case/best_case
        worst_case = torch.clamp(worst_case, min=min_bound, max=max_bound)
        best_case = torch.clamp(best_case, min=min_bound, max=max_bound)
        # print(worst_case - best_case)

        certified_k_lists = torch.cat((certified_k_lists, torch.max(abs(worst_case - nominal), abs(best_case - nominal))), axis=1)

    certified_k_lists = torch.cat((certified_k_lists, torch.full_like(nominal, max_bound - min_bound)), axis=1)
        
    #     l1 = torch.max(abs(worst_case - nominal), abs(best_case - nominal))
    #     l1_clamped = torch.clamp(l1, min=min_bound, max=max_bound)
    #     certified_k_lists = torch.cat((certified_k_lists, l1_clamped), axis=1)
    # print( torch.full_like(l1, 10))
    # certified_k_lists = torch.cat((certified_k_lists, torch.full_like(l1, 10)), axis=1)


    if check_flag:
        LOGGER.warning("Nominal parameters don't match for all k_private: check that you are seeding AGT correctly")
    return certified_k_lists

def compute_smooth_sensitivity(
    certified_matrix: np.ndarray,
    k_values: List[int],
    beta: float,
) -> torch.Tensor:
    """
    Compute the smooth sensitivity for each row in a 2D numpy array and return a PyTorch tensor.

    Args:
        certified_matrix (np.ndarray): 2D NumPy array of L1, shape [num_points, num_k_values].
        k_values (list[int]): The k values corresponding to each column of certified_matrix.
        beta (float): The beta-smooth sensitivity parameter.

    Returns:
        torch.Tensor: Smooth sensitivity values for each point (1D tensor).
    """
    
    assert certified_matrix.shape[1] == len(k_values) + 1, "Mismatch in k_values and matrix columns"

    k_values = torch.tensor(k_values)  # make sure k_values is a 1D tensor
    beta = torch.tensor(beta)          # beta should be a scalar tensor or float
    certified_matrix = certified_matrix.float()  # ensure same dtype if needed

    # Compute new_k
    new_k = k_values + 1
    new_k = torch.cat([torch.tensor([1]), new_k])

    # Compute exp_weights using PyTorch
    exp_weights = torch.exp(-beta * new_k).to(certified_matrix.device)  # shape: [num_k_values]
    
    # Reshape weights for broadcasting: [1, num_k_values]
    exp_weights = exp_weights.unsqueeze(0)

    # print(certified_matrix.device)
    # print(exp_weights.device)
    # Perform element-wise multiplication
    weighted_matrix = certified_matrix * exp_weights

    # Get max per row (dim=1)
    smooth_sensitivities = torch.max(weighted_matrix, dim=1).values
    return smooth_sensitivities