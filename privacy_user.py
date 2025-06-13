# This is a copy of the file, which should be directly imported into AGT

"""Certified privacy training."""

from __future__ import annotations
from collections.abc import Iterable
import logging
import gc

import torch

from abstract_gradient_training import training_utils
from abstract_gradient_training import interval_arithmetic
from abstract_gradient_training.gradient_accumulation import PrivacyGradientAccumulator
from abstract_gradient_training.configuration import AGTConfig
from abstract_gradient_training.bounded_models import BoundedModel

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def privacy_certified_training_user_level(
    bounded_model: BoundedModel,
    config: AGTConfig,
    dl_train: Iterable,
    dl_val: Iterable | None = None,
    dl_public: Iterable | None = None,
) -> BoundedModel:
    """
    Train the model with the given config and return a model with privacy-certified bounds on the parameters.

    Args:
        bounded_model (BoundedModel): Bounded version of the pytorch model to train.
        config (AGTConfig): Configuration object for the abstract gradient training module. See the configuration module
            for more details.
        dl_train (Iterable): Iterable of training data that returns (batch, users, labels) tuples at each iteration.
        dl_val (Iterable): Iterable for the validation data that returns (batch, users, labels) tuples at each iteration. This
            data is used to log the performance of the model at each training iteration.
        dl_public (Iterable, optional): Iterable of data considered 'public' for the purposes of privacy certification.

    Returns:
        BoundedModel: The trained model with the final bounds on the parameters.
    """
    # initialise hyperparameters, model, data, optimizer, logging
    bounded_model.to(config.device)
    optimizer = config.get_bounded_optimizer(bounded_model)
    training_utils.log_run_start(config, agt_type="privacy")

    # initialise the gradient accumulation class, which handles the logic of accumulating gradients across batch
    # fragments and computing the certified descent direction bounds.
    gradient_accumulator = PrivacyGradientAccumulator(config.k_private, config.clip_gamma, bounded_model.param_n)

    # returns an iterator of length n_epochs x batches_per_epoch to handle incomplete batch logic
    training_iterator = user_dataloader_pair_wrapper(dl_train, dl_public, config.n_epochs)
    val_iterator = training_utils.dataloader_cycle(dl_val) if dl_val is not None else None

    # main training loop
    for n, (batch, users, labels, batch_public, users_public, labels_public) in enumerate(training_iterator, 1):
        config.on_iter_start_callback(bounded_model)
        # possibly terminate early
        if config.early_stopping_callback(bounded_model):
            break

        # evaluate the network on the validation data and log the result
        if val_iterator is not None:
            loss = training_utils.compute_loss(bounded_model, config.get_val_loss_fn(), *next(val_iterator))
            LOGGER.info(f"Batch {n}. Loss ({config.val_loss}): {loss[0]:.3f} <= {loss[1]:.3f} <= {loss[2]:.3f}")

        # Initialize a dictionary to store the split data
        # The keys will be the user IDs
        # The values will be lists of (batch_sample, label_sample) tuples for that user
        split_data_by_user = {}

        # Iterate through the original batch and assign to respective user groups
        for i in range(batch.shape[0]):
            user_id = users[i].item() # .item() to get scalar from 0-dim tensor
            if user_id not in split_data_by_user:
                split_data_by_user[user_id] = {'batch': [], 'labels': []}

            split_data_by_user[user_id]['batch'].append(batch[i])
            split_data_by_user[user_id]['labels'].append(labels[i])
        
        # Convert the lists back to tensors for each user
        for user_id in split_data_by_user:
            split_data_by_user[user_id]['batch'] = torch.stack(split_data_by_user[user_id]['batch'])
            split_data_by_user[user_id]['labels'] = torch.stack(split_data_by_user[user_id]['labels'])
        
        # iterate through each user and compute the gradients
        for user_id, data in split_data_by_user.items():
            user_batch = data['batch']
            user_labels = data['labels']
            # compute the nominal gradients
            user_grads_n = training_utils.compute_batch_gradients(
                bounded_model, user_batch, user_labels, config, nominal=True
            )

            # compute the weight perturbed bounds
            user_grads_l, user_grads_u = training_utils.compute_batch_gradients(
                bounded_model, user_batch, user_labels, config, nominal=False
            )
            
            # find the mean of all gradients
            user_grads_n_mean = [torch.mean(g, dim=0, keepdim=True) for g in user_grads_n]
            user_grads_l_mean = [torch.mean(g, dim=0, keepdim=True) for g in user_grads_l]
            user_grads_u_mean = [torch.mean(g, dim=0, keepdim=True) for g in user_grads_u]
                    
            
            # apply gradient clipping
            user_grads_l_mean, user_grads_n_mean, user_grads_u_mean = training_utils.propagate_clipping(
                user_grads_l_mean, user_grads_n_mean, user_grads_u_mean, config.clip_gamma, config.clip_method
            )
            
            # accumulate the gradients
            gradient_accumulator.add_private_fragment_gradients(user_grads_n_mean, user_grads_l_mean, user_grads_u_mean)
            
            # the gpu memory allocations at each loop are not always collected, so we'll prompt pytorch to do so
            gc.collect()
            torch.cuda.empty_cache()

        # get the bounds on the descent direction, validate them, and apply the optimizer update
        batchsize = batch.size(0) if batch_public is None else batch.size(0) + batch_public.size(0)
        update_l, update_n, update_u = gradient_accumulator.concretize_gradient_update(batchsize)
        interval_arithmetic.validate_interval(update_l, update_u, update_n, "final_grad_bounds")
        optimizer.step(update_l, update_n, update_u)
        config.on_iter_end_callback(bounded_model)
    if val_iterator is not None:
        loss = training_utils.compute_loss(bounded_model, config.get_val_loss_fn(), *next(val_iterator))
        LOGGER.info(f"Final Eval. Loss ({config.val_loss}): {loss[0]:.3f} <= {loss[1]:.3f} <= {loss[2]:.3f}")

    training_utils.validate_bounded_model(bounded_model)
    LOGGER.info("=================== Finished Privacy Certified Training ===================")

    return bounded_model


def user_dataloader_pair_wrapper(
    dl_train: Iterable, dl_aux: Iterable | None, n_epochs: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
    """
    Return a new generator that iterates over the training dataloaders for a fixed number of epochs.
    The first dataloader contains the standard training data, while the second dataloader contains auxiliary data,
    which is e.g. clean data for poisoning or public data for privacy.
    For each combined batch, we return one batch from the clean dataloader and one batch from the poisoned dataloader.
    This includes a check to ensure that each batch is full and ignore any incomplete batches.
    We assume the first batch is full to set the batchsize and this is compared with all subsequent batches.

    Args:
        dl_train (Iterable): Dataloader that returns (batch, labels) tuples at each iteration.
        dl_aux (Iterable | None): Optional additional dataloader for auxiliary data that returns (batch, labels)
            tuples at each iteration.
        n_epochs (int): Maximum number of epochs.

    Yields:
        batch, labels, batch_aux, labels_aux: Tuples of post-processed (batch, labels, batch_aux, labels_aux)
            for each iteration.
    """
    # batchsize variable will be initialised in the first iteration
    full_batchsize = None
    # loop over epochs
    for n in range(n_epochs):
        LOGGER.info("Starting epoch %s", n + 1)
        # handle the case where there is no auxiliary dataloader by returning dummy values
        if dl_aux is None:
            data_iterator: Iterable = (((b, u, l), (None, None, None)) for b, u, l in dl_train)
        else:
            data_iterator = zip(dl_train, dl_aux)  # note that zip will stop at the shortest iterator
        t = -1  # possibly undefined loop variable
        for t, ((batch, user, labels), (batch_aux, user_aux, labels_aux)) in enumerate(data_iterator):
            batchsize = batch.size(0)
            if batch_aux is not None:
                batchsize += batch_aux.size(0)
            # initialise the batchsize variable if this is the first iteration
            if full_batchsize is None:
                full_batchsize = batchsize
                LOGGER.debug("Initialising dataloader batchsize to %s", full_batchsize)
            # check the batch is the correct size, otherwise skip it
            if batchsize != full_batchsize:
                LOGGER.debug(
                    "Skipping batch %s in epoch %s (expected batchsize %s, got %s)",
                    t + 1,
                    n + 1,
                    full_batchsize,
                    batchsize,
                )
                continue
            # return the batches for this iteration
            yield batch, user, labels, batch_aux, user_aux, labels_aux
        # check the number of batches we have processed and report the appropriate warnings
        assert t != -1, f"Dataloader is empty at epoch {n + 1}!"
        if n == 0 and t == 0:
            LOGGER.info("Dataloader has only one batch per epoch, effective batchsize may be smaller than expected.")