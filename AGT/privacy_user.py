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
    training_iterator = training_utils.dataloader_pair_wrapper(dl_train, dl_public, config.n_epochs)
    val_iterator = training_utils.dataloader_cycle(dl_val) if dl_val is not None else None

    # main training loop
    for n, (batch, users, labels, batch_public, labels_public) in enumerate(training_iterator, 1):
        config.on_iter_start_callback(bounded_model)
        # possibly terminate early
        if config.early_stopping_callback(bounded_model):
            break

        # evaluate the network on the validation data and log the result
        if val_iterator is not None:
            loss = training_utils.compute_loss(bounded_model, config.get_val_loss_fn(), *next(val_iterator))
            LOGGER.info(f"Batch {n}. Loss ({config.val_loss}): {loss[0]:.3f} <= {loss[1]:.3f} <= {loss[2]:.3f}")

        # NOTE: We have a lot of per-sample operations that must be performed on the entire batchsize, with typically
        # very large batchsizes. To conserve memory, we split the batch into fragments and process each fragment
        # separately and accumulate the gradients across fragments using the gradient_accumulator classes.

        # process the "public" data first
        batch_fragments = torch.split(batch_public, config.fragsize, dim=0) if batch_public is not None else []
        label_fragments = torch.split(labels_public, config.fragsize, dim=0) if labels_public is not None else []
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            # compute the nominal gradients
            frag_grads_n = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=True
            )
            # compute the weight perturbed bounds
            frag_grads_l, frag_grads_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False
            )
            # apply gradient clipping
            frag_grads_l, frag_grads_n, frag_grads_u = training_utils.propagate_clipping(
                frag_grads_l, frag_grads_n, frag_grads_u, config.clip_gamma, config.clip_method
            )
            # accumulate the gradients
            gradient_accumulator.add_public_fragment_gradients(frag_grads_n, frag_grads_l, frag_grads_u)
            # the gpu memory allocations at each loop are not always collected, so we'll prompt pytorch to do so
            gc.collect()
            torch.cuda.empty_cache()

        # stores the number of entries for the user so far
        user_count = {}
        # stores the user gradients [lower, nominal, upper]
        user_grads = {}

        # process the "private" data, taking the appropriate bounds
        # split the batch into fragments to avoid running out of GPU memory
        batch_fragments = torch.split(batch, config.fragsize, dim=0)
        users_fragments = torch.split(users, config.fragsize, dim=0)
        label_fragments = torch.split(labels, config.fragsize, dim=0)
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            # compute the nominal gradients
            frag_grads_n = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=True
            )

            # compute the weight perturbed bounds
            frag_grads_l, frag_grads_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False
            )

            # TODO: check dtype, turn pseudocode to code
            frag_size = len(frag_grads_n)
            user_sum = {}
            num_user = {}
            for i in range(frag_size):
                user = users_fragments[i]
                if user not in user_sum:
                    user_sum[user] = [0, 0, 0]
                    num_user[user] = 0

                user_sum[user] = [sum(x) for x in zip(user_sum[user], [frag_grads_l[i], frag_grads_n[i], frag_grads_u[i]])]
                num_user[user] += 1

            for u in user_sum:
                if u not in user_grads:
                    user_grads[u] = [0, 0, 0]
                    user_count[u] = 0
                user_grads[u] = [sum(x) for x in zip(user_grads[u], user_sum[user])]
                user_count[u] += num_user[u]


        for u in user_grads:
            user_grads[u] = [x / user_count[u] for x in user_grads[u]]


            # Assuming that ther is no change to the ordering


            # [u11-u19, u21-u28]
            # [u11, u12, u13, u21, u22]
            # []

        # batch [u1, u2, u3]

        # process the "private" data, taking the appropriate bounds
        # split the batch into fragments to avoid running out of GPU memory
        batch_fragments = torch.split(batch, config.fragsize, dim=0)
        label_fragments = torch.split(labels, config.fragsize, dim=0)
        for batch_frag, label_frag in zip(batch_fragments, label_fragments):
            # compute the nominal gradients
            frag_grads_n = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=True
            )
            # compute the weight perturbed bounds
            frag_grads_l, frag_grads_u = training_utils.compute_batch_gradients(
                bounded_model, batch_frag, label_frag, config, nominal=False
            )

        # for grads in user_grads:

        #     frag_grads_l, frag_grads_n, frag_grads_u = grads[0], grads[1], grads[2]
            
            # apply gradient clipping
            frag_grads_l, frag_grads_n, frag_grads_u = training_utils.propagate_clipping(
                frag_grads_l, frag_grads_n, frag_grads_u, config.clip_gamma, config.clip_method
            )
            # accumulate the gradients
            gradient_accumulator.add_private_fragment_gradients(frag_grads_n, frag_grads_l, frag_grads_u)
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
