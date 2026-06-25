import gc
import math
import os
import time
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

import sevenn._keys as KEY
from sevenn.error_recorder import AverageNumber, ErrorRecorder
from sevenn.logger import Logger
from sevenn.train.trainer import Trainer
from sevenn.util import unique_filepath


def update_config_for_batch_training(config: Dict[str, Any], loaders) -> None:
    """
    Called from sevenn/scripts/train.py, between dataset build and model build
    Update scheduler parameters for batch-level training.

    This converts epoch-based scheduler parameters to step-based parameters
    when using batch training mode.
    """
    train_loader = loaders['trainset']
    # convert float type `epoch` related parameters for batch training
    effective_batch_size = config[KEY.WORLD_SIZE] * config[KEY.BATCH_SIZE]
    steps_per_epoch = math.ceil(
        train_loader.sampler.total_size / effective_batch_size
    )

    scheduler_type = config.get(KEY.SCHEDULER, 'exponentiallr').lower()
    scheduler_param = config.get(KEY.SCHEDULER_PARAM, {})
    config[KEY.SCHEDULER_BATCH_MODE] = scheduler_param.pop(
        KEY.SCHEDULER_BATCH_MODE, False
    )

    if scheduler_type == 'onecyclelr':  # special case, always batch mode
        total_steps = scheduler_param.get('total_steps', None)
        if total_steps is None:
            # total_steps not given, automatically calculated
            # allow epochs to be float for SWA
            epochs = scheduler_param.get('epochs', None)
            if epochs is None:
                raise ValueError('One of total_steps or epochs should be given')
            total_steps = math.ceil(epochs * steps_per_epoch)
        config[KEY.SCHEDULER_PARAM]['total_steps'] = total_steps
        config[KEY.SCHEDULER_BATCH_MODE] = True

    elif config[KEY.SCHEDULER_BATCH_MODE]:
        scheduler_epoch_params = {
            'linearlr': ['total_iters', lambda x, y: math.ceil(x * y)],
            'cosineannealinglr': ['T_max', lambda x, y: math.ceil(x * y)],
            'exponentiallr': ['gamma', lambda x, y: x ** (1 / y)],
        }.get(scheduler_type, None)
        if scheduler_epoch_params is None:
            raise NotImplementedError(
                f'Scheduler batch mode not implemented for {scheduler_type}.'
            )

        config[KEY.SCHEDULER_PARAM][scheduler_epoch_params[0]] = (
            scheduler_epoch_params[1](
                config[KEY.SCHEDULER_PARAM][scheduler_epoch_params[0]],
                steps_per_epoch,
            )
        )


def processing_by_batch(
    config: dict,
    trainer: Trainer,
    loaders: dict,
    data_progress: dict,
    start_epoch: int = 1,
    train_loader_key: str = 'trainset',
    error_recorder: Optional[ErrorRecorder] = None,
    total_epoch: Optional[int] = None,
    per_epoch: Optional[float] = None,
    best_metric_loader_key: str = 'validset',
    best_metric: Optional[str] = None,
    write_csv: bool = True,
    working_dir: Optional[str] = None,
):
    """
    Batch-level training loop for large-scale training.

    Unlike epoch-level training (processing_epoch_v2), this function:
    - Saves checkpoints at configurable intervals within an epoch
    - Supports resuming from exact data position
    - Uses OrderedSampler for deterministic DDP training

    Args:
        config: Training configuration dictionary
        trainer: Trainer instance
        loaders: Dict of dataloaders (must include 'trainset')
        data_progress: Dict tracking data iteration progress
        start_epoch: Starting epoch number
        train_loader_key: Key for training dataloader
        error_recorder: ErrorRecorder instance (created from config if None)
        total_epoch: Total epochs to train (from config if None)
        per_epoch: Checkpoint frequency as fraction of epoch (from config if None)
        best_metric_loader_key: Key for validation loader used for best metric
        best_metric: Metric name for tracking best model
        write_csv: Whether to write CSV log
        working_dir: Working directory for checkpoints
    """
    log = Logger()
    write_csv = write_csv and log.rank == 0
    working_dir = working_dir or os.getcwd()
    prefix = f'{os.path.abspath(working_dir)}/'

    total_epoch = total_epoch or config[KEY.EPOCH]
    per_epoch = per_epoch or config.get(KEY.PER_EPOCH, 0.1)
    best_metric = best_metric or config.get(KEY.BEST_METRIC, 'TotalLoss')
    recorder = error_recorder or ErrorRecorder.from_config(
        config, trainer.loss_functions
    )
    recorders = {k: deepcopy(recorder) for k in loaders}

    best_val = float('inf')
    best_key = None
    if best_metric_loader_key in recorders:
        best_key = recorders[best_metric_loader_key].get_key_str(best_metric)
    if best_key is None:
        log.writeline(
            f'Failed to get error recorder key: {best_metric} or '
            + f'{best_metric_loader_key} is missing. There will be no best '
            + 'checkpoint.'
        )

    csv_path = unique_filepath(f'{prefix}/lc.csv')
    if write_csv:
        head = ['epoch', 'batch', 'lr']
        for k, rec in recorders.items():
            head.extend(list(rec.get_dct(prefix=k)))
        with open(csv_path, 'w') as f:
            f.write(','.join(head) + '\n')

    train_loader = loaders['trainset']

    if data_progress[KEY.TOTAL_DATA_NUM] < 0:  # fresh start or reset
        data_progress[KEY.TOTAL_DATA_NUM] = len(train_loader.sampler.sequence)
    else:  # continue
        train_loader.sampler.continue_from_data_progress(**data_progress)

    effective_batch_size = config[KEY.WORLD_SIZE] * config[KEY.BATCH_SIZE]
    start_batch = (data_progress[KEY.CURRENT_DATA_IDX]) // effective_batch_size
    total_step = data_progress[KEY.TOTAL_DATA_NUM] // effective_batch_size
    if data_progress[KEY.TOTAL_DATA_NUM] % effective_batch_size != 0:
        total_step += 1
    save_per_epoch = int(1 / per_epoch)
    save_batch_idx = np.linspace(1, total_step, save_per_epoch + 1)
    save_batch_idx = [int(idx) for idx in save_batch_idx[1:]]

    if data_progress[KEY.TOTAL_DATA_NUM] == data_progress[KEY.CURRENT_DATA_IDX]:
        start_epoch += 1  # continuing from end of epoch
        start_batch = 0
        train_loader.sampler.permutate_sequence()  # update rng state
        train_loader.sampler.refresh_sequence()  # make starting index to zero

    trainer.write_checkpoint(
        f'{prefix}/checkpoint_initial.pth',
        config=config,
        epoch=0,
        data_progress=data_progress,
    )

    scheduler_update_every_batch = (
        config.get(KEY.SCHEDULER_BATCH_MODE, False)
    )

    # --- periodic DataLoader-worker restart -----------------------------------
    # Within a single (huge) epoch the persistent_workers mechanism never fires,
    # so worker memory grows unbounded -> RAM OOM. We instead tear down and
    # respawn the workers at every checkpoint boundary, resuming the
    # OrderedSampler at the exact data position from the epoch-start rng state.
    # The emitted sample order is provably identical to an uninterrupted run
    # (proof: _proof_sampler_resume.py / _proof_dataloader_restart.py): same
    # permutation P0 from the same rng state, sliced from `current_data_index`,
    # so there is no duplication, skipping, or reordering of samples.
    _lk = config.get(KEY.LOADER_KWARGS, None) or {}
    effective_num_workers = _lk.get('num_workers', config.get(KEY.NUM_WORKERS, 0))
    persistent_workers = _lk.get('persistent_workers', False)
    restart_workers = effective_num_workers > 0 and not persistent_workers
    if restart_workers:
        log.writeline(
            'Periodic worker restart ENABLED: workers respawn every checkpoint '
            f'(~{save_batch_idx[0] if save_batch_idx else total_step} batches) '
            'to bound DataLoader RAM. Sample order is identical to a continuous '
            'run (data integrity preserved).'
        )

    # TODO: too long, refactor more
    log.writeline('Entering training loop')
    for epoch in range(start_epoch, total_epoch + 1):  # one indexing
        # rng state captured ONCE at epoch start; every leg resumes from this
        # so each leg reproduces the same permutation for this epoch.
        epoch_rng_state = train_loader.sampler.get_rng_state()
        data_progress[KEY.NUMPY_RNG_STATE] = epoch_rng_state
        log.timer_start('epoch')
        log.timer_start('batch')

        base_batch = start_batch if epoch == start_epoch else 0

        if restart_workers:
            # one leg per remaining checkpoint interval; each leg ends on a save
            leg_bounds = [b for b in save_batch_idx if b > base_batch]
            if not leg_bounds or leg_bounds[-1] < total_step:
                leg_bounds.append(total_step)
        else:
            leg_bounds = [total_step]  # single continuous pass (original behavior)

        dl_timing = AverageNumber()
        current_batch_idx = base_batch

        for leg_end in leg_bounds:
            if current_batch_idx >= leg_end:
                continue
            if restart_workers:
                # respawn fresh workers and resume sampler at the exact position
                consumed = min(
                    current_batch_idx * effective_batch_size,
                    data_progress[KEY.TOTAL_DATA_NUM],
                )
                train_loader.sampler.continue_from_data_progress(
                    numpy_rng_state=epoch_rng_state,
                    total_data_num=data_progress[KEY.TOTAL_DATA_NUM],
                    current_data_index=consumed,
                )
            loader_iter = iter(train_loader)  # spawns worker processes
            dl_end = time.time()
            while current_batch_idx < leg_end:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                dl_timing.update(time.time() - dl_end)
                current_batch_idx += 1
                save = current_batch_idx in save_batch_idx

                if save:
                    lr = trainer.get_lr()
                    log.bar()
                    log.writeline(
                        f'Epoch {epoch}/{total_epoch}  '
                        + f'Batch {current_batch_idx}/{total_step}  lr: {lr:8f}'
                    )
                    log.bar()

                trainer.train_one_batch(batch, recorders[train_loader_key])
                if scheduler_update_every_batch:  # onecyclelr
                    trainer.scheduler_step(best_val)

                if save:
                    csv_dct = {
                        'epoch': str(epoch),
                        'batch': str(current_batch_idx),
                        'lr': f'{trainer.get_lr():8f}',
                    }
                    errors = {}
                    for k, loader in loaders.items():
                        rec = recorders[k]
                        if k != train_loader_key:
                            print('valid', k, flush=True)
                            trainer.run_one_epoch(loader, False, rec)
                        if trainer.distributed:
                            trainer.recorder_all_reduce(rec)
                        csv_dct.update(rec.get_dct(prefix=k))
                        errors[k] = rec.epoch_forward()
                    log.write_full_table(list(errors.values()), list(errors))

                    batch_name = (
                        f'_{save_batch_idx.index(current_batch_idx) + 1}'
                        if current_batch_idx != save_batch_idx[-1]
                        else ''
                    )
                    data_progress[KEY.CURRENT_DATA_IDX] = min(
                        current_batch_idx * effective_batch_size,
                        data_progress[KEY.TOTAL_DATA_NUM],
                    )
                    trainer.write_checkpoint(
                        f'{prefix}/checkpoint_{epoch}{batch_name}.pth',
                        config=config,
                        epoch=epoch,
                        data_progress=data_progress,
                    )

                    if write_csv:
                        with open(csv_path, 'a') as f:
                            f.write(','.join(list(csv_dct.values())) + '\n')

                    if best_key and errors[best_metric_loader_key][best_key] < best_val:
                        trainer.write_checkpoint(
                            f'{prefix}/checkpoint_best.pth', config=config, epoch=epoch
                        )
                        best_val = errors[best_metric_loader_key][best_key]
                        log.writeline('Best checkpoint written')

                    log.timer_end(
                        'batch', message=f'Batch {current_batch_idx} elapsed'
                    )
                    if config[KEY.IS_DDP]:
                        dl_timing._ddp_reduce(trainer.device)
                    log.writeline(f'data loading, per (sec): {dl_timing.get():.4f}')
                    log.writeline(
                        f'data loading, sum *ALL* (sec): {dl_timing._sum:.4f}'
                    )
                    dl_timing = AverageNumber()
                    log.timer_start('batch')
                dl_end = time.time()
                # batch loop indent
            # leg finished: drop the iterator so the worker processes shut down
            # and release their accumulated memory before the next leg respawns
            # fresh ones. (no-op effect on data order: next leg resumes the
            # sampler at current_batch_idx from epoch_rng_state.)
            del loader_iter
            if restart_workers:
                gc.collect()

        if not scheduler_update_every_batch:
            trainer.scheduler_step(best_val)
        log.timer_end('epoch', message=f'Epoch {epoch} elapsed')
    return trainer
