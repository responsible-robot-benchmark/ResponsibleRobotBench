"""Train RGB policies with native/custom datasets or converted Zarr episodes."""
from copy import deepcopy
import hydra
import wandb
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, ListConfig
from torch.utils.data import DataLoader
from composer.trainer import Trainer
from composer.loggers import WandBLogger
from composer.callbacks import LRMonitor
from composer.algorithms import EMA

from diffusion_policy.model.common.lr_scheduler import get_scheduler
from pfp import DEVICE, DATA_DIRS, set_seeds
from pfp.data_utils.dataset_images import RobotDatasetImages
from pfp.data_utils.multitask_dataset import MultiTaskDataset


def resolve_training_config(cfg):
    """Apply the original experiment precedence and save the effective model.

    Checkpoint readers instantiate the top-level model configuration. Keeping
    overrides only inside ``experiment`` would train and restore different models.
    """
    cfg = deepcopy(cfg)
    experiment = cfg.get("experiment")
    if experiment is not None:
        for name in ("dataset", "dataloader", "model", "optimizer", "lr_scheduler"):
            override = experiment.get(name)
            if override is not None and (name != "dataset" or "_target_" in override):
                cfg[name] = deepcopy(override)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    return cfg


def image_dataset(data_root, task_name, train, **dataset_options):
    """Construct the existing RGB dataset for one or several tasks.

    Current conversions use ``val``; original converted datasets used ``valid``.
    Native custom datasets retain their own split handling.
    """
    root = Path(data_root) if data_root else DATA_DIRS.PFP
    multitask = isinstance(task_name, (list, tuple, ListConfig))
    tasks = list(task_name) if multitask else [task_name]
    datasets = []
    for task in tasks:
        path = root / task / ("train" if train else "val")
        if not train and not path.exists() and (root / task / "valid").exists():
            path = root / task / "valid"
        datasets.append(RobotDatasetImages(path, **dataset_options))
    return MultiTaskDataset(datasets, task_names=tasks) if multitask else datasets[0]


def build_datasets(cfg):
    """Honor a configured dataset factory's original train=True/False contract."""
    if "_target_" in cfg.dataset:
        return tuple(hydra.utils.instantiate(cfg.dataset, train=train)
                     for train in (True, False))
    return tuple(image_dataset(cfg.data_root, cfg.task_name, train, **cfg.dataset)
                 for train in (True, False))


def build_dataloaders(cfg, dataset_train, dataset_valid):
    options = dict(hydra.utils.instantiate(cfg.dataloader))
    options.setdefault("persistent_workers", options.get("num_workers", 0) > 0)
    return (DataLoader(dataset_train, shuffle=True, **options),
            DataLoader(dataset_valid, shuffle=False, **options))


def build_training_components(cfg, num_training_steps):
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    scheduler = get_scheduler(
        cfg.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return model, optimizer, scheduler


@hydra.main(version_base=None, config_path=str(Path(__file__).resolve().parents[1]/'conf'), config_name="train")
def main(cfg: OmegaConf):
    cfg = resolve_training_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seeds(cfg.seed)
    if cfg.obs_mode != "rgb":
        raise ValueError("Image FM training requires obs_mode='rgb'.")

    dataset_train, dataset_valid = build_datasets(cfg)
    if cfg.get("diagnose_dataset", True):
        if hasattr(dataset_train, "diagnose_coverage"):
            dataset_train.diagnose_coverage(num_samples=1000)
        if hasattr(dataset_train, "verify_watering_action_sampling"):
            dataset_train.verify_watering_action_sampling(num_episodes=5)
    dataloader_train, dataloader_valid = build_dataloaders(cfg, dataset_train, dataset_valid)
    composer_model, optimizer, lr_scheduler = build_training_components(
        cfg, len(dataloader_train) * cfg.epochs)

    wandb_logger = WandBLogger(
        project="pfp_training",
        init_kwargs={
            "config": OmegaConf.to_container(cfg),
            "mode": "online" if cfg.log_wandb else "disabled",
        },
    )

    trainer = Trainer(
        model=composer_model,
        train_dataloader=dataloader_train,
        eval_dataloader=dataloader_valid,
        max_duration=cfg.max_duration or cfg.epochs,
        load_path=cfg.load_path,
        train_subset_num_batches=cfg.train_subset_num_batches,
        eval_subset_num_batches=cfg.eval_subset_num_batches,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        step_schedulers_every_batch=True,
        device="gpu" if DEVICE.type == "cuda" else "cpu",
        loggers=[wandb_logger],
        callbacks=[LRMonitor()],
        save_folder=str(Path(cfg.checkpoint_dir)/"{run_name}"),
        save_interval=cfg.save_interval or f"{cfg.save_each_n_epochs}ep",
        save_num_checkpoints_to_keep=3,
        algorithms=[EMA()] if cfg.use_ema else None,
        run_name=cfg.run_name,
        autoresume=False,
        spin_dataloaders=False
    )
    # Save the used cfg for inference
    checkpoint_directory = Path(cfg.checkpoint_dir)/trainer.state.run_name
    checkpoint_directory.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(cfg, checkpoint_directory/"config.yaml")

    try:
        trainer.fit()
    finally:
        wandb.finish()
        trainer.close()

    return
