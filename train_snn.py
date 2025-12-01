import os

# CUDA_VISIBLE_DEVICES is handled by environment variables

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from callbacks.custom import get_ckpt_callback, get_viz_callback
from callbacks.gradflow import GradFlowLogCallback
from config.modifier import dynamically_modify_train_config
from data.utils.types import DatasetSamplingMode
from loggers.utils import get_wandb_logger, get_ckpt_path
from modules.utils.fetch import fetch_data_module, fetch_model_module


@hydra.main(config_path='config', config_name='train', version_base='1.2')
def train_snn(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # Set SNN-specific configuration (after config is printed, same structure as train.py)
    from omegaconf import open_dict
    with open_dict(config):
        if 'model' not in config:
            config.model = {}
        if 'backbone' not in config.model:
            config.model.backbone = {}
        config.model.backbone.type = 'fusion'
        config.model.backbone.name = 'snn_forward_fusion'
        config.model.backbone.backbone_type = 'snn'
        if 'snn_img_ch' not in config.model.backbone:
            config.model.backbone.snn_img_ch = 3
        if 'snn_evs_ch' not in config.model.backbone:
            config.model.backbone.snn_evs_ch = 2
        if 'snn_T' not in config.model.backbone:
            config.model.backbone.snn_T = 2
        if 'snn_img_T' not in config.model.backbone:
            config.model.backbone.snn_img_T = 1
        if 'snn_evs_T' not in config.model.backbone:
            config.model.backbone.snn_evs_T = 2
        if 'width' not in config.model.backbone:
            config.model.backbone.width = 0.5  # 从0.33提升到0.5
        if 'depth' not in config.model.backbone:
            config.model.backbone.depth = 0.33
        if 'max_channels' not in config.model.backbone:
            config.model.backbone.max_channels = 1024
        if 'verbose' not in config.model.backbone:
            config.model.backbone.verbose = False
        if 'enable_align' not in config.model.backbone:
            config.model.backbone.enable_align = False
        if 'using_align_loss' not in config.model.backbone:
            config.model.backbone.using_align_loss = False
        if 'fusion_type' not in config.model.backbone:
            config.model.backbone.fusion_type = 'cross_cbam'
        if 'memory_type' not in config.model.backbone:
            config.model.backbone.memory_type = 'lstm'
        if 'fpn' not in config.model:
            config.model.fpn = {}
        if 'name' not in config.model.fpn:
            config.model.fpn.name = 'PAFPN'
        if 'in_stages' not in config.model.fpn:
            config.model.fpn.in_stages = [2, 3, 4]
        if 'depth' not in config.model.fpn:
            config.model.fpn.depth = 0.67
        if 'depthwise' not in config.model.fpn:
            config.model.fpn.depthwise = False
        if 'act' not in config.model.fpn:
            config.model.fpn.act = 'silu'
        if 'head' not in config.model:
            config.model.head = {}
        if 'name' not in config.model.head:
            config.model.head.name = 'YoloX'

    # ---------------------
    # Reproducibility
    # ---------------------
    dataset_train_sampling = config.dataset.train.sampling
    assert dataset_train_sampling in iter(DatasetSamplingMode)
    disable_seed_everything = dataset_train_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.MIXED)
    if disable_seed_everything:
        print('Disabling PL seed everything because of unresolved issues                    with shuffling during training on streaming '
              'datasets')
    seed = config.reproduce.seed_everything
    if seed is not None and not disable_seed_everything:
        assert isinstance(seed, int)
        print(f'USING pl.seed_everything WITH {seed=}')
        pl.seed_everything(seed=seed, workers=True)

  
    # ---------------------
    # DDP
    # ---------------------
    # Debug: Check GPU availability
    print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")}')
    print(f'PyTorch detected {torch.cuda.device_count()} GPU(s)')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    print(f'Config requested GPUs: {gpus}')
    
    # Validate GPU indices
    available_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpu_count == 0:
        raise RuntimeError('No CUDA GPUs available!')
    
    # Filter out invalid GPU indices
    valid_gpus = [gpu for gpu in gpus if 0 <= gpu < available_gpu_count]
    invalid_gpus = [gpu for gpu in gpus if gpu < 0 or gpu >= available_gpu_count]
    
    if invalid_gpus:
        print(f'WARNING: Invalid GPU indices {invalid_gpus} requested. Available GPUs: 0-{available_gpu_count-1}')
        print(f'Using valid GPUs: {valid_gpus}')
    
    if not valid_gpus:
        raise RuntimeError(f'No valid GPU indices found! Requested: {gpus}, Available: 0-{available_gpu_count-1}')
    
    gpus = valid_gpus
    print(f'Final GPU configuration: {gpus}')
    
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None

    # ---------------------
    # Data
    # ---------------------
    
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = get_wandb_logger(config)
    ckpt_path = None
    # Allow direct checkpoint path specification (for local resuming)
    # First check environment variable, then config, then wandb artifact
    from pathlib import Path
    env_ckpt_path = os.environ.get('CHECKPOINT_PATH', None)
    if env_ckpt_path is not None:
        ckpt_path = Path(env_ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')
        print(f'Resuming from local checkpoint (env): {ckpt_path}')
    elif hasattr(config, 'checkpoint_path') and config.checkpoint_path is not None:
        ckpt_path = Path(config.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')
        print(f'Resuming from local checkpoint (config): {ckpt_path}')
    elif config.wandb.artifact_name is not None:
        ckpt_path = get_ckpt_path(logger, wandb_config=config.wandb)

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)
    if ckpt_path is not None and config.wandb.resume_only_weights:
        print('Resuming only the weights instead of the full training state')
        module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})
        ckpt_path = None

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    callbacks.append(get_ckpt_callback(config))
    callbacks.append(GradFlowLogCallback(config.logging.train.log_model_every_n_steps))
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if config.logging.train.high_dim.enable or config.logging.validation.high_dim.enable:
        viz_callback = get_viz_callback(config=config)
        callbacks.append(viz_callback)
    callbacks.append(ModelSummary(max_depth=2))

    logger.watch(model=module, log='all', log_freq=config.logging.train.log_model_every_n_steps, log_graph=True)

    class DummyCheckpoint:
        pass

    ckpt_callback = DummyCheckpoint()
    ckpt_callback.best_model_path = "your_best.ckpt"
    ckpt_callback.best_k_models = {"your_best.ckpt": 0.99}
    ckpt_callback.last_model_path = "your_last.ckpt"
    ckpt_callback.current_score = 0.88
    ckpt_callback.save_last = True
    ckpt_callback.save_top_k = 1
    ckpt_callback.best_model_score = 0.99

    # 传递弱引用
    logger.after_save_checkpoint(ckpt_callback)

    # ---------------------
    # Training
    # ---------------------

    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch
    assert val_check_interval is None or check_val_every_n_epoch is None

    # Handle max_steps: PyTorch Lightning doesn't accept None, so we omit it if None
    trainer_kwargs = {
        'accelerator': 'gpu',
        'callbacks': callbacks,
        'enable_checkpointing': True,
        'val_check_interval': val_check_interval,
        'check_val_every_n_epoch': check_val_every_n_epoch,
        'default_root_dir': None,
        'devices': gpus,
        'gradient_clip_val': config.training.gradient_clip_val,
        'gradient_clip_algorithm': 'value',
        'limit_train_batches': config.training.limit_train_batches,
        'limit_val_batches': config.validation.limit_val_batches,
        'logger': logger,
        'log_every_n_steps': config.logging.train.log_every_n_steps,
        'plugins': None,
        'precision': config.training.precision,
        'max_epochs': config.training.max_epochs,
        'strategy': strategy,
        'sync_batchnorm': False if strategy is None else True,
        'move_metrics_to_cpu': False,
        'benchmark': config.reproduce.benchmark,
        'deterministic': config.reproduce.deterministic_flag,
    }
    # Only add max_steps if it's not None
    if config.training.max_steps is not None:
        trainer_kwargs['max_steps'] = config.training.max_steps
    
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)


if __name__ == '__main__':
    train_snn()
