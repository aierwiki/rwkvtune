"""
Pretrain Trainer - follows trl-main design style.

Provides pretraining capabilities for RWKV models with support for
both standard and infinite context training modes.
"""
import os
import time
import torch
import lightning as pl
from lightning import Trainer
from pathlib import Path
from typing import Optional, Union, Any
from torch.utils.data import DataLoader

from rwkvtune.training.pretrain_config import PretrainConfig
from rwkvtune.training.lightning_module import RWKV7LightningModule
from rwkvtune.training.lightning_module_infctx import RWKV7InfctxLightningModule
from rwkvtune.training.callbacks import TrainingCallback
from rwkvtune.models.rwkv7.operators import load_wkv_operator

# HuggingFace Dataset support
try:
    from datasets import Dataset as HFDataset, IterableDataset as HFIterableDataset
    _has_datasets = True
except ImportError:
    _has_datasets = False
    HFDataset = None
    HFIterableDataset = None


class PretrainTrainer:
    """
    Pretrain Trainer - follows trl-main and GRPO design style.

    Features:
    1. Supports two initialization methods: new API (model parameter) and legacy API (config)
    2. User is responsible for data preprocessing, inputs processed HuggingFace Dataset
    3. Auto-loads model architecture parameters from model_config or model object

    Example (new API, recommended):
        ```python
        from rwkvtune.training import PretrainConfig, PretrainTrainer
        from rwkvtune import AutoModel, AutoTokenizer
        from datasets import Dataset

        config = PretrainConfig(
            ctx_len=2048,
            micro_bsz=4,
            epoch_count=10,
        )

        trainer = PretrainTrainer(
            model="/path/to/model",
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        trainer.train()
        ```
    """

    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        args: Optional[PretrainConfig] = None,
        train_dataset: Optional[Union[HFDataset, HFIterableDataset]] = None,
        processing_class: Optional[Any] = None,
        lora_config: Optional[dict] = None,
    ):
        """
        Args:
            model: Model path (string) or instantiated model object.
            args: Pretrain config.
            train_dataset: HuggingFace Dataset (required, must contain 'input_ids' and 'labels').
            processing_class: Tokenizer (optional).
            lora_config: LoRA config dict (optional), for saving LoRA weights.
        """
        if args is None:
            args = PretrainConfig()
        self.config = args
        self.lora_config = lora_config

        # Validate dataset
        if train_dataset is None:
            raise ValueError(
                "train_dataset is required!\n"
                "Please use HuggingFace Dataset, e.g.:\n"
                "  from datasets import Dataset\n"
                "  dataset = Dataset.from_list([{'input_ids': [...], 'labels': [...]}, ...])"
            )

        if not _has_datasets:
            raise ImportError("datasets library required! Run: pip install datasets")

        if not isinstance(train_dataset, (HFDataset, HFIterableDataset)):
            raise TypeError(f"train_dataset must be HuggingFace Dataset, got {type(train_dataset)}")

        col_names = getattr(train_dataset, "column_names", None)
        if not col_names:
            features = getattr(train_dataset, "features", None)
            if features is not None:
                try:
                    col_names = list(features.keys())
                except Exception:
                    col_names = None

        # Validate required fields
        if col_names:
            if 'input_ids' not in col_names:
                raise ValueError(f"train_dataset must contain 'input_ids' field! Current fields: {col_names}")
            if 'labels' not in col_names:
                raise ValueError(f"train_dataset must contain 'labels' field! Current fields: {col_names}")
        else:
            print("Warning: Cannot infer fields from train_dataset, skipping initial validation")

        try:
            dataset_size = len(train_dataset)
            print(f"Using HuggingFace Dataset: {dataset_size} samples")
        except TypeError:
            print(f"Using HuggingFace IterableDataset")
        print(f"  Fields: {col_names}")
        self.dataset = train_dataset

        # Create output directory
        Path(self.config.proj_dir).mkdir(parents=True, exist_ok=True)

        # Set random seed
        if self.config.seed >= 0:
            pl.seed_everything(self.config.seed)

        # Setup model
        print("Initializing model...")
        self.model = self._setup_model(model)
        self.model_path = model if isinstance(model, str) else None

        # Load model config for Lightning module
        self._load_model_config_for_lightning()

        # Setup tokenizer
        print("Initializing Tokenizer...")
        self.tokenizer = self._setup_tokenizer(processing_class)

        # Setup environment
        self._setup_environment()

        config = self.config

        # Create Lightning module
        print("Initializing model...")
        if config.train_type == 'infctx':
            print(f"Using infinite context training mode (train_type={config.train_type})")
            print(f"  Sequence length: {config.ctx_len}, chunk size: {config.chunk_ctx}")
            print(f"Loading RWKV7 infinite context operator...")
            from rwkvtune.models.rwkv7.operators import load_cuda_operator_infctx
            load_cuda_operator_infctx()
            self.lightning_module = RWKV7InfctxLightningModule(config)
        else:
            print(f"Using standard training mode (train_type={config.train_type})")
            device_type = 'cpu' if config.accelerator == 'cpu' else 'cuda'
            print(f"Loading RWKV7 {device_type.upper()} standard training operator...")
            load_wkv_operator(device=device_type, for_inference=False, head_size=config.head_size_a)
            self.lightning_module = RWKV7LightningModule(config)

        # Load pretrained weights
        load_model_path = getattr(config, 'load_model', None)
        if load_model_path and os.path.exists(load_model_path):
            print(f"Loading pretrained model: {load_model_path}")
            state_dict = torch.load(load_model_path, map_location="cpu")

            # Add model. prefix if needed
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith("model."):
                    new_state_dict[f"model.{k}"] = v
                else:
                    new_state_dict[k] = v
            self.lightning_module.load_state_dict(new_state_dict, strict=False)

        print(f"\nModel parameters: {sum(p.numel() for p in self.lightning_module.parameters()) / 1e6:.2f}M")
        print(f"Trainable parameters: {sum(p.numel() for p in self.lightning_module.parameters() if p.requires_grad) / 1e6:.2f}M")

        # Create Lightning Trainer
        self.trainer = self._create_trainer()
        print("PretrainTrainer initialization complete\n")

    def _setup_model(self, model: Union[str, torch.nn.Module]):
        """Setup model."""
        if isinstance(model, str):
            print(f"Loading model from path: {model}")
            from rwkvtune import AutoModel
            loaded_model = AutoModel.from_pretrained(model, device="cpu")
            print(f"Model loaded")
            print(f"  Model type: {type(loaded_model).__name__}")
            print(f"  Parameters: {sum(p.numel() for p in loaded_model.parameters()) / 1e6:.2f}M")
            return loaded_model
        elif isinstance(model, torch.nn.Module):
            print(f"Using provided model: {type(model).__name__}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
            return model
        else:
            raise TypeError(f"model must be string (path) or torch.nn.Module, got {type(model)}")

    def _setup_tokenizer(self, processing_class: Optional[Any]):
        """Setup Tokenizer."""
        if processing_class is not None:
            print(f"Using provided Tokenizer: {type(processing_class).__name__}")
            return processing_class
        elif self.model_path is not None:
            print(f"Loading Tokenizer from model path: {self.model_path}")
            from rwkvtune import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"Tokenizer loaded")
            return tokenizer
        else:
            print(f"No tokenizer provided (usually not needed for pretraining)")
            return None

    def _load_model_config_for_lightning(self):
        """Load model config for Lightning module and sync training parameters."""
        try:
            if hasattr(self.model, 'config'):
                model_config = self.model.config

                # Read architecture parameters from model config
                self.config.n_layer = model_config.n_layer
                self.config.n_embd = model_config.n_embd
                self.config.vocab_size = model_config.vocab_size
                self.config.head_size_a = model_config.head_size_a

                # LORA dimensions
                self.config.dim_att_lora = getattr(model_config, 'dim_att_lora', 32)
                self.config.dim_gate_lora = getattr(model_config, 'dim_gate_lora', 128)
                self.config.dim_mv_lora = getattr(model_config, 'dim_mv_lora', 32)

                # Derived parameters
                self.config.dim_att = self.config.n_embd
                self.config.dim_ffn = int((self.config.n_embd * 3.5) // 32 * 32)

                # Pass training parameters to model config
                model_config.grad_cp = self.config.grad_cp
                model_config.ctx_len = self.config.ctx_len

                print(f"Loaded config from model")
                print(f"  Architecture: {self.config.n_layer} layers, {self.config.n_embd} dim, {self.config.vocab_size} vocab")
                print(f"  LORA dims: att={self.config.dim_att_lora}, gate={self.config.dim_gate_lora}, mv={self.config.dim_mv_lora}")
                print(f"  Training params: grad_cp={self.config.grad_cp}, ctx_len={self.config.ctx_len}")
            else:
                raise ValueError("Model has no config attribute")
        except Exception as e:
            print(f"Error: Cannot load config from model: {str(e)}")
            raise

    def _setup_environment(self):
        """Setup environment variables."""
        config = self.config
        device_type = 'cpu' if config.accelerator == 'cpu' else 'cuda'

        if device_type == 'cpu':
            print("\n" + "Warning " * 10)
            print("Using CPU training mode (not optimized, slow performance)")
            print(f"Recommended config: n_layer<=6, n_embd<=512, ctx_len<=512, micro_bsz<=2")
            print(f"Current config: n_layer={config.n_layer}, n_embd={config.n_embd}, ctx_len={config.ctx_len}")
            print("Warning " * 10 + "\n")

        os.environ["RWKV_MY_TESTING"] = config.my_testing
        os.environ["RWKV_CTXLEN"] = str(config.ctx_len)
        os.environ["RWKV_HEAD_SIZE_A"] = str(config.head_size_a)
        os.environ["RWKV_FLOAT_MODE"] = config.precision
        os.environ["RWKV_JIT_ON"] = "0"
        os.environ["WKV"] = device_type
        os.environ["RWKV_DEVICE"] = device_type
        os.environ["FUSED_KERNEL"] = "0"
        os.environ["RWKV_TRAIN_TYPE"] = "none"

        # CUDA optimization (GPU mode only)
        if device_type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            if config.precision in ["bf16", "fp16"]:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
            else:
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False

    def _create_trainer(self) -> Trainer:
        """Create Lightning Trainer."""
        config = self.config

        accelerator = config.accelerator if config.accelerator in ['cpu', 'gpu'] else 'gpu'

        # Precision setup
        precision = config.precision
        if accelerator == 'cpu':
            if precision in ["bf16", "fp16"]:
                print(f"Warning: CPU training does not recommend {precision}, switching to fp32")
                precision = 32
            elif precision == "fp32":
                precision = 32
        else:
            if precision == "fp32":
                precision = 32
            elif precision == "fp16":
                precision = 16

        # Strategy setup
        strategy = config.strategy
        if accelerator == 'cpu':
            strategy = 'auto'
            devices = 1
        else:
            devices = config.devices

        # Create callback
        callbacks = [TrainingCallback(
            config=config,
            tokenizer=getattr(self, 'tokenizer', None),
            model_path=getattr(self, 'model_path', None),
            lora_config=getattr(self, 'lora_config', None),
        )]

        if config.train_type == 'infctx' and int(getattr(config, 'accumulate_grad_batches', 1)) != 1:
            raise ValueError(
                "train_type=infctx uses manual optimization (automatic_optimization=False). "
                "Lightning does not support Trainer(accumulate_grad_batches>1) in manual optimization. "
                "Please set --accumulate_grad_batches 1."
            )

        # Create trainer parameters
        trainer_kwargs = dict(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=1,
            precision=precision,
            max_epochs=config.epoch_count,
            accumulate_grad_batches=config.accumulate_grad_batches,
            callbacks=callbacks,
            enable_checkpointing=False,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )

        # infctx uses manual optimization, doesn't support auto gradient clipping
        if config.train_type != 'infctx':
            trainer_kwargs['gradient_clip_val'] = config.grad_clip

        if config.epoch_steps is not None:
            batches_per_device = int(config.epoch_steps) // int(getattr(config, 'devices', 1))
            trainer_kwargs['limit_train_batches'] = batches_per_device

            try:
                rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
            except Exception:
                rank = 0
            if rank == 0:
                print(f"Total steps per epoch (all GPUs): {config.epoch_steps}")
                print(f"Batches per GPU: {batches_per_device}")
                print(f"Optimizer updates per epoch: {batches_per_device // int(getattr(config, 'accumulate_grad_batches', 1))}")

        trainer = Trainer(**trainer_kwargs)
        return trainer

    def _collate_fn(self, batch):
        """DataLoader collate function."""
        import torch
        IGNORE_INDEX = -100
        pad_token_id = 0

        # Extract input_ids and labels, convert to tensor
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]

        # Pad to longest sequence in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # RWKV7 WKV operator requires sequence length to be multiple of 16
        CHUNK_LEN = 16
        seq_len = input_ids.size(1)
        if seq_len % CHUNK_LEN != 0:
            pad_len = CHUNK_LEN - (seq_len % CHUNK_LEN)
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_token_id)
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=IGNORE_INDEX)

        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    def train(self):
        """Start training."""
        print("\n" + "="*80)
        print("Starting pretraining")
        print("="*80 + "\n")

        dataset_is_iterable = hasattr(self.dataset, "__iter__") and not hasattr(self.dataset, "__len__")
        shuffle = getattr(self.config, 'dataloader_shuffle', True)
        if dataset_is_iterable and shuffle:
            if getattr(self.trainer, 'is_global_zero', False) or int(os.environ.get('RANK', '0')) == 0:
                print("Warning: IterableDataset does not support shuffle=True, disabling shuffle")
            shuffle = False

        # Create DataLoader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.micro_bsz,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            persistent_workers=True if self.config.num_workers > 0 else False,
            pin_memory=True if self.config.accelerator == 'gpu' else False,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

        # Handle checkpoint resume (Dummy Forward strategy)
        if self.config.resume_from_checkpoint:
            ckpt_path = self.config.resume_from_checkpoint

            if ckpt_path == 'latest':
                ckpt_path = self.config.proj_dir

            if os.path.isdir(ckpt_path) and os.path.exists(os.path.join(ckpt_path, 'trainer_state.ckpt')):
                ckpt_path = os.path.join(ckpt_path, 'trainer_state.ckpt')
            elif os.path.isdir(ckpt_path) and not os.path.exists(os.path.join(ckpt_path, 'checkpoint')):
                subdirs = [d for d in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, d))]
                valid_subdirs = []
                for d in subdirs:
                    if os.path.exists(os.path.join(ckpt_path, d, 'trainer_state.ckpt')):
                        valid_subdirs.append(d)
                if valid_subdirs:
                    valid_subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)), reverse=True)
                    latest_subdir = valid_subdirs[0]
                    ckpt_path = os.path.join(ckpt_path, latest_subdir, 'trainer_state.ckpt')

            print(f"\nPreparing to resume from checkpoint: {ckpt_path}")

            try:
                metadata_path = ckpt_path
                if os.path.isdir(ckpt_path):
                    candidates = [
                        os.path.join(ckpt_path, 'checkpoint', 'mp_rank_00_model_states.pt'),
                    ]
                    for candidate in candidates:
                        if os.path.isfile(candidate):
                            metadata_path = candidate
                            break
                    if metadata_path == ckpt_path:
                        for root, _, files in os.walk(ckpt_path):
                            for filename in files:
                                if filename.endswith('_model_states.pt'):
                                    metadata_path = os.path.join(root, filename)
                                    break
                            if metadata_path != ckpt_path:
                                break

                try:
                    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
                except Exception:
                    rank = 0
                if rank == 0:
                    print(f"  [DEBUG] DeepSpeed checkpoint directory, metadata path: {metadata_path}")

                checkpoint = torch.load(metadata_path, map_location='cpu', weights_only=False)

                saved_devices = checkpoint.get('devices', None)
                if saved_devices is None and os.path.isdir(ckpt_path):
                    try:
                        ds_checkpoint_root = os.path.dirname(os.path.dirname(metadata_path))
                        ds_state_dir = os.path.join(ds_checkpoint_root, 'checkpoint')
                        if os.path.isdir(ds_state_dir):
                            mp_files = [
                                f for f in os.listdir(ds_state_dir)
                                if f.startswith('mp_rank_') and f.endswith('_model_states.pt')
                            ]
                            if mp_files:
                                saved_devices = len(mp_files)
                    except Exception:
                        pass

                if 'samples_seen' in checkpoint:
                    samples_seen = checkpoint['samples_seen']
                    print(f"  Processed samples (samples_seen): {samples_seen}")
                else:
                    print("Warning: 'samples_seen' not found, estimating from global_step...")
                    global_step = checkpoint.get('global_step', 0)
                    samples_seen = global_step * self.config.micro_bsz * self.config.devices * self.config.accumulate_grad_batches
                    print(f"  Estimated samples: {samples_seen}")

                samples_per_step = self.config.micro_bsz * self.config.devices * self.config.accumulate_grad_batches
                target_global_step = samples_seen // samples_per_step
                print(f"  Samples per step: {samples_per_step}")
                print(f"  Target skip steps: {target_global_step}")

                self.lightning_module.set_skip_steps(target_global_step)
                elastic_mode = (saved_devices is not None and int(saved_devices) != int(self.config.devices))
                if elastic_mode:
                    print(
                        f"Warning: [Elastic Resume] Detected device change: saved_devices={saved_devices} -> current_devices={self.config.devices}. "
                        f"Skipping DeepSpeed optimizer state restore."
                    )
                    self.lightning_module.set_disable_optimizer_resume(True)
                else:
                    self.lightning_module.set_disable_optimizer_resume(False)
                self.lightning_module.set_resume_checkpoint_path(ckpt_path)

                print("  Loading model weights...")
                if os.path.isdir(ckpt_path):
                    base_dir = os.path.dirname(ckpt_path) if ckpt_path.endswith('trainer_state.ckpt') else ckpt_path
                    model_pth = os.path.join(base_dir, 'model.pth')
                    if os.path.isfile(model_pth):
                        state_dict = torch.load(model_pth, map_location='cpu', weights_only=False)
                        if hasattr(self.lightning_module, 'model'):
                            self.lightning_module.model.load_state_dict(state_dict, strict=False)
                        else:
                            self.lightning_module.load_state_dict(state_dict, strict=False)
                        print(f"Model weights loaded from {model_pth}")
                    else:
                        print(f"Warning: Weights file not found: {model_pth}")
                else:
                    state_dict = checkpoint['state_dict']
                    self.lightning_module.load_state_dict(state_dict, strict=False)
                    print("Model weights loaded")

            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("  Starting training from scratch...")

        # Start training
        self.trainer.fit(self.lightning_module, dataloader)

        print("\n" + "="*80)
        print("Training complete!")
        print("="*80)

    def save_model(self, output_dir: Optional[str] = None):
        """Save model (supports LoRA mode)."""
        from rwkvtune.training.model_save_utils import save_checkpoint_with_lora_support

        if output_dir is None:
            output_dir = self.config.proj_dir

        use_lora = getattr(self.config, 'use_lora', False)
        lora_save_mode = getattr(self.config, 'lora_save_mode', 'lora_only')

        print(f"\nSaving final model to: {output_dir}")
        if use_lora:
            print(f"  LoRA mode: {lora_save_mode}")

        try:
            save_checkpoint_with_lora_support(
                lightning_module=self.lightning_module,
                save_directory=output_dir,
                checkpoint_name="final",
                tokenizer=getattr(self, 'tokenizer', None),
                model_path=getattr(self, 'model_path', None),
                use_lora=use_lora,
                lora_save_mode=lora_save_mode,
                lora_config=getattr(self, 'lora_config', None),
            )
        except Exception as e:
            print(f"Warning: Failed to save complete model: {e}")
            print(f"  Falling back to weights-only save...")
            os.makedirs(output_dir, exist_ok=True)
            state_dict = {
                k.replace("model.", ""): v
                for k, v in self.lightning_module.state_dict().items()
            }
            save_path = os.path.join(output_dir, "final_model.pth")
            torch.save(state_dict, save_path)
            print(f"Weights saved to: {save_path}")
