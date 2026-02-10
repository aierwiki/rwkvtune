"""
SFT (Supervised Fine-Tuning) Trainer - Following trl-main design
"""
import os
import time
import torch
import lightning as pl
from lightning import Trainer
from pathlib import Path
from typing import Optional, Union, Any
from torch.utils.data import DataLoader

from rwkvtune.training.sft_config import SFTConfig
from rwkvtune.training.lightning_module import RWKV7LightningModule
from rwkvtune.training.lightning_module_infctx import RWKV7InfctxLightningModule
from rwkvtune.training.callbacks import TrainingCallback
from rwkvtune.models.rwkv7.operators import load_wkv_operator

# torch.compile configuration
# Following RWKV-PEFT: do not set allow_unspec_int_on_nn_module
# Benefits:
#   1. @torch.compile compatible with DeepSpeed checkpoint
#   2. Gradient checkpoint works properly (saves memory)
#   3. Some recompilation warnings (normal, can be ignored)



# HuggingFace Dataset support
try:
    from datasets import Dataset as HFDataset, IterableDataset as HFIterableDataset
    _has_datasets = True
except ImportError:
    _has_datasets = False
    HFDataset = None
    HFIterableDataset = None


class SFTTrainer:
    """
    SFT (Supervised Fine-Tuning) Trainer - Following trl-main and GRPO design
    
    Features:
    1. Supports two initialization methods: new API (model param) and old API (config)
    2. User handles data preprocessing, passes processed HuggingFace Dataset
    3. Automatically loads model architecture from model_config
    4. Supports chat format, instruction tuning, and various SFT scenarios
    
    Example 1 - New API (recommended, consistent with GRPOTrainer):
        ```python
        from rwkvtune.training import SFTConfig, SFTTrainer
        from rwkvtune import AutoModel, AutoTokenizer
        from datasets import Dataset
        
        # Create config
        config = SFTConfig(
            ctx_len=2048,
            micro_bsz=4,
            epoch_count=10,
        )
        
        # Create trainer (auto-loads model and tokenizer)
        trainer = SFTTrainer(
            model="/path/to/model",        # Model path
            args=config,                   # Config
            train_dataset=dataset,         # Dataset
            processing_class=tokenizer,    # Optional tokenizer
        )
        trainer.train()
        ```
    
    Example 2 - Old API (backward compatible):
        ```python
        from rwkvtune.training import SFTConfig, SFTTrainer
        
        # Create config
        config = SFTConfig(
            model_config="rwkv7-0.1b",
            load_model="/path/to/model.pth",
            ctx_len=2048,
            micro_bsz=4,
            epoch_count=10,
        )
        
        # Create trainer
        trainer = SFTTrainer(config=config, train_dataset=dataset)
        trainer.train()
        ```
    """
    
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        args: Optional[SFTConfig] = None,
        train_dataset: Optional[Union[HFDataset, HFIterableDataset]] = None,
        processing_class: Optional[Any] = None,
        lora_config: Optional[dict] = None,
    ):
        """
        Args:
            model: Model (required). Can be:
                - String: Model path (dir or .pth), calls AutoModel.from_pretrained()
                - torch.nn.Module: Instantiated model object
            args: SFT training config
            train_dataset: HuggingFace Dataset (required, must contain input_ids and labels)
                - User handles data loading and preprocessing
                - Must contain tokenized input_ids and labels
                - labels with -100 indicate positions to ignore (usually prompt)
            processing_class: Tokenizer (optional)
            lora_config: LoRA config dict (optional), used when saving LoRA weights
        """
        
        if args is None:
            args = SFTConfig()
        self.config = args
        self.lora_config = lora_config
        
        # Validate dataset
        if train_dataset is None:
            raise ValueError(
                "train_dataset is required!\n"
                "Please use HuggingFace Dataset, e.g.:\n"
                "  from datasets import Dataset\n"
                "  dataset = Dataset.from_list([{'input_ids': [...], 'labels': [...]}, ...])\n"
                "  trainer = SFTTrainer(model=model, train_dataset=dataset)\n"
            )
        
        if not _has_datasets:
            raise ImportError(
                "datasets library required!\n"
                "Please run: pip install datasets"
            )
        
        if not isinstance(train_dataset, (HFDataset, HFIterableDataset)):
            raise TypeError(
                f"train_dataset must be HuggingFace Dataset, got {type(train_dataset)}\n"
                "Please use Dataset.from_list(samples) to create dataset."
            )
        
        # Validate required fields
        if 'input_ids' not in train_dataset.column_names:
            raise ValueError(
                f"train_dataset must contain 'input_ids' field!\n"
                f"Current fields: {train_dataset.column_names}\n"
                "Ensure each sample has 'input_ids' and 'labels' fields."
            )
        
        if 'labels' not in train_dataset.column_names:
            raise ValueError(
                f"train_dataset must contain 'labels' field!\n"
                f"Current fields: {train_dataset.column_names}\n"
                "For SFT, labels typically use -100 for prompt and real tokens for completion."
            )
        
        print(f"Using HuggingFace Dataset: {len(train_dataset)} samples")
        print(f"  Fields: {train_dataset.column_names}")
        
        self.dataset = train_dataset
        
        
        Path(self.config.proj_dir).mkdir(parents=True, exist_ok=True)
        
        
        if self.config.seed >= 0:
            pl.seed_everything(self.config.seed)
        
        # Setup model
        print("Initializing model...")
        self.model = self._setup_model(model)
        self.model_path = model if isinstance(model, str) else None
        
        # Load model config into self.config (for Lightning module)
        self._load_model_config_for_lightning()
        
        # Setup tokenizer
        print("Initializing tokenizer...")
        self.tokenizer = self._setup_tokenizer(processing_class)
        
        # Setup environment variables
        self._setup_environment()
        
        # Create Lightning module
        print("Creating Lightning module...")
        
        # Select module based on training type
        if self.config.train_type == 'infctx':
            print(f"Using infinite context training mode (train_type={self.config.train_type})")
            print(f"  Sequence length: {self.config.ctx_len}, chunk size: {self.config.chunk_ctx}")
            print("Loading RWKV7 infinite context operator...")
            from rwkvtune.models.rwkv7.operators import load_cuda_operator_infctx
            load_cuda_operator_infctx()
            # infctx mode does not support direct model input
            self.lightning_module = RWKV7InfctxLightningModule(self.config)
            if model is not None:
                print("Loading model weights to Lightning module...")
                self.lightning_module.model.load_state_dict(self.model.state_dict(), strict=False)
                print("Weights loaded")
        else:
            print(f"Using standard training mode (train_type={self.config.train_type})")
            device_type = 'cpu' if self.config.accelerator == 'cpu' else 'cuda'
            print(f"Loading RWKV7 {device_type.upper()} standard training operator...")
            load_wkv_operator(device=device_type, for_inference=False, head_size=self.config.head_size_a)
            
            # Pass external model directly (supports LoRA)
            # Use the passed model directly instead of copying weights
            # This preserves LoRA frozen state and model structure
            if model is not None:
                print("Using passed model directly (preserving LoRA frozen state)...")
                self.lightning_module = RWKV7LightningModule(self.config, model=self.model)
                print("Model loaded to Lightning module")
            else:
                self.lightning_module = RWKV7LightningModule(self.config)
        
        print(f"\nModel parameters: {sum(p.numel() for p in self.lightning_module.parameters()) / 1e6:.2f}M")
        print(f"Trainable parameters: {sum(p.numel() for p in self.lightning_module.parameters() if p.requires_grad) / 1e6:.2f}M")
        
        # Create Lightning Trainer
        self.trainer = self._create_trainer()
        
        print("SFTTrainer initialization complete\n")
    
    def _setup_model(self, model: Union[str, torch.nn.Module]) -> torch.nn.Module:
        """
        Setup model（follows trl-main design）
        
        Args:
            model: Model path or instantiated model
                - String: Model path (dir or .pth), calls AutoModel.from_pretrained()
                - torch.nn.Module: Instantiated model object
            
        Returns:
            model: Model object
        """
        from rwkvtune import AutoModel
        
        if isinstance(model, str):
            # String: model path (dir or file)
            print(f"Loading model from path: {model}")
            # Load to CPU to avoid extra GPU memory usage
            # Lightning/DeepSpeed will distribute to GPUs automatically
            model_obj = AutoModel.from_pretrained(model, device="cpu")
            print("Model loaded (CPU)")
            return model_obj
            
        elif isinstance(model, torch.nn.Module):
            # Instantiated model
            print(f"Using instantiated model: {type(model).__name__}")
            return model
        else:
            raise TypeError(
                f"Invalid model type: {type(model)}\n"
                f"Expected str (model path) or torch.nn.Module (model object)"
            )
    
    def _setup_tokenizer(self, processing_class: Optional[Any]):
        """
        Setup tokenizer（follows trl-main design）
        
        Args:
            processing_class: User-provided tokenizer (optional)
            
        Returns:
            tokenizer object
        """
        if processing_class is not None:
            # User provided tokenizer
            print(f"Using user-provided tokenizer: {type(processing_class).__name__}")
            return processing_class
        elif self.model_path is not None:
            # Auto-load tokenizer from model_path
            print(f"Loading tokenizer from model path: {self.model_path}")
            from rwkvtune import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("Tokenizer loaded")
            return tokenizer
        else:
            # Cannot auto-load tokenizer, require user to provide
            raise ValueError(
                "Cannot auto-load tokenizer. Please provide via:\n"
                "  1. Pass processing_class parameter：SFTTrainer(model=..., processing_class=tokenizer, ...)\n"
                "  2. Pass model path (string)：SFTTrainer(model='path/to/model', ...)\n"
                "     （Will auto-load tokenizer from that path）"
            )
    
    def _load_model_config_for_lightning(self):
        """
        Load model config for Lightning module（new API）
        
        Extract config from loaded model and fill into config
        Also pass training params to model config (bidirectional sync)
        """
        try:
            # Get config from model
            if hasattr(self.model, 'config'):
                model_config = self.model.config
                
                # ========== 1. Read architecture params from model config ==========
                # Fill required architecture params
                self.config.n_layer = model_config.n_layer
                self.config.n_embd = model_config.n_embd
                self.config.vocab_size = model_config.vocab_size
                self.config.head_size_a = model_config.head_size_a
                
                # LORA dimensions
                self.config.dim_att_lora = getattr(model_config, 'dim_att_lora', 32)
                self.config.dim_gate_lora = getattr(model_config, 'dim_gate_lora', 128)
                self.config.dim_mv_lora = getattr(model_config, 'dim_mv_lora', 32)
                
                # Derived params
                self.config.dim_att = self.config.n_embd
                self.config.dim_ffn = int((self.config.n_embd * 3.5) // 32 * 32)
                
                # ========== 2. Pass training params to model config ==========
                # Model forward checks these params
                # If not passed, model uses defaults, training params won't take effect
                
                # Gradient checkpoint (to reduce activation memory)
                model_config.grad_cp = self.config.grad_cp
                
                # Context length (may differ from pretraining)
                model_config.ctx_len = self.config.ctx_len
                
                print("Config loaded from model")
                print(f"  Architecture: {self.config.n_layer}layers, {self.config.n_embd}dims, {self.config.vocab_size}vocab")
                print(f"  LORA dims: att={self.config.dim_att_lora}, gate={self.config.dim_gate_lora}, mv={self.config.dim_mv_lora}")
                print(f"  Training params: grad_cp={self.config.grad_cp}, ctx_len={self.config.ctx_len}")
            else:
                raise ValueError("Model has no config attribute, cannot load config")
                
        except Exception as e:
            print("Error: Cannot load config from model")
            print(f"   {str(e)}")
            raise
    
    def _setup_environment(self):
        """Setup environment variables"""
        config = self.config
        
        # Set device type based on accelerator
        device_type = 'cpu' if config.accelerator == 'cpu' else 'cuda'
        
        if device_type == 'cpu':
            print("\n" + "⚠️ " * 20)
            print("Using CPU training mode (not optimized, slow performance)")
            print(f"Recommended config: n_layer≤6, n_embd≤512, ctx_len≤512, micro_bsz≤2")
            print(f"Current config: n_layer={config.n_layer}, n_embd={config.n_embd}, ctx_len={config.ctx_len}")
            print("⚠️ " * 20 + "\n")
        
        os.environ["RWKV_MY_TESTING"] = config.my_testing
        os.environ["RWKV_CTXLEN"] = str(config.ctx_len)
        os.environ["RWKV_HEAD_SIZE_A"] = str(config.head_size_a)
        os.environ["RWKV_FLOAT_MODE"] = config.precision
        os.environ["RWKV_JIT_ON"] = "0"
        os.environ["WKV"] = device_type
        os.environ["RWKV_DEVICE"] = device_type
        os.environ["FUSED_KERNEL"] = "0"
        os.environ["RWKV_TRAIN_TYPE"] = "none"
        
        # CUDA optimizations (GPU mode only)
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
        """Create Lightning Trainer"""
        config = self.config
        
        # Adjust config based on accelerator type
        accelerator = config.accelerator if config.accelerator in ['cpu', 'gpu'] else 'gpu'
        
        # Precision settings
        precision = config.precision
        if accelerator == 'cpu':
            if precision in ["bf16", "fp16"]:
                print(f"⚠️  CPU training not recommended with{precision}, switched to fp32")
                precision = 32
            elif precision == "fp32":
                precision = 32
        else:
            if precision == "fp32":
                precision = 32
            elif precision == "fp16":
                precision = 16
            # Keep bf16 as string
        
        # Strategy settings
        strategy = config.strategy
        if accelerator == 'cpu':
            strategy = 'auto'
            devices = 1
        else:
            devices = config.devices
            # 🔧 CRITICAL FIX: Pass string directly to Lightning, do not manually create DeepSpeedStrategy
            # Following RWKV-PEFT train.py line 253: pass directly 'deepspeed_stage_2'
            # Lightning uses correct default DeepSpeed config (no offload optimizer)
            # 
            # Previous error: manually creating DeepSpeedStrategy with offload_optimizer=(devices>1)
            #    Caused optimizer state offload on multi-GPU:
            #    - Uneven memory (GPU4: 48GB, GPU5: 27GB, 21GB diff!)
            #    - Slow training (0.59 vs 0.96 it/s)
            #    - Loss not converging
            # 
            # Fix: pass string directly, Lightning auto-config (offload_optimizer=False)
            #    - Balanced memory (<2GB diff between GPUs)
            #    - Fast (~0.96 it/s)
            #    - Fully aligned with RWKV-PEFT!
            pass  # Keep strategy as string, no extra processing
        
        # Create callback（pass tokenizer, model_path and lora_config for saving）
        callbacks = [TrainingCallback(
            config=config, 
            tokenizer=getattr(self, 'tokenizer', None),
            model_path=getattr(self, 'model_path', None),
            lora_config=getattr(self, 'lora_config', None),
        )]
        
        # Config logger（SwanLab using official Lightning integration）
        pl_logger = False
        if config.report_to and config.report_to.lower() == "swanlab":
            try:
                from swanlab.integration.pytorch_lightning import SwanLabLogger
                run_name = config.run_name if config.run_name else f"rwkv7-{time.strftime('%Y%m%d-%H%M%S')}"
                pl_logger = SwanLabLogger(
                    project="rwkvtune",
                    experiment_name=run_name,
                    config=vars(config),
                )
                print(f"SwanLab Logger created: {run_name}")
            except ImportError:
                print("swanlab not installed: pip install swanlab")
            except Exception as e:
                print(f"SwanLab initialization failed: {e}")
        
        # Create trainer kwargs（following RWKV-PEFT best practices）
        trainer_kwargs = dict(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=1,
            precision=precision,
            max_epochs=config.epoch_count,
            accumulate_grad_batches=config.accumulate_grad_batches,
            gradient_clip_val=config.grad_clip,
            callbacks=callbacks,
            enable_checkpointing=False,
            logger=pl_logger,
            num_sanity_val_steps=0,
            # Log frequency（from config）
            log_every_n_steps=config.log_every_n_steps if pl_logger else int(1e20),
            # Disable validation check following RWKV-PEFT
            check_val_every_n_epoch=int(1e20),  # Avoid unnecessary validation
            # Disable ModelSummary for DeepSpeed Stage 3 compatibility
            enable_model_summary=False,
            # Progress bar enabled with SwanLabLogger
            # （）
            enable_progress_bar=True,
        )
        
        # epoch_steps meaning（following RWKV-PEFT trainer.py line 52）
        # epoch_steps is total physical batches across all GPUs
        # RWKV-PEFT formula: total_optimizer_steps = epoch_steps / devices / accumulate_grad_batches
        # Each GPU trains: epoch_steps / devices physical batches
        # Lightning limit_train_batches is batches per GPU
        #
        # When epoch_steps=None, do not set limit_train_batches
        # Lightning uses distributed sampler automatically(use_distributed_sampler=True, default), 
        # Distributes dataset evenly across GPUs, each GPU processes len(dataset) // devices samples
        if config.epoch_steps is not None:
            # User specified epoch_steps（total physical batches across all GPUs）
            # Calculate batches per GPU
            batches_per_device = config.epoch_steps // config.devices
            trainer_kwargs['limit_train_batches'] = batches_per_device
            print(f"Total steps per epoch (all GPUs): {config.epoch_steps}")
            print(f"Batches per GPU: {batches_per_device}")
            print(f"Optimizer updates per epoch: {batches_per_device // config.accumulate_grad_batches}")
        # else: do not set limit_train_batches, use full dataset（Lightning shards automatically）
        
        trainer = Trainer(**trainer_kwargs)
        
        return trainer
    
    def _collate_fn(self, batch):
        """
        DataLoader collate function
        
        Convert HuggingFace Dataset list format to tensor
        Handle variable length sequences: pad to longest in batch, max ctx_len, length multiple of 16
        """
        import torch
        
        IGNORE_INDEX = -100
        pad_token_id = 0  # RWKV tokenizer uses 0 as padding token
        CHUNK_LEN = 16  # RWKV7 WKV operator requires sequence length multiple of 16
        
        # Calculate max length: max ctx_len, multiple of 16
        max_len = (self.config.ctx_len // CHUNK_LEN) * CHUNK_LEN
        
        # Extract input_ids and labels, convert to tensor
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]
        
        # First truncate to max_len (prevent oversized samples)
        input_ids = [ids[:max_len] for ids in input_ids]
        labels = [lab[:max_len] for lab in labels]
        
        # Calculate longest sequence in batch
        batch_max_len = max(len(ids) for ids in input_ids) if input_ids else 0
        
        # Limit padding max length: max_len, multiple of 16
        pad_max_len = min(batch_max_len, max_len)
        
        # Round up to multiple of 16 (RWKV7 requirement)
        if pad_max_len % CHUNK_LEN != 0:
            pad_max_len = ((pad_max_len // CHUNK_LEN) + 1) * CHUNK_LEN
            
        # Ensure not exceeding max_len (truncate if rounded up)
        # Note: may truncate last few tokens, minor impact with large max_len
        if pad_max_len > max_len:
            pad_max_len = max_len
        
        # Padding helper function
        # Note: pad_sequence only pads to batch max length。
        # If pad_max_len > batch_max_len (due to rounding), pad_sequence is insufficient。
        # Therefore we need manual padding。
        def pad_to_len(t, length, value):
            if len(t) >= length:
                return t[:length]
            else:
                return torch.cat([t, torch.full((length - len(t),), value, dtype=t.dtype)], dim=0)
        
        input_ids = [pad_to_len(ids, pad_max_len, pad_token_id) for ids in input_ids]
        labels = [pad_to_len(lab, pad_max_len, IGNORE_INDEX) for lab in labels]
        
        # Convert to Tensor
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
    
    def train(self):
        """Start training"""
        print("\n" + "="*80)
        print("Starting SFT training")
        print("="*80 + "\n")
        
        # Create DataLoader (aligned with RWKV-PEFT)
        # 
        if not getattr(self.config, 'dataloader_shuffle', True):
            print("✓ DataLoader shuffle: False (using pre-shuffled/sorted data order)")
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.micro_bsz,
            shuffle=getattr(self.config, 'dataloader_shuffle', True),
            num_workers=self.config.num_workers,  # RWKV-PEFT uses 1, avoids uneven memory
            persistent_workers=False,  # RWKV-PEFT config: False
            pin_memory=True if self.config.accelerator == 'gpu' else False,
            drop_last=True,  # RWKV-PEFT config: drop incomplete batch
            collate_fn=self._collate_fn,  # Add collate function
        )
        
        # Handle resume training (Dummy Forward strategy)
        if self.config.resume_from_checkpoint:
            ckpt_path = self.config.resume_from_checkpoint
            
            # Must be a directory
            if not os.path.isdir(ckpt_path):
                raise ValueError(f"❌ Error: resume_from_checkpoint must be a directory: {ckpt_path}")

            # Case 1: Points directly to checkpoint directory (e.g. output/rwkv7-batch480), contains trainer_state.ckpt
            if os.path.exists(os.path.join(ckpt_path, 'trainer_state.ckpt')):
                ckpt_path = os.path.join(ckpt_path, 'trainer_state.ckpt')
                print(f"📂 Detected checkpoint directory, loading: {ckpt_path}")
            
            # Case 2: Points to parent directory with multiple checkpoints (e.g. output/)
            else:
                # 1. Find subdirs containing trainer_state.ckpt (new format)
                subdirs = [d for d in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, d))]
                valid_subdirs = []
                for d in subdirs:
                    if os.path.exists(os.path.join(ckpt_path, d, 'trainer_state.ckpt')):
                        valid_subdirs.append(d)
                
                if valid_subdirs:
                    # Sort subdirs by modification time
                    valid_subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)), reverse=True)
                    latest_subdir = valid_subdirs[0]
                    ckpt_path = os.path.join(ckpt_path, latest_subdir, 'trainer_state.ckpt')
                    print(f"📂 Detected new format checkpoint dir, selecting latest: {ckpt_path}")
                
                else:
                    # 2. Look for .ckpt files (old format compatibility)
                    ckpt_files = [f for f in os.listdir(ckpt_path) if f.endswith('.ckpt')]
                    
                    if ckpt_files:
                        # Prefer last.ckpt
                        if 'last.ckpt' in ckpt_files:
                            ckpt_path = os.path.join(ckpt_path, 'last.ckpt')
                        else:
                            # Sort by modification time
                            ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(ckpt_path, x)), reverse=True)
                            ckpt_path = os.path.join(ckpt_path, ckpt_files[0])
                        print(f"📂 Detected old format .ckpt file, selecting latest: {ckpt_path}")
                    
                    # 3. If no .ckpt, look for adapter_model.bin (LoRA)
                    elif os.path.exists(os.path.join(ckpt_path, 'adapter_model.bin')):
                        ckpt_path = os.path.join(ckpt_path, 'adapter_model.bin')
                        print(f"📂 Detected directory, selecting LoRA weights: {ckpt_path}")
                    
                    # 4. Otherwise assume directory is DeepSpeed checkpoint
                    else:
                        print(f"📂 Detected directory without .ckpt/bin, assuming DeepSpeed checkpoint: {ckpt_path}")

            print(f"\n🔄 Preparing to resume from checkpoint: {ckpt_path}")
            
            try:
                # 1. Load checkpoint metadata (CPU)
                # Note: PyTorch 2.6+ defaults weights_only=True, which does not support custom classes (like SFTConfig)
                # Therefore we explicitly set weights_only=False
                metadata_path = ckpt_path
                if os.path.isdir(ckpt_path):
                    # DeepSpeed Stage 2/3: Lightning ckpt is usually a directory, metadata in *_model_states.pt
                    candidates = [
                        os.path.join(ckpt_path, 'checkpoint', 'mp_rank_00_model_states.pt'),
                    ]
                    for candidate in candidates:
                        if os.path.isfile(candidate):
                            metadata_path = candidate
                            break

                    if metadata_path == ckpt_path:
                        # Fallback: scan directory for *_model_states.pt
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
                        print(f"   [DEBUG] DeepSpeed checkpoint dir, metadata load path: {metadata_path}")

                checkpoint = torch.load(metadata_path, map_location='cpu', weights_only=False)

                # 2.1 Read old checkpoint config (for elastic resume)
                saved_devices = checkpoint.get('devices', None)
                saved_micro_bsz = checkpoint.get('micro_bsz', None)
                saved_accum = checkpoint.get('accumulate_grad_batches', None)

                # 2.1.1 Backward compat: old checkpoint may not have saved devices。
                # For DeepSpeed checkpoint (dir), try to infer saved world size from mp_rank_XX_model_states.pt。
                if saved_devices is None:
                    try:
                        # metadata_path is usually: <ds_ckpt_root>/checkpoint/mp_rank_00_model_states.pt
                        # Therefore ds_ckpt_root = dirname(dirname(metadata_path))
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
                
                # 2. Get processed sample count
                if 'samples_seen' in checkpoint:
                    samples_seen = checkpoint['samples_seen']
                    print(f"   Processed samples (samples_seen): {samples_seen}")
                else:
                    # Fallback: estimate from global_step (assuming config unchanged)
                    # Note: estimate may be wrong if config changed
                    print("⚠️  Checkpoint missing 'samples_seen', Estimating from global_step...")
                    global_step = checkpoint.get('global_step', 0)
                    # Assuming config unchanged (strong assumption)
                    samples_seen = global_step * self.config.micro_bsz * self.config.devices * self.config.accumulate_grad_batches
                    print(f"   Estimated samples: {samples_seen} (based on current config)")

                # 3. Calculate steps to skip with current config
                # samples_per_step = micro_bsz * devices * accumulate_grad_batches
                samples_per_step = self.config.micro_bsz * self.config.devices * self.config.accumulate_grad_batches
                target_global_step = samples_seen // samples_per_step
                
                print(f"   Samples per step: {samples_per_step}")
                print(f"   Target skip steps (target_global_step): {target_global_step}")
                
                # 4. Setup model state
                self.lightning_module.set_skip_steps(target_global_step)
                self.lightning_module.samples_seen_at_resume = samples_seen

                # 4.1 Elastic resume (world size change) handling
                # DeepSpeed ZeRO2/3 optimizer state is sharded by world size。
                # If devices change on resume, forcing optimizer state restore often causes DeepSpeed internal state inconsistency
                # （e.g. backward stage IndexError / bucket mismatch）。
                # Therefore:
                # - devices unchanged: allow optimizer state restore (delayed load)
                # - devices changed: elastic mode, restore weights only, skip optimizer state
                elastic_mode = (saved_devices is not None and int(saved_devices) != int(self.config.devices))
                if elastic_mode:
                    print(
                        f"⚠️  [Elastic Resume] Detected devices change: saved_devices={saved_devices} -> current_devices={self.config.devices}. "
                        f"Will skip DeepSpeed optimizer state restore (weights only + Dummy Forward for data alignment)。"
                    )
                    self.lightning_module.set_disable_optimizer_resume(True)
                else:
                    self.lightning_module.set_disable_optimizer_resume(False)

                # Lazy load optimizer (if elastic_mode=True, LightningModule ignores this path)
                self.lightning_module.set_resume_checkpoint_path(ckpt_path)
                
                # 5. Load model weights (state_dict)
                # Note: We manually load weights instead of letting Trainer load, 
                # Because we start Dummy Forward from global_step=0
                print("   Loading model weights...")
                # Check if DeepSpeed checkpoint (directory)
                if os.path.isdir(ckpt_path):
                    # Our checkpoint structure：
                    #   rwkv7-batchXX/
                    #     trainer_state.ckpt        (Lightning/DeepSpeed training state)
                    #     model.pth                 (rank0 exported loadable weights)
                    # For DeepSpeed dir, prefer restoring weights from model.pth for Elastic Resume。
                    model_pth = os.path.join(os.path.dirname(ckpt_path), 'model.pth') if ckpt_path.endswith('trainer_state.ckpt') else os.path.join(ckpt_path, 'model.pth')
                    if os.path.isfile(model_pth):
                        state_dict = torch.load(model_pth, map_location='cpu', weights_only=False)
                        self.lightning_module.load_state_dict(state_dict, strict=False)
                        self.lightning_module.set_resume_model_weights_path(model_pth)
                        print(f"✅ Model weights loaded (from {model_pth})")
                    else:
                        print(
                            f"⚠️  Loadable weights file not found: {model_pth}. "
                            f"DeepSpeed sharded checkpoint dir does not support direct weight restore here。"
                        )
                else:
                    # Standard .ckpt file or .bin weights file
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        # Assuming pure weights file (e.g. adapter_model.bin)
                        print("⚠️  Checkpoint missing 'state_dict', Assuming file is state_dict itself (LoRA weights only)")
                        state_dict = checkpoint
                    
                    # Remove 'model.' prefix (if exists)
                    # LightningModule state_dict keys are usually "model.xxx" or "xxx"
                    # Our LightningModule has self.model
                    self.lightning_module.load_state_dict(state_dict, strict=False)
                    print("✅ Model weights loaded")

            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                print("   Will start training from scratch...")

        # Start training
        self.trainer.fit(self.lightning_module, dataloader)
        
        print("\n" + "="*80)
        print("Training complete！")
        print("="*80)
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        Save model (supports LoRA mode)
        
        Args:
            output_dir: Output directory. If None, uses config.proj_dir
        """
        from rwkvtune.training.model_save_utils import save_checkpoint_with_lora_support
        
        if output_dir is None:
            output_dir = self.config.proj_dir
        
        use_lora = getattr(self.config, 'use_lora', False)
        lora_save_mode = getattr(self.config, 'lora_save_mode', 'lora_only')
        
        print(f"\n💾 Saving final model to: {output_dir}")
        if use_lora:
            print(f"  LoRA mode: {lora_save_mode}")
        
        # Use LoRA-compatible save function
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
            # If saving full model fails, fall back to saving weights only
            print(f"⚠️  Failed to save full model: {e}")
            print(f"   Falling back to saving weights only...")
            
            os.makedirs(output_dir, exist_ok=True)
            state_dict = {
                k.replace("model.", ""): v 
                for k, v in self.lightning_module.state_dict().items()
            }
            
            save_path = os.path.join(output_dir, "final_model.pth")
            torch.save(state_dict, save_path)
            print(f"✓ Weights saved: {save_path}")
