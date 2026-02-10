"""
Training callbacks for Pretrain and SFT
"""
import os
import time
import json
import math
import shutil
import torch
from typing import Optional, Any
from lightning.pytorch.callbacks import Callback

from rwkvtune.models.rwkv7.config import SimpleTrainConfig


class TrainingCallback(Callback):
    """Training callback - handles logging and model saving"""
    
    def __init__(
        self, 
        config: SimpleTrainConfig, 
        tokenizer: Optional[Any] = None,
        model_path: Optional[str] = None,
        lora_config: Optional[dict] = None,
    ):
        """
        Args:
            config: Training configuration
            tokenizer: Tokenizer object (optional), saved with model if provided
            model_path: Original model path (optional), used to infer config
            lora_config: LoRA config dict (optional), used when saving LoRA weights
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.lora_config = lora_config
        self.loss_file = os.path.join(config.proj_dir, "loss_data.jsonl")
        
        if os.path.exists(self.loss_file):
            os.remove(self.loss_file)
        
        self.logger = None
        self.saved_checkpoints_queue = []
    
    def _init_logger(self):
        """Initialize logger"""
        if self.logger is not None:
            return

        try:
            rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
        except Exception:
            rank = 0
        if rank != 0:
            return

        config = self.config
        
        if config.report_to.lower() == "swanlab":
            self.logger = None
        
        elif config.report_to.lower() == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                run_name = config.run_name if config.run_name else f"rwkv7-{time.strftime('%Y%m%d-%H%M%S')}"
                log_dir = os.path.join(config.proj_dir, "tensorboard", run_name)
                self.logger = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("Error: tensorboard not installed, please run: pip install tensorboard")
                self.logger = None
        else:
            print(f"Warning: Unsupported report_to value: {config.report_to}, only 'swanlab' or 'tensorboard' supported")
            self.logger = None

    def _resolve_epoch_steps(self, trainer) -> int:
        epoch_steps = self.config.epoch_steps
        if epoch_steps is None:
            epoch_steps = getattr(trainer, 'num_training_batches', None)

        if epoch_steps is None:
            raise ValueError(
                "Unable to infer epoch_steps (trainer.num_training_batches is None). "
                "Please set --epoch_steps to a finite integer."
            )

        if isinstance(epoch_steps, float):
            if not math.isfinite(epoch_steps):
                raise ValueError(
                    "Unable to infer epoch_steps because trainer.num_training_batches is infinite/unknown. "
                    "Please set --epoch_steps to a finite integer (map-style dataset required)."
                )
            epoch_steps = int(epoch_steps)

        if not isinstance(epoch_steps, int) or epoch_steps <= 0:
            raise ValueError(f"Invalid epoch_steps={epoch_steps!r}. Please set --epoch_steps to a positive integer.")

        return epoch_steps
    
    def _log_metrics(self, metrics, step):
        """Log metrics to logging system"""
        if self.logger is None:
            return
        
        config = self.config
        
        if config.report_to.lower() == "swanlab":
            self.logger.log(metrics, step=step)
        elif config.report_to.lower() == "tensorboard":
            for key, value in metrics.items():
                self.logger.add_scalar(key, value, step)
    
    def write_data(self, loss_data, t_cost, kt_s):
        """Write training data to file"""
        with open(self.loss_file, 'a') as f:
            json.dump({
                "loss": float(loss_data),
                "t_cost": t_cost,
                "kt_s": kt_s
            }, f)
            f.write('\n')
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Update learning rate at batch start (following RWKV-PEFT)"""
        import math
        
        config = self.config
        epoch_steps = self._resolve_epoch_steps(trainer)
        real_step = (trainer.current_epoch + config.epoch_begin) * epoch_steps + (batch_idx + 1)
        total_training_steps = epoch_steps * config.epoch_count
        
        # Learning rate scheduling (cosine annealing + warmup)
        if config.warmup_steps > 0 and real_step < config.warmup_steps:
            lr = config.lr_init * real_step / config.warmup_steps
        else:
            if config.lr_final == config.lr_init or config.epoch_count == 0:
                lr = config.lr_init
            else:
                if config.warmup_steps >= total_training_steps:
                    lr = config.lr_init
                else:
                    progress = (real_step - config.warmup_steps) / max(1, total_training_steps - config.warmup_steps)
                    progress = min(max(progress, 0.0), 1.0)
                    lr = config.lr_final + (config.lr_init - config.lr_final) * (0.5 + 0.5 * math.cos(math.pi * progress))
        
        for param_group in trainer.optimizers[0].param_groups:
            if "my_lr_scale" in param_group:
                param_group["lr"] = lr * param_group["my_lr_scale"]
            else:
                param_group["lr"] = lr
        
        trainer.my_lr = lr
        
        if not hasattr(trainer, 'rwkvzen_callback_inited'):
            trainer.rwkvzen_callback_inited = True
            if trainer.is_global_zero:
                trainer.my_avg_loss = 0.0
                trainer.my_accumulation_counter = 0
                trainer.my_optimizer_step = 0
                trainer.my_time_ns = time.time_ns()
                trainer.my_loss_sum = 0.0
                trainer.my_loss_count = 0
                trainer.my_epoch_loss = 0.0
                
                trainer.my_log = open(
                    os.path.join(self.config.proj_dir, "train_log.txt"), "a"
                )
                trainer.my_log.write(f"\n{'='*80}\n")
                trainer.my_log.write(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                trainer.my_log.write(f"Config: {vars(self.config)}\n")
                trainer.my_log.write(f"{'='*80}\n")
                trainer.my_log.flush()
                
                if self.config.report_to:
                    self._init_logger()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """End of training batch"""
        config = self.config
        
        if isinstance(outputs, dict):
            loss = outputs['loss']
        else:
            loss = outputs
        
        # Multi-GPU sync (all_reduce is collective operation)
        if trainer.world_size > 1:
            import torch.distributed as dist
            loss_for_sync = loss.clone().detach()
            dist.all_reduce(loss_for_sync, op=dist.ReduceOp.SUM)
            loss = loss_for_sync / float(trainer.world_size)
        
        # Restore original loss value (Lightning auto-normalizes)
        loss = loss * float(trainer.accumulate_grad_batches)
        loss = loss.item()

        epoch_steps = self._resolve_epoch_steps(trainer)
        real_batch = batch_idx + 1 + trainer.current_epoch * epoch_steps
        
        if trainer.is_global_zero:
            if not hasattr(trainer, 'my_time_ns'):
                trainer.my_time_ns = time.time_ns()
            if not hasattr(trainer, 'my_loss_sum'):
                trainer.my_loss_sum = 0.0
            if not hasattr(trainer, 'my_loss_count'):
                trainer.my_loss_count = 0
            if not hasattr(trainer, 'my_epoch_loss'):
                trainer.my_epoch_loss = 0.0
            
            t_now = time.time_ns()
            kt_s = 0
            t_cost = 0
            
            try:
                token_per_step = config.ctx_len * config.micro_bsz * config.devices
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                t_cost = 1.0 / t_cost
            except:
                pass
            
            trainer.my_time_ns = t_now
            trainer.my_loss_sum += loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            
            pl_module.log("REAL it/s", t_cost, prog_bar=True, on_step=True, logger=False, sync_dist=False)
            pl_module.log("Kt/s", kt_s, prog_bar=True, on_step=True, logger=False, sync_dist=False)
            pl_module.log("lr", trainer.my_lr, prog_bar=True, on_step=True, logger=False, sync_dist=False)
            pl_module.log("sum_loss", trainer.my_epoch_loss, prog_bar=True, on_step=True, logger=False, sync_dist=False)
            pl_module.log("loss", loss, prog_bar=True, on_step=True, logger=False, sync_dist=False)
            
            epoch_steps = config.epoch_steps if config.epoch_steps is not None else trainer.num_training_batches
            real_step = (trainer.current_epoch + config.epoch_begin) * epoch_steps + (batch_idx + 1)
            real_batch = batch_idx + 1 + trainer.current_epoch * epoch_steps
            
            if trainer.accumulate_grad_batches is not None and trainer.accumulate_grad_batches > 1:
                if not hasattr(trainer, 'my_avg_loss'):
                    trainer.my_avg_loss = 0.0
                if not hasattr(trainer, 'my_accumulation_counter'):
                    trainer.my_accumulation_counter = 0
                
                trainer.my_accumulation_counter += 1
                trainer.my_avg_loss += loss / trainer.accumulate_grad_batches
                
                if trainer.my_accumulation_counter >= trainer.accumulate_grad_batches:
                    self.write_data(trainer.my_avg_loss, t_cost, kt_s)
                    
                    if trainer.logger is not None:
                        pl_module.log("train/loss", trainer.my_avg_loss, 
                                     prog_bar=False, logger=True, on_step=True, sync_dist=False)
                        pl_module.log("train/lr", trainer.my_lr, 
                                     prog_bar=False, logger=True, on_step=True, sync_dist=False)
                        if kt_s > 0:
                            pl_module.log("train/throughput_kt_s", kt_s, 
                                         prog_bar=False, logger=True, on_step=True, sync_dist=False)
                    
                    if self.logger is not None:
                        metrics = {
                            "train/loss": trainer.my_avg_loss,
                            "train/lr": trainer.my_lr,
                        }
                        if kt_s > 0:
                            metrics["train/throughput_kt_s"] = kt_s
                        self._log_metrics(metrics, trainer.my_optimizer_step)
                    
                    trainer.my_avg_loss = 0.0
                    trainer.my_accumulation_counter = 0
                    trainer.my_optimizer_step += 1
            else:
                self.write_data(loss, t_cost, kt_s)
                
                if trainer.logger is not None:
                    pl_module.log("train/loss", loss, 
                                 prog_bar=False, logger=True, on_step=True, sync_dist=False)
                    pl_module.log("train/lr", trainer.my_lr, 
                                 prog_bar=False, logger=True, on_step=True, sync_dist=False)
                    if kt_s > 0:
                        pl_module.log("train/throughput_kt_s", kt_s, 
                                     prog_bar=False, logger=True, on_step=True, sync_dist=False)
                
                if self.logger is not None:
                    metrics = {
                        "train/loss": loss,
                        "train/lr": trainer.my_lr,
                    }
                    if kt_s > 0:
                        metrics["train/throughput_kt_s"] = kt_s
                    self._log_metrics(metrics, trainer.my_optimizer_step)
                trainer.my_optimizer_step += 1

        # Save by batch count (all ranks must call for DeepSpeed collective ops)
        if config.save_every_n_batches > 0:
            if real_batch > 0 and real_batch % config.save_every_n_batches == 0:
                self._save_checkpoint(trainer, pl_module, batch=real_batch)
    
    def _save_checkpoint(self, trainer, pl_module, epoch=None, batch=None):
        """Save checkpoint (supports LoRA mode)"""
        # Skip saving during dummy forward phase using samples_seen_at_resume
        if batch is not None and hasattr(pl_module, 'samples_seen_at_resume'):
            current_samples = batch * self.config.micro_bsz * self.config.devices
            if current_samples <= pl_module.samples_seen_at_resume:
                return

        # Fallback protection logic
        if hasattr(pl_module, 'skip_until_step') and trainer.global_step < pl_module.skip_until_step:
            return

        from rwkvtune.training.model_save_utils import save_checkpoint_with_lora_support
        
        config = self.config
        
        if batch is not None:
            checkpoint_name = f"rwkv7-batch{batch}"
        elif epoch is not None:
            checkpoint_name = f"rwkv7-epoch{epoch + 1}"
        else:
            checkpoint_name = "rwkv7-final"
        
        if trainer.is_global_zero:
            print(f"\nSaving checkpoint: {checkpoint_name}")
        
        use_lora = getattr(config, 'use_lora', False)
        lora_save_mode = getattr(config, 'lora_save_mode', 'lora_only')
        lora_config = getattr(self, 'lora_config', None)
        
        if use_lora and trainer.is_global_zero:
            print(f"  LoRA mode: {lora_save_mode}")
        
        try:
            # Save PyTorch Lightning checkpoint for resume training
            # Place .ckpt in model directory to keep output dir clean
            model_dir = os.path.join(config.proj_dir, checkpoint_name)
            # All ranks need directory for DeepSpeed sharded writes
            os.makedirs(model_dir, exist_ok=True)
            
            ckpt_path = os.path.join(model_dir, "trainer_state.ckpt")
            tmp_path = f"{ckpt_path}.tmp"
            
            save_optimizer_state = getattr(config, 'save_optimizer_state', True)
            if trainer.is_global_zero:
                print(f"Saving training state: {ckpt_path} (tmp: {tmp_path})")
                print(f"   Save optimizer state: {save_optimizer_state}")
            
            # Use temp path to prevent file corruption on write interruption
            # trainer.save_checkpoint is a collective operation - all ranks must call
            trainer.strategy.barrier()
            
            trainer.save_checkpoint(tmp_path, weights_only=not save_optimizer_state)
            
            # Atomic rename - rank 0 only
            if trainer.is_global_zero:
                if os.path.exists(ckpt_path):
                    if os.path.isdir(ckpt_path):
                        shutil.rmtree(ckpt_path)
                    else:
                        os.remove(ckpt_path)

                # Move directory if DeepSpeed generated a folder
                if os.path.isdir(tmp_path):
                    shutil.move(tmp_path, ckpt_path)
                else:
                    os.rename(tmp_path, ckpt_path)

                print(f"Checkpoint saved: {ckpt_path}")

                self._rotate_checkpoints(model_dir)
            
            # Wait for rank 0 to complete file operations
            trainer.strategy.barrier()

            # Call on all ranks - internally handles collective ops and writes on rank 0 only
            save_checkpoint_with_lora_support(
                lightning_module=pl_module,
                save_directory=config.proj_dir,
                checkpoint_name=checkpoint_name,
                tokenizer=self.tokenizer,
                model_path=self.model_path,
                use_lora=use_lora,
                lora_save_mode=lora_save_mode,
                lora_config=lora_config,
                is_main_process=trainer.is_global_zero,
            )

            if str(getattr(config, 'strategy', '')).lower() == 'deepspeed_stage_3':
                export_ok = True
                export_err = None
                trainer.strategy.barrier()
                if trainer.is_global_zero:
                    try:
                        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

                        state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)

                        while any(k.startswith('model.') for k in state_dict.keys()):
                            state_dict = {k[6:]: v for k, v in state_dict.items()}
                        while any(k.startswith('module.') for k in state_dict.keys()):
                            state_dict = {k[7:]: v for k, v in state_dict.items()}
                        while any(k.startswith('_forward_module.') for k in state_dict.keys()):
                            state_dict = {k[len('_forward_module.'):]: v for k, v in state_dict.items()}
                        while any(k.startswith('net.') for k in state_dict.keys()):
                            state_dict = {k[4:]: v for k, v in state_dict.items()}

                        save_dtype = str(getattr(config, 'precision', 'fp32')).lower()
                        if save_dtype == 'bf16':
                            state_dict = {
                                k: (v.to(dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v)
                                for k, v in state_dict.items()
                            }
                        elif save_dtype == 'fp16':
                            state_dict = {
                                k: (v.to(dtype=torch.float16) if torch.is_tensor(v) and v.is_floating_point() else v)
                                for k, v in state_dict.items()
                            }

                        full_model_path = os.path.join(model_dir, 'model.pth')
                        torch.save(state_dict, full_model_path)
                        print(f"Full weights exported: {full_model_path}")
                    except Exception as e:
                        export_ok = False
                        export_err = e
                        print(f"Failed to export full weights: {e}")

                trainer.strategy.barrier()
                if (not export_ok) and trainer.is_global_zero:
                    raise export_err
            
        except Exception as e:
            # Fail-fast: print error, sync barrier, then re-raise
            if trainer.is_global_zero:
                print(f"Save failed: {e}")
            try:
                trainer.strategy.barrier()
            except Exception:
                pass
            raise
    
    def _rotate_checkpoints(self, new_ckpt_path):
        """Manage checkpoint count, delete old checkpoints"""
        limit = getattr(self.config, 'save_total_limit', 0)
        if limit <= 0:
            return
            
        self.saved_checkpoints_queue.append(new_ckpt_path)
        
        while len(self.saved_checkpoints_queue) > limit:
            old_ckpt = self.saved_checkpoints_queue.pop(0)
            if os.path.exists(old_ckpt):
                try:
                    print(f"Deleting old checkpoint (limit={limit}): {old_ckpt}")
                    if os.path.isdir(old_ckpt):
                        shutil.rmtree(old_ckpt)
                    else:
                        os.remove(old_ckpt)
                except Exception as e:
                    print(f"Failed to delete old checkpoint: {e}")
            else:
                print(f"Old checkpoint not found, skipping: {old_ckpt}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at training epoch end"""
        config = self.config
        
        # Reset epoch loss accumulators (all ranks need consistency)
        trainer.my_loss_sum = 0.0
        trainer.my_loss_count = 0
        trainer.my_epoch_loss = 0.0

        # Epoch save - all ranks must participate (collective operation)
        if (config.epoch_save > 0 and 
            trainer.current_epoch % config.epoch_save == 0) or \
           (trainer.current_epoch == config.epoch_count - 1):
            self._save_checkpoint(trainer, pl_module, epoch=trainer.current_epoch)

        # Logging only on rank 0
        if trainer.is_global_zero:
            if 'train_loss' in trainer.callback_metrics:
                avg_loss = trainer.callback_metrics['train_loss'].item()
            else:
                avg_loss = 0.0
                
            log_msg = (
                f"Epoch {trainer.current_epoch + 1} | "
                f"Loss: {avg_loss:.6f} | "
                f"Perplexity: {math.exp(avg_loss) if avg_loss > 0 else float('inf'):.4f} | "
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            trainer.my_log.write(log_msg)
            trainer.my_log.flush()
            
            if self.logger is not None:
                epoch_metrics = {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_perplexity": math.exp(avg_loss) if avg_loss > 0 else float('inf'),
                    "train/epoch": trainer.current_epoch + 1,
                }
                log_step = getattr(trainer, 'my_optimizer_step', trainer.global_step)
                self._log_metrics(epoch_metrics, log_step)
            
            if hasattr(trainer, 'my_accumulation_loss_sum'):
                trainer.my_accumulation_loss_sum = 0.0
                trainer.my_accumulation_loss_count = 0
    
    def on_train_end(self, trainer, pl_module):
        """Training end cleanup"""
        if trainer.is_global_zero and self.logger is not None:
            if self.config.report_to.lower() == "tensorboard":
                self.logger.close()
                print("TensorBoard logger closed")
            elif self.config.report_to.lower() == "swanlab":
                print("SwanLab logs saved")

