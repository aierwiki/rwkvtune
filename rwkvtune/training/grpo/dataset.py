"""
GRPO Dataset - Consistent design with SFT.

Design principles:
- User is responsible for data loading and preprocessing
- Only accepts HuggingFace Dataset
- Dataset must contain 'prompt' and 'input_ids' fields (consistent with SFT)
- Other fields are automatically passed to reward functions
- Tokenization is done during preprocessing, not during training
"""
import os
import torch
from typing import Dict, List, Any, Optional
from datasets import Dataset as HFDataset, IterableDataset as HFIterableDataset


class GRPODataBuffer:
    """
    GRPO data buffer for caching generated completions and computed advantages.
    Supports multiple iterations (num_iterations > 1) and manages data lifecycle.
    """

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_batch(
        self,
        prompts: List[str],
        prompt_ids: torch.Tensor,
        completions: List[str],
        completion_ids: torch.Tensor,
        completion_logps: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        masks: torch.Tensor,
    ):
        """Add a batch of generated data to the buffer."""
        B_times_G = len(prompts)
        for i in range(B_times_G):
            item = {
                'prompt': prompts[i],
                'prompt_ids': prompt_ids[i],
                'completion': completions[i],
                'completion_ids': completion_ids[i],
                'completion_logps': completion_logps[i],
                'reward': rewards[i],
                'advantage': advantages[i],
                'mask': masks[i],
            }
            self.buffer.append(item)
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a training batch from the buffer."""
        if len(self.buffer) == 0:
            return None
        indices = torch.randperm(len(self.buffer))[:batch_size]
        batch_items = [self.buffer[i] for i in indices]
        batch = {
            'prompt_ids': torch.stack([item['prompt_ids'] for item in batch_items]),
            'completion_ids': torch.stack([item['completion_ids'] for item in batch_items]),
            'completion_logps': torch.stack([item['completion_logps'] for item in batch_items]),
            'rewards': torch.stack([item['reward'] for item in batch_items]),
            'advantages': torch.stack([item['advantage'] for item in batch_items]),
            'masks': torch.stack([item['mask'] for item in batch_items]),
        }
        return batch

    def is_full(self) -> bool:
        return len(self.buffer) >= self.buffer_size

    def clear(self):
        self.buffer = []

    def __len__(self) -> int:
        return len(self.buffer)


class RepeatBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Repeat batch sampler - aligns with trl-main's "repeat batch" semantics.
    
    Each batch from the underlying sampler is yielded repeat_batch_count times.
    This ensures that GRPO can rollout once per prompt batch, then train for
    multiple steps without skipping any prompts.
    """

    def __init__(self, batch_sampler: torch.utils.data.Sampler[List[int]], repeat_batch_count: int):
        if repeat_batch_count <= 0:
            raise ValueError(f"repeat_batch_count must > 0, got {repeat_batch_count}")
        self._batch_sampler = batch_sampler
        self._repeat_batch_count = repeat_batch_count

    def __iter__(self):
        for batch in self._batch_sampler:
            for _ in range(self._repeat_batch_count):
                yield batch

    def __len__(self) -> int:
        return len(self._batch_sampler) * self._repeat_batch_count

    @property
    def batch_size(self) -> int:
        bs = getattr(self._batch_sampler, "batch_size", None)
        if bs is None:
            raise AttributeError("RepeatBatchSampler underlying batch_sampler has no attribute 'batch_size'")
        return int(bs)

    @property
    def drop_last(self) -> bool:
        return bool(getattr(self._batch_sampler, "drop_last", False))

    @property
    def sampler(self):
        return getattr(self._batch_sampler, "sampler", None)

    def set_epoch(self, epoch: int) -> None:
        sampler = getattr(self._batch_sampler, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)


def create_grpo_dataloader(
    dataset: HFDataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    repeat_batch_count: int = 1,
    seed: int = 42,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """
    Create GRPO dataloader with repeat batch semantics.
    
    Args:
        dataset: HuggingFace Dataset (must contain 'prompt' and 'input_ids' fields)
        batch_size: Number of prompts per batch (not B*G)
        num_workers: DataLoader worker count
        shuffle: Whether to shuffle data
        repeat_batch_count: Times each prompt batch repeats (typically generate_every)
        seed: Random seed for distributed sampler
        world_size: Number of processes (None = read from env)
        rank: Current process rank (None = read from env)
    
    Returns:
        DataLoader configured for GRPO training
    """
    from torch.utils.data import DataLoader
    from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
    from torch.utils.data.distributed import DistributedSampler

    if not isinstance(dataset, (HFDataset, HFIterableDataset)):
        raise TypeError(f"dataset must be HuggingFace Dataset, got {type(dataset)}")

    if 'prompt' not in dataset.column_names:
        raise ValueError(f"dataset must contain 'prompt' field! Current fields: {dataset.column_names}")

    if 'input_ids' not in dataset.column_names:
        raise ValueError(f"dataset must contain 'input_ids' field! Current fields: {dataset.column_names}")

    def collate_fn(batch):
        """GRPO collate function - handles input_ids and other fields."""
        prompts = [item['prompt'] for item in batch]
        input_ids_list = [item['input_ids'] for item in batch]
        result = {
            'prompts': prompts,
            'input_ids': input_ids_list,
        }
        for key in batch[0].keys():
            if key not in ['prompt', 'input_ids']:
                values = [item[key] for item in batch]
                result[key] = values
        return result

    if batch_size <= 0:
        raise ValueError(f"batch_size must > 0, got {batch_size}")

    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))

    if world_size > 1:
        base_sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )
    else:
        base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    base_batch_sampler = BatchSampler(base_sampler, batch_size=batch_size, drop_last=False)

    if repeat_batch_count != 1:
        batch_sampler = RepeatBatchSampler(base_batch_sampler, repeat_batch_count=repeat_batch_count)
    else:
        batch_sampler = base_batch_sampler

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader
