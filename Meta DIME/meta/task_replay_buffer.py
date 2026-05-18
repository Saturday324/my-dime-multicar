from collections import deque
from typing import Dict, Iterable, Optional

import numpy as np


class TaskReplayBuffer:
    def __init__(self, task_ids: Iterable[str], capacity_per_task: int):
        self.capacity_per_task = int(capacity_per_task)
        self.task_ids = [str(task_id) for task_id in task_ids]
        self.task_to_index = {
            task_id: task_idx
            for task_idx, task_id in enumerate(self.task_ids)
        }
        self._buffers = {
            str(task_id): deque(maxlen=self.capacity_per_task)
            for task_id in self.task_ids
        }

    def add(
        self,
        task_id: str,
        observation,
        action,
        next_observation,
        reward,
        done,
    ) -> None:
        task_id = str(task_id)
        if task_id not in self._buffers:
            raise KeyError(f"Unknown task_id: {task_id}")

        self._buffers[task_id].append(
            {
                "observation": np.asarray(observation, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "next_observation": np.asarray(next_observation, dtype=np.float32),
                "reward": np.float32(reward),
                "done": np.float32(done),
            }
        )

    def size(self, task_id: Optional[str] = None) -> int:
        if task_id is None:
            return int(sum(len(buffer) for buffer in self._buffers.values()))
        return int(len(self._buffers[str(task_id)]))

    def eligible_task_ids(self, min_transitions: int) -> list:
        min_transitions = int(min_transitions)
        return [
            task_id
            for task_id, buffer in self._buffers.items()
            if len(buffer) >= min_transitions
        ]

    def _sample_indices(
        self,
        data_size: int,
        batch_size: int,
        rng: Optional[np.random.RandomState] = None,
        recent: bool = False,
        exclude_indices: Optional[Iterable[int]] = None,
        replace: Optional[bool] = None,
    ) -> np.ndarray:
        if recent:
            start = max(int(data_size) - int(batch_size), 0)
            return np.arange(start, int(data_size), dtype=np.int64)

        rng = rng or np.random
        if exclude_indices is None:
            excluded = set()
        else:
            excluded = set(int(idx) for idx in np.asarray(exclude_indices).reshape(-1))
        candidates = np.asarray(
            [idx for idx in range(int(data_size)) if idx not in excluded],
            dtype=np.int64,
        )
        if len(candidates) == 0:
            return np.asarray([], dtype=np.int64)

        if replace is None:
            replace = len(candidates) < int(batch_size)
        return np.asarray(
            rng.choice(candidates, size=int(batch_size), replace=bool(replace)),
            dtype=np.int64,
        )

    def _batch_from_items(self, selected) -> Dict[str, np.ndarray]:
        return {
            "observations": np.stack(
                [item["observation"] for item in selected], axis=0
            ).astype(np.float32),
            "actions": np.stack(
                [item["action"] for item in selected], axis=0
            ).astype(np.float32),
            "next_observations": np.stack(
                [item["next_observation"] for item in selected], axis=0
            ).astype(np.float32),
            "rewards": np.asarray(
                [item["reward"] for item in selected], dtype=np.float32
            ),
            "dones": np.asarray(
                [item["done"] for item in selected], dtype=np.float32
            ),
        }

    def _sample_from_task(
        self,
        task_id: str,
        batch_size: int,
        rng: Optional[np.random.RandomState] = None,
        recent: bool = False,
        exclude_indices: Optional[Iterable[int]] = None,
        return_indices: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        task_id = str(task_id)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        data = list(self._buffers[task_id])
        if len(data) == 0:
            return None

        indices = self._sample_indices(
            len(data),
            batch_size,
            rng=rng,
            recent=recent,
            exclude_indices=exclude_indices,
        )
        if len(indices) == 0:
            return None

        selected = [data[int(idx)] for idx in np.asarray(indices).reshape(-1)]
        batch = self._batch_from_items(selected)
        if return_indices:
            batch["indices"] = indices.astype(np.int64)
        return batch

    def sample_context(
        self,
        task_id: str,
        batch_size: int,
        rng: Optional[np.random.RandomState] = None,
        recent: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        return self._sample_from_task(task_id, batch_size, rng=rng, recent=recent)

    def sample_query(
        self,
        task_id: str,
        batch_size: int,
        rng: Optional[np.random.RandomState] = None,
        exclude_indices: Optional[Iterable[int]] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        return self._sample_from_task(
            task_id,
            batch_size,
            rng=rng,
            recent=False,
            exclude_indices=exclude_indices,
        )

    def sample_meta_batch(
        self,
        meta_batch_tasks: int,
        context_batch_size: int,
        query_batch_size: int,
        rng: Optional[np.random.RandomState] = None,
        min_task_transitions: Optional[int] = None,
        disjoint_context_query: bool = True,
    ) -> Optional[Dict[str, np.ndarray]]:
        rng = rng or np.random
        min_required = max(int(context_batch_size), int(query_batch_size), int(min_task_transitions or 0))
        if disjoint_context_query:
            min_required = max(min_required, int(context_batch_size) + int(query_batch_size))
        eligible = self.eligible_task_ids(min_required)
        if len(eligible) == 0:
            return None

        num_tasks = min(int(meta_batch_tasks), len(eligible))
        chosen = list(rng.choice(eligible, size=num_tasks, replace=False))

        context_batches = []
        query_batches = []
        actual_task_ids = []
        task_indices = []
        for task_id in chosen:
            context_batch = self._sample_from_task(
                task_id,
                context_batch_size,
                rng=rng,
                return_indices=bool(disjoint_context_query),
            )
            exclude_indices = context_batch.get("indices") if context_batch is not None else None
            query_batch = self.sample_query(
                task_id,
                query_batch_size,
                rng=rng,
                exclude_indices=exclude_indices if disjoint_context_query else None,
            )
            if context_batch is None or query_batch is None:
                continue
            context_batch.pop("indices", None)
            context_batches.append(context_batch)
            query_batches.append(query_batch)
            actual_task_ids.append(task_id)
            task_indices.append(self.task_to_index[str(task_id)])

        if len(context_batches) == 0:
            return None

        def stack_batches(batches, key):
            return np.stack([batch[key] for batch in batches], axis=0).astype(np.float32)

        return {
            "task_ids": actual_task_ids,
            "task_indices": np.asarray(task_indices, dtype=np.int32),
            "context_observations": stack_batches(context_batches, "observations"),
            "context_actions": stack_batches(context_batches, "actions"),
            "context_next_observations": stack_batches(context_batches, "next_observations"),
            "context_rewards": stack_batches(context_batches, "rewards"),
            "context_dones": stack_batches(context_batches, "dones"),
            "query_observations": stack_batches(query_batches, "observations"),
            "query_actions": stack_batches(query_batches, "actions"),
            "query_next_observations": stack_batches(query_batches, "next_observations"),
            "query_rewards": stack_batches(query_batches, "rewards"),
            "query_dones": stack_batches(query_batches, "dones"),
        }
