from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

import numpy as np


PARTITION_VERSION = "1.0"
PARTITION_TYPES = {"iid", "non_iid"}


class DataPartitionError(ValueError):
    pass


@dataclass(frozen=True)
class PartitionAssignment:
    partition_type: str
    client_index: int
    client_count: int
    seed: int
    alpha: float | None
    indices: np.ndarray
    class_histogram: Dict[str, int]
    assignment_id: str

    def summary(self) -> Dict[str, Any]:
        return {
            "partition_version": PARTITION_VERSION,
            "partition_type": self.partition_type,
            "client_index": self.client_index,
            "client_count": self.client_count,
            "seed": self.seed,
            "alpha": self.alpha,
            "sample_count": int(len(self.indices)),
            "class_histogram": dict(self.class_histogram),
            "index_sha256": hashlib.sha256(
                np.asarray(self.indices, dtype=np.int64).tobytes()
            ).hexdigest(),
            "assignment_id": self.assignment_id,
            "disjoint_by_construction": True,
        }


def normalize_partition_type(value: Any) -> str:
    token = str(value or "iid").strip().lower().replace("-", "_")
    if token in {"noniid", "non_iid"}:
        token = "non_iid"
    if token not in PARTITION_TYPES:
        raise DataPartitionError(
            f"Unsupported data partition type {value!r}; choose iid or non_iid"
        )
    return token


def preferred_client_index(client_id: str, client_count: int) -> int | None:
    """Return zero-based slot for conventional client_N identifiers."""
    match = re.fullmatch(r"client[_-]?(\d+)", str(client_id).strip().lower())
    if not match:
        return None
    value = int(match.group(1)) - 1
    if 0 <= value < int(client_count):
        return value
    return None


def partition_policy_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    data = config.get("data", {}) or {}
    partition = data.get("partition", {}) or {}
    fed = config.get("federated", {}) or {}
    partition_type = normalize_partition_type(partition.get("type", "iid"))
    alpha = float(partition.get("alpha", 0.5))
    if partition_type == "non_iid" and alpha <= 0:
        raise DataPartitionError("Dirichlet alpha must be positive for non-IID")
    client_count = int(
        partition.get("client_count")
        or fed.get("expected_clients")
        or 1
    )
    if client_count <= 0:
        raise DataPartitionError("partition client_count must be positive")
    seed_value = partition.get("seed")
    seed = int(
        seed_value
        if seed_value is not None
        else config.get("execution", {}).get("seed", 42)
    )
    return {
        "partition_version": PARTITION_VERSION,
        "partition_type": partition_type,
        "partition_alpha": alpha if partition_type == "non_iid" else None,
        "partition_seed": seed,
        "partition_client_count": client_count,
    }


def _assignment_id(
    *,
    partition_type: str,
    client_index: int,
    client_count: int,
    seed: int,
    alpha: float | None,
    indices: np.ndarray,
) -> str:
    core = {
        "partition_version": PARTITION_VERSION,
        "partition_type": partition_type,
        "client_index": int(client_index),
        "client_count": int(client_count),
        "seed": int(seed),
        "alpha": alpha,
        "index_sha256": hashlib.sha256(
            np.asarray(indices, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _histogram(labels: np.ndarray, indices: np.ndarray) -> Dict[str, int]:
    if labels.size == 0 or indices.size == 0:
        return {}
    values, counts = np.unique(labels[indices], return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _iid_partitions(n_samples: int, client_count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples, dtype=np.int64)
    rng.shuffle(indices)
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(indices, client_count)]


def _dirichlet_partitions(
    labels: np.ndarray,
    client_count: int,
    seed: int,
    alpha: float,
) -> list[np.ndarray]:
    """Class-wise Dirichlet label-skew partition, disjoint by construction.

    Each class is independently divided among clients. A final deterministic
    repair prevents empty partitions on very small datasets without creating
    duplicate indices.
    """
    if alpha <= 0:
        raise DataPartitionError("Dirichlet alpha must be positive")
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(client_count)]

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label).astype(np.int64)
        rng.shuffle(class_indices)
        if class_indices.size == 0:
            continue
        proportions = rng.dirichlet(np.full(client_count, alpha, dtype=np.float64))
        counts = rng.multinomial(int(class_indices.size), proportions)
        cursor = 0
        for client_index, count in enumerate(counts.tolist()):
            if count:
                buckets[client_index].extend(
                    int(value) for value in class_indices[cursor:cursor + count]
                )
            cursor += count

    # Make every client usable when the dataset has at least one sample/client.
    # Move, never copy, an index from the largest donor partition.
    if len(labels) >= client_count:
        for empty_index in [i for i, values in enumerate(buckets) if not values]:
            donor = max(range(client_count), key=lambda i: len(buckets[i]))
            if len(buckets[donor]) <= 1:
                break
            position = int(rng.integers(0, len(buckets[donor])))
            buckets[empty_index].append(buckets[donor].pop(position))

    result: list[np.ndarray] = []
    for values in buckets:
        array = np.asarray(values, dtype=np.int64)
        rng.shuffle(array)
        result.append(array)
    return result


def make_partition_assignment(
    *,
    labels: Sequence[int] | np.ndarray,
    partition_type: str,
    client_index: int,
    client_count: int,
    seed: int,
    alpha: float | None = 0.5,
) -> PartitionAssignment:
    partition_type = normalize_partition_type(partition_type)
    client_count = int(client_count)
    client_index = int(client_index)
    seed = int(seed)
    if client_count <= 0:
        raise DataPartitionError("client_count must be positive")
    if not (0 <= client_index < client_count):
        raise DataPartitionError(
            f"client_index must be in [0,{client_count - 1}], got {client_index}"
        )

    label_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if label_array.size == 0:
        raise DataPartitionError("Cannot partition an empty dataset")

    if partition_type == "iid":
        partitions = _iid_partitions(len(label_array), client_count, seed)
        effective_alpha = None
    else:
        effective_alpha = float(alpha if alpha is not None else 0.5)
        partitions = _dirichlet_partitions(
            label_array, client_count, seed, effective_alpha
        )

    indices = partitions[client_index]
    assignment_id = _assignment_id(
        partition_type=partition_type,
        client_index=client_index,
        client_count=client_count,
        seed=seed,
        alpha=effective_alpha,
        indices=indices,
    )
    return PartitionAssignment(
        partition_type=partition_type,
        client_index=client_index,
        client_count=client_count,
        seed=seed,
        alpha=effective_alpha,
        indices=indices,
        class_histogram=_histogram(label_array, indices),
        assignment_id=assignment_id,
    )


def apply_partition_assignment(config: Dict[str, Any], descriptor: Dict[str, Any]) -> Dict[str, Any]:
    """Apply server-provided partition descriptor to a client config."""
    partition = config.setdefault("data", {}).setdefault("partition", {})
    partition_type = normalize_partition_type(descriptor.get("partition_type", "iid"))
    client_count = int(descriptor["client_count"])
    client_index = int(descriptor["client_index"])
    seed = int(descriptor["seed"])
    alpha = descriptor.get("alpha")
    if partition_type == "non_iid":
        alpha = float(alpha if alpha is not None else 0.5)
        if alpha <= 0:
            raise DataPartitionError("Dirichlet alpha must be positive")
    else:
        alpha = 0.5

    partition.update(
        {
            "type": partition_type,
            "alpha": alpha,
            "seed": seed,
            "client_count": client_count,
            "client_index": client_index,
            "assignment_id": None,
            "disjoint": True,
            "source": "server",
        }
    )
    return config


def partition_file_token(partition_type: str, alpha: float | None = None) -> str:
    partition_type = normalize_partition_type(partition_type)
    if partition_type == "iid":
        return "iid"
    value = float(alpha if alpha is not None else 0.5)
    token = (f"{value:g}").replace("-", "m").replace(".", "p")
    return f"non_iid_a{token}"
