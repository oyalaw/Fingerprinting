from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .workloads.base import Workload


@dataclass
class ClientUpdate:
    parameters: List[np.ndarray]
    num_examples: int
    metrics: Dict[str, float]


def fedavg(
    updates: List[ClientUpdate],
) -> List[np.ndarray]:
    if not updates:
        raise ValueError("FedAvg requires at least one client update")

    tensor_count = len(updates[0].parameters)
    if any(len(update.parameters) != tensor_count for update in updates):
        raise ValueError("Client updates have different tensor counts")

    weights = np.asarray(
        [max(int(update.num_examples), 1) for update in updates],
        dtype=np.float64,
    )
    total_weight = float(weights.sum())

    aggregated: List[np.ndarray] = []
    for tensor_index in range(tensor_count):
        reference = np.asarray(
            updates[0].parameters[tensor_index]
        )
        accumulator = np.zeros(
            reference.shape,
            dtype=np.float64,
        )

        for weight, update in zip(weights, updates):
            value = np.asarray(update.parameters[tensor_index])
            if value.shape != reference.shape:
                raise ValueError(
                    f"Parameter tensor {tensor_index} shape mismatch: "
                    f"{value.shape} != {reference.shape}"
                )
            accumulator += value.astype(np.float64) * float(weight)

        averaged = accumulator / total_weight

        if np.issubdtype(reference.dtype, np.integer):
            averaged = np.rint(averaged)

        aggregated.append(
            averaged.astype(reference.dtype, copy=False)
        )

    return aggregated


class SynchronousFedAvgCoordinator:
    """
    Minimal synchronous FedAvg coordinator used by the experiment server.

    Every client downloads the same global parameters for round r, performs
    local training, uploads one model update, and waits until all expected
    clients have submitted before the server advances to round r + 1.
    """

    def __init__(
        self,
        workload: Workload,
        rounds: int,
        expected_clients: int,
    ) -> None:
        self.workload = workload
        self.rounds = int(rounds)
        self.expected_clients = int(expected_clients)

        self._condition = threading.Condition()
        self._round = 0
        self._global_parameters = [
            value.copy()
            for value in workload.get_parameters()
        ]
        self._updates: Dict[int, Dict[str, ClientUpdate]] = {}

    @property
    def current_round(self) -> int:
        with self._condition:
            return self._round

    def get_global(
        self,
    ) -> Tuple[int, List[np.ndarray], bool]:
        with self._condition:
            if self._round >= self.rounds:
                return self._round, [], True

            return (
                self._round,
                [value.copy() for value in self._global_parameters],
                False,
            )

    def submit_update(
        self,
        round_index: int,
        client_id: str,
        parameters: List[np.ndarray],
        num_examples: int,
        metrics: Dict[str, float] | None = None,
    ) -> Tuple[int, bool]:
        with self._condition:
            if round_index != self._round:
                raise ValueError(
                    f"Stale or future federated update from {client_id!r}: "
                    f"round={round_index}, current_round={self._round}"
                )

            round_updates = self._updates.setdefault(
                round_index,
                {},
            )
            if client_id in round_updates:
                raise ValueError(
                    f"Duplicate update from client {client_id!r} "
                    f"for round {round_index}"
                )

            round_updates[client_id] = ClientUpdate(
                parameters=[value.copy() for value in parameters],
                num_examples=max(int(num_examples), 1),
                metrics=dict(metrics or {}),
            )

            if len(round_updates) == self.expected_clients:
                self._global_parameters = fedavg(
                    list(round_updates.values())
                )
                self.workload.set_parameters(
                    self._global_parameters
                )
                self._round += 1
                self._condition.notify_all()
            else:
                while (
                    self._round == round_index
                    and len(round_updates) < self.expected_clients
                ):
                    self._condition.wait()

            done = self._round >= self.rounds
            return self._round, done
