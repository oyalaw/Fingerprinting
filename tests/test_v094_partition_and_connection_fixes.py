from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_fingerprint.dataset_manager import DatasetManager
from ai_fingerprint.traffic.analysis import PacketRecord, connection_facing_packets
from ai_fingerprint.training_metrics import PerformanceLogWriter


class _DummySource:
    def __init__(self, labels):
        self.labels = np.asarray(labels, dtype=np.int64)
    def __len__(self):
        return len(self.labels)
    def get(self, index):
        return np.asarray([index], dtype=np.float32)
    def get_with_target(self, index):
        return self.get(index), np.asarray(self.labels[index], dtype=np.int64)


def _manager(kind: str, client_index: int, alpha: float = 0.5):
    manager = DatasetManager.__new__(DatasetManager)
    manager.config = {
        "data": {
            "split": "train",
            "shuffle": False,
            "max_samples": None,
            "partition": {
                "type": kind,
                "alpha": alpha,
                "seed": 1234,
                "client_count": 3,
                "client_index": client_index,
                "client_id": f"client_{client_index+1}",
                "disjoint": True,
            },
        },
        "execution": {"deployment": "federated", "seed": 42, "batch_size": 1},
        "node": {"role": "client"},
        "ai": {"application": "image_classification"},
    }
    manager.name = "dummy"
    manager.source = _DummySource([i % 10 for i in range(300)])
    manager.rng = np.random.default_rng(42)
    manager._indices = manager._build_partition_indices()
    manager._cursor = 0
    return manager


def test_iid_partitions_are_disjoint_and_cover_dataset():
    shards = [_manager("iid", i)._indices for i in range(3)]
    sets = [set(x.tolist()) for x in shards]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert set().union(*sets) == set(range(300))
    assert [len(x) for x in shards] == [100, 100, 100]


def test_dirichlet_non_iid_is_reproducible_and_disjoint():
    shards_a = [_manager("non_iid", i, 0.5)._indices for i in range(3)]
    shards_b = [_manager("non_iid", i, 0.5)._indices for i in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(shards_a, shards_b))
    sets = [set(x.tolist()) for x in shards_a]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert set().union(*sets) == set(range(300))
    histograms = []
    for manager in [_manager("non_iid", i, 0.5) for i in range(3)]:
        histograms.append(manager.partition_summary()["class_histogram"])
    assert len({tuple(sorted(h.items())) for h in histograms}) > 1


def _packet(i, src, dst, sport, dport, t):
    return PacketRecord(
        index=i, timestamp_epoch=t, frame_length=100,
        src_ip=src, dst_ip=dst, src_port=sport, dst_port=dport,
        transport_protocol="TCP", tcp_flags_hex="0x0010",
        tcp_syn=0, tcp_ack=1, tcp_fin=0, tcp_rst=0,
        retransmission=0, tls_record_lengths=(), direction="unknown",
    )


def test_connection_filter_does_not_merge_two_ports_from_same_ip():
    packets = [
        _packet(1, "10.42.0.145", "10.42.0.1", 57000, 8080, 1.0),
        _packet(2, "10.42.0.1", "10.42.0.145", 8080, 57000, 1.1),
        _packet(3, "10.42.0.145", "10.42.0.1", 51161, 8080, 10.0),
        _packet(4, "10.42.0.1", "10.42.0.145", 8080, 51161, 10.1),
    ]
    selected = connection_facing_packets(
        packets, client_ip="10.42.0.145", client_port=51161,
        proxy_ip="10.42.0.1", proxy_port=8080,
    )
    assert [p.index for p in selected] == [3, 4]


def test_partition_specific_metric_file_is_written(tmp_path: Path):
    config = {
        "experiment": {"output_dir": str(tmp_path)},
        "data": {"partition": {"type": "non_iid", "alpha": 0.5}},
    }
    writer = PerformanceLogWriter(config)
    writer.write_client_round({"experiment_id": "exp1", "client_id": "client_1", "round": 0})
    assert (tmp_path / "round_metrics.csv").exists()
    assert (tmp_path / "round_metrics_non_iid_alpha_0p5.csv").exists()
