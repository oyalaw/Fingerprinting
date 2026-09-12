from __future__ import annotations

import bisect
import csv
import json
import math
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .architecture_models import (
    FISHER_TOP_K,
    SIZE_NORMALIZED_EXACT_DROP,
    candidate_feature_columns,
    fisher_score_ranking,
)
from .fingerprinting_dataset import (
    GROUND_TRUTH_LABEL_FIELDS,
    OPTIONAL_CONTEXT_LABEL_FIELDS,
    PROXY_FEATURE_METADATA_FIELDS,
    _read_ground_truth_indices,
)
from .traffic import extract_feature_rows, read_packet_sequence_csv
from .traffic.analysis import PacketRecord, connection_facing_packets, client_facing_packets


ROUND_SCHEMA_VERSION = "1.0"
ROUND_INFERENCE_METHOD = "proxy_directional_burst_state_machine_v1"


class RoundFingerprintingError(RuntimeError):
    pass


@dataclass(frozen=True)
class RoundInferenceConfig:
    bin_sec: float = 0.25
    direction_dominance: float = 0.65
    min_bin_bytes: int = 2048
    bridge_gap_sec: float = 0.75
    min_major_transfer_bytes: int = 4_096
    min_training_gap_sec: float = 0.05
    max_round_sec: float = 3600.0
    max_transfer_size_ratio: float = 8.0
    min_round_packets: int = 20
    packet_information_threshold: int = 2
    burst_gap_sec: float = 0.05
    idle_threshold_sec: float = 0.5


@dataclass(frozen=True)
class TrafficBurst:
    direction: str
    start_epoch: float
    end_epoch: float
    bytes_up: int
    bytes_down: int
    packet_count: int

    @property
    def dominant_bytes(self) -> int:
        return self.bytes_down if self.direction == "down" else self.bytes_up

    @property
    def purity(self) -> float:
        total = self.bytes_up + self.bytes_down
        if total <= 0:
            return 0.0
        return float(self.dominant_bytes) / float(total)


@dataclass(frozen=True)
class InferredRound:
    inferred_round_index: int
    start_epoch: float
    end_epoch: float
    download_start_epoch: float
    download_end_epoch: float
    upload_start_epoch: float
    upload_end_epoch: float
    download_bytes: int
    upload_bytes: int
    packet_count: int
    confidence: float
    method: str = ROUND_INFERENCE_METHOD

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_epoch - self.start_epoch)

    @property
    def training_gap_sec(self) -> float:
        return max(0.0, self.upload_start_epoch - self.download_end_epoch)


@dataclass(frozen=True)
class GroundTruthRound:
    round_index: int
    start_epoch: float
    end_epoch: float
    download_start_epoch: Optional[float] = None
    download_end_epoch: Optional[float] = None
    training_start_epoch: Optional[float] = None
    training_end_epoch: Optional[float] = None
    upload_start_epoch: Optional[float] = None
    upload_end_epoch: Optional[float] = None


def _numeric(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _safe_slug(value: Any) -> str:
    text = str(value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned.strip("_") or "all"


def _manifest_artifact(manifest_path: Path, declared: Any) -> Optional[Path]:
    if not declared:
        return None
    declared_path = Path(str(declared))
    candidates = [
        declared_path,
        manifest_path.parent / declared_path.name,
        manifest_path.parent / declared_path,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    matches = list(manifest_path.parent.rglob(declared_path.name))
    return matches[0] if matches else None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _dedupe_packets(packets: Sequence[PacketRecord]) -> List[PacketRecord]:
    seen = set()
    result: List[PacketRecord] = []
    for packet in sorted(packets, key=lambda p: (p.timestamp_epoch, p.index)):
        key = (
            round(float(packet.timestamp_epoch), 6),
            int(packet.index),
            packet.src_ip,
            packet.dst_ip,
            int(packet.src_port),
            int(packet.dst_port),
            int(packet.frame_length),
            packet.direction,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(packet)
    return result


def discover_client_packet_sequences(
    collected_root: str | Path,
    *,
    valid_run_ids: Sequence[str],
    client_map: Mapping[tuple[str, str], str],
) -> Dict[tuple[str, str, str], List[PacketRecord]]:
    """Return {(run_id, capture_id, client_id): packets} from proxy manifests.

    Raw packet sequence files may be chunked because PCAP rotation is enabled.
    This function reconstructs one connection-specific packet sequence across
    all chunk manifests and rejects unconfirmed/stale proxy connections by
    requiring the same connection mapping already used by central dataset
    preparation.
    """
    root = Path(collected_root)
    allowed = {str(value) for value in valid_run_ids}
    # Collect all confirmed capture aliases belonging to the same
    # logical federated client.  Reconnects must be stitched before
    # round inference; different clients are never merged.
    grouped: Dict[tuple[str, str], List[PacketRecord]] = defaultdict(list)
    source_capture_ids: Dict[tuple[str, str], set[str]] = defaultdict(set)

    for manifest_path in sorted(root.rglob("*_manifest.json")):
        manifest = _load_json(manifest_path)
        run_id = str(manifest.get("experiment_id") or "").strip()
        if run_id not in allowed:
            continue
        outputs = manifest.get("outputs", {}) or {}
        per_client = outputs.get("per_client", {}) or {}
        if not isinstance(per_client, dict) or not per_client:
            continue
        raw_path = _manifest_artifact(manifest_path, outputs.get("packet_sequence_raw_csv"))
        if raw_path is None:
            continue

        for capture_id, item in per_client.items():
            if not isinstance(item, dict):
                continue
            capture_id = str(capture_id)
            client_id = client_map.get((run_id, capture_id))
            if not client_id:
                continue
            client_ip = str(item.get("client_ip") or "").strip()
            try:
                client_port = int(item.get("client_port") or 0)
            except (TypeError, ValueError):
                client_port = 0
            if not client_ip:
                continue

            # Re-read with this exact client as the direction reference.
            # read_packet_sequence_csv intentionally recomputes direction;
            # without a client/server reference every packet would become
            # "unknown", which would destroy round inference.
            try:
                client_side_packets = read_packet_sequence_csv(
                    raw_path,
                    client_ip=client_ip,
                    isolate_client_facing=True,
                )
            except Exception:
                continue

            if client_port > 0:
                selected = connection_facing_packets(
                    client_side_packets,
                    client_ip=client_ip,
                    client_port=client_port,
                )
            else:
                selected = client_facing_packets(client_side_packets, client_ip=client_ip)

            if selected:
                logical_key = (
                    run_id,
                    str(client_id),
                )
                grouped[logical_key].extend(selected)
                source_capture_ids[logical_key].add(capture_id)

    result: Dict[
        tuple[str, str, str],
        List[PacketRecord],
    ] = {}

    for (run_id, client_id), value in grouped.items():
        packets = _dedupe_packets(value)

        if not packets:
            continue

        aliases = sorted(
            source_capture_ids[
                (run_id, client_id)
            ]
        )

        if len(aliases) == 1:
            logical_capture_id = aliases[0]
        else:
            logical_capture_id = (
                "stitched_"
                + "_".join(aliases)
            )

        result[
            (
                run_id,
                logical_capture_id,
                client_id,
            )
        ] = packets

    return result


def _label_bins(
    packets: Sequence[PacketRecord],
    cfg: RoundInferenceConfig,
) -> tuple[float, List[Dict[str, Any]]]:
    if not packets:
        return 0.0, []
    if cfg.bin_sec <= 0:
        raise RoundFingerprintingError("bin_sec must be positive")
    if not 0.5 < cfg.direction_dominance <= 1.0:
        raise RoundFingerprintingError("direction_dominance must be in (0.5, 1]")

    t0 = float(packets[0].timestamp_epoch)
    duration = max(0.0, float(packets[-1].timestamp_epoch) - t0)
    n_bins = max(1, int(math.floor(duration / cfg.bin_sec)) + 1)
    bins = [
        {
            "up_bytes": 0,
            "down_bytes": 0,
            "unknown_bytes": 0,
            "up_packets": 0,
            "down_packets": 0,
            "unknown_packets": 0,
        }
        for _ in range(n_bins)
    ]

    for packet in packets:
        index = int(max(0.0, packet.timestamp_epoch - t0) // cfg.bin_sec)
        index = min(index, n_bins - 1)
        direction = packet.direction if packet.direction in {"up", "down"} else "unknown"
        bins[index][f"{direction}_bytes"] += int(packet.frame_length)
        bins[index][f"{direction}_packets"] += 1

    labels: List[str] = []
    for item in bins:
        up = int(item["up_bytes"])
        down = int(item["down_bytes"])
        total = up + down
        if total < int(cfg.min_bin_bytes):
            label = "quiet"
        elif down / max(total, 1) >= cfg.direction_dominance:
            label = "down"
        elif up / max(total, 1) >= cfg.direction_dominance:
            label = "up"
        else:
            # Mixed TLS/TCP bins still carry a direction signal by byte mass.
            ratio = max(up, down) / max(total, 1)
            if ratio >= 0.55:
                label = "down" if down > up else "up"
            else:
                label = "mixed"
        labels.append(label)

    # One-bin mixed islands between the same direction are transport noise,
    # not new FL states. Fill them conservatively.
    cleaned = list(labels)
    for i in range(1, len(labels) - 1):
        if labels[i] == "mixed" and labels[i - 1] == labels[i + 1] and labels[i - 1] in {"up", "down"}:
            cleaned[i] = labels[i - 1]

    for i, item in enumerate(bins):
        item["label"] = cleaned[i]
        item["start_epoch"] = t0 + i * cfg.bin_sec
        item["end_epoch"] = t0 + (i + 1) * cfg.bin_sec
    return t0, bins


def _directional_bursts(
    packets: Sequence[PacketRecord],
    bins: Sequence[Mapping[str, Any]],
    cfg: RoundInferenceConfig,
) -> List[TrafficBurst]:
    if not bins:
        return []
    max_gap_bins = max(0, int(round(cfg.bridge_gap_sec / cfg.bin_sec)))
    result: List[TrafficBurst] = []

    for direction in ("down", "up"):
        active = [i for i, item in enumerate(bins) if item.get("label") == direction]
        if not active:
            continue
        groups: List[List[int]] = [[active[0]]]
        opposite = "up" if direction == "down" else "down"
        for idx in active[1:]:
            previous = groups[-1][-1]
            gap = idx - previous - 1
            between = bins[previous + 1 : idx]
            has_opposite = any(item.get("label") == opposite for item in between)
            if gap <= max_gap_bins and not has_opposite:
                groups[-1].append(idx)
            else:
                groups.append([idx])

        for group in groups:
            first = group[0]
            last = group[-1]
            start = float(bins[first]["start_epoch"])
            end = float(bins[last]["end_epoch"])
            left = bisect.bisect_left([p.timestamp_epoch for p in packets], start)
            right = bisect.bisect_right([p.timestamp_epoch for p in packets], end)
            selected = packets[left:right]
            if not selected:
                continue
            up_bytes = sum(int(p.frame_length) for p in selected if p.direction == "up")
            down_bytes = sum(int(p.frame_length) for p in selected if p.direction == "down")
            result.append(
                TrafficBurst(
                    direction=direction,
                    start_epoch=float(selected[0].timestamp_epoch),
                    end_epoch=float(selected[-1].timestamp_epoch),
                    bytes_up=up_bytes,
                    bytes_down=down_bytes,
                    packet_count=len(selected),
                )
            )

    return sorted(result, key=lambda b: (b.start_epoch, b.end_epoch, b.direction))



def _adaptive_major_threshold(
    bursts: Sequence[TrafficBurst],
    direction: str,
    floor_bytes: int,
) -> float:
    """Estimate a trace-local major-transfer threshold.

    The floor is only a transport-noise floor. It is NOT a
    minimum model/update size.

    All positive directional bursts participate in threshold
    estimation so low-volume AI workloads are not discarded
    before adaptation occurs.
    """
    values = sorted(
        float(b.dominant_bytes)
        for b in bursts
        if (
            b.direction == direction
            and b.dominant_bytes > 0
        )
    )

    floor = max(
        1.0,
        float(floor_bytes),
    )

    if not values:
        return floor

    if len(values) < 4:
        typical = float(
            np.median(
                np.asarray(
                    values,
                    dtype=float,
                )
            )
        )

        return max(
            floor,
            0.50 * typical,
        )

    logs = np.log10(
        np.maximum(
            np.asarray(
                values,
                dtype=float,
            ),
            1.0,
        )
    )

    gaps = np.diff(logs)

    if gaps.size:
        split = int(
            np.argmax(gaps)
        )

        largest = float(
            gaps[split]
        )

        upper_count = (
            len(values)
            - split
            - 1
        )

        # Do not treat a single very large outlier as the
        # recurring FL transfer population.
        minimum_upper_population = max(
            3,
            int(
                math.ceil(
                    0.10 * len(values)
                )
            ),
        )

        if (
            largest >= 0.55
            and upper_count
            >= minimum_upper_population
        ):
            low = values[split]
            high = values[split + 1]

            return max(
                floor,
                math.sqrt(
                    low * high
                ),
            )

    # No strong two-population separation.
    #
    # Use the upper half of the observed directional burst
    # distribution as the trace-local transfer scale, then
    # retain bursts at least 25% of that scale.
    #
    # This preserves the previous relative rule while removing
    # the inappropriate 256-KiB model-size floor.
    upper = values[
        len(values) // 2 :
    ]

    typical = float(
        np.median(
            upper
            if upper
            else values
        )
    )

    return max(
        floor,
        0.25 * typical,
    )

def infer_round_boundaries(
    packets: Sequence[PacketRecord],
    config: Optional[RoundInferenceConfig] = None,
) -> tuple[List[InferredRound], Dict[str, Any]]:
    """Infer FL rounds from proxy-observable traffic only.

    No endpoint label, true round number, model family, architecture, dataset,
    or payload content is used. The detector looks for repeating large
    server->client (down) transfers followed by client->server (up) transfers
    separated by local-compute time.
    """
    cfg = config or RoundInferenceConfig()
    packets = sorted(packets, key=lambda p: (p.timestamp_epoch, p.index))
    if len(packets) < cfg.min_round_packets:
        return [], {"status": "insufficient_packets", "packet_count": len(packets)}

    _t0, bins = _label_bins(packets, cfg)
    bursts = _directional_bursts(packets, bins, cfg)
    down_threshold = _adaptive_major_threshold(bursts, "down", cfg.min_major_transfer_bytes)
    up_threshold = _adaptive_major_threshold(bursts, "up", cfg.min_major_transfer_bytes)
    major = [
        b for b in bursts
        if (
            b.direction == "down" and b.dominant_bytes >= down_threshold
        ) or (
            b.direction == "up" and b.dominant_bytes >= up_threshold
        )
    ]
    major.sort(key=lambda b: (b.start_epoch, b.end_epoch))

    raw_pairs: List[tuple[TrafficBurst, TrafficBurst]] = []
    pending_down: Optional[TrafficBurst] = None
    for burst in major:
        if burst.direction == "down":
            if pending_down is None:
                pending_down = burst
            else:
                # Multiple down candidates before an upload are typically
                # retries/control transfers. Keep the stronger model transfer.
                if burst.dominant_bytes >= pending_down.dominant_bytes:
                    pending_down = burst
            continue

        if pending_down is None:
            continue
        if burst.start_epoch <= pending_down.end_epoch:
            continue
        duration = burst.end_epoch - pending_down.start_epoch
        training_gap = burst.start_epoch - pending_down.end_epoch
        ratio = max(
            burst.dominant_bytes / max(pending_down.dominant_bytes, 1),
            pending_down.dominant_bytes / max(burst.dominant_bytes, 1),
        )
        if duration <= 0 or duration > cfg.max_round_sec:
            pending_down = None
            continue
        if ratio > cfg.max_transfer_size_ratio:
            # Do not permanently discard the down candidate on a small
            # unrelated upload; wait for the next major upload.
            continue
        if training_gap < -cfg.bin_sec:
            continue
        raw_pairs.append((pending_down, burst))
        pending_down = None

    if not raw_pairs:
        return [], {
            "status": "no_round_pairs",
            "packet_count": len(packets),
            "burst_count": len(bursts),
            "major_burst_count": len(major),
            "down_threshold_bytes": down_threshold,
            "up_threshold_bytes": up_threshold,
        }

    median_down = float(np.median([d.dominant_bytes for d, _ in raw_pairs]))
    median_up = float(np.median([u.dominant_bytes for _, u in raw_pairs]))
    timestamps = [float(p.timestamp_epoch) for p in packets]
    inferred: List[InferredRound] = []

    for index, (down, up) in enumerate(raw_pairs):
        left = bisect.bisect_left(timestamps, down.start_epoch)
        right = bisect.bisect_right(timestamps, up.end_epoch)
        selected = packets[left:right]
        if len(selected) < cfg.min_round_packets:
            continue

        size_similarity = math.exp(
            -abs(math.log(max(up.dominant_bytes, 1) / max(down.dominant_bytes, 1)))
        )
        down_consistency = math.exp(
            -abs(math.log(max(down.dominant_bytes, 1) / max(median_down, 1.0)))
        )
        up_consistency = math.exp(
            -abs(math.log(max(up.dominant_bytes, 1) / max(median_up, 1.0)))
        )
        gap = max(0.0, up.start_epoch - down.end_epoch)
        gap_score = 1.0 if gap >= cfg.min_training_gap_sec else max(0.0, gap / max(cfg.min_training_gap_sec, 1e-9))
        confidence = (
            0.20 * down.purity
            + 0.20 * up.purity
            + 0.25 * size_similarity
            + 0.20 * ((down_consistency + up_consistency) / 2.0)
            + 0.15 * gap_score
        )
        inferred.append(
            InferredRound(
                inferred_round_index=len(inferred),
                start_epoch=down.start_epoch,
                end_epoch=up.end_epoch,
                download_start_epoch=down.start_epoch,
                download_end_epoch=down.end_epoch,
                upload_start_epoch=up.start_epoch,
                upload_end_epoch=up.end_epoch,
                download_bytes=int(down.dominant_bytes),
                upload_bytes=int(up.dominant_bytes),
                packet_count=len(selected),
                confidence=float(max(0.0, min(1.0, confidence))),
            )
        )

    diagnostics = {
        "status": "ok" if inferred else "no_usable_rounds",
        "packet_count": len(packets),
        "bin_sec": cfg.bin_sec,
        "burst_count": len(bursts),
        "major_burst_count": len(major),
        "down_threshold_bytes": down_threshold,
        "up_threshold_bytes": up_threshold,
        "inferred_round_count": len(inferred),
        "median_download_bytes": median_down,
        "median_upload_bytes": median_up,
        "mean_confidence": float(np.mean([r.confidence for r in inferred])) if inferred else 0.0,
        "method": ROUND_INFERENCE_METHOD,
    }
    return inferred, diagnostics


def _phase_start(record: Mapping[str, Any]) -> Optional[float]:
    end = _numeric(record.get("timestamp_epoch"), default=float("nan"))
    duration_ms = _numeric(record.get("phase_time_ms"), default=float("nan"))
    if not math.isfinite(end) or not math.isfinite(duration_ms):
        return None
    return end - duration_ms / 1000.0


def read_ground_truth_rounds(
    ground_truth_paths: Sequence[str | Path],
) -> Dict[tuple[str, str], List[GroundTruthRound]]:
    """Read endpoint round boundaries for validation only.

    These timestamps are NEVER written into predictor X and are never used by
    infer_round_boundaries(). Existing logs are reconstructed from phase end
    timestamps plus phase durations. Future logs can add explicit boundaries
    without changing this API.
    """
    phases: Dict[tuple[str, str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    fallback_start_end: Dict[tuple[str, str, int], tuple[float, float]] = {}

    for path_value in ground_truth_paths:
        path = Path(path_value)
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        records: List[Dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        file_run_ids = {
            str(r.get("run_id") or "").strip()
            for r in records
            if str(r.get("run_id") or "").strip()
        }
        file_run_id = next(iter(file_run_ids)) if len(file_run_ids) == 1 else ""

        for record in records:
            if str(record.get("role") or "").lower() != "client":
                continue
            run_id = file_run_id or str(record.get("run_id") or record.get("experiment_id") or "").strip()
            client_id = str(record.get("client_id") or "").strip()
            if not run_id or not client_id:
                continue
            try:
                round_index = int(record.get("round"))
            except (TypeError, ValueError):
                continue

            event = str(record.get("event") or "")
            if event == "federated_phase":
                phase = str(record.get("phase") or "").strip().lower()
                if phase in {"download", "training", "upload", "idle"}:
                    phases[(run_id, client_id, round_index)][phase] = record
            elif event == "round_boundary_ground_truth":
                start = _numeric(record.get("round_start_epoch"), float("nan"))
                end = _numeric(record.get("round_end_epoch"), float("nan"))
                if math.isfinite(start) and math.isfinite(end) and end > start:
                    fallback_start_end[(run_id, client_id, round_index)] = (start, end)

    grouped: Dict[tuple[str, str], List[GroundTruthRound]] = defaultdict(list)
    keys = set(phases) | set(fallback_start_end)
    for key in sorted(keys):
        run_id, client_id, round_index = key
        phase_map = phases.get(key, {})
        explicit = fallback_start_end.get(key)

        def phase_bounds(name: str) -> tuple[Optional[float], Optional[float]]:
            record = phase_map.get(name)
            if not record:
                return None, None
            end = _numeric(record.get("timestamp_epoch"), float("nan"))
            start = _phase_start(record)
            return start, (end if math.isfinite(end) else None)

        d0, d1 = phase_bounds("download")
        t0, t1 = phase_bounds("training")
        u0, u1 = phase_bounds("upload")
        if explicit:
            start, end = explicit
        else:
            start_candidates = [x for x in (d0, t0, u0) if x is not None]
            end_candidates = [x for x in (u1, t1, d1) if x is not None]
            if not start_candidates or not end_candidates:
                continue
            start = min(start_candidates)
            end = max(end_candidates)
        if end <= start:
            continue
        grouped[(run_id, client_id)].append(
            GroundTruthRound(
                round_index=round_index,
                start_epoch=float(start),
                end_epoch=float(end),
                download_start_epoch=d0,
                download_end_epoch=d1,
                training_start_epoch=t0,
                training_end_epoch=t1,
                upload_start_epoch=u0,
                upload_end_epoch=u1,
            )
        )

    return {
        key: sorted(value, key=lambda r: r.round_index)
        for key, value in grouped.items()
    }


def _interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return overlap / union if union > 0 else 0.0




def validate_inferred_rounds(
    inferred: Sequence[InferredRound],
    truth: Sequence[GroundTruthRound],
) -> Dict[str, Any]:
    """Validate proxy-only round inference against endpoint ground truth.

    Ground truth is used ONLY for validation.

    A constant cross-machine clock offset is estimated from robust
    consensus among inferred/truth round-start timestamps. The
    estimator does not assume inferred round i corresponds to truth
    round i, which is important when some rounds are missed.

    Clock alignment never affects:
      * proxy-side round inference,
      * round feature extraction,
      * classifier predictors, or
      * classifier labels.

    Primary segmentation metrics use one-to-one interval matching
    at IoU >= 0.50. IoU >= 0.25 and >= 0.75 are also reported as
    sensitivity analyses.
    """
    thresholds = (
        0.25,
        0.50,
        0.75,
    )

    if not truth:
        return {
            "status":
                "ground_truth_unavailable"
        }

    if not inferred:
        result = {
            "status":
                "no_inferred_rounds",
            "truth_rounds":
                len(truth),
            "inferred_rounds":
                0,
            "matched_rounds":
                0,
            "boundary_precision":
                0.0,
            "boundary_recall":
                0.0,
            "interval_iou_mean":
                None,
            "clock_offset_sec_validation_only":
                None,
            "clock_offset_method":
                "unavailable_no_inferred_rounds",
            "matches":
                [],
        }

        for threshold in thresholds:
            pct = int(
                round(
                    threshold * 100
                )
            )

            result[
                f"matched_rounds_iou_{pct}"
            ] = 0

            result[
                f"boundary_precision_iou_{pct}"
            ] = 0.0

            result[
                f"boundary_recall_iou_{pct}"
            ] = 0.0

        return result

    # --------------------------------------------------------
    # Validation-only robust clock registration.
    #
    # Generate possible constant offsets from every observed
    # inferred/truth start-time difference. Correct matches form
    # a repeated cluster even when inferred round indices skip
    # ground-truth rounds.
    #
    # Candidate generation:
    #   0.25-s bins over pairwise start differences.
    #
    # Candidate selection:
    #   maximum one-to-one start matches within 1.0 s,
    #   then minimum median/mean start error.
    #
    # No interval IoU is used to choose the clock offset.
    # --------------------------------------------------------

    alignment_bin_sec = 0.25
    alignment_tolerance_sec = 0.25

    difference_bins: Dict[
        int,
        List[float],
    ] = defaultdict(list)

    for inf in inferred:
        for gt in truth:
            difference = (
                float(inf.start_epoch)
                - float(gt.start_epoch)
            )

            bucket = int(
                round(
                    difference
                    / alignment_bin_sec
                )
            )

            difference_bins[
                bucket
            ].append(
                difference
            )

    ranked_bins = sorted(
        difference_bins.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:64]

    candidate_offsets = [
        float(
            np.median(
                values
            )
        )
        for _bucket, values
        in ranked_bins
    ]

    # Include zero and the legacy indexed estimate only as
    # additional candidates/fallbacks.
    candidate_offsets.append(
        0.0
    )

    pair_count = min(
        len(inferred),
        len(truth),
        20,
    )

    if pair_count:
        candidate_offsets.append(
            float(
                np.median(
                    [
                        inferred[i].start_epoch
                        - truth[i].start_epoch
                        for i in range(
                            pair_count
                        )
                    ]
                )
            )
        )

    # Remove essentially duplicate candidates.
    unique_offsets: List[
        float
    ] = []

    seen_offset_keys = set()

    for offset in candidate_offsets:
        key = round(
            float(offset),
            6,
        )

        if key in seen_offset_keys:
            continue

        seen_offset_keys.add(
            key
        )

        unique_offsets.append(
            float(offset)
        )

    def start_alignment_score(
        offset: float,
    ) -> tuple[
        int,
        float,
        float,
    ]:
        candidates: List[
            tuple[
                float,
                int,
                int,
            ]
        ] = []

        for i, inf in enumerate(
            inferred
        ):
            adjusted_start = (
                float(
                    inf.start_epoch
                )
                - offset
            )

            for j, gt in enumerate(
                truth
            ):
                error = abs(
                    adjusted_start
                    - float(
                        gt.start_epoch
                    )
                )

                if (
                    error
                    <= alignment_tolerance_sec
                ):
                    candidates.append(
                        (
                            float(error),
                            i,
                            j,
                        )
                    )

        # Closest temporal starts first.
        candidates.sort(
            key=lambda item:
                item[0]
        )

        used_i: set[int] = set()
        used_j: set[int] = set()
        errors: List[float] = []

        for error, i, j in candidates:
            if (
                i in used_i
                or j in used_j
            ):
                continue

            used_i.add(i)
            used_j.add(j)

            errors.append(
                float(error)
            )

        if not errors:
            return (
                0,
                float("inf"),
                float("inf"),
            )

        return (
            len(errors),
            float(
                np.median(
                    errors
                )
            ),
            float(
                np.mean(
                    errors
                )
            ),
        )

    best_offset = 0.0
    best_support = -1
    best_median_error = float("inf")
    best_mean_error = float("inf")

    for offset in unique_offsets:
        (
            support,
            median_error,
            mean_error,
        ) = start_alignment_score(
            offset
        )

        better = False

        if support > best_support:
            better = True
        elif support == best_support:
            if (
                median_error
                < best_median_error
            ):
                better = True
            elif (
                median_error
                == best_median_error
                and mean_error
                < best_mean_error
            ):
                better = True

        if better:
            best_offset = float(
                offset
            )
            best_support = int(
                support
            )
            best_median_error = float(
                median_error
            )
            best_mean_error = float(
                mean_error
            )

    clock_offset = best_offset

    # --------------------------------------------------------
    # Strict one-to-one interval matching.
    # --------------------------------------------------------

    def match_at_threshold(
        minimum_iou: float,
    ) -> List[Dict[str, Any]]:

        candidates: List[
            tuple[
                float,
                float,
                int,
                int,
            ]
        ] = []

        for i, inf in enumerate(
            inferred
        ):
            a0 = (
                float(
                    inf.start_epoch
                )
                - clock_offset
            )

            a1 = (
                float(
                    inf.end_epoch
                )
                - clock_offset
            )

            for j, gt in enumerate(
                truth
            ):
                iou = _interval_iou(
                    a0,
                    a1,
                    float(
                        gt.start_epoch
                    ),
                    float(
                        gt.end_epoch
                    ),
                )

                if (
                    iou
                    < minimum_iou
                ):
                    continue

                start_error = abs(
                    a0
                    - float(
                        gt.start_epoch
                    )
                )

                candidates.append(
                    (
                        float(iou),
                        float(
                            start_error
                        ),
                        i,
                        j,
                    )
                )

        # Prefer maximum overlap. Start proximity breaks ties.
        candidates.sort(
            key=lambda item: (
                item[0],
                -item[1],
            ),
            reverse=True,
        )

        used_i: set[int] = set()
        used_j: set[int] = set()

        matches: List[
            Dict[str, Any]
        ] = []

        for (
            iou,
            _start_error,
            i,
            j,
        ) in candidates:

            if (
                i in used_i
                or j in used_j
            ):
                continue

            inf = inferred[i]
            gt = truth[j]

            a0 = (
                float(
                    inf.start_epoch
                )
                - clock_offset
            )

            a1 = (
                float(
                    inf.end_epoch
                )
                - clock_offset
            )

            start_error = abs(
                a0
                - float(
                    gt.start_epoch
                )
            )

            end_error = abs(
                a1
                - float(
                    gt.end_epoch
                )
            )

            used_i.add(i)
            used_j.add(j)

            matches.append(
                {
                    "inferred_index":
                        int(
                            inf.inferred_round_index
                        ),
                    "ground_truth_round":
                        int(
                            gt.round_index
                        ),
                    "iou":
                        float(
                            iou
                        ),
                    "start_error_sec":
                        float(
                            start_error
                        ),
                    "end_error_sec":
                        float(
                            end_error
                        ),
                }
            )

        matches.sort(
            key=lambda item:
                item[
                    "inferred_index"
                ]
        )

        return matches

    evaluated: Dict[
        float,
        Dict[str, Any],
    ] = {}

    for threshold in thresholds:
        matches = match_at_threshold(
            threshold
        )

        matched = len(
            matches
        )

        precision = (
            matched
            / len(inferred)
            if inferred
            else 0.0
        )

        recall = (
            matched
            / len(truth)
            if truth
            else 0.0
        )

        evaluated[
            threshold
        ] = {
            "matches":
                matches,
            "matched":
                matched,
            "precision":
                float(
                    precision
                ),
            "recall":
                float(
                    recall
                ),
        }

    primary = evaluated[
        0.50
    ]

    primary_matches = (
        primary[
            "matches"
        ]
    )

    ious = [
        float(
            item["iou"]
        )
        for item
        in primary_matches
    ]

    start_errors = [
        float(
            item[
                "start_error_sec"
            ]
        )
        for item
        in primary_matches
    ]

    end_errors = [
        float(
            item[
                "end_error_sec"
            ]
        )
        for item
        in primary_matches
    ]

    result: Dict[
        str,
        Any,
    ] = {
        "status":
            "evaluated",
        "truth_rounds":
            len(truth),
        "inferred_rounds":
            len(inferred),

        "clock_offset_sec_validation_only":
            float(
                clock_offset
            ),

        "clock_offset_method":
            (
                "pairwise_start_difference_"
                "consensus_one_to_one"
            ),

        "clock_alignment_bin_sec":
            float(
                alignment_bin_sec
            ),

        "clock_alignment_tolerance_sec":
            float(
                alignment_tolerance_sec
            ),

        "clock_alignment_support":
            int(
                best_support
            ),

        "clock_alignment_start_error_median_sec":
            (
                float(
                    best_median_error
                )
                if math.isfinite(
                    best_median_error
                )
                else None
            ),

        "clock_alignment_start_error_mean_sec":
            (
                float(
                    best_mean_error
                )
                if math.isfinite(
                    best_mean_error
                )
                else None
            ),

        "matching_rule":
            (
                "one_to_one_greedy_"
                "maximum_interval_iou"
            ),

        "primary_iou_threshold":
            0.50,

        "matched_rounds":
            int(
                primary[
                    "matched"
                ]
            ),

        "boundary_precision":
            float(
                primary[
                    "precision"
                ]
            ),

        "boundary_recall":
            float(
                primary[
                    "recall"
                ]
            ),

        "interval_iou_mean":
            (
                float(
                    np.mean(
                        ious
                    )
                )
                if ious
                else None
            ),

        "start_error_mean_sec":
            (
                float(
                    np.mean(
                        start_errors
                    )
                )
                if start_errors
                else None
            ),

        "start_error_median_sec":
            (
                float(
                    np.median(
                        start_errors
                    )
                )
                if start_errors
                else None
            ),

        "end_error_mean_sec":
            (
                float(
                    np.mean(
                        end_errors
                    )
                )
                if end_errors
                else None
            ),

        "matches":
            primary_matches,
    }

    for threshold in thresholds:
        pct = int(
            round(
                threshold
                * 100
            )
        )

        item = evaluated[
            threshold
        ]

        result[
            f"matched_rounds_iou_{pct}"
        ] = int(
            item[
                "matched"
            ]
        )

        result[
            f"boundary_precision_iou_{pct}"
        ] = float(
            item[
                "precision"
            ]
        )

        result[
            f"boundary_recall_iou_{pct}"
        ] = float(
            item[
                "recall"
            ]
        )

    return result

def _segment_packets(
    packets: Sequence[PacketRecord],
    start_epoch: float,
    end_epoch: float,
) -> List[PacketRecord]:
    timestamps = [float(p.timestamp_epoch) for p in packets]
    left = bisect.bisect_left(timestamps, float(start_epoch))
    right = bisect.bisect_right(timestamps, float(end_epoch))
    return list(packets[left:right])


def build_round_dataset(
    *,
    client_packets: Mapping[tuple[str, str, str], Sequence[PacketRecord]],
    client_labels: Mapping[tuple[str, str], Mapping[str, Any]],
    ground_truth_rounds: Mapping[tuple[str, str], Sequence[GroundTruthRound]],
    output_dir: str | Path,
    inference_config: Optional[RoundInferenceConfig] = None,
) -> Dict[str, Any]:
    """Create proxy-only round-level X and endpoint-label-only Y datasets."""
    cfg = inference_config or RoundInferenceConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    x_path = output / "round_X_proxy.csv"
    y_path = output / "round_Y_ground_truth.csv"
    schema_path = output / "round_schema.json"
    boundary_path = output / "round_boundaries.csv"
    validation_path = output / "round_boundary_validation.json"
    inference_path = output / "round_inference_diagnostics.json"

    x_rows: List[Dict[str, Any]] = []
    y_rows: List[Dict[str, Any]] = []
    boundary_rows: List[Dict[str, Any]] = []
    validation: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    predictor_columns: Optional[List[str]] = None
    row_id = 0

    metadata_names = set(PROXY_FEATURE_METADATA_FIELDS) | {
        "experiment_id",
        "client_capture_id",
        "row_type",
    }

    for (run_id, capture_id, client_id), packet_list in sorted(client_packets.items()):
        labels = client_labels.get((run_id, client_id))
        if labels is None:
            continue
        packets = sorted(packet_list, key=lambda p: (p.timestamp_epoch, p.index))
        inferred, diag = infer_round_boundaries(packets, cfg)
        diagnostics[f"{run_id}::{client_id}::{capture_id}"] = diag
        truth = list(ground_truth_rounds.get((run_id, client_id), []))
        validation[f"{run_id}::{client_id}::{capture_id}"] = validate_inferred_rounds(inferred, truth)

        for boundary in inferred:
            selected = _segment_packets(packets, boundary.start_epoch, boundary.end_epoch)
            if len(selected) < cfg.min_round_packets:
                continue
            features = extract_feature_rows(
                packets=selected,
                experiment_id=run_id,
                burst_gap_sec=cfg.burst_gap_sec,
                idle_threshold_sec=cfg.idle_threshold_sec,
                window_seconds=None,
                packet_information_threshold=cfg.packet_information_threshold,
            )[0]
            features["row_type"] = "round"

            current_predictors = [
                key for key in features.keys()
                if key not in metadata_names
            ]
            if predictor_columns is None:
                predictor_columns = current_predictors
            elif current_predictors != predictor_columns:
                raise RoundFingerprintingError(
                    "Round feature schema changed between samples"
                )

            row_id += 1
            x_row: Dict[str, Any] = {
                "row_id": row_id,
                "experiment_id": run_id,
                "client_capture_id": capture_id,
                "row_type": "round",
            }
            for name in current_predictors:
                x_row[name] = features.get(name, 0.0)

            y_row: Dict[str, Any] = {
                "row_id": row_id,
                "experiment_id": run_id,
                "client_capture_id": capture_id,
                "resolved_client_id": client_id,
                "inferred_round_index": boundary.inferred_round_index,
                "round_boundary_confidence": boundary.confidence,
            }
            for field in GROUND_TRUTH_LABEL_FIELDS:
                if field in labels:
                    y_row[field] = labels[field]
            for field in OPTIONAL_CONTEXT_LABEL_FIELDS:
                if field in labels:
                    y_row[field] = labels[field]

            x_rows.append(x_row)
            y_rows.append(y_row)
            boundary_rows.append(
                {
                    "experiment_id": run_id,
                    "client_capture_id": capture_id,
                    "resolved_client_id": client_id,
                    **asdict(boundary),
                    "duration_sec": boundary.duration_sec,
                    "training_gap_sec": boundary.training_gap_sec,
                }
            )

    if not x_rows or predictor_columns is None:
        raise RoundFingerprintingError(
            "No round-level samples were created. Check raw proxy packet sequence "
            "availability and round inference diagnostics."
        )

    x_fields = ["row_id", "experiment_id", "client_capture_id", "row_type", *predictor_columns]
    y_extra = [
        "row_id", "experiment_id", "client_capture_id", "resolved_client_id",
        "inferred_round_index", "round_boundary_confidence",
    ]
    y_label_fields = [
        field for field in [*GROUND_TRUTH_LABEL_FIELDS, *OPTIONAL_CONTEXT_LABEL_FIELDS]
        if any(field in row for row in y_rows)
    ]
    y_fields = y_extra + y_label_fields

    with x_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=x_fields)
        writer.writeheader()
        writer.writerows(x_rows)
    with y_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=y_fields)
        writer.writeheader()
        writer.writerows(y_rows)
    with boundary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(boundary_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(boundary_rows)

    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    inference_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")

    schema = {
        "schema_version": ROUND_SCHEMA_VERSION,
        "representation": "one inferred FL round for one client",
        "segmentation_source": "proxy_observable_traffic_only",
        "ground_truth_round_boundaries_used_as_predictors": False,
        "true_round_index_used_as_predictor": False,
        "inferred_round_index_used_as_predictor": False,
        "round_boundary_confidence_used_as_predictor": False,
        "predictor_columns": predictor_columns,
        "metadata_columns": ["row_id", "experiment_id", "client_capture_id", "row_type"],
        "label_columns": y_label_fields,
        "row_count": len(x_rows),
        "experiment_count": len({row["experiment_id"] for row in y_rows}),
        "client_trace_count": len({(row["experiment_id"], row["resolved_client_id"]) for row in y_rows}),
        "inference_config": asdict(cfg),
        "evaluation_rule": (
            "All rounds from one experiment remain in the same train/test fold. "
            "Random round/window splitting across the same experiment is forbidden."
        ),
    }
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "x_csv": str(x_path),
        "y_csv": str(y_path),
        "schema_json": str(schema_path),
        "boundaries_csv": str(boundary_path),
        "boundary_validation_json": str(validation_path),
        "inference_diagnostics_json": str(inference_path),
        "round_sample_count": len(x_rows),
        "experiment_count": schema["experiment_count"],
        "client_trace_count": schema["client_trace_count"],
        "predictor_count": len(predictor_columns),
    }


def _display_name(value: str) -> str:
    replacements = {
        "autoencoder": "Autoencoder",
        "cnn": "CNN",
        "rnn": "RNN",
        "transformer": "Transformer",
        "convolutional_autoencoder": "Convolutional AE",
        "dense_autoencoder": "Dense AE",
        "variational_autoencoder": "Variational AE",
        "lstm": "LSTM",
        "gru": "GRU",
        "efficientnet": "EfficientNet",
        "mobilenet": "MobileNet",
        "resnet": "ResNet",
    }
    return replacements.get(str(value), str(value).replace("_", " ").title())


def _require_ml():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            cohen_kappa_score,
            confusion_matrix,
            f1_score,
            log_loss,
            matthews_corrcoef,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
            auc,
            average_precision_score,
        )
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as exc:
        raise RoundFingerprintingError(
            "Round evaluation requires matplotlib and scikit-learn."
        ) from exc
    return locals()


def _read_round_xy(x_csv: str | Path, y_csv: str | Path):
    with Path(x_csv).open(newline="", encoding="utf-8") as handle:
        x_rows = list(csv.DictReader(handle))
    with Path(y_csv).open(newline="", encoding="utf-8") as handle:
        y_rows = list(csv.DictReader(handle))
    y_by_id = {str(row["row_id"]): row for row in y_rows}
    joined = []
    for x in x_rows:
        y = y_by_id.get(str(x.get("row_id") or ""))
        if not y:
            continue
        row = dict(x)
        row["_y"] = y
        joined.append(row)
    return joined


def _matrix(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[_numeric(row.get(name)) for name in columns] for row in rows],
        dtype=float,
    )


def _groups_per_class(rows, target: str) -> Dict[str, int]:
    mapping: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = str(row["_y"].get(target) or "")
        if label:
            mapping[label].add(str(row.get("experiment_id") or ""))
    return {key: len(value) for key, value in mapping.items()}



def _compact_confusion_count(value: int) -> str:
    """Compact count formatting for confusion-matrix annotations."""
    value = int(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def _write_confusion_artifacts(
    output_dir: Path,
    truth: Sequence[str],
    pred: Sequence[str],
    classes: Sequence[str],
    *,
    unit_name: str,
    evidence_count: Optional[int] = None,
    evidence_unit: Optional[str] = None,
) -> None:
    ml = _require_ml()
    plt = ml["plt"]
    cm = ml["confusion_matrix"](truth, pred, labels=list(classes))
    row_sum = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)

    with (output_dir / "confusion_matrix_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *classes])
        for label, values in zip(classes, cm):
            writer.writerow([label, *[int(v) for v in values]])
    with (output_dir / "confusion_matrix_normalized.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *classes])
        for label, values in zip(classes, normalized):
            writer.writerow([label, *[float(v) for v in values]])

    support = {label: int(sum(1 for value in truth if value == label)) for label in classes}
    with (output_dir / "confusion_matrix_classifier_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "N", "unit"])
        for label in classes:
            writer.writerow([label, support[label], unit_name])

    fig, ax = plt.subplots(figsize=(max(6.5, len(classes) * 1.45), max(5.5, len(classes) * 1.20)))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(
        [_display_name(c) for c in classes],
        rotation=25,
        ha="right",
    )
    ax.set_yticklabels(
        [_display_name(c) for c in classes]
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    if evidence_count is not None and evidence_unit:
        ax.set_title(
            f"n = {_compact_confusion_count(evidence_count)} {evidence_unit}",
            fontsize=11,
            pad=12,
        )

    for i in range(len(classes)):
        for j in range(len(classes)):
            value = normalized[i, j]
            count = int(cm[i, j])
            ax.text(
                j,
                i,
                f"{100.0 * value:.1f}%",
                ha="center",
                va="center",
                color="white" if value >= 0.5 else "black",
            )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def _metrics_from_predictions(truth, pred, probabilities, classes):
    ml = _require_ml()
    truth_arr = np.asarray(truth, dtype=object)
    pred_arr = np.asarray(pred, dtype=object)
    result = {
        "sample_count": int(len(truth)),
        "accuracy": float(ml["accuracy_score"](truth_arr, pred_arr)),
        "balanced_accuracy": float(ml["balanced_accuracy_score"](truth_arr, pred_arr)),
        "macro_precision": float(ml["precision_score"](truth_arr, pred_arr, average="macro", zero_division=0)),
        "macro_recall": float(ml["recall_score"](truth_arr, pred_arr, average="macro", zero_division=0)),
        "macro_f1": float(ml["f1_score"](truth_arr, pred_arr, average="macro", zero_division=0)),
        "mcc": float(ml["matthews_corrcoef"](truth_arr, pred_arr)),
        "cohen_kappa": float(ml["cohen_kappa_score"](truth_arr, pred_arr)),
    }
    if probabilities is not None and len(probabilities):
        result["log_loss"] = float(ml["log_loss"](truth_arr, probabilities, labels=list(classes)))
    return result


def _write_roc_pr(output_dir: Path, truth, probabilities, classes):
    ml = _require_ml()
    plt = ml["plt"]
    truth_arr = np.asarray(truth, dtype=object)
    probabilities = np.asarray(probabilities, dtype=float)
    roc_rows = []
    pr_rows = []

    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(classes):
        binary = (truth_arr == label).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        scores = probabilities[:, index]
        fpr, tpr, _ = ml["roc_curve"](binary, scores)
        roc_auc = float(ml["auc"](fpr, tpr))
        precision, recall, _ = ml["precision_recall_curve"](binary, scores)
        ap = float(ml["average_precision_score"](binary, scores))
        roc_rows.append({"class": label, "auroc": roc_auc})
        pr_rows.append({"class": label, "auprc": ap})
        ax_roc.plot(fpr, tpr, label=f"{_display_name(label)} ({roc_auc:.3f})")
        ax_pr.plot(recall, precision, label=f"{_display_name(label)} ({ap:.3f})")

    if roc_rows:
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.legend()
        fig_roc.tight_layout()
        fig_roc.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        fig_roc.savefig(output_dir / "roc_curve.pdf", bbox_inches="tight")
    if pr_rows:
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend()
        fig_pr.tight_layout()
        fig_pr.savefig(output_dir / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
        fig_pr.savefig(output_dir / "precision_recall_curve.pdf", bbox_inches="tight")
    plt.close(fig_roc)
    plt.close(fig_pr)

    if roc_rows:
        with (output_dir / "roc_auc_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["class", "auroc"])
            writer.writeheader(); writer.writerows(roc_rows)
    if pr_rows:
        with (output_dir / "pr_auc_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["class", "auprc"])
            writer.writeheader(); writer.writerows(pr_rows)



def _experiment_trace_balanced_weights(
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return training weights balanced by experiment and client trace.

    Weighting hierarchy:

      1. Every independent experiment receives equal total weight.
      2. Within an experiment, every logical client trace receives
         equal total weight.
      3. Within a client trace, that trace's weight is divided equally
         across its inferred-round observations.

    Thus, experiments with more inferred rounds do not receive more
    training influence merely because they produced more observations.
    """

    if not rows:
        return np.asarray([], dtype=float)

    trace_counts: Dict[
        tuple[str, str],
        int,
    ] = defaultdict(int)

    experiment_traces: Dict[
        str,
        set[str],
    ] = defaultdict(set)

    trace_keys: List[
        tuple[str, str]
    ] = []

    for row in rows:
        experiment_id = str(
            row.get("experiment_id")
            or ""
        )

        y = row.get("_y") or {}

        client_id = str(
            y.get("resolved_client_id")
            or row.get("client_capture_id")
            or ""
        )

        if not experiment_id:
            raise RoundFingerprintingError(
                "Training row missing experiment_id"
            )

        if not client_id:
            raise RoundFingerprintingError(
                "Training row missing logical client identity"
            )

        key = (
            experiment_id,
            client_id,
        )

        trace_keys.append(key)
        trace_counts[key] += 1
        experiment_traces[
            experiment_id
        ].add(client_id)

    weights = np.zeros(
        len(rows),
        dtype=float,
    )

    for i, (
        experiment_id,
        client_id,
    ) in enumerate(trace_keys):

        trace_count = trace_counts[
            (
                experiment_id,
                client_id,
            )
        ]

        traces_in_experiment = len(
            experiment_traces[
                experiment_id
            ]
        )

        if (
            trace_count <= 0
            or traces_in_experiment <= 0
        ):
            raise RoundFingerprintingError(
                "Invalid experiment/trace weighting state"
            )

        # Experiment total = 1.
        # Each client trace receives 1/T of that mass.
        # Each round receives an equal share of its trace mass.
        weights[i] = (
            1.0
            / traces_in_experiment
            / trace_count
        )

    # Scale to mean weight 1 for numerical convenience.
    # Relative weighting is unchanged.
    mean_weight = float(
        np.mean(weights)
    )

    if (
        not np.isfinite(mean_weight)
        or mean_weight <= 0
    ):
        raise RoundFingerprintingError(
            "Invalid training sample weights"
        )

    weights /= mean_weight

    return weights


def _weighted_fisher_score_ranking(
    X: np.ndarray,
    labels: Sequence[str],
    feature_columns: Sequence[str],
    sample_weight: Sequence[float],
) -> List[Dict[str, Any]]:
    """Weighted multiclass Fisher ranking.

    Only training-fold observations are used. The supplied weights
    equalize independent experiments and logical client traces.
    """

    if (
        X.ndim != 2
        or X.shape[1] != len(
            feature_columns
        )
    ):
        raise RoundFingerprintingError(
            "Invalid feature matrix for weighted Fisher score"
        )

    y = np.asarray(
        labels,
        dtype=object,
    )

    w = np.asarray(
        sample_weight,
        dtype=float,
    )

    if (
        len(w) != X.shape[0]
        or np.any(~np.isfinite(w))
        or np.any(w < 0)
        or float(np.sum(w)) <= 0
    ):
        raise RoundFingerprintingError(
            "Invalid Fisher sample weights"
        )

    classes = sorted(
        set(
            str(value)
            for value in labels
        )
    )

    if len(classes) < 2:
        return [
            {
                "feature": name,
                "score": 0.0,
            }
            for name in feature_columns
        ]

    total_weight = float(
        np.sum(w)
    )

    overall = (
        np.sum(
            X * w[:, None],
            axis=0,
        )
        / total_weight
    )

    numerator = np.zeros(
        X.shape[1],
        dtype=np.float64,
    )

    denominator = np.zeros(
        X.shape[1],
        dtype=np.float64,
    )

    for label in classes:
        mask = y == label

        subset = X[mask]
        class_weights = w[mask]

        if (
            subset.size == 0
            or float(
                np.sum(
                    class_weights
                )
            ) <= 0
        ):
            continue

        class_weight = float(
            np.sum(
                class_weights
            )
        )

        mean = (
            np.sum(
                subset
                * class_weights[:, None],
                axis=0,
            )
            / class_weight
        )

        variance = (
            np.sum(
                np.square(
                    subset - mean
                )
                * class_weights[:, None],
                axis=0,
            )
            / class_weight
        )

        numerator += (
            class_weight
            * np.square(
                mean - overall
            )
        )

        denominator += (
            class_weight
            * variance
        )

    scores = (
        numerator
        / np.maximum(
            denominator,
            1e-12,
        )
    )

    ranking = sorted(
        (
            {
                "feature": name,
                "score": float(score),
            }
            for name, score
            in zip(
                feature_columns,
                scores,
            )
        ),
        key=lambda item:
            item["score"],
        reverse=True,
    )

    return ranking


def _evaluate_stage_oof(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: str,
    feature_mode: str,
    output_dir: Path,
    parent_description: str,
    fisher_top_k: int = FISHER_TOP_K,
) -> Dict[str, Any]:
    ml = _require_ml()
    RandomForestClassifier = ml["RandomForestClassifier"]
    StratifiedGroupKFold = ml["StratifiedGroupKFold"]

    usable_all = [
        row
        for row in rows
        if str(row["_y"].get(target) or "").strip()
    ]

    original_groups_per_class = _groups_per_class(
        usable_all,
        target,
    )

    # Experiment-disjoint OOF requires at least two
    # independent experiments for each evaluated class.
    # Classes below that threshold are excluded rather than
    # suppressing the entire parent-level classification task.
    eligible_classes = sorted(
        label
        for label, count
        in original_groups_per_class.items()
        if count >= 2
    )

    excluded_classes = {
        label: count
        for label, count
        in original_groups_per_class.items()
        if count < 2
    }

    eligible_set = set(eligible_classes)

    usable = [
        row
        for row in usable_all
        if str(row["_y"].get(target)) in eligible_set
    ]

    classes = eligible_classes

    groups_per_class = _groups_per_class(
        usable,
        target,
    )

    min_groups = (
        min(groups_per_class.values())
        if groups_per_class
        else 0
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(classes) < 2 or min_groups < 2:
        metrics = {
            "status": "insufficient_independent_runs",
            "target": target,
            "classes": classes,
            "independent_groups_per_class": groups_per_class,
            "original_independent_groups_per_class":
                original_groups_per_class,
            "excluded_classes_insufficient_independent_runs":
                excluded_classes,
            "sample_count": len(usable),
            "original_sample_count": len(usable_all),
            "parent": parent_description,
        }

        (output_dir / "metrics.json").write_text(
            json.dumps(
                metrics,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return metrics

    fieldnames = [key for key in usable[0].keys() if key != "_y"]
    candidates = candidate_feature_columns(fieldnames, feature_mode)
    labels = np.asarray([str(row["_y"][target]) for row in usable], dtype=object)
    groups = np.asarray([str(row.get("experiment_id") or "") for row in usable], dtype=object)
    n_splits = min(5, min_groups)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    class_index = {label: i for i, label in enumerate(classes)}

    oof_pred = np.empty(len(usable), dtype=object)
    oof_prob = np.zeros((len(usable), len(classes)), dtype=float)
    seen = np.zeros(len(usable), dtype=bool)
    selected_by_fold = []

    dummy_X = np.zeros((len(usable), 1), dtype=float)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy_X, labels, groups), start=1):
        train_rows = [usable[i] for i in train_idx]
        X_train_all = _matrix(train_rows, candidates)
        y_train = labels[train_idx]
        train_weights = _experiment_trace_balanced_weights(
            train_rows
        )

        ranking = _weighted_fisher_score_ranking(
            X_train_all,
            y_train,
            candidates,
            sample_weight=train_weights,
        )

        selected = [
            item["feature"]
            for item in ranking[
                : max(
                    1,
                    min(
                        fisher_top_k,
                        len(ranking),
                    ),
                )
            ]
        ]
        selected_by_fold.append({"fold": fold, "selected_features": selected, "ranking": ranking})

        X_train = _matrix(train_rows, selected)
        X_test = _matrix([usable[i] for i in test_idx], selected)
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
        )
        pred = model.predict(X_test)
        raw_prob = model.predict_proba(X_test)
        aligned = np.full((len(test_idx), len(classes)), 1e-15, dtype=float)
        for source_index, label in enumerate(model.classes_):
            aligned[:, class_index[str(label)]] = raw_prob[:, source_index]
        aligned /= aligned.sum(axis=1, keepdims=True)
        oof_pred[test_idx] = pred
        oof_prob[test_idx] = aligned
        seen[test_idx] = True

    if not bool(np.all(seen)):
        raise RoundFingerprintingError("Grouped OOF did not cover every round sample")

    truth = [str(value) for value in labels]
    pred = [str(value) for value in oof_pred]
    metrics = _metrics_from_predictions(truth, pred, oof_prob, classes)
    metrics.update({
        "status": "evaluated",
        "target": target,
        "feature_mode": feature_mode,
        "folds": n_splits,
        "classes": classes,
        "independent_groups_per_class": groups_per_class,
        "original_independent_groups_per_class":
            original_groups_per_class,
        "excluded_classes_insufficient_independent_runs":
            excluded_classes,
        "parent": parent_description,
        "evaluation_unit": "inferred_round",
        "fisher_selection": "inside_each_training_fold",
        "grouping": "experiment_id",
        "training_weighting":
            "equal_experiment_then_equal_client_trace_then_equal_round",
        "fisher_weighting":
            "equal_experiment_then_equal_client_trace_then_equal_round",
    })
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "selected_features_by_fold.json").write_text(json.dumps(selected_by_fold, indent=2, sort_keys=True), encoding="utf-8")

    with (output_dir / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "row_id", "experiment_id", "resolved_client_id", "inferred_round_index",
            "true_label", "predicted_label", *[f"prob_{c}" for c in classes],
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(usable):
            y = row["_y"]
            writer.writerow({
                "row_id": row.get("row_id"),
                "experiment_id": row.get("experiment_id"),
                "resolved_client_id": y.get("resolved_client_id"),
                "inferred_round_index": y.get("inferred_round_index"),
                "true_label": truth[i],
                "predicted_label": pred[i],
                **{f"prob_{c}": float(oof_prob[i, class_index[c]]) for c in classes},
            })

    _write_confusion_artifacts(
        output_dir,
        truth,
        pred,
        classes,
        unit_name="rounds",
        evidence_count=len(usable),
        evidence_unit="rounds",
    )
    _write_roc_pr(output_dir, truth, oof_prob, classes)

    # Late fusion: one final prediction per client trace, using only OOF round
    # probabilities generated by models that did not train on that experiment.
    fusion_groups: Dict[tuple[str, str], List[int]] = defaultdict(list)
    for i, row in enumerate(usable):
        y = row["_y"]
        key = (str(row.get("experiment_id") or ""), str(y.get("resolved_client_id") or row.get("client_capture_id") or ""))
        fusion_groups[key].append(i)

    fusion_truth = []
    fusion_pred = []
    fusion_prob = []
    fusion_rows = []
    for (experiment_id, client_id), indices in sorted(fusion_groups.items()):
        truths = {truth[i] for i in indices}
        if len(truths) != 1:
            raise RoundFingerprintingError(f"Conflicting round labels for {experiment_id}/{client_id}: {sorted(truths)}")
        averaged = np.mean(oof_prob[indices], axis=0)
        label = classes[int(np.argmax(averaged))]
        true_label = next(iter(truths))
        fusion_truth.append(true_label)
        fusion_pred.append(label)
        fusion_prob.append(averaged)
        fusion_rows.append({
            "experiment_id": experiment_id,
            "resolved_client_id": client_id,
            "round_count": len(indices),
            "true_label": true_label,
            "predicted_label": label,
            **{f"prob_{c}": float(averaged[class_index[c]]) for c in classes},
        })

    # output_dir is <feature_mode>/round/<level>/<parent>.
    # Place fusion beside round, not underneath it.
    fusion_dir = output_dir.parents[2] / "round_fusion" / output_dir.parent.name / output_dir.name
    fusion_dir.mkdir(parents=True, exist_ok=True)
    fusion_prob_array = np.asarray(fusion_prob, dtype=float)
    fusion_metrics = _metrics_from_predictions(fusion_truth, fusion_pred, fusion_prob_array, classes)
    fusion_metrics.update({
        "status": "evaluated",
        "target": target,
        "feature_mode": feature_mode,
        "classes": classes,
        "parent": parent_description,
        "evaluation_unit": "client_trace_after_round_probability_fusion",
        "fusion_method": "mean_oof_probability_across_inferred_rounds",
        "grouping": "experiment_id",
        "round_predictions_are_oof": True,
    })
    (fusion_dir / "metrics.json").write_text(json.dumps(fusion_metrics, indent=2, sort_keys=True), encoding="utf-8")
    with (fusion_dir / "oof_fused_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(fusion_rows[0].keys()) if fusion_rows else ["experiment_id"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(fusion_rows)
    _write_confusion_artifacts(
        fusion_dir,
        fusion_truth,
        fusion_pred,
        classes,
        unit_name="client traces",
        evidence_count=len(usable),
        evidence_unit="rounds",
    )
    _write_roc_pr(fusion_dir, fusion_truth, fusion_prob_array, classes)

    return {"round": metrics, "round_fusion": fusion_metrics}


def evaluate_round_hierarchy(
    *,
    x_csv: str | Path,
    y_csv: str | Path,
    output_root: str | Path,
    feature_modes: Sequence[str] = ("full", "size_normalized"),
) -> Dict[str, Any]:
    rows = _read_round_xy(x_csv, y_csv)
    if not rows:
        raise RoundFingerprintingError("Round X/Y dataset is empty")
    output_root = Path(output_root)
    summary: Dict[str, Any] = {}

    for feature_mode in feature_modes:
        mode_root = output_root / feature_mode
        results: Dict[str, Any] = {}
        results["family"] = _evaluate_stage_oof(
            rows,
            target="family",
            feature_mode=feature_mode,
            output_dir=mode_root / "round" / "family" / "all",
            parent_description="all",
        )

        families = sorted({str(row["_y"].get("family") or "") for row in rows if row["_y"].get("family")})
        arch_results = {}
        for family in families:
            subset = [row for row in rows if str(row["_y"].get("family") or "") == family]
            arch_results[family] = _evaluate_stage_oof(
                subset,
                target="architecture",
                feature_mode=feature_mode,
                output_dir=mode_root / "round" / "architecture" / _safe_slug(family),
                parent_description=f"family={family}; conditional_on_true_parent",
            )
        results["architecture_by_family"] = arch_results

        parent_pairs = sorted({
            (str(row["_y"].get("family") or ""), str(row["_y"].get("architecture") or ""))
            for row in rows if row["_y"].get("family") and row["_y"].get("architecture")
        })
        variant_results = {}
        for family, architecture in parent_pairs:
            subset = [
                row for row in rows
                if str(row["_y"].get("family") or "") == family
                and str(row["_y"].get("architecture") or "") == architecture
            ]
            key = f"{family}::{architecture}"
            variant_results[key] = _evaluate_stage_oof(
                subset,
                target="variant",
                feature_mode=feature_mode,
                output_dir=mode_root / "round" / "variant" / _safe_slug(key),
                parent_description=f"family={family}, architecture={architecture}; conditional_on_true_parent",
            )
        results["variant_by_parent"] = variant_results
        summary[feature_mode] = results

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "round_evaluation_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary_json": str(path), "feature_modes": list(feature_modes)}
