from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_fingerprint.traffic import (
    FeatureExtractionError,
    repair_packet_sequence_artifacts,
)


def _discover_raw_sequences(root: Path):
    return sorted(
        path
        for path in root.rglob("*_packet_sequence.csv")
        if (
            "fingerprint_sequence" not in path.name
            and "_repaired" not in path.parts
            and "fingerprinting_dataset" not in path.parts
        )
    )


def _choose_file(paths):
    if not paths:
        raise SystemExit(
            "No raw *_packet_sequence.csv file was found under the current "
            "directory. Copy the proxy packet-sequence CSV into the project "
            "or pass --input."
        )
    if len(paths) == 1:
        return paths[0]

    print("Raw packet-sequence files:")
    for index, path in enumerate(paths, start=1):
        print(f"  {index}. {path}")
    while True:
        raw = input("Selection: ").strip()
        try:
            selected = int(raw)
        except ValueError:
            selected = 0
        if 1 <= selected <= len(paths):
            return paths[selected - 1]
        print("Enter one of the displayed numbers.")


def _infer_experiment_id(path: Path) -> str:
    suffix = "_packet_sequence.csv"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def _read_manifest_defaults(path: Path):
    suffix = "_packet_sequence.csv"
    stem = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
    manifest = path.with_name(f"{stem}_manifest.json")
    if not manifest.exists():
        return {}, None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}, None

    isolation = data.get("capture_isolation", {}) or {}
    client_ips = isolation.get("configured_client_ips", []) or []
    aliases = isolation.get("client_aliases", {}) or {}
    if not client_ips:
        direction = data.get("direction_reference", {}) or {}
        value = direction.get("client_ips", []) or []
        client_ips = value if isinstance(value, list) else []
    return {str(k): str(v) for k, v in aliases.items()}, [str(v) for v in client_ips]


def _parse_aliases(text: str, client_ips):
    aliases = {}
    if not text.strip():
        return aliases
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError("Aliases must use IP=client_id syntax")
        ip, alias = piece.split("=", 1)
        ip = ip.strip()
        alias = alias.strip()
        if ip not in client_ips:
            raise ValueError(f"Alias IP {ip!r} is not in client IP list")
        if not alias:
            raise ValueError(f"Empty alias for {ip!r}")
        aliases[ip] = alias
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repair an existing broad proxy packet sequence without the "
            "original PCAP. Removes the upstream duplicate leg, reconstructs "
            "TCP flags, and produces per-client/windowed fingerprint files."
        )
    )
    parser.add_argument("--input", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--client-ips", default=None)
    parser.add_argument("--aliases", default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(".").resolve()
    source = (
        Path(args.input).expanduser().resolve()
        if args.input
        else _choose_file(_discover_raw_sequences(root))
    )
    experiment_id = args.experiment_id or _infer_experiment_id(source)

    manifest_aliases, manifest_ips = _read_manifest_defaults(source)
    if args.client_ips:
        client_ips = [
            value.strip()
            for value in args.client_ips.split(",")
            if value.strip()
        ]
    elif manifest_ips:
        client_ips = manifest_ips
        print("Client IPs from manifest:", ", ".join(client_ips))
    else:
        while True:
            raw = input(
                "Participating client IPs, comma-separated "
                "(required to remove the upstream duplicate leg): "
            ).strip()
            client_ips = [
                value.strip()
                for value in raw.split(",")
                if value.strip()
            ]
            if client_ips:
                break
            print("At least one client IP is required.")

    if args.aliases is not None:
        aliases = _parse_aliases(args.aliases, client_ips)
    elif manifest_aliases:
        aliases = manifest_aliases
    else:
        alias_text = input(
            "Optional aliases as IP=client_id pairs, comma-separated. "
            "For federated learning, use the exact federated client IDs: "
        ).strip()
        aliases = _parse_aliases(alias_text, client_ips)

    if args.window_seconds is not None:
        window_seconds = args.window_seconds
    else:
        raw = input("Feature window seconds [5.0; enter 0 for overall only]: ").strip()
        window_seconds = float(raw or "5.0")
    if window_seconds <= 0:
        window_seconds = None

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else source.parent / f"{experiment_id}_repaired"
    )

    print("\nRepair policy:")
    print("  - retain only packets involving configured client IPs")
    print("  - recompute direction from client membership")
    print("  - derive SYN/ACK/FIN/RST from tcp_flags_hex")
    print("  - strip endpoint identity from classifier-safe sequences")
    print("  - generate separate per-client feature/sequence files")
    print("  - preserve an overall row plus configured time windows")

    try:
        result = repair_packet_sequence_artifacts(
            raw_packet_csv=source,
            experiment_id=experiment_id,
            client_ips=client_ips,
            client_aliases=aliases,
            output_dir=output_dir,
            window_seconds=window_seconds,
        )
    except (FeatureExtractionError, ValueError) as exc:
        raise SystemExit(f"Repair failed: {exc}") from exc

    print("\nRepair complete:")
    for key in (
        "repair_output_dir",
        "packet_sequence_csv",
        "fingerprint_sequence_csv",
        "features_csv",
        "manifest_json",
        "packet_count",
        "feature_row_count",
    ):
        if key in result:
            print(f"  {key}: {result[key]}")

    per_client = result.get("per_client_artifacts", {}) or {}
    if per_client:
        print("  per_client_artifacts:")
        for alias, details in per_client.items():
            print(
                f"    {alias}: packets={details.get('packet_count')} "
                f"features={details.get('features_csv')} "
                f"sequence={details.get('fingerprint_sequence_csv')}"
            )


if __name__ == "__main__":
    main()
