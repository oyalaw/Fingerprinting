from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import registry
from .capture import run_capture
from .config import (
    DEFAULT_CONFIG,
    ConfigError,
    generate_experiment_id,
    load_config,
    save_config,
)
from .dataset_catalog import (
    DATASETS,
    SIZE_ORDER,
    automatic_datasets,
    dataset_names,
    get_dataset_spec,
)
from .dataset_manager import DatasetError, prepare_dataset
from .experiment_output import (
    ExistingExperimentError,
    archive_existing_outputs,
    find_existing_outputs,
)
from .offload import CaptureOffloadError
from .proxy import (
    BlindTCPProxy,
    DEFAULT_PROXY_CONFIG,
    ProxyError,
    load_proxy_config,
    save_proxy_config,
)
from .runner import run
from .traffic import FeatureExtractionError, extract_capture_artifacts
from .tls import TLSConfigurationError


def choose(prompt: str, choices: List[str]) -> str:
    if not choices:
        raise ValueError(f"No choices available for {prompt}")
    print()
    print(prompt)
    for index, choice in enumerate(choices, start=1):
        print(f"  {index}. {choice}")
    while True:
        raw = input("Selection: ").strip()
        try:
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1]
        except ValueError:
            pass
        print("Enter one of the displayed numbers.")


def ask_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if raw:
        return raw
    return "" if default is None else default


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = ask_text(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Enter an integer.")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")



def resolve_existing_outputs_interactive(
    config: Dict[str, Any],
    role: str,
) -> Dict[str, Any]:
    """
    Prevent accidental mixing of repeated runs in the no-argument workflow.

    If output files already exist for the same experiment_id and role, the
    user must choose a new ID, archive the old role-specific files, or cancel.
    """
    while True:
        existing = find_existing_outputs(
            config,
            role=role,
        )
        if not existing:
            return config

        experiment_id = str(
            config["experiment"]["experiment_id"]
        )
        print()
        print(
            f"Experiment ID {experiment_id!r} already has "
            f"{role} output files:"
        )
        for path in existing:
            print(f"  - {path}")

        action = choose(
            "Existing experiment output detected",
            [
                "use_new_experiment_id",
                "archive_existing_run",
                "cancel",
            ],
        )

        if action == "use_new_experiment_id":
            new_id = ask_text(
                "New experiment ID; use auto for a timestamped ID",
                "auto",
            ).strip()
            if not new_id or new_id == "auto":
                new_id = generate_experiment_id()
            config["experiment"]["experiment_id"] = new_id
            continue

        if action == "archive_existing_run":
            archive_dir = archive_existing_outputs(
                config,
                role=role,
                paths=existing,
            )
            print(
                f"Archived {len(existing)} existing file(s) "
                f"to {archive_dir}"
            )
            return config

        raise KeyboardInterrupt

def interactive_configure(
    forced_role: str | None = None,
) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)

    config["experiment"]["experiment_id"] = ask_text(
        "Experiment ID; use the same ID on client, server, and proxy",
        "auto",
    )
    config["experiment"]["output_dir"] = ask_text(
        "Experiment output directory",
        "experiments/results",
    )

    task = choose("Select experiment task", registry.EXECUTION_TASKS)
    config["execution"]["task"] = task

    deployment_options = registry.deployments_for_task(task)
    if forced_role == "server":
        deployment_options = [
            value
            for value in deployment_options
            if value != "local"
        ]

    deployment = choose(
        "Select deployment",
        deployment_options,
    )
    config["execution"]["deployment"] = deployment

    role_options = (
        ["client"]
        if deployment == "local"
        else ["client", "server"]
    )
    if forced_role is not None:
        if forced_role not in role_options:
            raise ValueError(
                f"Role {forced_role!r} is not valid for "
                f"task={task!r}, deployment={deployment!r}"
            )
        role = forced_role
    else:
        role = choose("Select node role", role_options)

    config["node"]["role"] = role

    framework = choose("Select framework", registry.frameworks())
    config["ai"]["framework"] = framework

    runtime_options = ["native"] if task == "training" else registry.RUNTIMES
    runtime = choose("Select execution runtime", runtime_options)
    config["ai"]["runtime"] = runtime

    family = choose(
        "Select model family",
        registry.families_for_framework(framework, runtime),
    )
    config["ai"]["family"] = family

    architecture = choose(
        "Select architecture",
        registry.architectures_for(framework, family, runtime),
    )
    config["ai"]["architecture"] = architecture

    variant = choose(
        "Select architecture variant",
        registry.variants_for(
            framework,
            family,
            architecture,
            runtime,
        ),
    )
    config["ai"]["variant"] = variant

    application = choose(
        "Select application",
        registry.applications_for(architecture, variant),
    )
    config["ai"]["application"] = application

    dataset = choose(
        "Select dataset",
        registry.datasets_for(architecture, application, variant),
    )
    config["ai"]["dataset"] = dataset

    dataset_spec = get_dataset_spec(dataset)
    if dataset_spec.num_classes is not None:
        config["ai"]["num_classes"] = dataset_spec.num_classes

    config["data"]["root"] = ask_text("Dataset root directory", "datasets")
    default_split = "train" if task == "training" else dataset_spec.default_split
    config["data"]["split"] = ask_text("Dataset split", default_split)
    config["data"]["auto_download"] = ask_yes_no(
        "Download public datasets automatically when missing",
        True,
    )

    if dataset_spec.acquisition == "manual":
        local_path = ask_text(
            f"Local path for {dataset}; expected {dataset_spec.manual_layout}"
        )
        config["data"]["local_paths"][dataset] = local_path

    if registry.requires_artifact(framework, runtime, variant):
        artifact = ask_text("Path to local model artifact")
        config["ai"]["model_artifact"] = artifact or None

    config["device"]["label"] = choose("Select device label", registry.DEVICES)
    if config["device"]["label"] == "custom":
        config["device"]["label"] = ask_text("Custom device label", "custom")

    config["device"]["operating_system"] = ask_text(
        "Operating system label",
        "unknown",
    )

    default_host = "0.0.0.0" if role == "server" else "127.0.0.1"
    if deployment != "local":
        config["node"]["host"] = ask_text(
            (
                "Listen host"
                if role == "server"
                else "Remote endpoint IP or hostname (server or proxy)"
            ),
            default_host,
        )
        config["node"]["port"] = ask_int("Port", 5000)

        # TLS is presented first for new network experiments. TCP remains
        # available for controlled baseline/debug runs.
        transport_kind = choose(
            "Select network transport",
            ["tls", "tcp"],
        )
        config["transport"]["kind"] = transport_kind

        if transport_kind == "tls":
            config["transport"][
                "minimum_tls_version"
            ] = choose(
                "Minimum TLS version",
                ["TLSv1_2", "TLSv1_3"],
            )

            if role == "server":
                certfile = ask_text(
                    "TLS certificate path; leave blank to auto-generate "
                    "a research self-signed certificate",
                    "",
                ).strip()
                if certfile:
                    keyfile = ask_text(
                        "TLS private-key path",
                        "",
                    ).strip()
                    if not keyfile:
                        raise ValueError(
                            "TLS private-key path is required when "
                            "a certificate path is supplied"
                        )
                    config["transport"]["certfile"] = certfile
                    config["transport"]["keyfile"] = keyfile
                    config["transport"][
                        "auto_generate_self_signed"
                    ] = False
                else:
                    config["transport"][
                        "auto_generate_self_signed"
                    ] = True
                    config["transport"][
                        "self_signed_common_name"
                    ] = ask_text(
                        "Self-signed certificate name",
                        "ai-fingerprint-server",
                    )
            else:
                verify_peer = ask_yes_no(
                    "Verify the server TLS certificate",
                    False,
                )
                config["transport"][
                    "verify_peer"
                ] = verify_peer
                if verify_peer:
                    cafile = ask_text(
                        "CA/certificate file used to verify the server",
                        "",
                    ).strip()
                    hostname = ask_text(
                        "TLS server hostname from the certificate",
                        "ai-fingerprint-server",
                    ).strip()
                    config["transport"]["cafile"] = (
                        cafile or None
                    )
                    config["transport"][
                        "server_hostname"
                    ] = hostname or None

    config["ai"]["input_size"] = ask_int("Image input size", 224)
    config["execution"]["batch_size"] = ask_int("Batch size", 1)

    if task == "inference":
        config["execution"]["repetitions"] = ask_int("Repetitions", 20)
        config["execution"]["warmup"] = ask_int("Warmup runs", 2)
        config["execution"]["interval_ms"] = ask_int(
            "Interval between requests in milliseconds",
            250,
        )
    else:
        config["execution"]["learning_rate"] = float(
            ask_text("Learning rate", "0.001")
        )
        if deployment in {"local", "remote"}:
            config["execution"]["epochs"] = ask_int("Training epochs", 1)
            config["execution"]["steps_per_epoch"] = ask_int(
                "Training steps per epoch",
                20,
            )
        else:
            config["federated"]["rounds"] = ask_int(
                "Federated rounds",
                10,
            )
            config["federated"]["local_epochs"] = ask_int(
                "Local epochs per round",
                1,
            )
            config["federated"]["steps_per_epoch"] = ask_int(
                "Local steps per epoch",
                10,
            )
            if role == "server":
                config["federated"]["expected_clients"] = ask_int(
                    "Expected clients",
                    2,
                )
            else:
                config["federated"]["client_id"] = ask_text(
                    "Federated client ID",
                    "client_1",
                )

    monitor_enabled = ask_yes_no(
        "Enable client/server resource telemetry",
        True,
    )
    config["resource_monitor"]["enabled"] = monitor_enabled
    if monitor_enabled:
        config["resource_monitor"]["interval_ms"] = ask_int(
            "Resource telemetry interval in milliseconds",
            500,
        )
        interface = ask_text(
            "Network interface for byte counters; leave blank for all",
            "",
        )
        config["resource_monitor"]["network_interface"] = interface or None
        config["resource_monitor"]["gpu_index"] = ask_int(
            "NVIDIA GPU index",
            0,
        )
        config["resource_monitor"]["power_enabled"] = ask_yes_no(
            "Collect power and energy when hardware counters are available",
            True,
        )

    return config


def interactive_proxy_configure() -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_PROXY_CONFIG)

    experiment_id = ask_text(
        "Proxy experiment ID; use auto for timestamped ID",
        "auto",
    )
    config["experiment"]["experiment_id"] = experiment_id
    config["experiment"]["output_dir"] = ask_text(
        "Proxy output directory",
        "proxy_results",
    )

    config["proxy"]["listen_host"] = ask_text(
        "Proxy listen host",
        "0.0.0.0",
    )
    config["proxy"]["listen_port"] = ask_int(
        "Proxy listen port",
        5000,
    )
    config["proxy"]["upstream_host"] = ask_text(
        "Real upstream server IP or hostname",
        "127.0.0.1",
    )
    config["proxy"]["upstream_port"] = ask_int(
        "Real upstream server port",
        5000,
    )

    capture_enabled = ask_yes_no(
        "Capture client-facing traffic for fingerprinting",
        True,
    )
    config["capture"]["enabled"] = capture_enabled

    if capture_enabled:
        config["capture"]["interface"] = ask_text(
            "Client-facing capture interface",
            "wlan0",
        )
        config["capture"]["snaplen_bytes"] = ask_int(
            "PCAP snapshot length in bytes; 0 stores full frames",
            256,
        )

        manage_offloads = ask_yes_no(
            "Automatically disable GRO/GSO/TSO/LRO during capture",
            True,
        )
        config["capture"]["offload_management"][
            "enabled"
        ] = manage_offloads
        if manage_offloads:
            config["capture"]["offload_management"][
                "required"
            ] = ask_yes_no(
                "Abort capture if offloads cannot be verified disabled",
                True,
            )
            config["capture"]["offload_management"][
                "allow_sudo_noninteractive"
            ] = ask_yes_no(
                "Allow non-interactive sudo fallback for ethtool only",
                True,
            )
            config["capture"]["offload_management"][
                "restore_on_exit"
            ] = True

        while True:
            client_text = ask_text(
                "Participating client IPs for capture isolation "
                "(comma-separated)",
                "",
            )
            client_ips = [
                value.strip()
                for value in client_text.split(",")
                if value.strip()
            ]
            if client_ips:
                break
            print(
                "At least one client IP is required. This prevents "
                "capturing the proxy-to-upstream duplicate leg."
            )

        config["capture"]["client_ip"] = None
        config["capture"]["client_ips"] = list(
            dict.fromkeys(client_ips)
        )

        alias_text = ask_text(
            "Optional capture aliases as IP=alias pairs. Leave blank to "
            "use neutral trace IDs; FL client IDs are auto-resolved later "
            "from client network-registration ground truth",
            "",
        )
        aliases: Dict[str, str] = {}
        if alias_text:
            for piece in alias_text.split(","):
                piece = piece.strip()
                if not piece:
                    continue
                if "=" not in piece:
                    raise ValueError(
                        "Client aliases must use IP=client_id syntax"
                    )
                ip, alias = piece.split("=", 1)
                ip = ip.strip()
                alias = alias.strip()
                if ip not in config["capture"]["client_ips"]:
                    raise ValueError(
                        f"Alias IP {ip!r} is not in the configured "
                        "client IP list"
                    )
                if not alias:
                    raise ValueError(
                        f"Empty client alias for {ip!r}"
                    )
                aliases[ip] = alias
        config["capture"]["client_aliases"] = aliases

        config["capture"]["per_client_artifacts"] = ask_yes_no(
            "Generate separate classifier-safe sequence/features per client",
            True,
        )

        config["capture"]["extract_after"] = ask_yes_no(
            "Extract packet sequence and handcrafted features on stop",
            True,
        )

        window_text = ask_text(
            "Feature window in seconds; enter 0 for overall only",
            "5.0",
        )
        window_value = float(window_text)
        config["capture"]["window_seconds"] = (
            window_value if window_value > 0 else None
        )

    return config


def run_interactive_proxy() -> None:
    output = "proxy_config.yaml"
    config = interactive_proxy_configure()

    if config["experiment"]["experiment_id"] in {
        None,
        "",
        "auto",
    }:
        # Reuse the loader's automatic ID generation by saving once,
        # then loading the resolved label-blind config.
        save_proxy_config(config, output)
        config = load_proxy_config(output)
    else:
        save_proxy_config(config, output)
        config = load_proxy_config(output)

    config = resolve_existing_outputs_interactive(
        config,
        role="proxy",
    )
    save_proxy_config(config, output)

    print(f"Proxy configuration saved to {output}")
    BlindTCPProxy(config).serve_forever()

def print_dataset_table() -> None:
    header = (
        f"{'DATASET':30} {'APPLICATION':24} {'SOURCE':18} "
        f"{'ACQUISITION':11} {'SIZE':11} {'CLASSES':7}"
    )
    print(header)
    print("=" * len(header))
    for name in dataset_names():
        spec = DATASETS[name]
        applications = ",".join(spec.applications)
        classes = "-" if spec.num_classes is None else str(spec.num_classes)
        print(
            f"{name:30} {applications:24} {spec.source:18} "
            f"{spec.acquisition:11} {spec.size_tier:11} {classes:7}"
        )


def print_dataset_info(name: str) -> None:
    spec = get_dataset_spec(name)
    print(f"Name: {spec.name}")
    print(f"Applications: {', '.join(spec.applications)}")
    print(f"Modality: {spec.modality}")
    print(f"Source: {spec.source}")
    print(f"Acquisition: {spec.acquisition}")
    print(f"Size tier: {spec.size_tier}")
    print(f"Classes: {spec.num_classes}")
    print(f"Default split: {spec.default_split}")
    print(f"Description: {spec.description}")
    if spec.manual_layout:
        print(f"Expected local layout: {spec.manual_layout}")


def download_datasets(
    names: List[str],
    root: str,
    split: str | None,
    input_size: int,
) -> None:
    succeeded = 0
    failed = 0
    skipped = 0

    for name in names:
        print(f"\n[{name}]")
        try:
            result = prepare_dataset(
                name=name,
                root=root,
                split=split,
                input_size=input_size,
            )
            status = result["status"]
            if status in {"manual", "synthetic"}:
                skipped += 1
            else:
                succeeded += 1
            for key, value in result.items():
                print(f"  {key}: {value}")
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    print()
    print(
        f"Dataset preparation finished: "
        f"{succeeded} ready, {skipped} skipped, {failed} failed."
    )


def _print_artifacts(result: Dict[str, Any]) -> None:
    print()
    print("Generated artifacts")
    for key in (
        "pcap",
        "packet_sequence_csv",
        "fingerprint_sequence_csv",
        "features_csv",
        "manifest_json",
        "packet_count",
        "feature_count",
        "feature_row_count",
    ):
        if key in result:
            print(f"  {key}: {result[key]}")

    per_client = result.get("per_client_artifacts", {}) or {}
    if per_client:
        print("  per_client_artifacts:")
        for alias, artifacts in per_client.items():
            print(
                f"    {alias}: packets={artifacts.get('packet_count')} "
                f"features={artifacts.get('features_csv')} "
                f"sequence={artifacts.get('fingerprint_sequence_csv')}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI workload fingerprinting experiment controller"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure = subparsers.add_parser(
        "configure",
        help="Interactively create a YAML configuration",
    )
    configure.add_argument(
        "--output",
        default="config.yaml",
        help="Output YAML path",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a client or server from a configuration file",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration",
    )

    proxy_parser = subparsers.add_parser(
        "proxy",
        help="Run the label-blind inline TCP/TLS forwarding proxy",
    )
    proxy_parser.add_argument(
        "--config",
        required=True,
        help="Path to label-blind proxy YAML configuration",
    )

    capture = subparsers.add_parser(
        "capture",
        help=(
            "Capture traffic and automatically export packet sequences, "
            "handcrafted features, and a manifest"
        ),
    )
    capture.add_argument("--interface", required=True)
    capture.add_argument(
        "--host",
        default=None,
        help=(
            "Optional capture filter host. When --server-ip is omitted, "
            "this host is also treated as the server direction reference."
        ),
    )
    capture.add_argument("--port", type=int, default=None)
    capture.add_argument("--experiment-id", required=True)
    capture.add_argument("--output", default=None)
    capture.add_argument("--output-dir", default=None)
    capture.add_argument("--server-ip", default=None)
    capture.add_argument("--client-ip", default=None)
    capture.add_argument(
        "--client-ips",
        default=None,
        help=(
            "Comma-separated participating client IPs. Prefer this for "
            "inline-proxy captures so the upstream duplicate leg is "
            "excluded."
        ),
    )
    capture.add_argument("--burst-gap-sec", type=float, default=0.05)
    capture.add_argument("--idle-threshold-sec", type=float, default=0.5)
    capture.add_argument("--window-seconds", type=float, default=None)
    capture.add_argument(
        "--snaplen-bytes",
        type=int,
        default=256,
        help=(
            "Capture only the first N bytes of each frame while preserving "
            "the original frame length. Use 0 for full frames."
        ),
    )
    capture.add_argument(
        "--no-manage-offloads",
        action="store_true",
        help=(
            "Do not automatically disable GRO/GSO/TSO/LRO. "
            "Not recommended for packet-size fingerprinting."
        ),
    )
    capture.add_argument(
        "--offload-warning-only",
        action="store_true",
        help=(
            "Attempt offload management but do not abort if the "
            "disabled state cannot be verified."
        ),
    )
    capture.add_argument(
        "--no-sudo-offload-fallback",
        action="store_true",
        help=(
            "Do not retry ethtool -K through sudo -n when direct "
            "CAP_NET_ADMIN permission is unavailable."
        ),
    )
    capture.add_argument(
        "--no-extract",
        action="store_true",
        help="Capture PCAP only and skip automatic extraction",
    )

    extract = subparsers.add_parser(
        "extract",
        help=(
            "Extract packet sequence CSV, handcrafted feature CSV, "
            "and manifest from an existing PCAP"
        ),
    )
    extract.add_argument("--pcap", required=True)
    extract.add_argument("--experiment-id", required=True)
    extract.add_argument("--output-dir", default=None)
    extract.add_argument("--server-ip", default=None)
    extract.add_argument("--client-ip", default=None)
    extract.add_argument(
        "--client-ips",
        default=None,
        help=(
            "Comma-separated client IPs used to post-filter a broad PCAP "
            "to client-facing traffic only."
        ),
    )
    extract.add_argument("--burst-gap-sec", type=float, default=0.05)
    extract.add_argument("--idle-threshold-sec", type=float, default=0.5)
    extract.add_argument("--window-seconds", type=float, default=None)

    datasets_parser = subparsers.add_parser(
        "datasets",
        help="List, inspect, or download experiment datasets",
    )
    dataset_subparsers = datasets_parser.add_subparsers(
        dest="dataset_command",
        required=True,
    )

    dataset_subparsers.add_parser(
        "list",
        help="List the complete dataset catalog",
    )

    info = dataset_subparsers.add_parser(
        "info",
        help="Show details about one dataset",
    )
    info.add_argument("--name", required=True, choices=dataset_names())

    download = dataset_subparsers.add_parser(
        "download",
        help="Prepare one dataset or a batch of automatic datasets",
    )
    target = download.add_mutually_exclusive_group(required=True)
    target.add_argument("--name", choices=dataset_names())
    target.add_argument("--all", action="store_true")
    download.add_argument("--root", default="datasets")
    download.add_argument("--split", default=None)
    download.add_argument("--input-size", type=int, default=224)
    download.add_argument(
        "--max-tier",
        choices=list(SIZE_ORDER),
        default="medium",
        help="For --all, download automatic datasets up to this size tier",
    )

    return parser


def main() -> None:
    # No-argument execution is the default interactive workflow.
    # The first choice selects whether this machine acts as a client,
    # server, or label-blind inline proxy.
    if len(sys.argv) == 1 or sys.argv[1:] == ["--interactive"]:
        role = choose(
            "Select experiment node role",
            ["client", "server", "proxy"],
        )

        if role == "proxy":
            run_interactive_proxy()
            return

        output = "config.yaml"
        config = interactive_configure(
            forced_role=role,
        )
        save_config(config, output)
        config = load_config(output)
        config = resolve_existing_outputs_interactive(
            config,
            role=role,
        )
        save_config(config, output)
        print(f"Configuration saved to {output}")
        run(config)
        return

    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "configure":
            config = interactive_configure()
            save_config(config, args.output)
            print(f"Configuration saved to {args.output}")
            return

        if args.command == "run":
            config = load_config(args.config)
            run(config)
            return

        if args.command == "proxy":
            config = load_proxy_config(args.config)
            BlindTCPProxy(config).serve_forever()
            return

        if args.command == "capture":
            output = args.output or f"captures/{args.experiment_id}.pcapng"
            client_ips = (
                [
                    value.strip()
                    for value in args.client_ips.split(",")
                    if value.strip()
                ]
                if args.client_ips
                else []
            )
            result = run_capture(
                interface=args.interface,
                output=output,
                host=args.host,
                hosts=client_ips,
                port=args.port,
                snaplen_bytes=args.snaplen_bytes,
                experiment_id=args.experiment_id,
                extract_after=not args.no_extract,
                output_dir=args.output_dir,
                server_ip=args.server_ip,
                client_ip=args.client_ip,
                burst_gap_sec=args.burst_gap_sec,
                idle_threshold_sec=args.idle_threshold_sec,
                window_seconds=args.window_seconds,
                manage_offloads=not args.no_manage_offloads,
                require_offloads_disabled=(
                    not args.offload_warning_only
                ),
                allow_sudo_noninteractive=(
                    not args.no_sudo_offload_fallback
                ),
            )
            _print_artifacts(result)
            return

        if args.command == "extract":
            client_ips = (
                [
                    value.strip()
                    for value in args.client_ips.split(",")
                    if value.strip()
                ]
                if args.client_ips
                else []
            )
            result = extract_capture_artifacts(
                pcap_path=args.pcap,
                experiment_id=args.experiment_id,
                output_dir=args.output_dir,
                server_ip=args.server_ip,
                client_ip=args.client_ip,
                client_ips=client_ips,
                burst_gap_sec=args.burst_gap_sec,
                idle_threshold_sec=args.idle_threshold_sec,
                window_seconds=args.window_seconds,
            )
            _print_artifacts(result)
            return

        if args.command == "datasets":
            if args.dataset_command == "list":
                print_dataset_table()
                return

            if args.dataset_command == "info":
                print_dataset_info(args.name)
                return

            if args.dataset_command == "download":
                names = (
                    automatic_datasets(args.max_tier)
                    if args.all
                    else [args.name]
                )
                download_datasets(
                    names=names,
                    root=args.root,
                    split=args.split,
                    input_size=args.input_size,
                )
                return

    except (
        CaptureOffloadError,
        TLSConfigurationError,
        ConfigError,
        DatasetError,
        ExistingExperimentError,
        FeatureExtractionError,
        ProxyError,
        KeyError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\nStopped.")
