from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import registry
from .capture import run_capture
from .config import DEFAULT_CONFIG, ConfigError, load_config, save_config
from .dataset_catalog import (
    DATASETS,
    SIZE_ORDER,
    automatic_datasets,
    dataset_names,
    get_dataset_spec,
)
from .dataset_manager import DatasetError, prepare_dataset
from .proxy import (
    BlindTCPProxy,
    DEFAULT_PROXY_CONFIG,
    ProxyError,
    load_proxy_config,
    save_proxy_config,
)
from .runner import run
from .traffic import FeatureExtractionError, extract_capture_artifacts


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

        client_ip = ask_text(
            "Client IP for capture isolation; leave blank if unknown",
            "",
        )
        config["capture"]["client_ip"] = client_ip or None

        if not client_ip:
            proxy_ip = ask_text(
                "Proxy IP visible to the client for direction inference",
                "",
            )
            config["capture"]["proxy_client_facing_ip"] = (
                proxy_ip or None
            )

        config["capture"]["extract_after"] = ask_yes_no(
            "Extract packet sequence and handcrafted features on stop",
            True,
        )

        window_text = ask_text(
            "Optional feature window in seconds; blank for overall only",
            "",
        )
        config["capture"]["window_seconds"] = (
            float(window_text)
            if window_text
            else None
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
        # then loading and saving the resolved label-blind config.
        save_proxy_config(config, output)
        config = load_proxy_config(output)
        save_proxy_config(config, output)
    else:
        save_proxy_config(config, output)
        config = load_proxy_config(output)

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
    capture.add_argument("--burst-gap-sec", type=float, default=0.05)
    capture.add_argument("--idle-threshold-sec", type=float, default=0.5)
    capture.add_argument("--window-seconds", type=float, default=None)
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
            result = run_capture(
                interface=args.interface,
                output=output,
                host=args.host,
                port=args.port,
                experiment_id=args.experiment_id,
                extract_after=not args.no_extract,
                output_dir=args.output_dir,
                server_ip=args.server_ip,
                client_ip=args.client_ip,
                burst_gap_sec=args.burst_gap_sec,
                idle_threshold_sec=args.idle_threshold_sec,
                window_seconds=args.window_seconds,
            )
            _print_artifacts(result)
            return

        if args.command == "extract":
            result = extract_capture_artifacts(
                pcap_path=args.pcap,
                experiment_id=args.experiment_id,
                output_dir=args.output_dir,
                server_ip=args.server_ip,
                client_ip=args.client_ip,
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
        ConfigError,
        DatasetError,
        FeatureExtractionError,
        ProxyError,
        KeyError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\nStopped.")
