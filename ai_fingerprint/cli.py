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
from .experiment_layout import (
    ExperimentLayoutError,
    materialize_role_metadata,
    write_role_status,
    apply_hierarchical_layout,
    apply_proxy_locator_layout,
    branch_directory,
    locator_for,
    next_experiment_number,
    normalize_experiment_id,
)
from .networking import (
    candidate_interfaces,
    detect_consensus_interface,
    detect_local_interface,
    detect_route_interface,
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


def _select_interface_fallback(prompt: str) -> str:
    choices = candidate_interfaces()
    if not choices:
        raise ValueError(
            "No usable IPv4 network interfaces were detected. "
            "Check that the experiment interface is up."
        )
    return choose(prompt, choices)


def _auto_resource_interface(config: Dict[str, Any]) -> str:
    role = str(config.get("node", {}).get("role", ""))
    detection = None
    if role == "client":
        detection = detect_route_interface(str(config["node"]["host"]))
    elif role == "server":
        host = str(config["node"].get("host", "")).strip()
        if host and host not in {"0.0.0.0", "::"}:
            detection = detect_local_interface(host)

    if detection is not None:
        detail = f" ({detection.local_ip})" if detection.local_ip else ""
        print(
            f"Automatically detected network interface: "
            f"{detection.interface}{detail} "
            f"[{detection.method}]"
        )
        return detection.interface

    return _select_interface_fallback(
        "Automatic interface detection was inconclusive; "
        "select the experiment interface"
    )


def _configure_experiment_storage(
    config: Dict[str, Any],
    role: str,
) -> None:
    root = ask_text(
        "Experiment results root",
        "experiments/results",
    )
    branch = branch_directory(root, config)
    suggested = next_experiment_number(branch)
    print()
    print(f"Experiment branch: {branch}")
    print(f"Next available experiment in this branch: exp{suggested}")
    prompt = (
        "Experiment number; press Enter to use the next available number"
        if role == "server" or config.get("execution", {}).get("deployment") == "local"
        else "Coordinated experiment number; use the SAME number as the server"
    )
    raw = ask_text(prompt, str(suggested)).strip()
    exp_id = normalize_experiment_id(raw)
    role_token = role
    if role == "client" and config.get("execution", {}).get("deployment") == "federated":
        client_id = str(config.get("federated", {}).get("client_id", "client_1"))
        role_token = client_id
    apply_hierarchical_layout(
        config,
        root=root,
        experiment_id=exp_id,
        role_token=role_token,
    )
    print(f"Coordinated experiment ID: {config['experiment']['experiment_id']}")
    print(f"Experiment storage locator: {locator_for(config)}")
    print(f"Role output directory: {config['experiment']['output_dir']}")



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
            experiment = config.get("experiment", {})
            if experiment.get("layout_version") == "1.0" and "ai" in config:
                branch = Path(experiment["branch_dir"])
                suggested = next_experiment_number(branch)
                new_id = normalize_experiment_id(
                    ask_text(
                        "New experiment number",
                        str(suggested),
                    )
                )
                role_token = role
                if role == "client" and config.get("execution", {}).get("deployment") == "federated":
                    role_token = str(config.get("federated", {}).get("client_id", "client_1"))
                apply_hierarchical_layout(
                    config,
                    root=experiment.get("results_root", "experiments/results"),
                    experiment_id=new_id,
                    role_token=role_token,
                )
            elif experiment.get("layout_version") == "1.0" and role == "proxy":
                new_locator = ask_text(
                    "New coordinated storage locator from the server",
                    "",
                ).strip()
                if not new_locator:
                    raise KeyboardInterrupt
                apply_proxy_locator_layout(
                    config,
                    root=experiment.get("results_root", "experiments/results"),
                    locator=new_locator,
                )
            else:
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

    executable_families = registry.families_for_framework(framework, runtime)
    catalog_only_families = sorted(
        set(registry.families()) - set(executable_families)
    )
    if runtime == "native" and catalog_only_families:
        print(
            "Catalog-only families not selectable for native execution: "
            + ", ".join(catalog_only_families)
            + ". Use `python main.py models --framework "
            + framework
            + "` to inspect their status."
        )
    family = choose(
        "Select model family",
        executable_families,
    )
    config["ai"]["family"] = family

    executable_architectures = registry.architectures_for(
        framework, family, runtime
    )
    catalog_architectures = registry.architectures_for_family(family)
    catalog_only_architectures = sorted(
        set(catalog_architectures) - set(executable_architectures)
    )
    if runtime == "native" and catalog_only_architectures:
        print(
            f"Catalog-only {family} architectures not selectable natively: "
            + ", ".join(catalog_only_architectures)
        )
    architecture = choose(
        "Select architecture",
        executable_architectures,
    )
    config["ai"]["architecture"] = architecture

    executable_variants = registry.variants_for(
        framework, family, architecture, runtime
    )
    catalog_variants = registry.variants_for_architecture(architecture)
    catalog_only_variants = sorted(
        set(catalog_variants) - set(executable_variants)
    )
    if runtime == "native" and catalog_only_variants:
        print(
            f"Catalog-only {architecture} variants not selectable natively: "
            + ", ".join(catalog_only_variants)
        )
    variant = choose(
        "Select architecture variant",
        executable_variants,
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

    operating_system = choose(
        "Select operating system",
        registry.operating_systems_for_device(config["device"]["label"]),
    )
    if operating_system == "custom":
        operating_system = ask_text(
            "Custom operating system label",
            "custom",
        )
    config["device"]["operating_system"] = operating_system

    if deployment != "local":
        if role == "server":
            config["node"]["host"] = ask_text(
                "FL server bind IP",
                "10.42.0.195",
            )
            config["node"]["port"] = ask_int(
                "FL server port",
                8080,
            )
        else:
            config["node"]["host"] = ask_text(
                "Remote endpoint IP or hostname (proxy)",
                "10.42.0.1",
            )
            config["node"]["port"] = ask_int(
                "Proxy port",
                8080,
            )
            print(
                "Network path: Client -> "
                f"{config['node']['host']}:{config['node']['port']} (Proxy) "
                "-> 10.42.0.195:8080 (FL Server)"
            )

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

    default_input_size = (
        32
        if family in {"autoencoder", "mlp"}
        and application in {
            "image_classification",
            "reconstruction",
            "anomaly_detection",
        }
        else 224
    )
    federated_client = (
        task == "training"
        and deployment == "federated"
        and role == "client"
    )

    # For controlled FL experiments the server is the single authority for
    # model/training-shape controls. A client receives these six values during
    # the FL handshake rather than independently typing them.
    if not federated_client:
        config["ai"]["input_size"] = ask_int(
            "Image input size",
            default_input_size,
        )
        config["execution"]["batch_size"] = ask_int("Batch size", 1)
    else:
        print(
            "Federated training policy will be received from the server: "
            "input size, batch size, learning rate, global rounds, "
            "local epochs, and local steps."
        )

    if task == "inference":
        config["execution"]["repetitions"] = ask_int("Repetitions", 20)
        config["execution"]["warmup"] = ask_int("Warmup runs", 2)
        config["execution"]["interval_ms"] = ask_int(
            "Interval between requests in milliseconds",
            250,
        )
    else:
        if deployment in {"local", "remote"}:
            config["execution"]["learning_rate"] = float(
                ask_text("Learning rate", "0.001")
            )
            config["execution"]["epochs"] = ask_int("Training epochs", 1)
            config["execution"]["steps_per_epoch"] = ask_int(
                "Training steps per epoch",
                20,
            )
        elif role == "server":
            config["execution"]["learning_rate"] = float(
                ask_text("Learning rate", "0.001")
            )
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
            config["federated"]["expected_clients"] = ask_int(
                "Expected clients",
                2,
            )
            config["federated"]["policy_source"] = "server"
        else:
            config["federated"]["client_id"] = ask_text(
                "Federated client ID",
                "client_1",
            )
            config["federated"]["policy_source"] = "server"

    if task == "training" and deployment == "federated":
        performance_enabled = ask_yes_no(
            "Log per-round training performance",
            True,
        )
        config["performance_logging"]["enabled"] = performance_enabled
        if performance_enabled:
            if role == "client":
                config["performance_logging"]["client_round_probe"] = ask_yes_no(
                    "Evaluate one held-out probe batch before/after each local round",
                    True,
                )
            elif role == "server":
                config["performance_logging"]["server_eval_batches"] = ask_int(
                    "Global evaluation batches after each federated round",
                    10,
                )
                config["performance_logging"]["server_eval_split"] = ask_text(
                    "Global evaluation dataset split",
                    "test",
                ).strip().lower()

    _configure_experiment_storage(config, role)

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
        config["resource_monitor"]["network_interface"] = (
            _auto_resource_interface(config)
        )
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

    results_root = ask_text(
        "Experiment results root",
        "experiments/results",
    )
    while True:
        locator = ask_text(
            "Coordinated experiment storage locator; paste the value "
            "printed by the server",
            "",
        ).strip()
        if locator:
            break
        print(
            "The storage locator is required so proxy output is placed "
            "under the same family/architecture/variant/application/"
            "dataset/framework/expN hierarchy."
        )
    apply_proxy_locator_layout(
        config,
        root=results_root,
        locator=locator,
    )
    print(f"Coordinated experiment ID: {config['experiment']['experiment_id']}")
    print(f"Proxy output directory: {config['experiment']['output_dir']}")

    config["proxy"]["listen_host"] = ask_text(
        "Proxy listen IP",
        "10.42.0.1",
    )
    config["proxy"]["listen_port"] = ask_int(
        "Proxy listen port",
        8080,
    )
    config["proxy"]["upstream_host"] = ask_text(
        "FL server upstream IP",
        "10.42.0.195",
    )
    config["proxy"]["upstream_port"] = ask_int(
        "FL server upstream port",
        8080,
    )
    print(
        "Network path: Clients -> "
        f"{config['proxy']['listen_host']}:{config['proxy']['listen_port']} "
        "(Proxy) -> "
        f"{config['proxy']['upstream_host']}:{config['proxy']['upstream_port']} "
        "(FL Server)"
    )

    capture_enabled = ask_yes_no(
        "Capture client-facing traffic for fingerprinting",
        True,
    )
    config["capture"]["enabled"] = capture_enabled

    if capture_enabled:
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

        # v0.9.2: participating clients are discovered automatically from
        # actual proxy connections. The archival/live capture filter excludes
        # the known upstream FL server, and discovered peer IPs are used only
        # for post-capture grouping/isolation (never as classifier features).
        config["capture"]["client_ip"] = None
        config["capture"]["client_ips"] = []
        config["capture"]["client_aliases"] = {}
        config["capture"]["client_discovery_mode"] = "automatic"
        print(
            "Participating client discovery: automatic "
            "(from accepted proxy connections)."
        )
        print(
            "Upstream FL server excluded from capture: "
            f"{config['proxy']['upstream_host']}"
        )

        listen_ip = str(config["proxy"].get("listen_host", "")).strip()
        detection = None
        if listen_ip and listen_ip not in {"0.0.0.0", "::"}:
            detection = detect_local_interface(listen_ip)
        if detection is not None:
            config["capture"]["interface"] = detection.interface
            detail = f" ({detection.local_ip})" if detection.local_ip else ""
            print(
                f"Automatically detected client-facing capture interface: "
                f"{detection.interface}{detail} [{detection.method}]"
            )
        else:
            config["capture"]["interface"] = _select_interface_fallback(
                "Automatic capture-interface detection was inconclusive; "
                "select the client-facing interface"
            )

        config["capture"]["per_client_artifacts"] = ask_yes_no(
            "Generate separate classifier-safe sequence/features per client",
            True,
        )

        config["capture"]["extract_after"] = ask_yes_no(
            "Extract packet sequence and handcrafted features on stop",
            True,
        )

        use_default_scales = ask_yes_no(
            "Use default multi-scale fingerprinting windows "
            "(0.5 s, 1 s, 2 s, 5 s)",
            True,
        )
        if use_default_scales:
            window_sizes = [0.5, 1.0, 2.0, 5.0]
            config["capture"]["allow_single_scale"] = False
        else:
            window_choice = choose(
                "Select window configuration",
                [
                    "0.5_seconds_only",
                    "1_second_only",
                    "2_seconds_only",
                    "5_seconds_only",
                    "custom",
                ],
            )
            preset = {
                "0.5_seconds_only": [0.5],
                "1_second_only": [1.0],
                "2_seconds_only": [2.0],
                "5_seconds_only": [5.0],
            }
            if window_choice == "custom":
                window_text = ask_text(
                    "Custom feature window scales in seconds, comma-separated",
                    "0.5,1,2,5",
                )
                window_sizes = sorted(
                    {
                        float(value.strip())
                        for value in window_text.split(",")
                        if value.strip()
                    }
                )
            else:
                window_sizes = preset[window_choice]
            if not window_sizes or any(value <= 0 for value in window_sizes):
                raise ValueError(
                    "At least one positive feature window scale is required"
                )
            config["capture"]["allow_single_scale"] = len(window_sizes) == 1
        config["capture"]["window_sizes_sec"] = window_sizes
        # Keep the largest scale in the legacy field for older utilities.
        config["capture"]["window_seconds"] = max(window_sizes)
        print(
            "Fingerprinting window scales: "
            + ", ".join(f"{value:g} s" for value in window_sizes)
        )

        inference_enabled = ask_yes_no(
            "Enable architecture fingerprinting "
            "(real-time + end-of-experiment)",
            True,
        )
        config["architecture_inference"][
            "enabled"
        ] = inference_enabled

        if inference_enabled:
            config["architecture_inference"][
                "realtime_enabled"
            ] = ask_yes_no(
                "Enable real-time progressive architecture inference",
                True,
            )
            config["architecture_inference"][
                "final_enabled"
            ] = ask_yes_no(
                "Enable final complete-trace architecture inference",
                True,
            )
            config["architecture_inference"][
                "realtime_required"
            ] = ask_yes_no(
                "Abort the run if the real-time tshark monitor cannot start",
                True,
            )
            config["architecture_inference"][
                "model_root"
            ] = ask_text(
                "Architecture model directory",
                "fingerprinting_models",
            )
            mode_text = ask_text(
                "Architecture feature modes "
                "(full,size_normalized)",
                "full,size_normalized",
            )
            modes = [
                value.strip()
                for value in mode_text.split(",")
                if value.strip()
            ]
            unknown = set(modes) - {
                "full",
                "size_normalized",
            }
            if unknown:
                raise ValueError(
                    f"Unknown architecture feature modes: {sorted(unknown)}"
                )
            config["architecture_inference"][
                "feature_modes"
            ] = modes
            threshold = float(
                ask_text(
                    "Real-time confidence threshold",
                    "0.90",
                )
            )
            if not 0.0 < threshold <= 1.0:
                raise ValueError(
                    "Confidence threshold must be in (0,1]"
                )
            config["architecture_inference"][
                "confidence_threshold"
            ] = threshold
            config["architecture_inference"][
                "stable_windows"
            ] = ask_int(
                "Consecutive confident windows before stable decision",
                3,
            )

    return config


def _run_proxy_with_status(config: Dict[str, Any]) -> Dict[str, Any]:
    materialize_role_metadata(config)
    write_role_status(config, "RUNNING")
    try:
        result = BlindTCPProxy(config).serve_forever()
    except KeyboardInterrupt:
        write_role_status(config, "STOPPED")
        raise
    except Exception as exc:
        write_role_status(
            config,
            "FAILED",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    else:
        write_role_status(config, "COMPLETE")
        return result


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
    _run_proxy_with_status(config)

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



def print_model_catalog(framework: str) -> None:
    rows = registry.model_catalog_rows(framework)
    header = (
        f"{'FAMILY':16} {'ARCHITECTURE':28} {'VARIANT':36} "
        f"{'EXECUTION':14} {'OPTIONAL DEPENDENCY'}"
    )
    print(header)
    print("=" * len(header))
    for row in rows:
        execution = "native" if row["native"] else "artifact-only"
        dependencies = ",".join(row["optional_dependencies"]) or "-"
        print(
            f"{str(row['family']):16} {str(row['architecture']):28} "
            f"{str(row['variant']):36} {execution:14} {dependencies}"
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

    models_parser = subparsers.add_parser(
        "models",
        help="Show the complete family/architecture/variant catalog and execution status",
    )
    models_parser.add_argument(
        "--framework",
        choices=registry.frameworks(),
        default="pytorch",
    )

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
            _run_proxy_with_status(config)
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

        if args.command == "models":
            print_model_catalog(args.framework)
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
        ExperimentLayoutError,
        FeatureExtractionError,
        ProxyError,
        KeyError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\nStopped.")
