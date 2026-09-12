from __future__ import annotations

import argparse
import copy
import threading
import time
from pathlib import Path
from typing import Any, Dict

from ai_fingerprint.cli import (
    interactive_proxy_configure,
    resolve_existing_outputs_interactive,
)
from ai_fingerprint.experiment_coordination import (
    ExperimentCoordinationError,
    query_active_run,
    wait_for_active_run,
)
from ai_fingerprint.experiment_layout import (
    apply_proxy_staging_layout,
    materialize_role_metadata,
    write_role_status,
)
from ai_fingerprint.proxy import (
    BlindTCPProxy,
    load_proxy_config,
    save_proxy_config,
)
from ai_fingerprint.result_collection import auto_upload_result_copy


DEFAULT_CONFIG_PATH = "proxy_config.yaml"


class ProxySessionError(RuntimeError):
    pass


def _run_one_session(
    config: Dict[str, Any],
    *,
    poll_interval_sec: float = 0.5,
    query_timeout_sec: float = 1.5,
) -> tuple[Dict[str, Any], str | None]:
    """
    Run one proxy session.

    Returns:
        (proxy_result, next_run_id)

    next_run_id is None when the proxy stopped normally.
    When the server coordinator exposes a different neutral run ID, the
    current proxy is stopped cleanly and that new ID is returned.
    """
    materialize_role_metadata(config)
    write_role_status(config, "RUNNING")

    current_run_id = str(
        config.get("experiment", {}).get("run_id") or ""
    ).strip()
    if not current_run_id:
        raise ProxySessionError("Proxy configuration has no neutral run_id")

    coordination = config.get("coordination", {}) or {}
    host = str(
        coordination.get("server_host")
        or config.get("proxy", {}).get("upstream_host")
        or ""
    ).strip()
    port = int(coordination.get("server_port", 8081))

    if not host:
        raise ProxySessionError("No server coordinator host is configured")

    proxy = BlindTCPProxy(config)
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def target() -> None:
        try:
            result_box["result"] = proxy.serve_forever()
        except BaseException as exc:
            error_box["error"] = exc

    thread = threading.Thread(
        target=target,
        name=f"aifp-proxy-{current_run_id}",
        daemon=True,
    )
    thread.start()

    next_run_id: str | None = None
    unreachable_since: float | None = None
    last_unreachable_notice = 0.0

    try:
        while thread.is_alive():
            time.sleep(max(0.2, poll_interval_sec))

            try:
                active = query_active_run(
                    host,
                    port,
                    timeout_sec=max(0.2, query_timeout_sec),
                )
            except ExperimentCoordinationError as exc:
                # A server restart creates a temporary coordinator outage.
                # Do NOT close the proxy merely because the server is down.
                now = time.monotonic()
                if unreachable_since is None:
                    unreachable_since = now
                if (
                    now - unreachable_since >= 15.0
                    and now - last_unreachable_notice >= 15.0
                ):
                    print(
                        "[proxy-supervisor] server coordinator is temporarily "
                        "unavailable; keeping the current proxy capture open: "
                        f"{exc}"
                    )
                    last_unreachable_notice = now
                continue

            unreachable_since = None

            if active.run_id == current_run_id:
                continue

            next_run_id = active.run_id
            print()
            print(
                "[proxy-supervisor] SERVER RUN CHANGED:"
                f" {current_run_id} -> {next_run_id}"
            )
            print(
                "[proxy-supervisor] Finalizing the old proxy capture before "
                "starting the new run."
            )

            # BlindTCPProxy.stop() is graceful: the listener exits, forwarding
            # loops observe stop_event, capture stops, chunks are extracted,
            # and experiment_status.json is written.
            proxy.stop()
            break

    except KeyboardInterrupt:
        print()
        print("[proxy-supervisor] Ctrl+C received; stopping proxy cleanly.")
        proxy.stop()
        next_run_id = None

    thread.join(timeout=30.0)
    if thread.is_alive():
        raise ProxySessionError(
            f"Proxy session {current_run_id} did not stop within 30 seconds"
        )

    if "error" in error_box:
        write_role_status(
            config,
            "FAILED",
            error=(
                f"{type(error_box['error']).__name__}: "
                f"{error_box['error']}"
            ),
        )
        raise error_box["error"]

    result = dict(result_box.get("result") or {})

    # Preserve a specific CAPTURE_INCOMPLETE status written by BlindTCPProxy.
    write_role_status(config, "COMPLETE")
    auto_upload_result_copy(config)

    result["supervisor_run_id"] = current_run_id
    result["supervisor_rollover"] = bool(next_run_id)
    result["supervisor_next_run_id"] = next_run_id

    return result, next_run_id


def _prepare_interactive_config(config_path: Path) -> Dict[str, Any]:
    config = interactive_proxy_configure()
    save_proxy_config(config, config_path)

    config = load_proxy_config(config_path)
    config = resolve_existing_outputs_interactive(
        config,
        role="proxy",
    )
    save_proxy_config(config, config_path)
    return config


def _rebind_to_run(
    config: Dict[str, Any],
    *,
    new_run_id: str,
    config_path: Path,
) -> Dict[str, Any]:
    updated = copy.deepcopy(config)

    staging_root = str(
        updated.get("experiment", {}).get(
            "results_root",
            "experiments/staging",
        )
    )

    apply_proxy_staging_layout(
        updated,
        staging_root=staging_root,
        run_id=new_run_id,
    )

    # A neutral run ID is unique, so the next session should never silently
    # overwrite an existing proxy output directory.
    updated["experiment"]["existing_output_policy"] = "error"

    save_proxy_config(updated, config_path)
    return load_proxy_config(config_path)


def run_resilient_proxy(
    config: Dict[str, Any],
    *,
    config_path: Path,
    poll_interval_sec: float = 0.5,
) -> None:
    current = copy.deepcopy(config)

    while True:
        current_run_id = str(
            current.get("experiment", {}).get("run_id") or ""
        ).strip()

        print()
        print("=" * 78)
        print("RESILIENT PROXY SESSION")
        print("=" * 78)
        print(f"Neutral run ID: {current_run_id}")
        print(
            "Output: "
            f"{current.get('experiment', {}).get('output_dir')}"
        )
        print(
            "The proxy will remain active across server restarts and "
            "automatically roll over when a new run ID appears."
        )
        print("=" * 78)

        _result, next_run_id = _run_one_session(
            current,
            poll_interval_sec=poll_interval_sec,
        )

        if not next_run_id:
            print("[proxy-supervisor] Proxy stopped normally.")
            return

        # The detected ID is already positive evidence from the new server.
        # Re-query only if a race left us with the same ID.
        if next_run_id == current_run_id:
            coordination = current.get("coordination", {}) or {}
            host = str(
                coordination.get("server_host")
                or current.get("proxy", {}).get("upstream_host")
                or ""
            ).strip()
            port = int(coordination.get("server_port", 8081))

            while next_run_id == current_run_id:
                active = wait_for_active_run(host, port)
                next_run_id = active.run_id

        current = _rebind_to_run(
            current,
            new_run_id=next_run_id,
            config_path=config_path,
        )

        print(
            "[proxy-supervisor] New proxy session prepared:"
            f" {current_run_id} -> {next_run_id}"
        )
        print(
            "[proxy-supervisor] New staging directory: "
            f"{current['experiment']['output_dir']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the label-blind proxy persistently across FL server/client "
            "restarts while preserving one proxy capture per neutral run ID."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Proxy YAML configuration. If it does not exist, the normal "
            "interactive proxy configuration wizard is started."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.5,
        help="How often to check the server's neutral active run ID.",
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Run the interactive proxy configuration wizard even if config exists.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)

    if args.reconfigure or not config_path.exists():
        config = _prepare_interactive_config(config_path)
    else:
        config = load_proxy_config(config_path)

        # If an old proxy_config.yaml points to a run that is no longer the
        # active server run, correct it before opening a new capture.
        coordination = config.get("coordination", {}) or {}
        host = str(
            coordination.get("server_host")
            or config.get("proxy", {}).get("upstream_host")
            or ""
        ).strip()
        port = int(coordination.get("server_port", 8081))

        active = wait_for_active_run(host, port)
        configured_run = str(
            config.get("experiment", {}).get("run_id") or ""
        ).strip()

        if configured_run != active.run_id:
            print(
                "[proxy-supervisor] Existing proxy config belongs to "
                f"{configured_run or '<none>'}; active server run is "
                f"{active.run_id}. Rebinding before capture starts."
            )
            config = _rebind_to_run(
                config,
                new_run_id=active.run_id,
                config_path=config_path,
            )

    run_resilient_proxy(
        config,
        config_path=config_path,
        poll_interval_sec=max(0.2, float(args.poll_seconds)),
    )


if __name__ == "__main__":
    main()
