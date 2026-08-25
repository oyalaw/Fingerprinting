from __future__ import annotations

import ipaddress
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict


class TLSConfigurationError(RuntimeError):
    pass


def _subject_alt_name(common_name: str) -> str:
    try:
        ipaddress.ip_address(common_name)
    except ValueError:
        return f"DNS:{common_name}"
    return f"IP:{common_name}"


def ensure_server_tls_material(
    config: Dict[str, Any],
) -> Dict[str, str]:
    transport = config.get("transport", {})
    if str(transport.get("kind", "tcp")).lower() != "tls":
        return {}

    certfile = str(transport.get("certfile") or "").strip()
    keyfile = str(transport.get("keyfile") or "").strip()
    if certfile and keyfile:
        if not Path(certfile).exists():
            raise TLSConfigurationError(
                f"TLS certificate does not exist: {certfile}"
            )
        if not Path(keyfile).exists():
            raise TLSConfigurationError(
                f"TLS private key does not exist: {keyfile}"
            )
        return {
            "certfile": certfile,
            "keyfile": keyfile,
            "generated": "false",
        }

    if not bool(
        transport.get("auto_generate_self_signed", False)
    ):
        raise TLSConfigurationError(
            "TLS server requires transport.certfile and "
            "transport.keyfile, or "
            "transport.auto_generate_self_signed=true"
        )

    openssl = shutil.which("openssl")
    if not openssl:
        raise TLSConfigurationError(
            "openssl is required to auto-generate the research TLS "
            "certificate. Install openssl or supply certfile/keyfile."
        )

    output_dir = Path(
        config["experiment"]["output_dir"]
    )
    tls_dir = output_dir / "_tls"
    tls_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = str(
        config["experiment"]["experiment_id"]
    )
    cert_path = tls_dir / f"{experiment_id}_server_cert.pem"
    key_path = tls_dir / f"{experiment_id}_server_key.pem"

    if cert_path.exists() and key_path.exists():
        transport["certfile"] = str(cert_path)
        transport["keyfile"] = str(key_path)
        return {
            "certfile": str(cert_path),
            "keyfile": str(key_path),
            "generated": "existing",
        }

    common_name = str(
        transport.get("self_signed_common_name")
        or "ai-fingerprint-server"
    ).strip()
    days = int(
        transport.get("self_signed_valid_days", 30)
    )
    if days <= 0:
        raise TLSConfigurationError(
            "transport.self_signed_valid_days must be positive"
        )

    command = [
        openssl,
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-sha256",
        "-nodes",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        str(days),
        "-subj",
        f"/CN={common_name}",
        "-addext",
        f"subjectAltName={_subject_alt_name(common_name)}",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        # Some older OpenSSL builds do not support -addext. Retry without it.
        fallback = command[:-2]
        completed = subprocess.run(
            fallback,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    if completed.returncode != 0:
        raise TLSConfigurationError(
            "Unable to generate self-signed TLS material: "
            + (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "openssl failed"
            )
        )

    try:
        key_path.chmod(0o600)
        cert_path.chmod(0o644)
    except OSError:
        pass

    transport["certfile"] = str(cert_path)
    transport["keyfile"] = str(key_path)
    return {
        "certfile": str(cert_path),
        "keyfile": str(key_path),
        "generated": "true",
    }
