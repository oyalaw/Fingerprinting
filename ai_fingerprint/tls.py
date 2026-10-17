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
    auto = bool(transport.get("auto_generate_self_signed", False))
    if certfile and keyfile and not (Path(certfile).exists() and Path(keyfile).exists()) and auto:
        # Configured testbed certificate not provisioned on this machine: fall back to a
        # per-run self-signed certificate, so the connection is still encrypted.
        print(f"[tls] {certfile} not found; generating a self-signed certificate "
              "(run `main.py tls-provision` for verified TLS)")
        certfile = keyfile = ""
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


TESTBED_COMMON_NAME = "ai-fingerprint-server"


def provision_testbed_tls(
    out_dir: str | Path,
    hosts: list[str],
    common_name: str = TESTBED_COMMON_NAME,
    days: int = 825,
) -> Dict[str, str]:
    """Create a testbed CA and a server certificate signed by it.

    The server certificate names `common_name` and every address in `hosts` (server and proxy
    IPs, hostnames), so a client that verifies against ca.pem accepts it whether it connects
    to the server directly or through the proxy. Copy ca.pem, server.pem and server.key to the
    same relative path on every machine; ca.key stays on the provisioning machine.
    """
    openssl = shutil.which("openssl")
    if not openssl:
        raise TLSConfigurationError("openssl is required for tls-provision")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / v for k, v in (("ca_key", "ca.key"), ("ca", "ca.pem"), ("key", "server.key"),
                                      ("csr", "server.csr"), ("cert", "server.pem"), ("ext", "server_ext.cnf"))}
    sans = [f"DNS:{common_name}"] + [_subject_alt_name(h) for h in hosts if h and h != common_name]
    paths["ext"].write_text(
        "basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\nsubjectAltName=" + ",".join(dict.fromkeys(sans)) + "\n")

    def run(cmd):
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise TLSConfigurationError((done.stderr or done.stdout or "openssl failed").strip())

    run([openssl, "req", "-x509", "-newkey", "rsa:3072", "-sha256", "-nodes", "-keyout", str(paths["ca_key"]),
         "-out", str(paths["ca"]), "-days", str(days), "-subj", "/CN=AI Fingerprinting Testbed CA"])
    run([openssl, "req", "-newkey", "rsa:2048", "-sha256", "-nodes", "-keyout", str(paths["key"]),
         "-out", str(paths["csr"]), "-subj", f"/CN={common_name}"])
    run([openssl, "x509", "-req", "-in", str(paths["csr"]), "-CA", str(paths["ca"]), "-CAkey", str(paths["ca_key"]),
         "-CAcreateserial", "-out", str(paths["cert"]), "-days", str(days), "-sha256",
         "-extfile", str(paths["ext"])])
    for key in ("ca_key", "key"):
        try:
            paths[key].chmod(0o600)
        except OSError:
            pass
    return {k: str(v) for k, v in paths.items()}
