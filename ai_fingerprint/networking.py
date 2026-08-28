from __future__ import annotations

import ipaddress
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import Iterable

import psutil


@dataclass(frozen=True)
class InterfaceDetection:
    interface: str
    local_ip: str | None
    method: str
    peer: str | None = None


def _resolve_ipv4(host: str) -> str:
    text = str(host or "").strip()
    if not text:
        raise ValueError("host is empty")
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return socket.gethostbyname(text)


def interface_for_local_ip(local_ip: str) -> str | None:
    target = str(local_ip or "").strip()
    if not target:
        return None
    for name, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address == target:
                return name
    return None


def detect_route_interface(peer: str) -> InterfaceDetection | None:
    try:
        resolved = _resolve_ipv4(peer)
    except Exception:
        return None

    ip_cmd = shutil.which("ip")
    if ip_cmd:
        try:
            result = subprocess.run(
                [ip_cmd, "route", "get", resolved],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                line = (result.stdout or "").strip().splitlines()[0]
                dev_match = re.search(r"(?:^|\s)dev\s+(\S+)", line)
                src_match = re.search(r"(?:^|\s)src\s+(\S+)", line)
                if dev_match:
                    return InterfaceDetection(
                        interface=dev_match.group(1),
                        local_ip=src_match.group(1) if src_match else None,
                        method="ip_route_get",
                        peer=resolved,
                    )
        except Exception:
            pass

    # Cross-platform fallback: UDP connect selects the local source address
    # without sending application data. Map that address to a psutil interface.
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((resolved, 9))
            local_ip = sock.getsockname()[0]
        finally:
            sock.close()
        interface = interface_for_local_ip(local_ip)
        if interface:
            return InterfaceDetection(
                interface=interface,
                local_ip=local_ip,
                method="udp_source_address",
                peer=resolved,
            )
    except Exception:
        pass
    return None


def detect_local_interface(local_ip: str) -> InterfaceDetection | None:
    interface = interface_for_local_ip(local_ip)
    if not interface:
        return None
    return InterfaceDetection(
        interface=interface,
        local_ip=local_ip,
        method="local_ip_mapping",
        peer=None,
    )


def detect_consensus_interface(peers: Iterable[str]) -> InterfaceDetection | None:
    results = [detect_route_interface(peer) for peer in peers]
    results = [item for item in results if item is not None]
    if not results:
        return None
    names = {item.interface for item in results}
    if len(names) != 1:
        return None
    first = results[0]
    return InterfaceDetection(
        interface=first.interface,
        local_ip=first.local_ip,
        method="peer_route_consensus",
        peer=",".join(str(item.peer) for item in results if item.peer),
    )


def candidate_interfaces() -> list[str]:
    stats = psutil.net_if_stats()
    excluded_prefixes = ("lo", "docker", "veth", "virbr", "br-")
    names: list[str] = []
    for name in sorted(psutil.net_if_addrs()):
        if name.startswith(excluded_prefixes):
            continue
        stat = stats.get(name)
        if stat is not None and not stat.isup:
            continue
        has_ipv4 = any(
            addr.family == socket.AF_INET
            for addr in psutil.net_if_addrs().get(name, [])
        )
        if has_ipv4:
            names.append(name)
    return names
