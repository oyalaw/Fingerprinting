from __future__ import annotations

import json
import socket
import struct
from io import BytesIO
from typing import Any, Dict, Tuple

import numpy as np


class ProtocolError(RuntimeError):
    pass


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, header: Dict[str, Any], payload: bytes = b"") -> None:
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!I", len(header_bytes)))
    sock.sendall(header_bytes)
    sock.sendall(struct.pack("!Q", len(payload)))
    if payload:
        sock.sendall(payload)


def recv_frame(sock: socket.socket) -> Tuple[Dict[str, Any], bytes]:
    header_size = struct.unpack("!I", recv_exact(sock, 4))[0]
    if header_size > 1_000_000:
        raise ProtocolError("Header is unexpectedly large")
    header = json.loads(recv_exact(sock, header_size).decode("utf-8"))
    payload_size = struct.unpack("!Q", recv_exact(sock, 8))[0]
    if payload_size > 10_000_000_000:
        raise ProtocolError("Payload is unexpectedly large")
    payload = recv_exact(sock, payload_size) if payload_size else b""
    return header, payload


def array_to_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def bytes_to_array(payload: bytes) -> np.ndarray:
    return np.load(BytesIO(payload), allow_pickle=False)


def arrays_to_bytes(arrays: list[np.ndarray]) -> bytes:
    stream = BytesIO()
    payload = {
        f"arr_{index}": np.asarray(array)
        for index, array in enumerate(arrays)
    }
    np.savez(stream, **payload)
    return stream.getvalue()


def bytes_to_arrays(payload: bytes) -> list[np.ndarray]:
    stream = BytesIO(payload)
    with np.load(stream, allow_pickle=False) as archive:
        keys = sorted(
            archive.files,
            key=lambda name: int(name.split("_")[-1]),
        )
        return [np.asarray(archive[key]) for key in keys]


def training_batch_to_bytes(
    inputs: np.ndarray,
    targets: np.ndarray,
) -> bytes:
    return arrays_to_bytes(
        [np.asarray(inputs), np.asarray(targets)]
    )


def bytes_to_training_batch(
    payload: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = bytes_to_arrays(payload)
    if len(arrays) != 2:
        raise ProtocolError(
            f"Expected input and target arrays, got {len(arrays)} arrays"
        )
    return arrays[0], arrays[1]
