from __future__ import annotations

import socket
from pathlib import Path


def unix_request(path: Path, request: bytes, *, timeout: float) -> bytes:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(path))
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while chunk := client.recv(64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        client.close()


def socket_ready(path: Path) -> bool:
    if not path.is_socket():
        return False
    try:
        unix_request(path, b"", timeout=0.5)
    except OSError:
        return False
    return True
