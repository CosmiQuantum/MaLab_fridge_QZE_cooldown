"""Standalone Yokogawa GS200 driver copied from MaLab's ``voltsource.py``.

Upstream source:
https://github.com/ma-quantumlab/malab/blob/main/instruments/voltsource.py

The upstream ``YokogawaGS200`` inherits MaLab's ``SocketInstrument``. This copy
keeps its public SCPI interface but uses the Python standard library so this
repository does not need the complete ``malab`` package. Current ramps enforce
an absolute safety limit and always include the endpoint.
"""

import socket
import time
from typing import Optional


class YokogawaGS200:
    default_port = 7655

    def __init__(self, address: str, port: int = default_port,
                 timeout: float = 10.0, max_current: float = 0.015):
        self.address = address
        self.port = port
        self.max_current = max_current
        self.query_sleep = 0.01
        self._socket = socket.create_connection((address, port), timeout=timeout)
        self._socket.settimeout(timeout)

    def close(self) -> None:
        self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, command: str) -> None:
        self._socket.sendall((command.rstrip() + "\n").encode("ascii"))

    def query(self, command: str) -> str:
        self.write(command)
        time.sleep(self.query_sleep)
        chunks = []
        while True:
            chunk = self._socket.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        return b"".join(chunks).decode("ascii").strip()

    def get_id(self) -> str:
        return self.query("*IDN?")

    def set_output(self, state: bool = True) -> None:
        self.write(":OUTPUT:STATE %d" % int(state))

    def get_output(self) -> bool:
        return bool(int(self.query(":OUTPUT:STATE?")))

    def set_mode(self, mode: str) -> None:
        normalized = mode.upper()
        if normalized not in ("CURR", "CURRENT"):
            raise ValueError("Flux-bias driver only permits current mode")
        self.write(":SOURCE:FUNCTION CURRENT")

    def get_mode(self) -> str:
        return self.query(":SOURCE:FUNCTION?").upper()

    def set_level(self, level) -> None:
        self.write(":SOURCE:LEVEL %s" % level)

    def get_level(self) -> float:
        return float(self.query(":SOURCE:LEVEL?"))

    def set_range(self, value: float) -> None:
        self.write(":SOURCE:RANGE %f" % value)

    def get_range(self) -> float:
        return float(self.query(":SOURCE:RANGE?"))

    def set_voltage_limit(self, limit: float) -> None:
        self.write(":SOURCE:PROTECTION:VOLT %f" % limit)

    def get_voltage_limit(self) -> float:
        return float(self.query(":SOURCE:PROTECTION:VOLT?"))

    def get_current(self) -> float:
        if not self.get_mode().startswith("CURR"):
            raise RuntimeError("Yokogawa is not in current-source mode")
        return self.get_level()

    def set_current(self, current: float, safety_level: Optional[float] = None) -> None:
        limit = self.max_current if safety_level is None else min(
            abs(safety_level), self.max_current
        )
        if abs(current) > limit:
            raise ValueError(
                "Requested current %.6f A exceeds absolute limit %.6f A"
                % (current, limit)
            )
        if not self.get_mode().startswith("CURR"):
            raise RuntimeError("Yokogawa is not in current-source mode")
        self.set_level("%.9fA" % current)

    def ramp_current(self, current: float, sweeprate: float = 0.0005,
                     interval: float = 0.1) -> None:
        """Ramp in amperes at ``sweeprate`` A/s, matching MaLab's API."""
        if sweeprate <= 0 or interval < 0.01:
            raise ValueError("sweeprate must be positive and interval at least 10 ms")
        start = self.get_current()
        distance = current - start
        step = sweeprate * interval
        count = max(1, int(abs(distance) / step))
        for index in range(1, count + 1):
            self.set_current(start + distance * index / count)
            time.sleep(interval)
        self.set_current(current)
