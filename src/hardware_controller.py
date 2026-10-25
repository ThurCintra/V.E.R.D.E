"""Comandos para o microcontrolador que abre a comporta da lixeira certa (Arduino/ESP32, esteira…).

Protocolo: um caractere por categoria seguido de ``\\n`` (P plástico, C papel, M metal, V vidro, O orgânico/
rejeito). O mesmo caractere vai pela serial (USB) ou como payload MQTT. O firmware de exemplo está em
``firmware/lixeira/lixeira.ino``.

O backend vem de ``HARDWARE_BACKEND``: ``sim`` (padrão, só registra), ``serial`` ou ``mqtt``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from src.recycling import CATEGORIES

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Action:
    category: str
    command: str
    bin: str  # ex.: "Plástico (lixeira vermelha)"
    label: str
    confidence: float


def choose_action(detections) -> Action | None:
    """O item que a comporta deve receber: o resíduo detectado com maior confiança."""
    waste = [d for d in detections if d.category]
    if not waste:
        return None
    best = max(waste, key=lambda d: d.confidence)
    cat = CATEGORIES[best.category]
    return Action(cat.name, cat.command, f"{cat.name} (lixeira {cat.color_name})", best.label, best.confidence)


class Backend(Protocol):
    name: str

    def send(self, command: str) -> None: ...


class SimulatedBackend:
    name = "sim"

    def __init__(self):
        self.sent: list[str] = []

    def send(self, command: str) -> None:
        self.sent.append(command)
        log.info("[simulado] comporta %s", command)


class SerialBackend:
    """USB/serial (Arduino, ESP32). ``serial_factory`` permite trocar o pyserial nos testes."""

    name = "serial"

    def __init__(
        self, port: str, baudrate: int = 115200, timeout: float = 1.0, serial_factory=None, reset_wait_s: float = 2.0
    ):
        if serial_factory is None:
            import serial

            serial_factory = serial.Serial
        self._serial = serial_factory(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(reset_wait_s)  # o Arduino reinicia ao abrir a porta e perde o que chegar antes do boot

    def send(self, command: str) -> None:
        self._serial.write(f"{command}\n".encode("ascii"))
        self._serial.flush()


class MqttBackend:
    """Publica o comando num tópico MQTT (ESP32 com Wi-Fi, Node-RED, Home Assistant…)."""

    name = "mqtt"

    def __init__(self, host: str, port: int = 1883, topic: str = "lixeira/comando", client=None):
        if client is None:
            import paho.mqtt.client as mqtt

            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="waste-detector")
            client.connect(host, port)
            client.loop_start()
        self._client = client
        self.topic = topic

    def send(self, command: str) -> None:
        self._client.publish(self.topic, command, qos=1)


class HardwareController:
    def __init__(self, backend: Backend | None = None, cooldown_s: float = 1.5, clock: Callable[[], float] = time.monotonic):
        self.backend = backend or SimulatedBackend()
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._last: tuple[str, float] | None = None

    def actuate(self, category: str | None) -> bool:
        """Abre a comporta da categoria. Não repete o mesmo comando dentro do cooldown (o mesmo item em
        frames seguidos). Devolve se o comando foi enviado."""
        if not category or category not in CATEGORIES:
            return False
        command = CATEGORIES[category].command
        now = self._clock()
        if self._last and self._last[0] == command and now - self._last[1] < self.cooldown_s:
            return False
        try:
            self.backend.send(command)
        except Exception:  # um cabo solto não pode derrubar a API
            log.exception("Falha ao enviar o comando %s ao hardware", command)
            return False
        self._last = (command, now)
        return True


def controller_from_env() -> HardwareController:
    kind = os.environ.get("HARDWARE_BACKEND", "sim").lower()
    cooldown = float(os.environ.get("HARDWARE_COOLDOWN_S", "1.5"))
    if kind == "serial":
        backend = SerialBackend(
            os.environ.get("SERIAL_PORT", "/dev/ttyUSB0"), int(os.environ.get("SERIAL_BAUD", "115200"))
        )
    elif kind == "mqtt":
        backend = MqttBackend(
            os.environ.get("MQTT_HOST", "localhost"),
            int(os.environ.get("MQTT_PORT", "1883")),
            os.environ.get("MQTT_TOPIC", "lixeira/comando"),
        )
    else:
        backend = SimulatedBackend()
    return HardwareController(backend, cooldown)
