from src.detector import Detection
from src.hardware_controller import (
    HardwareController,
    MqttBackend,
    SerialBackend,
    SimulatedBackend,
    choose_action,
    controller_from_env,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_escolhe_o_residuo_mais_confiante():
    action = choose_action(
        [
            Detection("person", 0.99, (0, 0, 1, 1)),
            Detection("bottle", 0.6, (0, 0, 1, 1), "Plástico", 30),
            Detection("scissors", 0.8, (0, 0, 1, 1), "Metal", 50),
        ]
    )
    assert action.category == "Metal" and action.command == "M"
    assert action.bin == "Metal (lixeira amarela)"
    assert choose_action([Detection("person", 0.9, (0, 0, 1, 1))]) is None


def test_simulado_envia_e_respeita_o_cooldown():
    clock = FakeClock()
    controller = HardwareController(SimulatedBackend(), cooldown_s=2, clock=clock)
    assert controller.actuate("Plástico")
    assert not controller.actuate("Plástico")  # mesmo item no frame seguinte
    assert controller.actuate("Metal")  # outro material passa direto
    clock.t = 5
    assert controller.actuate("Metal")
    assert controller.backend.sent == ["P", "M", "M"]
    assert not controller.actuate(None)
    assert not controller.actuate("Desconhecida")


def test_serial_escreve_o_caractere_com_quebra_de_linha():
    class FakeSerial:
        def __init__(self, **kwargs):
            self.kwargs, self.written = kwargs, b""

        def write(self, data):
            self.written += data

        def flush(self):
            pass

    backend = SerialBackend("/dev/ttyUSB0", 9600, serial_factory=FakeSerial, reset_wait_s=0)
    HardwareController(backend).actuate("Vidro")
    assert backend._serial.written == b"V\n"
    assert backend._serial.kwargs["port"] == "/dev/ttyUSB0" and backend._serial.kwargs["baudrate"] == 9600


def test_mqtt_publica_no_topico():
    class FakeClient:
        def __init__(self):
            self.published = []

        def publish(self, topic, payload, qos):
            self.published.append((topic, payload, qos))

    client = FakeClient()
    HardwareController(MqttBackend("broker", topic="fabrica/esteira1", client=client)).actuate("Papel")
    assert client.published == [("fabrica/esteira1", "C", 1)]


def test_falha_do_hardware_nao_derruba():
    class Broken:
        name = "serial"

        def send(self, command):
            raise OSError("cabo solto")

    assert HardwareController(Broken()).actuate("Plástico") is False


def test_backend_padrao_e_simulado(monkeypatch):
    monkeypatch.delenv("HARDWARE_BACKEND", raising=False)
    assert controller_from_env().backend.name == "sim"
