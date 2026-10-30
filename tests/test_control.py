from raiders_helper.control.buttons import Button, ControllerState, StickVector
from raiders_helper.control.dummy import DummyBackend
from raiders_helper.control.easycon import EasyConBackend


class FakeSerial:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.writes = []
        self.is_open = True

    def reset_input_buffer(self):
        return None

    def reset_output_buffer(self):
        return None

    def write(self, data):
        self.writes.append(bytes(data))
        return len(data)

    def flush(self):
        return None

    def read(self, size):
        del size
        if not self.responses:
            return b""
        return bytes((self.responses.pop(0),))

    def close(self):
        self.is_open = False


def test_neutral_packet_encoding() -> None:
    packet = EasyConBackend.pack_report(ControllerState.neutral())

    assert packet == bytes.fromhex("0000010804020180")
    assert len(packet) == 8
    assert packet[-1] & 0x80


def test_button_mask_and_dpad() -> None:
    buttons = frozenset(
        {
            Button.A,
            Button.Y,
            Button.L,
            Button.ZR,
            Button.DPAD_UP,
            Button.DPAD_RIGHT,
        }
    )

    assert EasyConBackend.button_mask(buttons) == (1 << 2 | 1 << 0 | 1 << 4 | 1 << 7)
    assert EasyConBackend.dpad_hat(buttons) == 1
    assert EasyConBackend.dpad_hat(frozenset({Button.DPAD_DOWN})) == 4
    assert EasyConBackend.dpad_hat(frozenset()) == 8


def test_axis_direction_and_center() -> None:
    assert EasyConBackend.to_axis(-32767) == 0
    assert EasyConBackend.to_axis(0) == 128
    assert EasyConBackend.to_axis(32767) == 255
    assert EasyConBackend.to_axis(32767, invert=True) == 0


def test_easycon_connect_send_and_neutral() -> None:
    fake = FakeSerial(
        [
            EasyConBackend.REPLY_HELLO,
            EasyConBackend.REPLY_SCRIPT_ACK,
            EasyConBackend.REPLY_ERROR,
        ]
    )
    backend = EasyConBackend(serial_factory=lambda **kwargs: fake)

    assert backend.connect("COM_TEST")
    state = ControllerState(
        buttons=frozenset({Button.A}),
        left_stick=StickVector(32767, 0),
        right_stick=StickVector(0, -32767),
    )
    assert backend.send_state(state)
    backend.disconnect()

    assert fake.writes[0] == bytes(
        (
            EasyConBackend.CMD_READY,
            EasyConBackend.CMD_READY,
            EasyConBackend.CMD_HELLO,
        )
    )
    assert fake.writes[1] == bytes(
        (EasyConBackend.CMD_READY, EasyConBackend.CMD_SCRIPT_STOP)
    )
    assert len(fake.writes[2]) == 8
    assert fake.writes[3] == bytes.fromhex("0000010804020180")
    assert fake.responses == [EasyConBackend.REPLY_ERROR]
    assert not backend.connected


def test_dummy_backend_records_state_history() -> None:
    backend = DummyBackend()
    assert backend.connect()
    state = ControllerState(buttons=frozenset({Button.B}))

    assert backend.send_state(state)
    assert backend.state == state
    assert backend.history[-1] == state
    backend.disconnect()
    assert not backend.connected
    assert backend.state == ControllerState.neutral()

