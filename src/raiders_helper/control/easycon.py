from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from raiders_helper.control.backend import ControllerBackend, SerialPortInfo
from raiders_helper.control.buttons import Button, ControllerState, StickVector


class EasyConError(RuntimeError):
    pass


class EasyConBackend(ControllerBackend):
    """EasyCon ESP32-S3 115200-baud, 8N1 controller transport."""

    BAUD_RATE = 115200
    AXIS_CENTER = 128
    DPAD_CENTER = 8

    CMD_READY = 0xA5
    CMD_HELLO = 0x81
    CMD_SCRIPT_STOP = 0x84

    REPLY_ERROR = 0x00
    REPLY_ACK = 0xFF
    REPLY_BUSY = 0xFE
    REPLY_HELLO = 0x80
    REPLY_SCRIPT_ACK = 0x83

    BUTTON_BITS = {
        Button.Y: 0,
        Button.B: 1,
        Button.A: 2,
        Button.X: 3,
        Button.L: 4,
        Button.R: 5,
        Button.ZL: 6,
        Button.ZR: 7,
        Button.MINUS: 8,
        Button.PLUS: 9,
        Button.L_STICK_PRESS: 10,
        Button.R_STICK_PRESS: 11,
        Button.HOME: 12,
        Button.CAPTURE: 13,
    }

    def __init__(
        self,
        serial_factory: Callable[..., Any] | None = None,
        timeout_seconds: float = 0.5,
    ):
        self._serial_factory = serial_factory or self._default_serial_factory
        self._timeout_seconds = timeout_seconds
        self._port: Any | None = None
        self._lock = threading.RLock()
        self._state = ControllerState.neutral()
        self.last_error = ""

    @property
    def connected(self) -> bool:
        return bool(self._port is not None and self._port.is_open)

    @staticmethod
    def list_ports() -> tuple[SerialPortInfo, ...]:
        try:
            from serial.tools import list_ports
        except ImportError:
            return ()
        return tuple(
            SerialPortInfo(device=port.device, description=port.description)
            for port in list_ports.comports()
        )

    def connect(self, port: str | None = None) -> bool:
        if not port:
            self.last_error = "未指定串口"
            return False
        with self._lock:
            self.disconnect()
            try:
                self._port = self._serial_factory(
                    port=port,
                    baudrate=self.BAUD_RATE,
                    bytesize=8,
                    parity="N",
                    stopbits=1,
                    timeout=self._timeout_seconds,
                    write_timeout=self._timeout_seconds,
                )
                self._port.reset_input_buffer()
                self._port.reset_output_buffer()
                if not self._handshake():
                    self.last_error = "EasyCon 握手失败"
                    self._close_transport()
                    return False
                if not self._stop_script():
                    self.last_error = "EasyCon 脚本停止命令未确认"
                    self._close_transport()
                    return False
                self.last_error = ""
                self._state = ControllerState.neutral()
                return True
            except Exception as error:
                self.last_error = f"连接 {port} 失败: {error}"
                self._close_transport()
                return False

    def disconnect(self) -> None:
        with self._lock:
            if self.connected:
                try:
                    self._send_report(ControllerState.neutral())
                except Exception:
                    pass
            self._close_transport()
            self._state = ControllerState.neutral()

    def send_state(self, state: ControllerState) -> bool:
        with self._lock:
            if not self.connected:
                self.last_error = "控制器未连接"
                return False
            try:
                if not self._send_report(state):
                    return False
                self._state = state
                self.last_error = ""
                return True
            except Exception as error:
                self.last_error = f"发送控制器状态失败: {error}"
                return False

    def neutral(self) -> bool:
        return self.send_state(ControllerState.neutral())

    @classmethod
    def button_mask(cls, buttons: frozenset[Button]) -> int:
        mask = 0
        for button in buttons:
            bit = cls.BUTTON_BITS.get(button)
            if bit is not None:
                mask |= 1 << bit
        return mask

    @classmethod
    def dpad_hat(cls, buttons: frozenset[Button]) -> int:
        horizontal = int(Button.DPAD_RIGHT in buttons) - int(
            Button.DPAD_LEFT in buttons
        )
        vertical = int(Button.DPAD_DOWN in buttons) - int(
            Button.DPAD_UP in buttons
        )
        return {
            (0, -1): 0,
            (1, -1): 1,
            (1, 0): 2,
            (1, 1): 3,
            (0, 1): 4,
            (-1, 1): 5,
            (-1, 0): 6,
            (-1, -1): 7,
            (0, 0): 8,
        }[(horizontal, vertical)]

    @classmethod
    def to_axis(cls, value: int, invert: bool = False) -> int:
        normalized = max(-1.0, min(1.0, value / 32767.0))
        if invert:
            normalized = -normalized
        return max(0, min(255, round(cls.AXIS_CENTER + normalized * 128)))

    @classmethod
    def pack_report(cls, state: ControllerState) -> bytes:
        payload = [
            cls.button_mask(state.buttons) >> 8,
            cls.button_mask(state.buttons) & 0xFF,
            cls.dpad_hat(state.buttons),
            cls.to_axis(state.left_stick.x),
            cls.to_axis(state.left_stick.y, invert=True),
            cls.to_axis(state.right_stick.x),
            cls.to_axis(state.right_stick.y, invert=True),
        ]
        packet: list[int] = []
        accumulator = 0
        bit_count = 0
        for value in payload:
            accumulator = (accumulator << 8) | value
            bit_count += 8
            while bit_count >= 7:
                bit_count -= 7
                packet.append((accumulator >> bit_count) & 0x7F)
                accumulator &= (1 << bit_count) - 1
        if bit_count:
            packet.append((accumulator << (7 - bit_count)) & 0x7F)
        packet[-1] |= 0x80
        return bytes(packet)

    def _send_report(self, state: ControllerState) -> bool:
        packet = self.pack_report(state)
        # EasyCon controller reports are asynchronous writes. Only command
        # packets use ACK replies; reports do not have a per-report response.
        self._write(packet)
        return True

    def _handshake(self) -> bool:
        command = bytes((self.CMD_READY, self.CMD_READY, self.CMD_HELLO))
        self._write(command)
        return self._read_byte() == self.REPLY_HELLO

    def _stop_script(self) -> bool:
        self._write(bytes((self.CMD_READY, self.CMD_SCRIPT_STOP)))
        return self._read_byte() == self.REPLY_SCRIPT_ACK

    def _write(self, data: bytes) -> None:
        if not self._port or not self._port.is_open:
            raise EasyConError("串口未打开")
        self._port.write(data)
        self._port.flush()

    def _read_byte(self) -> int | None:
        if not self._port or not self._port.is_open:
            return None
        value = self._port.read(1)
        return value[0] if value else None

    def _close_transport(self) -> None:
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None

    @staticmethod
    def _default_serial_factory(**kwargs):
        try:
            import serial
        except ImportError as error:
            raise EasyConError(
                "缺少 pyserial，请安装项目 controller extra"
            ) from error
        return serial.Serial(**kwargs)

