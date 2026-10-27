from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence

KEY_CODES = {
    "ESC": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "0": 11,
    "Q": 16,
    "W": 17,
    "E": 18,
    "R": 19,
    "T": 20,
    "Y": 21,
    "U": 22,
    "I": 23,
    "O": 24,
    "P": 25,
    "A": 30,
    "S": 31,
    "D": 32,
    "F": 33,
    "G": 34,
    "H": 35,
    "J": 36,
    "K": 37,
    "L": 38,
    "Z": 44,
    "X": 45,
    "C": 46,
    "V": 47,
    "B": 48,
    "N": 49,
    "M": 50,
    "SPACE": 57,
    "ENTER": 28,
    "TAB": 15,
    "BACKSPACE": 14,
    "LEFT": 105,
    "RIGHT": 106,
    "UP": 103,
    "DOWN": 108,
    "F1": 59,
    "F2": 60,
    "F3": 61,
    "F4": 62,
    "F5": 63,
    "F6": 64,
    "F7": 65,
    "F8": 66,
    "F9": 67,
    "F10": 68,
    "F11": 87,
    "F12": 88,
    "CTRL": 29,
    "ALT": 56,
    "SHIFT": 42,
    "SUPER": 125,
}
MODIFIERS = {"CTRL", "ALT", "SHIFT", "SUPER"}
ACTION_TYPES = {
    "move_cursor",
    "click",
    "keypress",
    "minimize_window",
    "maximize_window",
    "close_window",
    "move_window",
}


def parse_key_combo(value: str) -> list[int]:
    keys = [part.strip().upper() for part in value.split("+")]
    if not keys or any(not key for key in keys):
        raise ValueError("Informe teclas separadas por +, por exemplo CTRL+ALT+T.")
    if len(set(keys)) != len(keys):
        raise ValueError("A combinação não pode repetir teclas.")
    try:
        codes = [KEY_CODES[key] for key in keys]
    except KeyError as error:
        raise ValueError(f"Tecla não suportada: {error.args[0]}") from error
    if len(keys) > 1 and not any(key in MODIFIERS for key in keys[:-1]):
        raise ValueError("Combinações devem começar com um modificador, como CTRL+C.")
    return codes


class ActionExecutor:
    def __init__(self, screen_size: tuple[int, int]) -> None:
        executable = shutil.which("ydotool")
        if executable is None:
            raise RuntimeError("ydotool não foi encontrado. Instale-o para controlar o desktop.")
        self.executable = executable
        self.screen_width, self.screen_height = screen_size
        self.active_window_move = False

    def _run(self, *arguments: str) -> None:
        try:
            result = subprocess.run(
                [self.executable, *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("ydotool não respondeu dentro do tempo limite.") from error
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"ydotool falhou: {details or 'erro sem detalhes'}")

    def _press_keys(self, key_codes: Sequence[int]) -> None:
        presses = [f"{code}:1" for code in key_codes]
        releases = [f"{code}:0" for code in reversed(key_codes)]
        self._run("key", *presses, *releases)

    def move_cursor(self, x: float, y: float) -> None:
        screen_x = round(min(max(x, 0.0), 1.0) * (self.screen_width - 1))
        screen_y = round(min(max(y, 0.0), 1.0) * (self.screen_height - 1))
        self._run(
            "mousemove",
            "--absolute",
            "-x",
            str(screen_x),
            "-y",
            str(screen_y),
        )

    def perform(self, action: str, value: str, cursor: tuple[float, float] | None) -> None:
        if action not in ACTION_TYPES:
            raise ValueError(f"Ação desconhecida: {action}")
        if action == "move_cursor":
            if cursor is None:
                return
            self.move_cursor(*cursor)
        elif action == "click":
            button = value.strip().lower()
            if button not in {"left", "right", "middle"}:
                raise ValueError("O botão do mouse deve ser left, right ou middle.")
            self._run("click", {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}[button])
        elif action == "keypress":
            self._press_keys(parse_key_combo(value))
        elif action == "minimize_window":
            self._press_keys([KEY_CODES["ALT"], KEY_CODES["F9"]])
        elif action == "maximize_window":
            self._press_keys([KEY_CODES["ALT"], KEY_CODES["F10"]])
        elif action == "close_window":
            self._press_keys([KEY_CODES["ALT"], KEY_CODES["F4"]])
        elif action == "move_window":
            self._press_keys([KEY_CODES["ALT"], KEY_CODES["F7"]])
            self.active_window_move = True

    def update_window_move(self, cursor: tuple[float, float] | None) -> None:
        if not self.active_window_move:
            return
        if cursor is not None:
            self.move_cursor(*cursor)
        else:
            self._press_keys([KEY_CODES["ENTER"]])
            self.active_window_move = False
