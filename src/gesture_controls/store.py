from __future__ import annotations

import json
from pathlib import Path

from gesture_controls.gestures import Gesture

GESTURES_FILE = Path.home() / ".config" / "gesture-controls" / "gestures.json"


class GestureStore:
    def __init__(self, path: Path = GESTURES_FILE) -> None:
        self.path = path

    def load(self) -> list[Gesture]:
        if not self.path.exists():
            return []
        try:
            contents = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(contents, list):
                raise ValueError("O arquivo de gestos precisa conter uma lista.")
            return [Gesture.from_dict(item) for item in contents]
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"Não foi possível carregar {self.path}: {error}") from error

    def save(self, gestures: list[Gesture]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps([gesture.to_dict() for gesture in gestures], indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
