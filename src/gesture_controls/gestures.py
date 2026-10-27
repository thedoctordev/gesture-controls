from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

SEQUENCE_LENGTH = 24
MIN_PATH_LENGTH = 0.08
MIN_EXCURSION = 0.035
MATCH_THRESHOLD = 0.13
JITTER_THRESHOLD = 0.0035


def normalize_trajectory(
    points: Sequence[Sequence[float]],
) -> list[list[float]] | None:
    """Resample a hand path by distance and normalize its start and scale."""
    if len(points) < 2:
        return None
    path: list[tuple[float, float]] = []
    for point in points:
        if len(point) < 2 or not all(math.isfinite(value) for value in point[:2]):
            return None
        current = (float(point[0]), float(point[1]))
        if not path or math.dist(path[-1], current) >= JITTER_THRESHOLD:
            path.append(current)
    if len(path) < 2:
        return None

    distances = [0.0]
    for previous, current in zip(path, path[1:]):
        distances.append(distances[-1] + math.dist(previous, current))
    path_length = distances[-1]
    origin = path[0]
    scale = max(math.dist(origin, point) for point in path)
    if path_length < MIN_PATH_LENGTH or scale < MIN_EXCURSION:
        return None

    normalized: list[list[float]] = []
    segment = 0
    for index in range(SEQUENCE_LENGTH):
        target = path_length * index / (SEQUENCE_LENGTH - 1)
        while segment < len(distances) - 2 and distances[segment + 1] < target:
            segment += 1
        start_distance = distances[segment]
        end_distance = distances[segment + 1]
        portion = (target - start_distance) / (end_distance - start_distance)
        start = path[segment]
        end = path[segment + 1]
        x = start[0] + (end[0] - start[0]) * portion
        y = start[1] + (end[1] - start[1]) * portion
        normalized.append([(x - origin[0]) / scale, (y - origin[1]) / scale])
    return normalized


@dataclass(frozen=True)
class Gesture:
    name: str
    action: str
    value: str
    trajectory: list[list[float]]

    @property
    def needs_recapture(self) -> bool:
        return len(self.trajectory) != SEQUENCE_LENGTH

    @classmethod
    def from_dict(cls, data: object) -> Gesture:
        if not isinstance(data, dict):
            raise ValueError("Definição de gesto inválida.")
        name = data.get("name")
        action = data.get("action")
        value = data.get("value", "")
        trajectory = data.get("trajectory")
        if trajectory is None and isinstance(data.get("pose"), list):
            pose = data["pose"]
            if len(pose) != 63 or any(
                not isinstance(item, (int, float)) or not math.isfinite(item)
                for item in pose
            ):
                raise ValueError("Definição antiga de gesto inválida.")
            trajectory = []
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(action, str)
            or not isinstance(value, str)
            or not isinstance(trajectory, list)
        ):
            raise ValueError("Definição de gesto inválida.")
        if trajectory and (
            len(trajectory) != SEQUENCE_LENGTH
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    not isinstance(item, (int, float)) or not math.isfinite(item)
                    for item in point
                )
                for point in trajectory
            )
        ):
            raise ValueError("Trajetória de gesto inválida.")
        return cls(
            name.strip(),
            action,
            value,
            [[float(item) for item in point] for point in trajectory],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "action": self.action,
            "value": self.value,
            "trajectory": self.trajectory,
        }


def _trajectory_distance(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> float:
    rows = len(first)
    columns = len(second)
    costs = [[float("inf")] * (columns + 1) for _ in range(rows + 1)]
    lengths = [[0] * (columns + 1) for _ in range(rows + 1)]
    costs[0][0] = 0.0
    band = max(4, abs(rows - columns))

    for row in range(1, rows + 1):
        for column in range(max(1, row - band), min(columns, row + band) + 1):
            candidates = (
                (costs[row - 1][column], lengths[row - 1][column]),
                (costs[row][column - 1], lengths[row][column - 1]),
                (costs[row - 1][column - 1], lengths[row - 1][column - 1]),
            )
            previous_cost, previous_length = min(candidates, key=lambda item: item[0])
            point_distance = math.dist(first[row - 1], second[column - 1])
            costs[row][column] = previous_cost + point_distance
            lengths[row][column] = previous_length + 1
    return costs[rows][columns] / max(lengths[rows][columns], 1)


def classify_trajectory(
    points: Sequence[Sequence[float]],
    gestures: Sequence[Gesture],
    threshold: float = MATCH_THRESHOLD,
) -> Gesture | None:
    trajectory = normalize_trajectory(points)
    if trajectory is None:
        return None
    closest: Gesture | None = None
    closest_distance = float("inf")
    for gesture in gestures:
        if gesture.needs_recapture:
            continue
        distance = _trajectory_distance(trajectory, gesture.trajectory)
        if distance < closest_distance:
            closest = gesture
            closest_distance = distance
    return closest if closest_distance <= threshold else None
