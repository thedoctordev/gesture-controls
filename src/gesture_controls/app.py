from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import mediapipe as mp
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gesture_controls.actions import ActionExecutor, parse_key_combo
from gesture_controls.gestures import Gesture, classify_trajectory, normalize_trajectory
from gesture_controls.store import GestureStore

ACTION_LABELS = {
    "Mover cursor": "move_cursor",
    "Clique do mouse": "click",
    "Pressionar tecla(s)": "keypress",
    "Minimizar janela": "minimize_window",
    "Maximizar janela": "maximize_window",
    "Fechar janela": "close_window",
    "Mover janela": "move_window",
}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path.home() / ".cache" / "gesture-controls" / "hand_landmarker.task"
MAX_MODEL_SIZE = 20 * 1024 * 1024
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def ensure_hand_model() -> Path:
    if MODEL_PATH.is_file() and 0 < MODEL_PATH.stat().st_size <= MAX_MODEL_SIZE:
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".task.tmp")
    try:
        request = Request(MODEL_URL, headers={"User-Agent": "gesture-controls/0.1"})
        size = 0
        with urlopen(request, timeout=45) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MODEL_SIZE:
                    raise RuntimeError("O modelo baixado excede o limite de tamanho.")
                output.write(chunk)
        if size == 0:
            raise RuntimeError("O servidor retornou um modelo vazio.")
        temporary.replace(MODEL_PATH)
    except (OSError, URLError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Não foi possível baixar o modelo MediaPipe: {error}") from error
    except RuntimeError:
        temporary.unlink(missing_ok=True)
        raise
    return MODEL_PATH


def draw_hand(frame: Any, landmarks: list[Any]) -> None:
    height, width = frame.shape[:2]
    points = [
        (round(point.x * width), round(point.y * height))
        for point in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (80, 220, 100), 2)
    for point in points:
        cv2.circle(frame, point, 4, (30, 120, 255), -1)


class CameraWorker(QObject):
    frame_ready = Signal(QImage)
    status_changed = Signal(str)
    capture_finished = Signal(object)
    capture_failed = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, executor: ActionExecutor, gestures: list[Gesture]) -> None:
        super().__init__()
        self.executor = executor
        self.gestures = gestures
        self.camera: cv2.VideoCapture | None = None
        self.landmarker: Any = None
        self.timer: Any = None
        self._capture_request: tuple[str, str, str] | None = None
        self._capture_samples: list[tuple[float, float]] = []
        self._capture_started = 0.0
        self._last_capture_progress = 0.0
        self._motion_history: deque[tuple[float, tuple[float, float]]] = deque(maxlen=120)
        self._last_action = ""
        self._last_recognized_at = 0.0
        self._move_gesture_active = False
        self._last_cursor_update = 0.0
        self._last_status = ""
        self._awaiting_hand_release = False
        self._hand_missing_since: float | None = None
        self._finished = False

    @Slot(str, str, str)
    def request_capture(self, name: str, action: str, value: str) -> None:
        self._capture_request = (name, action, value)
        self._capture_samples = []
        self._capture_started = time.monotonic()
        self._last_capture_progress = 0.0
        self.status_changed.emit("Execute o movimento do gesto continuamente por 2 segundos.")

    @Slot()
    def stop(self) -> None:
        if self.timer is not None:
            self.timer.stop()
        self._finish()

    @Slot(object)
    def set_gestures(self, gestures: list[Gesture]) -> None:
        self.gestures = gestures

    @Slot()
    def start(self) -> None:
        try:
            self.status_changed.emit("Verificando o modelo de reconhecimento…")
            model_path = ensure_hand_model()
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError(
                    "Não foi possível abrir a webcam. Verifique a câmera e as permissões do sistema."
                )

            vision = mp.tasks.vision
            options = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.process_frame)
            self.timer.start(20)
        except (RuntimeError, ValueError, OSError) as error:
            self.error_occurred.emit(str(error))
            self._finish()

    @Slot()
    def process_frame(self) -> None:
        try:
            if self.camera is None or self.landmarker is None:
                return
            ok, frame = self.camera.read()
            if not ok:
                raise RuntimeError("A webcam parou de fornecer imagens.")
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect_for_video(
                mp_image, time.monotonic_ns() // 1_000_000
            )
            hand_position: tuple[float, float] | None = None
            cursor: tuple[float, float] | None = None

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)
                landmarks = [[point.x, point.y, point.z] for point in hand]
                palm_points = [landmarks[index] for index in (0, 5, 9, 13, 17)]
                hand_position = (
                    sum(point[0] for point in palm_points) / len(palm_points),
                    sum(point[1] for point in palm_points) / len(palm_points),
                )
                cursor = (
                    min(max(landmarks[8][0], 0.0), 1.0),
                    min(max(landmarks[8][1], 0.0), 1.0),
                )

            now = time.monotonic()
            if hand_position is not None and not self._awaiting_hand_release:
                self._motion_history.append((now, hand_position))
            elif hand_position is None:
                self._motion_history.clear()
            while self._motion_history and now - self._motion_history[0][0] > 2.0:
                self._motion_history.popleft()

            if self._capture_request is not None and hand_position is not None:
                self._capture_samples.append(hand_position)
                elapsed = now - self._capture_started
                if elapsed - self._last_capture_progress >= 0.2:
                    self.status_changed.emit(f"Gravando movimento: {min(elapsed, 2.0):.1f} / 2.0 s")
                    self._last_capture_progress = elapsed
                if (
                    len(self._capture_samples) >= 20
                    and elapsed >= 2.0
                ):
                    name, action, value = self._capture_request
                    self._capture_request = None
                    trajectory = normalize_trajectory(self._capture_samples)
                    if trajectory is None:
                        self._capture_samples = []
                        self._awaiting_hand_release = True
                        self._hand_missing_since = None
                        self._motion_history.clear()
                        self.capture_failed.emit(
                            "Movimento insuficiente. Durante o cadastro, mova a mão "
                            "por pelo menos 8% da largura da imagem."
                        )
                    else:
                        gesture = Gesture(name, action, value, trajectory)
                        self.gestures = [
                            existing for existing in self.gestures if existing.name != name
                        ]
                        self.gestures.append(gesture)
                        self._awaiting_hand_release = True
                        self._hand_missing_since = None
                        self._motion_history.clear()
                        self.capture_finished.emit(gesture)
            if (
                self._capture_request is not None
                and now - self._capture_started >= 15.0
            ):
                self._capture_request = None
                self._capture_samples = []
                self._awaiting_hand_release = True
                self._hand_missing_since = None
                self._motion_history.clear()
                self.capture_failed.emit(
                    "Cadastro cancelado: não foi possível detectar a mão por 15 segundos."
                )

            if self._awaiting_hand_release:
                if hand_position is None:
                    if self._hand_missing_since is None:
                        self._hand_missing_since = now
                    elif now - self._hand_missing_since >= 0.75:
                        self._awaiting_hand_release = False
                        self._hand_missing_since = None
                        self._motion_history.clear()
                else:
                    self._hand_missing_since = None

            recognized = (
                classify_trajectory(
                    [point for _, point in self._motion_history],
                    self.gestures,
                )
                if len(self._motion_history) >= 20
                and now - self._motion_history[0][0] >= 1.8
                and self._capture_request is None
                and not self._awaiting_hand_release
                else None
            )
            if recognized is not None:
                self._last_recognized_at = now
                status = f"Gesto: {recognized.name}"
                if recognized.action == "move_window":
                    self._last_action = ""
                    if not self._move_gesture_active:
                        self.executor.perform(recognized.action, recognized.value, cursor)
                        self._move_gesture_active = True
                    if now - self._last_cursor_update >= 0.06:
                        self.executor.update_window_move(cursor)
                        self._last_cursor_update = now
                elif recognized.action == "move_cursor":
                    self._last_action = ""
                    if now - self._last_cursor_update >= 0.06:
                        self.executor.perform(recognized.action, recognized.value, cursor)
                        self._last_cursor_update = now
                elif recognized.name != self._last_action:
                    self.executor.perform(recognized.action, recognized.value, cursor)
                    self._last_action = recognized.name
            else:
                status = (
                    "Execute o movimento do gesto"
                    if self._capture_request is not None
                    else (
                        "Afaste a mão por um instante para ativar o gesto"
                        if self._awaiting_hand_release
                        else "Mostre uma mão para a câmera"
                    )
                )
                if now - self._last_recognized_at >= 0.75:
                    self._last_action = ""
                if self._move_gesture_active and now - self._last_recognized_at >= 0.75:
                    self.executor.update_window_move(None)
                    self._move_gesture_active = False
            if status != self._last_status:
                self.status_changed.emit(status)
                self._last_status = status

            height, width, _ = frame.shape
            image = QImage(
                frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888
            ).copy()
            self.frame_ready.emit(image)
        except (RuntimeError, ValueError, OSError) as error:
            self.error_occurred.emit(str(error))
            self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._move_gesture_active:
            try:
                self.executor.update_window_move(None)
            except (RuntimeError, OSError) as error:
                self.error_occurred.emit(str(error))
            self._move_gesture_active = False
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.finished.emit()


class GestureControlsWindow(QMainWindow):
    capture_requested = Signal(str, str, str)
    stop_requested = Signal()
    gestures_updated = Signal(object)

    def __init__(self, store: GestureStore, gestures: list[Gesture]) -> None:
        super().__init__()
        self.store = store
        self.gestures = gestures
        self.setWindowTitle("Gesture Controls")
        self.resize(900, 700)

        self.preview = QLabel("Inicializando webcam…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("background: #202020; color: white;")
        self.status = QLabel("Iniciando")
        self.gesture_list = QListWidget()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Ex.: Mão aberta")
        self.action_combo = QComboBox()
        self.action_combo.addItems(ACTION_LABELS)
        self.value_input = QLineEdit()
        self.value_input.setPlaceholderText("Para clique: left/right/middle; teclas: CTRL+ALT+T")
        self.action_combo.currentTextChanged.connect(self._update_value_help)
        self._update_value_help(self.action_combo.currentText())

        form = QFormLayout()
        form.addRow("Nome do gesto", self.name_input)
        form.addRow("Ação", self.action_combo)
        form.addRow("Parâmetro", self.value_input)
        self.capture_button = QPushButton("Gravar movimento do gesto (2 s)")
        self.capture_button.clicked.connect(self._capture)
        self.remove_button = QPushButton("Remover gesto selecionado")
        self.remove_button.clicked.connect(self._remove)

        right = QVBoxLayout()
        right.addWidget(QLabel("Gestos cadastrados"))
        right.addWidget(self.gesture_list, 1)
        right.addLayout(form)
        right.addWidget(self.capture_button)
        right.addWidget(self.remove_button)
        right.addWidget(QLabel(
            "Durante o cadastro, execute o movimento por 2 segundos. "
            "Mover o cursor usa a ponta do indicador; mover a janela usa a trajetória "
            "do punho enquanto o gesto é reconhecido."
        ))

        content = QWidget()
        layout = QHBoxLayout(content)
        layout.addWidget(self.preview, 2)
        layout.addLayout(right, 1)
        self.setCentralWidget(content)
        self.statusBar().addWidget(self.status)
        self._refresh_gesture_list()

        screen = QApplication.primaryScreen()
        if screen is None:
            raise RuntimeError("Não foi possível obter a resolução do desktop.")
        size = screen.geometry()
        executor = ActionExecutor((size.width(), size.height()))

        self.thread = QThread(self)
        self.worker = CameraWorker(executor, self.gestures)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.capture_requested.connect(self.worker.request_capture)
        self.stop_requested.connect(self.worker.stop)
        self.gestures_updated.connect(self.worker.set_gestures)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.status_changed.connect(self.status.setText)
        self.worker.capture_finished.connect(self._capture_complete)
        self.worker.capture_failed.connect(self._capture_failed)
        self.worker.error_occurred.connect(self._show_error)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._camera_stopped)
        self.thread.start()

    def _update_value_help(self, label: str) -> None:
        action = ACTION_LABELS[label]
        if action == "click":
            self.value_input.setPlaceholderText("left, right ou middle")
            self.value_input.setEnabled(True)
        elif action == "keypress":
            self.value_input.setPlaceholderText("Ex.: CTRL+ALT+T ou CTRL+C")
            self.value_input.setEnabled(True)
        else:
            self.value_input.clear()
            self.value_input.setPlaceholderText("Esta ação não precisa de parâmetro")
            self.value_input.setEnabled(False)

    def _capture(self) -> None:
        name = self.name_input.text().strip()
        if not name:
            self._show_error("Informe um nome para o gesto.")
            return
        action = ACTION_LABELS[self.action_combo.currentText()]
        value = self.value_input.text().strip()
        try:
            if action == "click" and value.lower() not in {"left", "right", "middle"}:
                raise ValueError("Escolha left, right ou middle para o clique.")
            if action == "keypress":
                parse_key_combo(value)
        except ValueError as error:
            self._show_error(str(error))
            return
        self._previous_gestures = self.gestures.copy()
        self.capture_button.setEnabled(False)
        self.capture_requested.emit(name, action, value)

    @Slot(object)
    def _capture_complete(self, gesture: Gesture) -> None:
        updated_gestures = [
            existing for existing in self.gestures if existing.name != gesture.name
        ]
        updated_gestures.append(gesture)
        try:
            self.store.save(updated_gestures)
        except OSError as error:
            self.gestures_updated.emit(self._previous_gestures)
            self._refresh_gesture_list()
            self.capture_button.setEnabled(True)
            self._show_error(f"Não foi possível salvar o gesto: {error}")
            return
        self.gestures = updated_gestures
        self.capture_button.setEnabled(True)
        self._refresh_gesture_list()
        self.status.setText(f"Gesto cadastrado: {gesture.name}")

    @Slot(str)
    def _capture_failed(self, message: str) -> None:
        self.capture_button.setEnabled(True)
        self._show_error(message)

    def _remove(self) -> None:
        selected = self.gesture_list.currentItem()
        if selected is None:
            return
        selected_name = selected.data(Qt.ItemDataRole.UserRole)
        updated_gestures = [
            gesture for gesture in self.gestures if gesture.name != selected_name
        ]
        try:
            self.store.save(updated_gestures)
        except OSError as error:
            self._show_error(f"Não foi possível salvar a alteração: {error}")
            return
        self.gestures = updated_gestures
        self.gestures_updated.emit(self.gestures)
        self._refresh_gesture_list()

    def _refresh_gesture_list(self) -> None:
        self.gesture_list.clear()
        for gesture in self.gestures:
            action = next(
                (label for label, identifier in ACTION_LABELS.items() if identifier == gesture.action),
                gesture.action,
            )
            recapture_note = " (recadastre)" if gesture.needs_recapture else ""
            self.gesture_list.addItem(f"{gesture.name}{recapture_note} → {action}")
        # Keep the actual gesture name accessible when removing a row with an arrow/action label.
        for index, gesture in enumerate(self.gestures):
            self.gesture_list.item(index).setData(Qt.ItemDataRole.UserRole, gesture.name)

    @Slot(QImage)
    def _show_frame(self, image: QImage) -> None:
        self.preview.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @Slot(str)
    def _show_error(self, message: str) -> None:
        self.status.setText(message)

    @Slot()
    def _camera_stopped(self) -> None:
        self.capture_button.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.thread.isRunning():
            self.stop_requested.emit()
            self.thread.wait()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    store = GestureStore()
    try:
        gestures = store.load()
        window = GestureControlsWindow(store, gestures)
    except (OSError, ValueError, RuntimeError) as error:
        QMessageBox.critical(None, "Erro ao iniciar", str(error))
        raise SystemExit(1) from error
    window.show()
    raise SystemExit(app.exec())
