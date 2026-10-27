import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gesture_controls.gestures import Gesture
from gesture_controls.store import GestureStore


class StoreTests(unittest.TestCase):
    def test_store_round_trips_gestures(self):
        with TemporaryDirectory() as directory:
            store = GestureStore(Path(directory) / "gestures.json")
            trajectory = [[index / 23, 0.0] for index in range(24)]
            gesture = Gesture("Deslizar", "click", "left", trajectory)
            store.save([gesture])

            self.assertEqual(store.load(), [gesture])

    def test_store_loads_old_static_gestures_as_needing_recapture(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gestures.json"
            path.write_text(
                '[{"name":"Pose","action":"click","value":"left","pose":'
                + str([0.1] * 63).replace("'", '"')
                + "}]",
                encoding="utf-8",
            )

            gesture = GestureStore(path).load()[0]

            self.assertTrue(gesture.needs_recapture)

    def test_store_reports_corrupt_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gestures.json"
            path.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Não foi possível carregar"):
                GestureStore(path).load()
