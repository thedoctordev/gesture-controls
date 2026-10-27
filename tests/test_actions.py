import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gesture_controls.actions import ActionExecutor, parse_key_combo


class ActionTests(unittest.TestCase):
    def test_parse_key_combo_accepts_supported_modifiers_and_keys(self):
        self.assertEqual(parse_key_combo("CTRL+ALT+T"), [29, 56, 20])

    def test_parse_key_combo_rejects_invalid_input(self):
        for value in ("", "CTRL++C", "CTRL+UNKNOWN", "A+B", "CTRL+CTRL+C"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_key_combo(value)

    @patch("gesture_controls.actions.subprocess.run")
    def test_cursor_move_uses_ydotool_absolute_coordinates(self, run):
        run.return_value = SimpleNamespace(returncode=0, stderr="", stdout="")
        executor = object.__new__(ActionExecutor)
        executor.executable = "/usr/bin/ydotool"
        executor.screen_width = 1000
        executor.screen_height = 800

        executor.move_cursor(0.5, 0.25)

        self.assertEqual(
            run.call_args.args[0],
            ["/usr/bin/ydotool", "mousemove", "--absolute", "-x", "500", "-y", "200"],
        )
