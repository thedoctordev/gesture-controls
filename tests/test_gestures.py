import unittest

from gesture_controls.gestures import (
    SEQUENCE_LENGTH,
    Gesture,
    classify_trajectory,
    normalize_trajectory,
)


class GestureTests(unittest.TestCase):
    def line(self, start, end, count=40):
        return [
            [
                start[0] + (end[0] - start[0]) * index / (count - 1),
                start[1] + (end[1] - start[1]) * index / (count - 1),
            ]
            for index in range(count)
        ]

    def test_normalize_trajectory_is_translation_and_scale_invariant(self):
        original = self.line((0.1, 0.2), (0.8, 0.7))
        transformed = [[x * 2 + 0.3, y * 2 + 0.1] for x, y in original]

        actual = normalize_trajectory(original)
        expected = normalize_trajectory(transformed)
        self.assertEqual(len(actual), SEQUENCE_LENGTH)
        for actual_point, expected_point in zip(actual, expected):
            for actual_value, expected_value in zip(actual_point, expected_point):
                self.assertAlmostEqual(actual_value, expected_value)

    def test_normalize_trajectory_rejects_stationary_and_invalid_input(self):
        self.assertIsNone(normalize_trajectory([[0.1, 0.2]] * 20))
        self.assertIsNone(normalize_trajectory([[float("nan"), 0.2], [0.2, 0.3]]))
        jitter = [[0.5 + (-1) ** index * 0.002, 0.5] for index in range(100)]
        self.assertIsNone(normalize_trajectory(jitter))

    def test_classifier_matches_same_path_at_different_scale_and_speed(self):
        recorded_path = self.line((0.1, 0.5), (0.7, 0.5))
        recorded = normalize_trajectory(recorded_path)
        gesture = Gesture("Swipe", "click", "left", recorded)
        slower_path = self.line((0.4, 0.3), (0.7, 0.3), count=70)

        self.assertEqual(classify_trajectory(slower_path, [gesture]), gesture)

    def test_classifier_distinguishes_opposite_motion(self):
        recorded = normalize_trajectory(self.line((0.1, 0.5), (0.8, 0.5)))
        gesture = Gesture("Swipe right", "click", "left", recorded)
        reverse = self.line((0.8, 0.5), (0.1, 0.5))

        self.assertIsNone(classify_trajectory(reverse, [gesture]))
