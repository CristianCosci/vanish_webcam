import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tests import test_teleport as teleport_tests
from vanishcam import app
from vanishcam.effects.container import (
    DROP_LANDS, DROP_SECONDS, FALL_STARTS, LIFT_RELEASES, LIFT_SECONDS, container_anchor,
    load_container, pointing_up_score, render_drop, render_lift, render_rest,
)
from vanishcam.effects.teleport import HOLD_SECONDS
from tests.test_gestures import hand_points, landmarks


def pointing_up(fingers=(True, False, False, False)):
    """Image pixels (y down) with the extended fingers pointing up."""
    points = hand_points(fingers)
    points[:, 1] *= -1
    return points


def gesture_result(score=None, hands=()):
    gestures = [[SimpleNamespace(category_name="Pointing_Up", score=score)]] if score else []
    return SimpleNamespace(gestures=gestures, hand_landmarks=[landmarks(hand) for hand in hands])


class PointingUpTests(unittest.TestCase):
    def test_only_a_straight_index_pointing_up_counts(self):
        self.assertGreater(pointing_up_score(pointing_up()), 0.9)
        self.assertEqual(pointing_up_score(hand_points((True, False, False, False))), 0.0)  # In giù.
        sideways = pointing_up()[:, [1, 0, 2]]
        self.assertEqual(pointing_up_score(sideways), 0.0)
        self.assertEqual(pointing_up_score(pointing_up((True, True, False, False))), 0.0)  # Segno a V.
        self.assertEqual(pointing_up_score(pointing_up((True, True, True, True))), 0.0)
        self.assertEqual(pointing_up_score(pointing_up((False, False, False, False))), 0.0)

    def test_detector_accepts_the_classifier_or_the_landmarks(self):
        recognizer = Mock()
        recognizer.recognize_for_video.side_effect = [
            gesture_result(score=0.8),
            gesture_result(score=0.4),
            gesture_result(hands=[pointing_up()]),
            gesture_result(hands=[pointing_up((True, True, False, False))]),
        ]
        detector = app.HeldGestureDetector(recognizer, *app.HELD_GESTURES["container"])
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((480, 640, 3), np.uint8))
        self.assertEqual([detector.recognize_pose(image, ms) for ms in (100, 200, 300, 400)],
                         [True, False, True, False])
        self.assertIn("mani=1", detector.last_report)
        detector.close()
        recognizer.close.assert_called_once()

    def test_container_tasks_use_only_the_video_gesture_recognizer(self):
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app.vision.ImageSegmenter, "create_from_options"))
            gestures = mocks.enter_context(patch.object(app.vision.GestureRecognizer, "create_from_options"))
            _, detector = app.create_tasks("container")
        self.assertIsInstance(detector, app.HeldGestureDetector)
        self.assertEqual(detector.category, "Pointing_Up")
        gestures.assert_called_once()
        # Una allowlist farebbe sparire dal risultato le mani con altri gesti.
        options = gestures.call_args.args[0]
        self.assertFalse(options.canned_gesture_classifier_options.category_allowlist)

    def test_container_needs_the_pose_held_for_one_second(self):
        state = app.AppState(phase="ready")
        self.assertFalse(app.start_on_hold(state, HOLD_SECONDS - 0.01, 3.0, "crushing"))
        self.assertTrue(app.start_on_hold(state, HOLD_SECONDS, 3.1, "crushing"))
        self.assertEqual((state.phase, state.vanish_started), ("crushing", 3.1))

    def test_fourth_menu_key_selects_container(self):
        with ExitStack() as mocks:
            for name in ("namedWindow", "setMouseCallback", "imshow", "destroyWindow"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "waitKey", return_value=ord("4")))
            self.assertEqual(app.choose_mode(), "container")


class ContainerEffectTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((240, 320, 3), 200, np.uint8)
        self.background = np.full((240, 320, 3), 40, np.uint8)
        self.mask = np.zeros((240, 320), np.float32)
        self.mask[40:, 120:200] = 1.0
        # Plain blue box: only the person has a bright red channel.
        self.sprite = np.zeros((20, 84, 3), np.uint8)
        self.sprite[:] = (255, 0, 0)
        self.anchor = container_anchor(self.mask)
        self.person = app.alpha_composite(self.frame, self.background, self.mask)

    def person_box(self, image):
        """Rows and columns where the (possibly shadowed) person is visible."""
        visible = image[:, :, 2] > 60
        return int(visible.any(axis=1).sum()), int(visible.any(axis=0).sum())

    def test_container_lands_on_the_person_centre_and_floor(self):
        self.assertEqual(self.anchor[:2], (160, 240))
        rest = render_rest(self.background, self.sprite, self.anchor)
        size = self.anchor[2]
        self.assertTrue(np.all(rest[-size:, 160] == (255, 0, 0)))
        np.testing.assert_array_equal(rest[:-size], self.background[:-size])

    def test_drop_starts_on_the_person_and_leaves_only_the_container(self):
        render = lambda age: render_drop(self.frame, self.background, self.mask, self.sprite,
                                         self.anchor, age)
        np.testing.assert_array_equal(render(0.0), self.person)
        np.testing.assert_array_equal(render(DROP_SECONDS),
                                      render_rest(self.background, self.sprite, self.anchor))

    def test_person_gets_flat_and_wide_under_the_container(self):
        boxes = [
            self.person_box(render_drop(self.frame, self.background, self.mask, self.sprite,
                                        self.anchor, age))
            for age in (FALL_STARTS + 0.2, 0.75, DROP_LANDS - 0.01)
        ]
        self.assertEqual(boxes[0], (200, 80))
        self.assertLess(boxes[1][0], boxes[0][0])
        self.assertLess(boxes[2][0], boxes[1][0])
        self.assertLess(boxes[2][0], 12)
        self.assertGreater(boxes[1][1], 80)

    def test_crane_lifts_the_container_and_stretches_the_person_back(self):
        render = lambda age: render_lift(self.frame, self.background, self.mask, self.sprite,
                                         self.anchor, age)
        np.testing.assert_array_equal(render(0.0), render_rest(self.background, self.sprite, self.anchor))
        stuck = self.person_box(render(LIFT_RELEASES - 0.01))
        self.assertGreater(stuck[0], 200)  # Ancora attaccato al container: più alto.
        self.assertLess(stuck[1], 80)  # E più magro.
        np.testing.assert_array_equal(render(LIFT_SECONDS), self.person)

    def test_spreader_is_drawn_while_the_crane_grabs_the_box(self):
        locked = render_lift(self.frame, self.background, self.mask, self.sprite, self.anchor, 0.8)
        rest = render_rest(self.background, self.sprite, self.anchor)
        size = self.anchor[2]
        changed = np.any(locked != rest, axis=2)
        self.assertTrue(changed[:-size].any())  # Spreader e funi sopra il container.
        self.assertFalse(changed[:, :5].any())  # Niente fuori dalla lunghezza del container.


class ContainerLoopTests(unittest.TestCase):
    run_video = teleport_tests.GestureLoopTests.run_video

    def test_container_drops_and_the_crane_brings_you_back(self):
        times = [0.1, 4.1] + [round(4.2 + 0.1 * step, 1) for step in range(11)]  # Indice in alto.
        times += [5.6, 6.0, 6.3, 7.0]  # Caduta, schiacciamento, polvere.
        times += [7.1, 7.3, 7.5, 7.7]  # Sotto il container: mano abbassata.
        times += [round(8.0 + 0.1 * step, 1) for step in range(11)]  # Di nuovo l'indice: la gru.
        times += [9.1, 9.8, 10.2, 10.6, 11.3, 11.4]  # Sollevamento, poi webcam normale.
        recognizer = Mock(last_report="")
        recognizer.recognize_pose.side_effect = lambda image, ms: 4150 < ms < 5350 or 7950 < ms < 9150
        frames, preview, sound = self.run_video("container", times, len(times), recognizer)
        self.assertEqual(sound.play.call_count, 2)
        anchor = container_anchor(np.ones((80, 120), np.float32))
        rest = render_rest(frames[1], load_container(), anchor)
        for time in (7.0, 7.5, 9.0):
            np.testing.assert_array_equal(preview[times.index(time)], rest)  # Solo il container.
        crushed = times.index(6.0)
        self.assertTrue(np.any(preview[crushed] != frames[crushed]))
        lifting = times.index(10.2)
        self.assertTrue(np.any(preview[lifting] != rest))
        self.assertTrue(np.any(preview[lifting] != frames[lifting]))
        np.testing.assert_array_equal(preview[-1], frames[-1])


if __name__ == "__main__":
    unittest.main()
