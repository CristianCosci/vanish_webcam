import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from vanishcam import app
from vanishcam.effects.naruto import (
    EFFECT_SECONDS, landed_log_anchor, load_log, overlay_log, render_return,
)
from vanishcam.effects.teleport import (
    HOLD_SECONDS, TELEPORT_SECONDS, PoseTimer, fingers_on_forehead, forehead_from_keypoints,
    make_speed_lines, render_teleport, two_fingers_score,
)
from tests.test_gestures import hand_points, landmarks
from tests.test_video_output import arguments


FACE = [SimpleNamespace(x=x, y=y) for x, y in (
    (0.45, 0.40), (0.55, 0.40), (0.50, 0.47), (0.50, 0.53), (0.38, 0.42), (0.62, 0.42),
)]


def pose_points():
    """Index and middle straight and side by side, ring and little finger folded."""
    points = hand_points((True, True, False, False))
    for base, (start, end) in ((5, (-0.45, -0.17)), (9, (0.0, 0.17))):
        points[base : base + 4] = [
            (start + (end - start) * step, 1.0 + 1.5 * step, 0.0) for step in (0.0, 0.47, 0.77, 1.0)
        ]
    return points


def on_forehead(points, lift=0.0, width=640, height=480):
    """Image pixels with the finger pads on the FACE forehead, moved up by lift eye distances."""
    center, _, scale = forehead_from_keypoints(FACE, width, height)
    size = scale * 1.3 / 1.5
    pads = points[[7, 8, 11, 12], :2].mean(axis=0)
    pixels = np.zeros_like(points)
    pixels[:, 0] = center[0] + (points[:, 0] - pads[0]) * size
    pixels[:, 1] = center[1] - (points[:, 1] - pads[1]) * size - lift * scale
    return pixels


class TeleportPoseTests(unittest.TestCase):
    def test_two_fingers_side_by_side_but_not_v_or_other_hands(self):
        self.assertGreater(two_fingers_score(pose_points()), 0.9)
        self.assertEqual(two_fingers_score(hand_points()), 0.0)  # Segno a V.
        self.assertEqual(two_fingers_score(hand_points((True, False, False, False))), 0.0)
        self.assertEqual(two_fingers_score(hand_points((True, True, True, True))), 0.0)

    def test_fingers_must_rest_on_the_forehead(self):
        forehead = forehead_from_keypoints(FACE, 640, 480)
        self.assertTrue(fingers_on_forehead(on_forehead(pose_points()), forehead))
        self.assertFalse(fingers_on_forehead(on_forehead(pose_points(), lift=-2.0), forehead))
        self.assertFalse(fingers_on_forehead(on_forehead(pose_points(), lift=3.0), forehead))
        sideways = on_forehead(pose_points())
        sideways[:, 0] += forehead[2] * 2.0
        self.assertFalse(fingers_on_forehead(sideways, forehead))

    def test_detector_remembers_the_forehead_while_the_hand_covers_the_face(self):
        pixels = on_forehead(pose_points())
        hands, faces = Mock(), Mock()
        hands.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=x / 640, y=y / 480, z=0.0) for x, y, _ in pixels]],
            hand_world_landmarks=[landmarks(pose_points())],
        )
        face = SimpleNamespace(keypoints=FACE, bounding_box=SimpleNamespace(width=90, height=90))
        faces.detect_for_video.side_effect = [
            SimpleNamespace(detections=[face]),
            SimpleNamespace(detections=[]),
            SimpleNamespace(detections=[]),
        ]
        detector = app.TeleportPoseDetector(hands, faces)
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(detector.recognize_pose(image, 100))
        self.assertTrue(detector.recognize_pose(image, 900))  # Volto coperto dalla mano.
        self.assertFalse(detector.recognize_pose(image, 2000))  # Fronte dimenticata.
        detector.close()
        hands.close.assert_called_once()
        faces.close.assert_called_once()

    def test_short_gaps_are_forgiven_but_breaking_the_pose_restarts(self):
        timer = PoseTimer()
        self.assertEqual(timer.update(True, 10.0), 0.0)
        self.assertAlmostEqual(timer.update(True, 10.2), 0.2)
        self.assertAlmostEqual(timer.update(False, 10.3), 0.3)
        self.assertAlmostEqual(timer.update(True, 10.4), 0.4)
        self.assertEqual(timer.update(False, 10.8), 0.0)
        self.assertEqual(timer.update(True, 10.9), 0.0)


class ReturnGestureTests(unittest.TestCase):
    def test_same_gesture_brings_you_back_only_after_lowering_the_hand(self):
        state = app.AppState(phase="ready")
        self.assertTrue(app.start_vanish_on_gesture(state, "Victory", 0.9, 1.0))
        state.phase = "vanished"
        self.assertFalse(app.start_return_on_gesture(state, True, 5.0))  # Mano ancora alzata.
        for now in (5.0, 5.1, 5.4):
            app.track_release(state, now == 5.0, now)
        self.assertFalse(state.released)
        app.track_release(state, False, 5.7)
        self.assertTrue(state.released)
        self.assertTrue(app.start_return_on_gesture(state, True, 5.8))
        self.assertEqual((state.phase, state.reappear_started), ("reappearing", 5.8))
        state.phase = "ready"
        self.assertFalse(app.start_vanish_on_gesture(state, "Victory", 0.9, 6.0))
        self.assertFalse(app.start_naruto_on_join(state, True, 6.0))

    def test_teleport_needs_the_pose_held_for_one_second(self):
        state = app.AppState(phase="ready")
        self.assertFalse(app.start_on_hold(state, HOLD_SECONDS - 0.01, 3.0, "teleporting"))
        self.assertTrue(app.start_on_hold(state, HOLD_SECONDS, 3.1, "teleporting"))
        self.assertEqual((state.phase, state.vanish_started), ("teleporting", 3.1))

    def test_third_menu_key_selects_teleport(self):
        with ExitStack() as mocks:
            for name in ("namedWindow", "setMouseCallback", "imshow", "destroyWindow"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "waitKey", return_value=ord("3")))
            self.assertEqual(app.choose_mode(), "teleport")


class EffectTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((90, 120, 3), 200, np.uint8)
        self.background = np.full((90, 120, 3), 30, np.uint8)
        self.mask = np.zeros((90, 120), np.float32)
        self.mask[20:80, 50:70] = 1.0
        self.lines = make_speed_lines()

    def render(self, age, lines=None, arriving=False):
        lines = self.lines if lines is None else lines
        return render_teleport(self.frame, self.background, self.mask, lines, age, arriving)

    def test_teleport_starts_on_the_person_and_ends_on_the_background(self):
        person = app.alpha_composite(self.frame, self.background, self.mask)
        np.testing.assert_array_equal(self.render(0.0), person)
        np.testing.assert_array_equal(self.render(TELEPORT_SECONDS), self.background)
        np.testing.assert_array_equal(self.render(0.0, arriving=True), self.background)
        np.testing.assert_array_equal(self.render(TELEPORT_SECONDS, arriving=True), person)

    def test_person_squeezes_into_a_beam_before_vanishing(self):
        widths = []
        for age in (0.0, 0.36, 0.40):
            changed = np.any(self.render(age, lines=self.lines[:0]) != self.background, axis=2)
            widths.append(int(changed.any(axis=0).sum()))
        self.assertEqual(widths[0], 20)
        self.assertLess(widths[1], widths[0])
        self.assertLess(widths[2], widths[1])

    def test_lines_linger_only_around_the_silhouette(self):
        changed = np.any(self.render(0.5) != self.background, axis=2)
        self.assertTrue(changed[20:80, 50:70].any())
        self.assertFalse(changed[:, :20].any())
        self.assertFalse(changed[:, 100:].any())

    def test_naruto_return_swallows_the_log_and_brings_the_person_back(self):
        frame = np.full((240, 320, 3), 200, np.uint8)
        background = np.full((240, 320, 3), 40, np.uint8)
        mask = np.zeros((240, 320), np.float32)
        mask[40:220, 120:200] = 1.0
        sprite, anchor = load_log(), (160, 130, 110)
        vanished = background.copy()
        overlay_log(vanished, sprite, landed_log_anchor(anchor, 240))
        person = app.alpha_composite(frame, background, mask)
        np.testing.assert_array_equal(render_return(frame, background, mask, sprite, anchor, 0.0), vanished)
        self.assertTrue(np.any(render_return(frame, background, mask, sprite, anchor, 0.3) != person))
        np.testing.assert_array_equal(
            render_return(frame, background, mask, sprite, anchor, EFFECT_SECONDS), person
        )


class GestureLoopTests(unittest.TestCase):
    def run_video(self, mode, times, frame_count, recognizer, **patches):
        frames = [np.full((80, 120, 3), 10 + 3 * index, np.uint8) for index in range(frame_count)]
        capture = Mock()
        capture.read.side_effect = [(True, frame) for frame in frames]
        sound = Mock()
        virtual = Mock(width=120, height=80)
        preview = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(mode=mode)))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(Mock(), recognizer)))
            mocks.enter_context(patch.object(app, "VanishSound", return_value=sound))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=virtual))
            mocks.enter_context(patch.object(app, "person_mask", return_value=np.ones((80, 120), np.float32)))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=(0, 0, *times)))
            for name, value in patches.items():
                mocks.enter_context(patch.object(app, name, value))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: preview.append(frame.copy())))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=[0] * (frame_count - 1) + [ord("q")]))
            mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("Overlay vietato")))
            mocks.enter_context(patch("builtins.print"))
            self.assertEqual(app.run(), 0)
        self.assertEqual(len(preview), frame_count)
        for sent, shown in zip(virtual.send.call_args_list, preview):
            np.testing.assert_array_equal(sent.args[0], shown)
        return frames, preview, sound

    def test_teleport_out_and_back_with_the_same_pose(self):
        times = [0.1, 4.1] + [round(4.2 + 0.1 * step, 1) for step in range(12)]  # Posa tenuta.
        times += [5.6, 5.9, 6.3]  # Teletrasporto.
        times += [6.4, 6.5, 6.7, 6.9, 7.1]  # Sparito: mano abbassata.
        times += [round(7.2 + 0.1 * step, 1) for step in range(12)]  # Posa di nuovo.
        times += [8.6, 8.9, 9.4, 9.5]  # Ritorno, poi webcam normale.
        recognizer = Mock(last_report="")
        recognizer.recognize_pose.side_effect = (
            lambda image, ms: 4150 < ms < 6450 or 7150 < ms < 8350
        )
        arrivals = []

        def recording(*args, **kwargs):
            arrivals.append(kwargs.get("arriving", False))
            return render_teleport(*args, **kwargs)

        frames, preview, sound = self.run_video(
            "teleport", times, len(times), recognizer, render_teleport=recording
        )
        self.assertEqual(sound.play.call_count, 2)
        self.assertIn(False, arrivals)
        self.assertIn(True, arrivals)
        np.testing.assert_array_equal(preview[times.index(6.7)], frames[1])  # Solo sfondo.
        np.testing.assert_array_equal(preview[-1], frames[-1])

    def test_normal_mode_rebuilds_the_person_with_the_v_sign(self):
        # Il dissolvimento legge l'orologio una volta in più a ogni fotogramma.
        times = [0.1, 4.1, 4.3, 4.31, 8.0, 8.0, 8.1, 8.3, 8.9, 9.1, 10.0, 12.8, 12.9]

        def victory(recognizer, image, ms):
            raised = ms <= 4350 or 8050 <= ms <= 8150 or ms >= 9050
            return ("Victory", 0.9) if raised else ("No_Hand", 0.0)

        frames, preview, _ = self.run_video(
            "normal", times, 11, Mock(), recognize_gesture=Mock(side_effect=victory)
        )
        np.testing.assert_array_equal(preview[3], frames[1])  # Dissolto: solo sfondo.
        np.testing.assert_array_equal(preview[6], frames[1])  # Mano abbassata, ancora sparito.
        self.assertTrue(np.any(preview[8] != frames[1]))  # Ricomparsa in corso.
        self.assertTrue(np.any(preview[8] != frames[8]))
        np.testing.assert_array_equal(preview[9], frames[9])  # Ricomposto.
        np.testing.assert_array_equal(preview[10], frames[10])  # La V ancora alzata non fa sparire.


if __name__ == "__main__":
    unittest.main()
