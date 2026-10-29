import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tests import test_teleport as teleport_tests
from vanishcam.effects import tung as tung_effect
from vanishcam import app
from vanishcam.effects.teleport import HOLD_SECONDS
from tests.test_gestures import hand_points, landmarks
from vanishcam.effects.tung import (
    BEAT_SECONDS, RELEASE_AT, RESCUE_SECONDS, SIZES, TUNG_VOICE_PATH, TungSprites,
    beating_anchor, load_tung, render_beating, render_rescue, render_squashed, thumb_down_score,
    voice_path,
)


def thumb_down(fingers=(False, False, False, False)):
    """Image pixels (y down): a fist with the thumb pointing at the floor."""
    points = hand_points(fingers)
    points[:, 1] *= -1
    points[1:5] = [(0.3, -0.6, 0.0), (0.35, -0.2, 0.0), (0.4, 0.4, 0.0), (0.45, 1.0, 0.0)]
    return points


def invisible_sprites():
    """Same sizes as the real sprites, fully transparent: only the person shows."""
    return TungSprites(*(np.zeros(sprite.shape, np.float32) for sprite in load_tung()))


class ThumbDownTests(unittest.TestCase):
    def test_only_a_thumb_pointing_down_under_a_fist_counts(self):
        self.assertGreater(thumb_down_score(thumb_down()), 0.9)
        thumb_up = thumb_down()
        thumb_up[1:5, 1] *= -1
        self.assertEqual(thumb_down_score(thumb_up), 0.0)
        tucked = thumb_down()
        tucked[4] = (0.2, -0.1, 0.0)
        self.assertEqual(thumb_down_score(tucked), 0.0)
        self.assertEqual(thumb_down_score(thumb_down((True, False, False, False))), 0.0)

    def test_tung_mode_watches_the_thumb_down_gesture(self):
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app.vision.ImageSegmenter, "create_from_options"))
            gestures = mocks.enter_context(patch.object(app.vision.GestureRecognizer, "create_from_options"))
            _, detector = app.create_tasks("tung")
        self.assertIsInstance(detector, app.HeldGestureDetector)
        self.assertEqual(detector.category, "Thumb_Down")
        self.assertFalse(gestures.call_args.args[0].canned_gesture_classifier_options.category_allowlist)
        recognizer = Mock()
        recognizer.recognize_for_video.side_effect = [
            SimpleNamespace(gestures=[[SimpleNamespace(category_name="Thumb_Down", score=0.8)]],
                            hand_landmarks=[]),
            SimpleNamespace(gestures=[[SimpleNamespace(category_name="Thumb_Up", score=0.9)]],
                            hand_landmarks=[]),
            SimpleNamespace(gestures=[], hand_landmarks=[landmarks(thumb_down())]),
        ]
        detector = app.HeldGestureDetector(recognizer, *app.HELD_GESTURES["tung"])
        image = app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                             data=np.zeros((480, 640, 3), np.uint8))
        self.assertEqual([detector.recognize_pose(image, ms) for ms in (100, 200, 300)],
                         [True, False, True])
        self.assertIn("pollice in giù=1.00", detector.last_report)

    def test_beating_needs_the_pose_held_for_one_second(self):
        state = app.AppState(phase="ready")
        self.assertFalse(app.start_on_hold(state, HOLD_SECONDS - 0.01, 3.0, "beating"))
        self.assertTrue(app.start_on_hold(state, HOLD_SECONDS, 3.1, "beating"))
        self.assertEqual(state.phase, "beating")

    def test_own_clips_in_voci_replace_the_synthesized_voices(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(tung_effect, "VOICES_DIR", Path(folder)):
            self.assertEqual(voice_path("tung", TUNG_VOICE_PATH), TUNG_VOICE_PATH)
            (Path(folder) / "tung.mp3.download").touch()  # Download non finito.
            self.assertEqual(voice_path("tung", TUNG_VOICE_PATH), TUNG_VOICE_PATH)
            (Path(folder) / "tung.MP3").touch()
            self.assertEqual(voice_path("tung", TUNG_VOICE_PATH), Path(folder) / "tung.MP3")
            self.assertEqual(voice_path("bombardiro", TUNG_VOICE_PATH), TUNG_VOICE_PATH)

    def test_fifth_menu_key_selects_tung(self):
        with ExitStack() as mocks:
            for name in ("namedWindow", "setMouseCallback", "imshow", "destroyWindow"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "waitKey", return_value=ord("5")))
            self.assertEqual(app.choose_mode(), "tung")


class TungEffectTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((240, 320, 3), 200, np.uint8)
        self.background = np.full((240, 320, 3), 40, np.uint8)
        self.mask = np.zeros((240, 320), np.float32)
        self.mask[40:, 180:260] = 1.0
        self.anchor = beating_anchor(self.mask)
        self.person = app.alpha_composite(self.frame, self.background, self.mask)
        self.sprites = load_tung()

    def person_box(self, image):
        """Rows and columns where the person (plain light grey) is visible."""
        visible = (image > 175).all(axis=2)
        return int(visible.any(axis=1).sum()), int(visible.any(axis=0).sum())

    def test_character_comes_from_the_side_with_more_room(self):
        self.assertEqual(self.anchor, (220, 40, -1))
        mask = self.mask[:, ::-1].copy()
        self.assertEqual(beating_anchor(mask)[2], 1)
        mirrored = render_beating(self.frame, self.background, mask, self.sprites, beating_anchor(mask), 1.0)
        changed = np.any(mirrored != render_squashed(self.frame, self.background, mask), axis=2)
        self.assertTrue(changed[:, 220:].any())  # Tung Tung Tung Sahur a destra.

    def test_beating_starts_on_the_person_and_leaves_them_squashed(self):
        render = lambda age: render_beating(self.frame, self.background, self.mask, self.sprites,
                                            self.anchor, age)
        np.testing.assert_array_equal(render(0.0), self.person)
        np.testing.assert_array_equal(render(BEAT_SECONDS),
                                      render_squashed(self.frame, self.background, self.mask))

    def test_every_hit_makes_the_person_smaller(self):
        sprites = invisible_sprites()
        # Between hits, once the stars of the previous one are gone.
        heights = [
            self.person_box(render_beating(self.frame, self.background, self.mask, sprites,
                                           self.anchor, age))[0]
            for age in (0.70, 1.05, 1.45, 1.85, 2.5)
        ]
        self.assertEqual(heights[0], 200)
        self.assertEqual(heights, sorted(heights, reverse=True))
        self.assertEqual(len(set(heights)), 5)
        self.assertLessEqual(heights[-1], round(200 * SIZES[-1][1]) + 1)

    def test_bombardiro_hooks_the_squashed_person_and_pulls_them_back(self):
        render = lambda age, sprites=self.sprites: render_rescue(
            self.frame, self.background, self.mask, sprites, self.anchor, age)
        np.testing.assert_array_equal(render(0.0), render_squashed(self.frame, self.background, self.mask))
        tall, wide = self.person_box(render(RELEASE_AT - 0.01, invisible_sprites()))
        self.assertGreater(tall, 200)  # Tirato dal cavo: più alto.
        self.assertLess(wide, 80)  # E più magro.
        np.testing.assert_array_equal(render(RESCUE_SECONDS), self.person)


class TungLoopTests(unittest.TestCase):
    run_video = teleport_tests.GestureLoopTests.run_video

    def test_tung_beats_you_down_and_bombardiro_brings_you_back(self):
        times = [0.1, 4.1] + [round(4.2 + 0.1 * step, 1) for step in range(11)]  # Pollice verso.
        times += [5.6, 6.2, 6.8, 7.4, 8.1]  # Le bastonate.
        times += [8.2, 8.4, 8.6, 8.8]  # Schiacciato: mano abbassata.
        times += [round(9.0 + 0.1 * step, 1) for step in range(11)]  # Di nuovo il pollice.
        times += [10.1, 10.9, 11.7, 12.5, 13.3, 13.4]  # Bombardiro Crocodilo, poi webcam normale.
        recognizer = Mock(last_report="")
        recognizer.recognize_pose.side_effect = lambda image, ms: 4150 < ms < 5350 or 8950 < ms < 10150
        frames, preview, sound = self.run_video("tung", times, len(times), recognizer)
        self.assertEqual(sound.play.call_count, 4)  # Effetto e voce, all'andata e al ritorno.
        landed = times.index(8.1)
        squashed = render_squashed(frames[landed], frames[1], np.ones((80, 120), np.float32))
        for time in (8.1, 8.6, 10.0):
            np.testing.assert_array_equal(preview[times.index(time)], squashed)
        beaten = times.index(6.8)
        self.assertTrue(np.any(preview[beaten] != frames[beaten]))
        pulled = times.index(11.7)
        self.assertTrue(np.any(preview[pulled] != squashed))
        self.assertTrue(np.any(preview[pulled] != frames[pulled]))
        np.testing.assert_array_equal(preview[-1], frames[-1])


if __name__ == "__main__":
    unittest.main()
