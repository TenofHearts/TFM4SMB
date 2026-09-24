"""Contract tests use synthetic RAM and no weights, network, or emulator."""

import io
import json
import os
from contextlib import redirect_stderr
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tfm4mario.actions import button_names, to_nes_action, validate_action
from tfm4mario.dataset import load_table, prepare
from tfm4mario.features import (
    ACTION_FEATURE_NAMES,
    FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    extract_features,
    feature_dict,
    tile_at,
)
from tfm4mario.game import rollout
from tfm4mario.metadata_cache import CACHE_SCHEMA, build_cache
from tfm4mario.online import OnlineReplay
from tfm4mario.policy import Policy as FittedPolicy
from tfm4mario.ram import (
    PNG_SIGNATURE,
    OutcomeMismatchError,
    decode_ram,
    parse_frame,
    read_frame,
)
from tfm4mario.teacher import DATASET_ACTIONS, _preprocess, _write_shard
from tfm4mario.teacher_data import prepare_teacher_decisions
from tfm4mario.cli import parse_args


def png_bytes(ram, action, outcome=2):
    # A minimal structural PNG fixture; image decoding is intentionally unused.
    return (PNG_SIGNATURE + struct.pack(">I", 0) + b"IEND" + b"\x00" * 4
            + struct.pack(">I", 2052) + b"tEXtRAM\x00"
            + bytes(ram).replace(b"\r", b"\r\n") + b"\x00" * 4
            + struct.pack(">I", 5) + b"tEXtBP1\x00" + bytes([action]) + b"\x00" * 4
            + struct.pack(">I", 9) + b"tEXtOUTCOME\x00" + bytes([outcome]) + b"\x00" * 4)


def game_ram():
    ram = np.zeros(2048, dtype=np.uint8)
    ram[0x770] = 1
    ram[0x0E] = 8
    ram[0xB5] = 1
    return ram


class RamTests(unittest.TestCase):
    def test_cr_expansion_is_inverted_without_corrupting_real_crlf(self):
        original = bytes(range(256)) * 8
        encoded = original.replace(b"\r", b"\r\n")
        self.assertEqual(decode_ram(encoded).tobytes(), original)
        self.assertEqual(decode_ram(original, "raw").tobytes(), original)
        with self.assertRaises(ValueError):
            decode_ram(encoded + b"\x00")

    def test_metadata_and_filename_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p_s_e0_1-1_f10_a148_date.win.png"
            ram = game_ram()
            ram[500:503] = [13, 10, 13]
            path.write_bytes(png_bytes(ram, 148))
            np.testing.assert_array_equal(read_frame(parse_frame(path)), ram)
            path.write_bytes(png_bytes(ram, 20))
            with self.assertRaisesRegex(ValueError, "BP1"):
                read_frame(parse_frame(path))
            path.write_bytes(png_bytes(ram, 148, outcome=1))
            with self.assertRaises(OutcomeMismatchError):
                read_frame(parse_frame(path))


class ConfigTests(unittest.TestCase):
    def test_adapt_accepts_argmax_action_selection(self):
        args = parse_args(["adapt", "--action-selection", "argmax"])
        self.assertEqual(args.action_selection, "argmax")

    def test_shared_paths_resolve_relative_to_config_and_cli_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text('[paths]\ncontext="rows.npz"\nmodel="policy"\n'
                              '[runtime]\ndevice="cpu"\n[train]\nn_estimators=2\n')
            args = parse_args(["train", "--config", str(config), "--n-estimators", "3"])
            self.assertEqual(args.context, Path(directory) / "rows.npz")
            self.assertEqual(args.output, Path(directory) / "policy")
            self.assertEqual(args.device, "cpu")
            self.assertEqual(args.n_estimators, 3)

    def test_boolean_override_and_input_override(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text('[paths]\nmodel="policy"\nrollout="rollout"\n'
                              '[play]\nrender=true\n[predict]\nram="snapshot.bin"\n')
            args = parse_args(["--config", str(config), "play", "--no-render"])
            self.assertFalse(args.render)
            args = parse_args(["predict", "--config", str(config), "--png", "sample.png"])
            self.assertIsNone(args.ram)
            self.assertEqual(args.png, Path("sample.png"))

    def test_typos_and_invalid_numbers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            for content in ['[prepare]\nmax_rows=0',
                            '[adapt]\nepsilon=1.1',
                            '[adapt]\nonline_min_progress_delta=0',
                            '[predict]\nprevious_action=1',
                            '[train]\nn_estimator=1',
                            '[play]\nrender="false"', '[prepare]\nlabel_offset=2']:
                config.write_text(content)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["doctor", "--config", str(config)])


class TeacherCollectionTests(unittest.TestCase):
    def test_action_order_matches_simple_movement(self):
        self.assertEqual(DATASET_ACTIONS, (0, 4, 132, 20, 148, 128, 32))

    def test_preprocess_is_normalized_four_stack_frame(self):
        image = np.zeros((240, 256, 3), dtype=np.uint8)
        image[..., 0] = 255
        processed = _preprocess(image)
        self.assertEqual(processed.shape, (84, 84))
        self.assertEqual(processed.dtype, np.float32)
        np.testing.assert_allclose(processed, 76 / 255, atol=1e-6)

    def test_teacher_shard_is_prepare_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "teacher-cache"
            cache.mkdir()
            ram = np.stack([game_ram(), game_ram(), game_ram()])
            ram[1, 0x86], ram[2, 0x86] = 4, 8
            _write_shard(
                cache,
                {
                    "ram": ram,
                    "actions": np.asarray([0, 20, 148], dtype=np.uint8),
                },
                0,
            )
            (cache / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": CACHE_SCHEMA,
                        "outcome": "win",
                        "ram_encoding": "raw",
                        "complete": True,
                        "shards": 1,
                        "frames": 3,
                    }
                ),
                encoding="utf-8",
            )
            context = root / "context.npz"
            metadata = prepare(
                cache,
                context,
                outcome="win",
                encoding="raw",
                include_level="1-3",
                max_rows=10,
            )
            self.assertEqual(metadata["trajectory_count"], 1)
            self.assertEqual(metadata["levels"], ["1-3"])
            with np.load(context, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["y"], [20, 148])

    def test_decision_rows_include_reset_and_only_four_frame_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "teacher-cache"
            cache.mkdir()
            reset_ram = game_ram()
            ram = np.stack([game_ram() for _ in range(12)])
            ram[:, 0x86] = [1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8]
            actions = np.asarray([20] * 4 + [132] * 4 + [4] * 4, dtype=np.uint8)
            _write_shard(
                cache,
                {"reset_ram": reset_ram, "ram": ram, "actions": actions},
                0,
            )
            (cache / "manifest.json").write_text(
                json.dumps({
                    "schema": CACHE_SCHEMA, "complete": True,
                    "source": "roclark/super-mario-bros-dqn",
                    "environment": "SuperMarioBros-1-3-v0",
                    "teacher_action_repeat": 4, "ram_encoding": "raw",
                }),
                encoding="utf-8",
            )
            output = root / "teacher-decisions.npz"
            metadata = prepare_teacher_decisions(cache, output, value_mode="transition")
            self.assertEqual(metadata["reset_state_rows"], 1)
            with np.load(output, allow_pickle=False) as table:
                np.testing.assert_array_equal(table["frames"], [0, 4, 8])
                np.testing.assert_array_equal(table["y"], [20, 132, 4])
                np.testing.assert_array_equal(table["previous_actions"], [0, 20, 132])
                np.testing.assert_array_equal(table["action_values"], [1, 0, 1])
                np.testing.assert_array_equal(
                    table["X"][:, FEATURE_NAMES.index("desired_action_value")],
                    [1, 0, 1],
                )
                self.assertEqual(table["X"].shape, (3, len(FEATURE_NAMES)))
                self.assertEqual(
                    table["X"][0, FEATURE_NAMES.index("previous_action_right")], 0
                )
                self.assertEqual(
                    table["X"][1, FEATURE_NAMES.index("previous_action_right")], 1
                )

            positive_output = root / "teacher-decisions-positive.npz"
            positive_metadata = prepare_teacher_decisions(
                cache, positive_output, value_mode="trajectory-positive"
            )
            self.assertEqual(
                positive_metadata["action_value_mode"], "trajectory-positive"
            )
            with np.load(positive_output, allow_pickle=False) as table:
                np.testing.assert_array_equal(table["action_values"], [1, 1, 1])
                np.testing.assert_array_equal(
                    table["X"][:, FEATURE_NAMES.index("desired_action_value")],
                    [1, 1, 1],
                )

            judged_output = root / "teacher-decisions-student-judge.npz"
            judged_metadata = prepare_teacher_decisions(
                cache,
                judged_output,
                value_mode="student-judge",
                judge_horizon_frames=8,
                judge_min_progress_delta=3,
            )
            self.assertIsNone(judged_metadata["judge_capacity"])
            self.assertEqual(judged_metadata["judge_horizon_frames"], 8)
            self.assertEqual(judged_metadata["death_lookback_actions"], 4)
            self.assertEqual(judged_metadata["min_progress_delta"], 3)
            with np.load(judged_output, allow_pickle=False) as table:
                # Every action gets its own eight raw frames. Ordinary forward
                # progress reward inside each horizon makes all three positive.
                np.testing.assert_array_equal(table["action_values"], [1, 1, 1])

    def test_teacher_policy_requires_matching_playback_repeat(self):
        class TeacherPolicy:
            manifest = {"context": {"teacher_action_repeat": 4}}

        with self.assertRaisesRegex(ValueError, "action_repeat=4"):
            rollout(None, TeacherPolicy(), action_repeat=1)


class FeatureTests(unittest.TestCase):
    def test_signed_speed_relative_objects_and_tile_pages(self):
        ram = game_ram()
        ram[0x6D], ram[0x86], ram[0xCE] = 1, 8, 80
        ram[0x71A] = 1
        ram[0x57], ram[0x9F] = 255, 128
        ram[0x7F8:0x7FB] = [3, 9, 8]
        ram[0x7A0] = 4
        ram[0x0F], ram[0x16], ram[0x6E], ram[0x87] = 1, 6, 1, 40
        ram[0xB6], ram[0xCF] = 1, 64
        ram[0x5D0 + 16 * 3] = 42
        features = feature_dict(ram)
        self.assertEqual(features["player_speed_x_raw"], -1)
        self.assertEqual(features["player_speed_y_raw"], -128)
        self.assertEqual(features["player_screen_x"], 8)
        self.assertEqual(features["game_timer"], 398)
        self.assertEqual(features["screen_timer"], 4)
        self.assertEqual(features["object_0_dx"], 32)
        self.assertEqual(features["object_0_dy"], -16)
        self.assertEqual(features["tile_row_3_dx_0"], 42)
        self.assertTrue(np.isnan(features["object_1_dx"]))
        self.assertTrue(np.isnan(features["tile_row_3_dx_-3"]))
        self.assertEqual(tile_at(ram, 16, 3), 42)
        self.assertEqual(tile_at(ram, 48, 3), 42)

    def test_controller_and_identifiers_do_not_change_features(self):
        ram = game_ram()
        before = extract_features(ram)
        for address in [0x0A, 0x0B, 0x0C, 0x0D, 0x6FC, 0x6FD, 0x74A, 0x758,
                        0x09, 0x75C, 0x75F, 0x7DD]:
            ram[address] = 255
        np.testing.assert_array_equal(before, extract_features(ram))
        self.assertEqual(len(before), len(FEATURE_NAMES))

    def test_two_frame_window_and_action_value_condition(self):
        previous = game_ram()
        current = game_ram()
        previous[0x86] = 9
        current[0x86] = 10
        values = extract_features(
            current, previous, previous_action=148, action_value=-1
        )
        self.assertEqual(
            len(values),
            len(STATE_FEATURE_NAMES) * 2 + len(ACTION_FEATURE_NAMES) + 1,
        )
        self.assertEqual(values[FEATURE_NAMES.index("previous_player_subtile_x")], 9)
        self.assertEqual(values[FEATURE_NAMES.index("current_player_subtile_x")], 10)
        self.assertEqual(values[FEATURE_NAMES.index("desired_action_value")], -1)
        self.assertEqual(values[FEATURE_NAMES.index("previous_action_A")], 1)
        self.assertEqual(values[FEATURE_NAMES.index("previous_action_B")], 1)
        self.assertEqual(values[FEATURE_NAMES.index("previous_action_right")], 1)
        self.assertEqual(values[FEATURE_NAMES.index("previous_action_left")], 0)
        np.testing.assert_array_equal(
            values,
            extract_features(
                bytes(current),
                bytes(previous),
                previous_action=148,
                action_value=-1,
            ),
        )
        padded = extract_features(current, action_value=1)
        np.testing.assert_array_equal(
            padded[:len(STATE_FEATURE_NAMES)],
            padded[len(STATE_FEATURE_NAMES):2 * len(STATE_FEATURE_NAMES)],
        )
        with self.assertRaisesRegex(ValueError, "action_value"):
            extract_features(current, action_value=2)
        with self.assertRaisesRegex(ValueError, "START/SELECT"):
            extract_features(current, previous_action=1)

    def test_invalid_ram_rejected(self):
        for value in [np.zeros(2047), np.zeros(2048), np.full(2048, 256), np.full(2048, -1)]:
            with self.assertRaises(ValueError):
                extract_features(value)


class DatasetTests(unittest.TestCase):
    def test_progress_success_requires_large_delta_and_new_maximum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            # 0->2 is a small new maximum. 5->8 is a large advance, but it
            # remains below the earlier maximum of 10. Neither is successful.
            positions = [0, 2, 10, 5, 8]
            for number, position in enumerate(positions, 1):
                ram = game_ram()
                ram[0x86] = position
                path = data / f"p_s_e0_1-1_f{number}_a20_date.win.png"
                path.write_bytes(png_bytes(ram, 20))
            output = root / "values.npz"
            metadata = prepare(
                data,
                output,
                stride=1,
                max_rows=10,
                label_offset=1,
                min_progress_delta=3,
            )
            with np.load(output, allow_pickle=False) as table:
                self.assertEqual(table["frames"].tolist(), [1, 2, 3, 4])
                self.assertEqual(table["action_values"].tolist(), [0, 1, 0, 0])
            self.assertEqual(metadata["min_progress_delta"], 3)

    def test_next_frame_alignment_numeric_order_and_no_cross_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            originals = {}
            for episode in [0, 1]:
                for number, action in [(9, 0), (10, 20), (11, 148), (13, 16)]:
                    ram = game_ram()
                    ram[0x86] = number
                    path = data / f"p_s_e{episode}_1-1_f{number}_a{action}_date.win.png"
                    originals[path] = png_bytes(ram, action)
                    path.write_bytes(originals[path])
            output = root / "context.npz"
            prepare(data, output, stride=1, max_rows=100, label_offset=1)
            X, y, meta = load_table(output)
            np.testing.assert_array_equal(y, [20, 148, 20, 148])
            np.testing.assert_array_equal(
                X[:, FEATURE_NAMES.index("current_player_subtile_x")],
                [9, 10, 9, 10],
            )
            np.testing.assert_array_equal(
                X[:, FEATURE_NAMES.index("previous_player_subtile_x")],
                [9, 9, 9, 9],
            )
            np.testing.assert_array_equal(
                X[:, FEATURE_NAMES.index("previous_action_B")],
                [0, 1, 0, 1],
            )
            np.testing.assert_array_equal(
                X[:, FEATURE_NAMES.index("previous_action_right")],
                [0, 1, 0, 1],
            )
            with np.load(output, allow_pickle=False) as table:
                np.testing.assert_array_equal(
                    table["previous_actions"], [0, 20, 0, 20]
                )
            self.assertEqual(meta["trajectory_count"], 2)
            self.assertEqual(meta["skipped"]["missing_target_frame"], 4)
            self.assertEqual(meta["priority_rows"], 4)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                prepare(data, output)

    def test_npz_metadata_cache_matches_png_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for episode in [0, 1]:
                for number, action in [(9, 0), (10, 20), (11, 148)]:
                    ram = game_ram()
                    ram[0x86] = number
                    path = data / f"p_s_e{episode}_1-1_f{number}_a{action}_date.win.png"
                    path.write_bytes(png_bytes(ram, action))
            bad = data / "p_s_e2_1-1_f1_a0_date.fail.png"
            bad.write_bytes(png_bytes(game_ram(), 0, outcome=2))
            direct = root / "direct.npz"
            cached = root / "cached.npz"
            cache = root / "metadata-cache"
            prepare(data, direct, outcome="win", stride=1, max_rows=100, label_offset=1)
            result = build_cache(data, cache, outcome="all", workers=2)
            resumed = build_cache(data, cache, outcome="all", workers=2)
            prepare(cache, cached, outcome="win", stride=1, max_rows=100, label_offset=1)
            direct_x, direct_y, direct_meta = load_table(direct)
            cached_x, cached_y, cached_meta = load_table(cached)
            np.testing.assert_array_equal(cached_x, direct_x)
            np.testing.assert_array_equal(cached_y, direct_y)
            self.assertEqual(result["frames"], 6)
            self.assertEqual(len(result["skipped_trajectories"]), 1)
            self.assertEqual(result["skipped_trajectories"][0]["episode"], "p_s_e2_1-1")
            self.assertEqual(resumed["new_shards"], 0)
            self.assertEqual(cached_meta["source_format"], "npz-cache")
            self.assertEqual(direct_meta["source_format"], "png")
            with np.load(next(cache.glob("*.npz")), allow_pickle=False) as shard:
                self.assertEqual(set(shard.files), {"ram", "actions", "frames", "paths", "metadata"})

    def test_explicit_level_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for level in ("1-1", "8-4"):
                for number, action in ((1, 20), (2, 148)):
                    ram = game_ram()
                    world, stage = level.split("-")
                    path = data / f"p_s_e{world}{stage}_{level}_f{number}_a{action}_date.win.png"
                    path.write_bytes(png_bytes(ram, action))
            output = root / "context.npz"
            metadata = prepare(data, output, stride=1, max_rows=100,
                               label_offset=0, exclude_level="8-4")
            _, _, loaded = load_table(output)
            self.assertEqual(metadata["levels"], ["1-1"])
            self.assertEqual(loaded["excluded_level"], "8-4")
            self.assertNotIn("8-4", loaded["levels"])
            with self.assertRaisesRegex(ValueError, "exclude_level"):
                prepare(data, root / "bad.npz", exclude_level="world-8")

            included = root / "included.npz"
            metadata = prepare(
                data,
                included,
                stride=1,
                max_rows=100,
                label_offset=0,
                include_level="8-4",
            )
            self.assertEqual(metadata["levels"], ["8-4"])
            self.assertEqual(metadata["included_level"], "8-4")
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                prepare(
                    data,
                    root / "both.npz",
                    include_level="1-1",
                    exclude_level="8-4",
                )

    def test_opening_rows_are_preferred_during_trajectory_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for episode in range(2):
                for number in range(1, 21):
                    ram = game_ram()
                    ram[0x86] = number
                    action = 20 if number > 4 else 0
                    path = data / f"p_s_e{episode}_1-1_f{number}_a{action}_date.win.png"
                    path.write_bytes(png_bytes(ram, action))
            output = root / "context.npz"
            metadata = prepare(data, output, stride=1, max_rows=8, label_offset=0,
                               head_rows_per_trajectory=2)
            _, _, loaded = load_table(output)
            with np.load(output, allow_pickle=False) as table:
                selected = list(zip(table["episodes"].tolist(), table["frames"].tolist()))
            for episode in ("p_s_e0_1-1", "p_s_e1_1-1"):
                self.assertIn((episode, 1), selected)
                self.assertIn((episode, 2), selected)
            self.assertEqual(metadata["priority_rows"], 4)
            self.assertEqual(loaded["effective_head_rows_per_trajectory"], 2)

    def test_trajectory_selection_respects_capacity_and_action_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for number in range(1, 11):
                ram = game_ram()
                action = 20 if number <= 7 else 148
                path = data / f"p_s_e0_1-1_f{number}_a{action}_date.win.png"
                path.write_bytes(png_bytes(ram, action))
            output = root / "selected.npz"
            metadata = prepare(
                data,
                output,
                stride=1,
                max_rows=6,
                label_offset=0,
                include_level="1-1",
            )
            self.assertEqual(metadata["selected_rows"], 6)
            self.assertEqual(metadata["selection_discarded_rows"], 4)
            with np.load(output, allow_pickle=False) as table:
                self.assertIn(8, set(map(int, table["frames"])))

    def test_stride_never_drops_action_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for number in range(1, 9):
                ram = game_ram()
                action = 148 if number == 3 else 20
                path = data / f"p_s_e0_1-1_f{number}_a{action}_date.win.png"
                path.write_bytes(png_bytes(ram, action))
            output = root / "context.npz"
            metadata = prepare(
                data,
                output,
                stride=4,
                max_rows=20,
                label_offset=0,
            )
            with np.load(output, allow_pickle=False) as table:
                selected_frames = set(map(int, table["frames"]))
            self.assertIn(3, selected_frames)
            self.assertIn(4, selected_frames)
            self.assertEqual(metadata["mandatory_action_change_rows"], 2)

    def test_per_action_values_are_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            specifications = [("win", 2), ("fail", 1)]
            for outcome, embedded in specifications:
                for episode in range(2):
                    for number in range(1, 9):
                        ram = game_ram()
                        ram[0x86] = 4 if number >= 2 else 1
                        if outcome == "fail" and number == 8:
                            ram[0x0E] = 0x0B
                        action = 20 if number % 2 == 0 else 148
                        path = data / (
                            f"p_s_e{episode}_1-1_f{number}_a{action}_date.{outcome}.png"
                        )
                        path.write_bytes(png_bytes(ram, action, outcome=embedded))
            output = root / "values.npz"
            metadata = prepare(
                data,
                output,
                outcome="all",
                stride=1,
                max_rows=28,
                label_offset=1,
                head_rows_per_trajectory=1,
                pre_death_frames=2,
            )
            with np.load(output, allow_pickle=False) as table:
                values = table["X"][:, FEATURE_NAMES.index("desired_action_value")]
                outcomes = table["outcomes"].tolist()
                stored_values = table["action_values"].tolist()
            self.assertEqual(set(outcomes), {"win", "fail"})
            self.assertEqual(set(values.tolist()), {-1.0, 0.0, 1.0})
            self.assertEqual(values.tolist(), stored_values)
            self.assertEqual(metadata["detected_death_trajectories"], 2)
            self.assertEqual(
                metadata["selection"],
                "mandatory-changes-trajectory-round-robin",
            )


class ActionAndRolloutTests(unittest.TestCase):
    def test_unconditional_epsilon_sampling(self):
        class Model:
            classes_ = np.asarray([0, 20, 148])

            def __init__(self, probabilities):
                self.probabilities = np.asarray([probabilities])

            def predict_proba(self, X):
                self.last_X = X.copy()
                return self.probabilities

        policy = FittedPolicy.__new__(FittedPolicy)
        policy._previous_ram = None
        policy.model = Model([0.40, 0.35, 0.25])
        explored = policy.predict_ram(
            game_ram(),
            selection="epsilon_sample",
            epsilon=1.0,
            rng=np.random.default_rng(3),
        )
        self.assertTrue(explored["explored"])
        self.assertEqual(explored["max_confidence"], 0.4)

        policy.model = Model([0.0, 1.0, 0.0])
        sampled = policy.predict_ram(
            game_ram(),
            previous_action=148,
            selection="epsilon_sample",
            epsilon=0.0,
            rng=np.random.default_rng(3),
        )
        self.assertFalse(sampled["explored"])
        self.assertEqual(sampled["action"], 20)
        self.assertEqual(sampled["previous_action"], 148)
        self.assertEqual(
            policy.model.last_X[0, FEATURE_NAMES.index("previous_action_A")], 1
        )
        self.assertEqual(
            policy.model.last_X[0, FEATURE_NAMES.index("previous_action_B")], 1
        )
        self.assertEqual(
            policy.model.last_X[0, FEATURE_NAMES.index("previous_action_right")], 1
        )

        policy.predict_ram(
            game_ram(), selection="sample", rng=np.random.default_rng(3)
        )
        self.assertEqual(
            policy.model.last_X[0, FEATURE_NAMES.index("previous_action_B")], 1
        )
        self.assertEqual(
            policy.model.last_X[0, FEATURE_NAMES.index("previous_action_right")], 1
        )
        decision = policy.predict_ram(game_ram(), selection="argmax")
        self.assertEqual(decision["selection"], "argmax")
        self.assertEqual(decision["action"], 20)

    def test_action_translation(self):
        self.assertEqual(button_names(148), ["A", "B", "right"])
        self.assertEqual(to_nes_action(148), 131)
        self.assertEqual(to_nes_action(20), 130)
        self.assertEqual(to_nes_action(64), 16)
        self.assertEqual(to_nes_action(2), 32)
        for action in [1, 8, 36, 66, 256, -1]:
            with self.assertRaises(ValueError):
                validate_action(action)

    def test_stops_inside_repeat_and_preserves_ram_snapshot(self):
        class Env:
            unwrapped = property(lambda self: self)
            ram = game_ram()

            def reset(self, seed=None):
                self.actions = []
                return None, {}

            def step(self, action):
                self.actions.append(action)
                return None, 1.0, len(self.actions) == 2, False, {"flag_get": len(self.actions) == 2}

        class Policy:
            def __init__(self):
                self.previous = []
                self.previous_actions = []

            def reset_history(self):
                self.previous.append("reset")

            def predict_ram(self, ram, **kwargs):
                self.previous.append(kwargs["previous_ram"])
                self.previous_actions.append(kwargs["previous_action"])
                return {"action": 148, "predict_seconds": .01, "buttons": ["A", "B", "right"]}

        env = Env()
        policy = Policy()
        class Online:
            def __init__(self):
                self.observations = []

            def begin_episode(self, initial_ram):
                self.initial_ram = initial_ram.copy()

            def observe(self, *args, **kwargs):
                self.observations.append((args, kwargs))

            def end_episode(self, **kwargs):
                return kwargs

            def summary(self):
                return {"observations": len(self.observations)}

        online = Online()
        trace = io.StringIO()
        result = rollout(
            env,
            policy,
            max_frames=20,
            action_repeat=4,
            trace=trace,
            online=online,
        )
        self.assertEqual(env.actions, [131, 131])
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["decisions"], 1)
        self.assertTrue(result["flag_get"])
        self.assertEqual(json.loads(trace.getvalue())["action"], 148)
        self.assertEqual(policy.previous, ["reset", None])
        self.assertEqual(policy.previous_actions, [0])
        self.assertEqual(len(online.observations), 1)
        self.assertEqual(online.observations[0][0][2], 148)
        self.assertEqual(online.observations[0][1]["previous_action"], 0)

    def test_video_receives_initial_and_stepped_frames(self):
        class Env:
            unwrapped = property(lambda self: self)
            ram = game_ram()

            def reset(self, seed=None):
                self.steps = 0
                return None, {}

            def step(self, action):
                self.steps += 1
                return None, 0.0, self.steps == 2, False, {}

            def render(self):
                return np.full((4, 5, 3), self.steps, dtype=np.uint8)

        class Policy:
            def predict_ram(self, ram, **kwargs):
                return {"action": 0, "predict_seconds": 0.0, "buttons": []}

        class Video:
            def __init__(self):
                self.frames = []

            def write(self, frame):
                self.frames.append(frame.copy())

        video = Video()
        result = rollout(Env(), Policy(), max_frames=5, video=video)
        self.assertEqual(result["frames"], 2)
        self.assertEqual([int(frame[0, 0, 0]) for frame in video.frames], [0, 1, 2])


class OnlineReplayTests(unittest.TestCase):
    def test_cache_write_retries_transient_windows_file_lock(self):
        class Policy:
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "online.npz"
            replay = OnlineReplay(Policy(), self._context(root), cache)
            real_replace = os.replace
            attempts = []

            def replace(source, target):
                attempts.append((source, target))
                if len(attempts) == 1:
                    raise PermissionError("temporary file lock")
                real_replace(source, target)

            with patch("tfm4mario.online.os.replace", side_effect=replace), patch(
                "tfm4mario.online.time.sleep"
            ):
                replay._write()
            self.assertEqual(len(attempts), 2)
            with np.load(cache, allow_pickle=False) as saved:
                self.assertEqual(len(saved["y"]), 0)

    def _context(self, root):
        data = root / "data"
        data.mkdir()
        for number, action in ((1, 20), (2, 148), (3, 20), (4, 148)):
            ram = game_ram()
            ram[0x86] = number
            path = data / f"p_s_e0_1-1_f{number}_a{action}_date.win.png"
            path.write_bytes(png_bytes(ram, action))
        context = root / "context.npz"
        prepare(data, context, stride=1, max_rows=10, label_offset=0)
        return context

    def test_full_cache_gets_one_delayed_value_without_mid_episode_refit(self):
        class Policy:
            def __init__(self):
                self.refits = []

            def refit_context(self, X, y):
                self.refits.append((X.copy(), y.copy()))
                return 0.5

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root)
            policy = Policy()
            cache = root / "online.npz"
            replay = OnlineReplay(
                policy, context, cache, capacity=3, min_progress_delta=3
            )
            initial = game_ram()
            replay.begin_episode(initial)
            current = initial.copy()
            flushed = None
            previous_action = 0
            for index, action in enumerate((0, 16, 20), 1):
                after = current.copy()
                after[0x86] = index
                flushed = replay.observe(
                    current,
                    current,
                    action,
                    after,
                    previous_action=previous_action,
                )
                current = after
                previous_action = action

            self.assertEqual(flushed["rows"], 1)
            self.assertEqual(flushed["assigned_action_value"], 1)
            self.assertEqual(replay.summary()["pending_rows"], 2)
            self.assertEqual(replay.summary()["accumulated_context_rows"], 1)
            self.assertEqual(policy.refits, [])
            with np.load(cache, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["action_values"], [1])
                np.testing.assert_array_equal(
                    saved["X"][:, FEATURE_NAMES.index("previous_action_B")],
                    [0],
                )
                metadata = json.loads(str(saved["metadata"]))
                self.assertEqual(metadata["min_progress_delta"], 3)

            replay.observe(current, current, 16, current)
            replay.observe(current, current, 0, current)
            self.assertEqual(replay.summary()["pending_rows"], 2)
            self.assertEqual(policy.refits, [])
            update = replay.end_episode(success=True)
            self.assertTrue(update["refit_performed"])
            self.assertEqual(len(policy.refits), 1)
            self.assertEqual(update["flushed_batches"], 4)
            self.assertEqual(update["accumulated_context_rows"], 5)
            np.testing.assert_array_equal(
                policy.refits[0][1][-5:], [0, 16, 20, 16, 0]
            )
            with np.load(cache, allow_pickle=False) as saved:
                np.testing.assert_array_equal(
                    saved["action_values"], [1, 0, 0, 0, 0]
                )

    def test_online_progress_must_reach_configured_milestone_delta(self):
        class Policy:
            def refit_context(self, X, y):
                return 0.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root)
            replay = OnlineReplay(
                Policy(),
                context,
                root / "online.npz",
                capacity=2,
                min_progress_delta=3,
            )
            initial = game_ram()
            replay.begin_episode(initial)
            one_pixel = initial.copy()
            one_pixel[0x86] += 1
            replay.observe(initial, initial, 20, one_pixel)
            two_pixels = initial.copy()
            two_pixels[0x86] += 2
            update = replay.observe(one_pixel, one_pixel, 20, two_pixels)
            self.assertEqual(update["assigned_action_value"], 0)

            three_pixels = initial.copy()
            three_pixels[0x86] += 3
            replay.observe(two_pixels, two_pixels, 20, three_pixels)
            five_pixels = initial.copy()
            five_pixels[0x86] += 5
            update = replay.observe(three_pixels, three_pixels, 20, five_pixels)
            self.assertEqual(update["assigned_action_value"], 1)

    def test_any_positive_reward_in_actions_horizon_is_positive(self):
        class Policy:
            def refit_context(self, X, y):
                return 0.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = OnlineReplay(
                Policy(), self._context(root), root / "online.npz", capacity=2
            )
            state = game_ram()
            replay.begin_episode(state)
            replay.observe(state, state, 0, state)
            update = replay.observe(
                state, state, 4, state, positive_reward=True
            )
            self.assertEqual(update["assigned_action_value"], 1)
            with np.load(root / "online.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["action_values"], [1])

    def test_online_progress_is_relative_to_each_actions_start(self):
        class Policy:
            def refit_context(self, X, y):
                return 0.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = OnlineReplay(
                Policy(),
                self._context(root),
                root / "online.npz",
                capacity=2,
                min_progress_delta=2,
            )
            initial = game_ram()
            replay.begin_episode(initial)
            five = initial.copy()
            five[0x86] += 5
            three = initial.copy()
            three[0x86] += 3
            replay.observe(initial, initial, 20, five)
            update = replay.observe(five, five, 20, three)
            self.assertEqual(update["assigned_action_value"], 1)

            four = initial.copy()
            four[0x86] += 4
            update = replay.observe(three, three, 20, four)
            self.assertEqual(update["assigned_action_value"], 0)
            update = replay.observe(four, four, 20, five)
            self.assertEqual(update["assigned_action_value"], 1)

    def test_partial_cache_is_marked_at_episode_end_and_persists(self):
        class Policy:
            def __init__(self):
                self.refits = []

            def refit_context(self, X, y):
                self.refits.append((X.copy(), y.copy()))
                return 1.25

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root)
            policy = Policy()
            cache = root / "online.npz"
            replay = OnlineReplay(
                policy, context, cache, capacity=16, death_lookback_actions=4
            )
            initial = game_ram()
            replay.begin_episode(initial)
            for action in (0, 4, 20, 32, 128):
                replay.observe(initial, initial, action, initial)
            replay.observe(initial, initial, 148, initial, death=True)
            self.assertEqual(policy.refits, [])
            update = replay.end_episode(death=True)
            self.assertEqual(len(policy.refits), 1)
            self.assertEqual(update["accumulated_context_rows"], 4)
            self.assertEqual(update["action_value_counts"], {-1: 4})
            with np.load(cache, allow_pickle=False) as saved:
                self.assertEqual(saved["X"].shape, (4, len(FEATURE_NAMES)))
                np.testing.assert_array_equal(saved["y"], [20, 32, 128, 148])
                np.testing.assert_array_equal(saved["action_values"], [-1] * 4)

            restored_policy = Policy()
            restored = OnlineReplay(
                restored_policy,
                context,
                cache,
                capacity=16,
                death_lookback_actions=4,
            )
            self.assertEqual(len(restored_policy.refits), 1)
            self.assertEqual(restored.summary()["accumulated_context_rows"], 4)

    def test_context_is_not_refit_after_final_episode(self):
        class Policy:
            def __init__(self):
                self.refits = []

            def refit_context(self, X, y):
                self.refits.append((X.copy(), y.copy()))
                return 1.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root)
            policy = Policy()
            replay = OnlineReplay(policy, context, root / "online.npz", capacity=2)
            initial = game_ram()
            replay.begin_episode(initial)
            replay.observe(initial, initial, 20, initial)
            update = replay.end_episode(success=True, refit=False)
            self.assertFalse(update["refit_performed"])
            self.assertEqual(policy.refits, [])
            self.assertEqual(update["accumulated_context_rows"], 1)


if __name__ == "__main__":
    unittest.main()
