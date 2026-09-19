"""Contract tests use synthetic RAM and no weights, network, or emulator."""

import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from tfm4mario.actions import button_names, to_nes_action, validate_action
from tfm4mario.dataset import load_table, prepare
from tfm4mario.features import FEATURE_NAMES, extract_features, feature_dict, tile_at
from tfm4mario.game import rollout
from tfm4mario.ram import PNG_SIGNATURE, decode_ram, parse_frame, read_frame
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
            with self.assertRaisesRegex(ValueError, "mismatch"):
                read_frame(parse_frame(path))


class ConfigTests(unittest.TestCase):
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
            for content in ['[prepare]\nmax_rows=0', '[train]\nn_estimator=1',
                            '[play]\nrender="false"', '[prepare]\nlabel_offset=2']:
                config.write_text(content)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["doctor", "--config", str(config)])


class FeatureTests(unittest.TestCase):
    def test_signed_speed_relative_objects_and_tile_pages(self):
        ram = game_ram()
        ram[0x6D], ram[0x86], ram[0xCE] = 1, 8, 80
        ram[0x71A] = 1
        ram[0x57], ram[0x9F] = 255, 128
        ram[0x0F], ram[0x16], ram[0x6E], ram[0x87] = 1, 6, 1, 40
        ram[0xB6], ram[0xCF] = 1, 64
        ram[0x5D0 + 16 * 3] = 42
        features = feature_dict(ram)
        self.assertEqual(features["player_speed_x_raw"], -1)
        self.assertEqual(features["player_speed_y_raw"], -128)
        self.assertEqual(features["player_screen_x"], 8)
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
                        0x09, 0x75C, 0x75F, 0x7DD, 0x7F8]:
            ram[address] = 255
        np.testing.assert_array_equal(before, extract_features(ram))
        self.assertEqual(len(before), len(FEATURE_NAMES))

    def test_invalid_ram_rejected(self):
        for value in [np.zeros(2047), np.zeros(2048), np.full(2048, 256), np.full(2048, -1)]:
            with self.assertRaises(ValueError):
                extract_features(value)


class DatasetTests(unittest.TestCase):
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
            np.testing.assert_array_equal(X[:, FEATURE_NAMES.index("player_subtile_x")], [9, 10, 9, 10])
            self.assertEqual(meta["trajectory_count"], 2)
            self.assertEqual(meta["skipped"]["missing_target_frame"], 4)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                prepare(data, output)


class ActionAndRolloutTests(unittest.TestCase):
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
            def predict_ram(self, ram):
                return {"action": 148, "predict_seconds": .01, "buttons": ["A", "B", "right"]}

        env = Env()
        trace = io.StringIO()
        result = rollout(env, Policy(), max_frames=20, action_repeat=4, trace=trace)
        self.assertEqual(env.actions, [131, 131])
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["decisions"], 1)
        self.assertTrue(result["flag_get"])
        self.assertEqual(json.loads(trace.getvalue())["action"], 148)


if __name__ == "__main__":
    unittest.main()
