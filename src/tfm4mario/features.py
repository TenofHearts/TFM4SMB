"""Named state features shared by dataset preparation and live inference.

Addresses follow doppelganger's SMB1 disassembly. No controller registers,
outcomes, frame numbers, score, world/level IDs, or absolute level progress.
"""

import numpy as np

SCHEMA = "smb1-semantic-ram-v2"
TILE_COLUMNS = range(-3, 8)
TILE_ROWS = range(13)


def checked_ram(ram) -> np.ndarray:
    if isinstance(ram, (bytes, bytearray, memoryview)):
        ram = np.frombuffer(ram, dtype=np.uint8)
    ram = np.asarray(ram)
    if ram.shape != (2048,) or not np.issubdtype(ram.dtype, np.integer):
        raise ValueError("RAM must be a one-dimensional array of 2048 integer bytes")
    if np.any(ram < 0) or np.any(ram > 255):
        raise ValueError("RAM values must be in [0, 255]")
    return ram.astype(np.int32)


def signed(value: int) -> int:
    return int(value) if value < 128 else int(value) - 256


def tile_at(ram: np.ndarray, column: int, row: int) -> float:
    if column < 0 or not 0 <= row < 13:
        return float("nan")
    # Two alternating 16-column pages, each with 13 rows of metatile IDs.
    address = 0x500 + ((column // 16) % 2) * 0xD0 + row * 16 + column % 16
    return float(ram[address])


def feature_dict(ram) -> dict[str, float]:
    r = checked_ram(ram)
    px = int(r[0x6D]) * 256 + int(r[0x86])
    py = (int(r[0xB5]) - 1) * 256 + int(r[0xCE])
    camera_x = int(r[0x71A]) * 256 + int(r[0x71C])
    features = {
        "player_screen_x": px - camera_x,
        "player_y": py,
        "player_subtile_x": px % 16,
        "player_subtile_y": py % 16,
        "player_speed_x_raw": signed(r[0x57]),
        "player_speed_y_raw": signed(r[0x9F]),
        "player_state": r[0x1D],
        "player_facing": r[0x33],
        "player_moving_direction": r[0x45],
        "player_size": r[0x754],
        "player_powerup": r[0x756],
        "swimming": r[0x704],
        "crouching": r[0x714],
        "collision_bits": r[0x490],
        "area_type": r[0x74E],
        "injury_timer": r[0x79E],
        "star_timer": r[0x79F],
        "game_timer": r[0x7F8] * 100 + r[0x7F9] * 10 + r[0x7FA],
        "screen_timer": r[0x7A0],
    }
    # Six object slots; high-bit flags alias other slots and are not independent
    # enemies. Keep active objects ordered by distance rather than slot number.
    enemies = []
    for slot in range(6):
        flag = int(r[0x0F + slot])
        if flag == 0 or flag & 0x80:
            continue
        ex = int(r[0x6E + slot]) * 256 + int(r[0x87 + slot])
        ey = (int(r[0xB6 + slot]) - 1) * 256 + int(r[0xCF + slot])
        dx, dy = ex - px, ey - py
        enemies.append(
            (
                dx * dx + dy * dy,
                slot,
                {
                    "present": 1,
                    "type": r[0x16 + slot],
                    "state": r[0x1E + slot],
                    "dx": dx,
                    "dy": dy,
                    "speed_x_raw": signed(r[0x58 + slot]),
                    "speed_y_raw": signed(r[0xA0 + slot]),
                },
            )
        )
    enemies.sort(key=lambda item: (item[0], item[1]))
    for rank in range(6):
        values = (
            enemies[rank][2]
            if rank < len(enemies)
            else {
                "present": 0,
                "type": np.nan,
                "state": np.nan,
                "dx": np.nan,
                "dy": np.nan,
                "speed_x_raw": np.nan,
                "speed_y_raw": np.nan,
            }
        )
        features.update(
            {f"object_{rank}_{key}": value for key, value in values.items()}
        )
    # Full vertical range, player-relative horizontal columns. Retain metatile
    # IDs rather than incorrectly equating every nonzero tile with solid ground.
    for row in TILE_ROWS:
        for dx in TILE_COLUMNS:
            column = px // 16 + dx
            # Outside the visible viewport the circular buffer may be stale.
            visible = column * 16 + 15 >= camera_x and column * 16 < camera_x + 256
            features[f"tile_row_{row}_dx_{dx}"] = (
                tile_at(r, column, row) if visible else np.nan
            )
    return {key: float(value) for key, value in features.items()}


FEATURE_NAMES = tuple(feature_dict(np.zeros(2048, dtype=np.uint8)))
CATEGORICAL_INDICES = tuple(
    i
    for i, name in enumerate(FEATURE_NAMES)
    if name.startswith("tile_")
    or name.endswith(("_type", "_state", "_facing", "_direction", "_size", "_powerup"))
    or name in {"swimming", "crouching", "collision_bits"}
)


def extract_features(ram) -> np.ndarray:
    values = feature_dict(ram)
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
