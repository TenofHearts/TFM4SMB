"""Keep dataset actions distinct from nes-py's controller bit mask."""

# Dataset README, corrected by the author in issue #2. START/SELECT were
# explicitly uncertain in that discussion, so reject either during gameplay.
BUTTONS = {"A": 128, "up": 64, "left": 32, "B": 16, "right": 4, "down": 2}
NES_BITS = {"A": 1, "B": 2, "up": 16, "down": 32, "left": 64, "right": 128}


def validate_action(action: int) -> int:
    action = int(action)
    if not 0 <= action <= 255:
        raise ValueError(f"Action outside byte range: {action}")
    if action & 9:
        raise ValueError(f"START/SELECT action is not a gameplay label: {action}")
    if action & 36 == 36 or action & 66 == 66:
        raise ValueError(f"Opposing directions in action: {action}")
    return action


def button_names(action: int) -> list[str]:
    validate_action(action)
    return [name for name, bit in BUTTONS.items() if action & bit]


def to_nes_action(action: int) -> int:
    return sum(NES_BITS[name] for name in button_names(action))
