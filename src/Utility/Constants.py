from enum import Enum

RIGHT_HAND = 0
LEFT_HAND = 1

std_velocity = 64
internal_ticks = 24
external_ticks = 480


class Difficulty(Enum):
    EASY = 1,
    MEDIUM = 2,
    HARD = 3
