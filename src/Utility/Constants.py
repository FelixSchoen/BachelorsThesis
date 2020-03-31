from enum import Enum

RIGHT_HAND = 0
LEFT_HAND = 1

std_velocity = 64
internal_ticks = 24
external_ticks = 480

class Difficulty(Enum):
    NOVICE = 1,
    APPRENTICE = 2,
    ADEPT = 3,
    EXPERT = 4,
    MASTER = 5
