from enum import Enum

RIGHT_HAND = 0
LEFT_HAND = 1

START_WORD = 201
END_WORD = 202
PADDING = 0

std_velocity = 64
internal_ticks = 24
external_ticks = 480


class Complexity(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __add__(self, other):
        return self.value + other.value

    def __str__(self):
        return self.name
