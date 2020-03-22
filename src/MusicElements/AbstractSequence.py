from abc import ABC, abstractmethod
from mido import MidiTrack


class AbstractSequence(ABC):

    def __init__(self, numerator=4, denominator=4, name="Track"):
        self.elements = []
        self.numerator = numerator
        self.denominator = denominator
        self.name = name

    def __str__(self) -> str:
        return "(" + self.name + ", Numerator: " + str(self.numerator) + ", Denominator: " + str(
        self.denominator) + ", Elements: " + str(self.elements) + ")"
