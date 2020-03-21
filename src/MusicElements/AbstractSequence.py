from abc import ABC, abstractmethod
from mido import MidiTrack


class AbstractSequence(ABC):

    def __init__(self, numerator=4, denominator=4):
        self.elements = []
        self.numerator = numerator
        self.denominator = denominator

    def __str__(self) -> str:
        return "(Numerator: " + str(self.numerator) + ", Denominator: " + str(
            self.denominator) + ", Elements: " + str(self.elements) + ")"
