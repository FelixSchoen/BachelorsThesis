from __future__ import annotations
from src.MusicElements import *
from src.Utility import *


class SequenceAbsolute(AbstractSequence):

    def __init__(self, numerator=4, denominator=4):
        super().__init__(numerator, denominator)

    def to_relative_sequence(self) -> SequenceRelative:
        from src.MusicElements.SequenceRelative import SequenceRelative
        seq_relative = SequenceRelative(self.numerator, self.denominator)
        wait = 0

        for element in sorted(self.elements, key=lambda item: item[1]):
            if element[1] > wait:
                seq_relative.elements.append(Element(MessageType.wait, element[1] - wait, std_velocity))
                wait = element[1]
            seq_relative.elements.append(element[0])

        return seq_relative

    def quantize(self) -> SequenceAbsolute:
        quantized_elements = []

        for element in self.elements:
            quantized_elements.append((element[0], self.quantize_value(element[1])))

        self.elements = quantized_elements
        return self

    @staticmethod
    def quantize_value(wait: float) -> int:
        unit_normal = internal_ticks / 8
        unit_triplet = unit_normal * 2 / 3

        # Steps are used to track position of the notes
        step_normal = (wait // unit_normal) % 2
        wait_quantized = unit_normal * (wait // unit_normal)

        distance = (2 - step_normal) * unit_normal - (2 - step_normal) * unit_triplet
        remainder = unit_normal - distance

        if wait_quantized + distance > wait:
            if wait_quantized + distance / 2 >= wait:
                return int(wait_quantized)
            else:
                return int(wait_quantized + distance)
        else:
            if wait_quantized + distance + remainder / 2 > wait:
                return int(wait_quantized + distance)
            else:
                return int(wait_quantized + unit_normal)

    def merge(self, sequence: SequenceAbsolute) -> SequenceAbsolute:
        merged_elements = []

        i = 0
        j = 0

        while i < len(self.elements) or j < len(sequence.elements):
            el_this = self.elements[i] if i < len(self.elements) else None
            el_that = sequence.elements[j] if j < len(sequence.elements) else None

            if el_that is None or (el_this is not None and el_this[1] < el_that[1]):
                merged_elements.append(el_this)
                i += 1
            elif el_this is None or (el_that is not None and el_that[1] < el_this[1]):
                merged_elements.append(el_that)
                j += 1
            else:
                if el_that[0].message_type == MessageType.stop and el_this[0].message_type != MessageType.stop:
                    merged_elements.append(el_that)
                    j += 1
                else:
                    merged_elements.append(el_this)
                    i += 1

        self.elements = merged_elements
        return self

    def cutoff(self, force: bool=False, max_val: int = 24) -> SequenceAbsolute:
        '''
        Inserts a stop message for every played message max_val time units after being played. This ensures that no
        notes stretch for too long.
        :param max_val: Max time units for any given note
        '''
        elements_to_merge = {}

        for entry in sorted(self.elements, key=lambda item: item[1]):
            element = entry[0]
            if element.message_type == MessageType.play:
                if force:
                    elements_to_merge.update(
                        {Element(MessageType.stop, element.value, element.velocity): entry[1] + max_val})
                else:
                    if not any(elem.message_type == MessageType.stop and elem.value == element.value and time > entry[1] for elem, time in self.elements):
                        elements_to_merge.update(
                            {Element(MessageType.stop, element.value, element.velocity): entry[1] + max_val})

        sequence_to_merge = SequenceAbsolute(self.numerator, self.denominator)
        sequence_to_merge.elements = sorted(elements_to_merge.items(), key=lambda item: item[1])
        self.merge(sequence_to_merge)
        return self
