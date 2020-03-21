from __future__ import annotations

from mido import MidiTrack, Message

from src.MusicElements import *


class SequenceRelative(AbstractSequence):

    def __init__(self, numerator=4, denominator=4):
        super().__init__(numerator, denominator)

    @staticmethod
    def from_midi_track(midi_track: MidiTrack, modifier: float = internal_ticks / external_ticks) -> SequenceRelative:
        sequence = SequenceRelative()
        wait_buffer = 0

        for message in midi_track:
            # Generate wait messages
            if message.type == "note_on" or message.type == "note_off" or message.type == "control_change":
                wait_buffer += message.time * modifier
                if wait_buffer != 0 and message.type != "control_change":
                    sequence.elements.append(Element(MessageType.wait, wait_buffer, std_velocity))
                    wait_buffer = 0

            # Generate play and stop messages
            if message.type == "note_on":
                if message.velocity > 0:
                    sequence.elements.append(Element(MessageType.play, message.note, message.velocity))
                else:
                    sequence.elements.append(Element(MessageType.stop, message.note, message.velocity))
            elif message.type == "note_off":
                sequence.elements.append(Element(MessageType.stop, message.note, message.velocity))

        return sequence

    def to_midi_track(self, modifier: float = external_ticks / internal_ticks) -> MidiTrack:
        track = MidiTrack()
        wait_buffer = 0

        for element in self.elements:
            if element.message_type == MessageType.wait:
                wait_buffer += element.value
            elif element.message_type == MessageType.play:
                track.append(
                    Message("note_on", note=element.value, velocity=element.velocity, time=int(wait_buffer * modifier)))
                wait_buffer = 0
            elif element.message_type == MessageType.stop:
                track.append(
                    Message("note_off", note=element.value, velocity=element.velocity,
                            time=int(wait_buffer * modifier)))
                wait_buffer = 0

        return track

    def to_absolute_sequence(self) -> SequenceAbsolute:
        elements = {}
        seq_absolute = SequenceAbsolute(self.numerator, self.denominator)
        wait = 0

        for element in self.elements:
            if element.message_type == MessageType.wait:
                wait += element.value
            else:
                elements.update({element: wait})

        seq_absolute.elements = sorted(elements.items(), key=lambda item: item[1])
        return seq_absolute

    def transpose(self, steps: int) -> SequenceRelative:
        for i, element in enumerate(self.elements):
            if element.message_type != MessageType.play and element.message_type != MessageType.stop:
                continue
            self.elements.pop(i)
            element.value += steps
            while element.value < 21:
                element.value += 12
            while element.value > 108:
                element.value -= 12
            self.elements.insert(i, element)
        return self

    @staticmethod
    def average_complexity(sequences: list):
        complexity = list()
        occurrences = 0
        for sequence in sequences:
            complexity.append(sequence.complexity_breakdown())
            occurrences += 1

        if occurrences == 0:
            occurrences = 1
        avg_complexity = sum([pair[0] for pair in complexity]) / occurrences
        dict = {"Note Values": sum([pair[1]["Note Values"] for pair in complexity]) / occurrences,
                "Note Classes": sum([pair[1]["Note Classes"] for pair in complexity]) / occurrences,
                "Concurrent Notes": sum([pair[1]["Concurrent Notes"] for pair in complexity]) / occurrences}
        return avg_complexity, dict

    def complexity(self):
        return self.complexity_breakdown()[0]

    def complexity_breakdown(self):
        complex_note_values = self.__complexity_note_values()
        weight_note_values = 7
        complex_note_classes = self.__complexity_note_classes()
        weight_note_classes = 6
        complex_concurrent_notes = self.__complexity_concurrent_notes()
        weight_concurrent_notes = 3

        weight_sum = weight_note_values + weight_note_classes + weight_concurrent_notes
        complexity = (weight_note_values / weight_sum * complex_note_values) + \
                     (weight_note_classes / weight_sum * complex_note_classes) + \
                     (weight_concurrent_notes / weight_sum * complex_concurrent_notes)

        dict = {"Note Values": complex_note_values,
                "Note Classes": complex_note_classes,
                "Concurrent Notes": complex_concurrent_notes}

        return complexity, dict

    def __complexity_note_values(self):
        time = 0
        occurrences = 0

        for element in self.elements:
            if element.message_type == MessageType.wait:
                time += element.value
                occurrences += 1
        average_time = time / occurrences if occurrences else time

        x = average_time
        value = 6 - 0.1 * x + 8.5E-4 * x ** 2 - 2.425E-6 * x ** 3
        return self.util_adjust_rating(value)

    def __complexity_note_classes(self):
        classes = set()

        for element in self.elements:
            if element.message_type == MessageType.play:
                classes.add(element.value)

        x = len(classes)
        value = 1 / 3 * x + 1 / 3
        return self.util_adjust_rating(value)

    def __complexity_note_amount(self):
        amount = 0
        for element in self.elements:
            if element.message_type == MessageType.play:
                amount += 1

        x = amount
        value = -0.1 + 3.25E-1 * x - 1.0E-2 * x ** 2 + 1.5E-4 * x ** 3
        return self.util_adjust_rating(value)

    def __complexity_concurrent_notes(self):
        notes = 0
        occurrences = 0
        last_element = None

        for element in self.elements:
            if element.message_type == MessageType.play:
                notes += 1
                if last_element is None or not last_element.message_type == MessageType.play:
                    occurrences += 1
            last_element = element

        x = notes / occurrences if occurrences else notes
        value = 5 - 1.65E1 * x + 1.925E1 * x ** 2 - 8 * x ** 3 + 1.15 * x ** 4
        value *= self.__complexity_note_amount() / 3
        return self.util_adjust_rating(value)

    @staticmethod
    def util_adjust_rating(value: float):
        return min(5, max(1, value))

    def adjust(self) -> SequenceRelative:
        active_notes = set()
        elements_adjusted = []

        for i, element in enumerate(self.elements):
            if element.message_type == MessageType.wait:
                if element.value > internal_ticks:
                    for j in range(0, int(element.value // internal_ticks)):
                        elements_adjusted.append((Element(MessageType.wait, internal_ticks, std_velocity)))
                    if element.value % internal_ticks > 0:
                        elements_adjusted.append(
                            Element(MessageType.wait, int(element.value % internal_ticks), std_velocity))
                else:
                    elements_adjusted.append(Element(MessageType.wait, int(element.value), element.velocity))
            elif element.message_type == MessageType.stop:
                if element.value not in active_notes:
                    continue
                else:
                    elements_adjusted.append(element)
                    active_notes.remove(element.value)
            elif element.message_type == MessageType.play:
                if element.value in active_notes:
                    continue
                else:
                    elements_adjusted.append(element)
                    active_notes.add(element.value)

        for value in active_notes:
            elements_adjusted.append(Element(MessageType.stop, value, std_velocity))

        self.elements = elements_adjusted
        return self

    def util_first_notes(self, steps: int) -> list:
        first_notes = list()

        for element in self.elements:
            if element.message_type == MessageType.wait and len(first_notes) >= steps:
                return first_notes
            if element.message_type == MessageType.play:
                first_notes.append(Note.from_note_value(element.value % 12))

        return first_notes

    def util_last_notes(self, steps: int) -> list:
        last_notes = list()

        elements = self.elements.copy()
        elements.reverse()
        for element in elements:
            if element.message_type == MessageType.wait and len(last_notes) >= steps:
                return last_notes
            if element.message_type == MessageType.play:
                last_notes.append(Note.from_note_value(element.value % 12))

        return last_notes
