from __future__ import annotations
from mido import MidiTrack, Message
from src.MusicElements import *
from src.Utility import *
import re
import math


class SequenceRelative(AbstractSequence, Persistable):

    def __init__(self, numerator=4, denominator=4):
        super().__init__(numerator, denominator)

    # Conversion Related Functions

    @staticmethod
    def from_midi_track(midi_track: MidiTrack, modifier: float = internal_ticks / external_ticks) -> SequenceRelative:
        sequence = SequenceRelative()
        wait_buffer = 0

        for message in midi_track:
            if message.type == "track_name":
                sequence.name = message.name

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

    @staticmethod
    def from_neural_representation(representation: list[int], numerator: int = 4, denominator: int = 4):
        sequence = SequenceRelative(numerator=numerator, denominator=denominator)
        for value in representation:
            if value == Constants.START_WORD or value == Constants.END_WORD or value == Constants.PADDING:
                continue
            sequence.elements.append(Element.from_neuron_representation(value))
        return sequence

    def to_neural_representation(self) -> list[int]:
        representation = []
        for element in self.elements:
            representation.append(element.to_neuron_representation())
        return representation

    # Transformative Functions

    def adjust(self) -> SequenceRelative:
        """
        Adjusts the Sequence to be a valid midi representation. Checks if all opened notes are closed at the end. Does
        not allow for closing or opening the same note without the respectively other operation. Sorts play and stop
        operations that occur at the same time
        :return:
        """
        active_notes = set()
        elements_adjusted = []
        wait_buffer = 0
        note_buffer = []
        total_wait_time = 0

        for i, element in enumerate(self.elements):
            # Consolidates wait messages and split into equal spaced parts of max value of internal_ticks, append notes
            # played at the same point in time, sorted by their note value
            if element.message_type == MessageType.wait:
                elements_adjusted.extend(map(lambda x: x.element, sorted(note_buffer)))
                note_buffer = []
                wait_buffer += element.value
            else:
                elements_adjusted.extend(self.ut_generate_wait_message(wait_buffer))
                total_wait_time += wait_buffer
                wait_buffer = 0

            # Check if action is legal, if so, add or remove note from list of open notes and append to notes played at
            # the same time. The latter list is used to guarantee an order to the final string
            if element.message_type == MessageType.stop:
                if element.value not in active_notes:
                    continue
                else:
                    note_buffer.append(WrappedElement(element))
                    active_notes.remove(element.value)
            elif element.message_type == MessageType.play:
                if element.value in active_notes:
                    continue
                else:
                    note_buffer.append(WrappedElement(element))
                    active_notes.add(element.value)

        # Append last wait
        elements_adjusted.extend(self.ut_generate_wait_message(wait_buffer))
        total_wait_time += wait_buffer
        # Append last notes
        elements_adjusted.extend(map(lambda x: x.element, note_buffer))

        # Stop still active notes
        for value in active_notes:
            elements_adjusted.append(Element(MessageType.stop, value, std_velocity))

        self.elements = elements_adjusted

        # Fill bar
        if len(self.elements) > 0 and self.elements[-1].message_type == MessageType.wait:
            wait_message = self.elements.pop(-1)
            total_wait_time -= wait_message.value
        ticks_in_bar = 4 * internal_ticks * self.numerator / self.denominator
        remaining = (ticks_in_bar - (total_wait_time % ticks_in_bar)) % ticks_in_bar
        if len(self.elements) == 0:
            remaining = ticks_in_bar
        self.elements.extend(self.ut_generate_wait_message(int(remaining)))

        return self

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

    def stitch(self, sequence) -> SequenceRelative:
        self.elements.extend(sequence.elements)
        return self

    def split(self, capacity: int) -> tuple[SequenceRelative, SequenceRelative]:
        # Queue for elements to cover
        initial_queue = self.elements.copy()
        # Queue for elements to carry to tail sequence (e.g. (play, stop, wait) carries (play))
        carry_queue = list()

        # Head contains elements of length = capacity
        seq_head = SequenceRelative()
        # Tail contains all other elements
        seq_tail = None

        seq = seq_head

        # Duration that has passed up to now
        duration = 0
        # Determines if carry queue is to be used (e.g. switch to new section)
        flag_carry = False
        # Stores open notes that have to be replayed at the start of a new bar
        open_notes = set()

        while len(initial_queue) != 0 or len(carry_queue) != 0:
            if flag_carry and seq_tail is None:
                # Second sequence is not empty
                seq_tail = SequenceRelative()
                seq = seq_tail
            if len(carry_queue) != 0 and flag_carry:
                element = carry_queue.pop(0)
            else:
                flag_carry = False
                element = initial_queue.pop(0)

            if element.message_type == MessageType.play:
                if duration < capacity or capacity == -1:
                    # Can play note
                    seq.elements.append(element)
                    open_notes.add(WrappedElement(element))
                else:
                    # Carry to tail
                    carry_queue.append(element)
            elif element.message_type == MessageType.stop:
                seq.elements.append(element)
                if WrappedElement(element) in open_notes:
                    open_notes.remove(WrappedElement(element))
            elif element.message_type == MessageType.wait:
                if duration + element.value <= capacity or capacity == -1:
                    # Fits in its entirety
                    duration += element.value
                    seq.elements.append(element)
                else:
                    # Wait does not fit entirely
                    fit_duration = capacity - duration
                    remainder_duration = element.value - fit_duration
                    duration += fit_duration
                    if fit_duration > 0:
                        seq.elements.append(Element(MessageType.wait, int(fit_duration), element.velocity))
                    carry_queue.append(Element(MessageType.wait, int(remainder_duration), element.velocity))
                    # Add open notes to carry queue
                    for wrapped_element in sorted(open_notes, reverse=True):
                        carry_queue.insert(0, wrapped_element.element)
                    flag_carry = True
                    capacity = -1

        return seq_head, seq_tail

    def split_to_bars(self) -> list[SequenceRelative]:
        sequences = []
        split_capacity = internal_ticks * 4 * (self.numerator / self.denominator)
        split = self.split(split_capacity)

        while split[1] is not None:
            sequences.append(split[0].adjust())
            split = split[1].split(split_capacity)

        if split[0] is not None:
            sequences.append(split[0])

        return sequences

    # Information Retrieval Functions

    def is_empty(self) -> bool:
        for element in self.elements:
            if element.message_type == MessageType.play or element.message_type == MessageType.stop:
                return False
        return True

    def to_neuron_representation(self) -> list[int]:
        representation = []
        for element in self.elements:
            representation.append(element.to_neuron_representation())
        return representation

    # Complexity Related Functions

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

    def complexity(self) -> Complexity:
        if self.is_empty():
            return Complexity.EASY

        complexity_note_values = self.__complexity_note_values()
        complexity_note_classes = self.__complexity_note_classes()
        complexity_concurrent_notes = self.__complexity_concurrent_notes()
        complexity_pattern_relative = self.__complexity_pattern("".join(self.ut_repr_relative()), min_pattern_length=1)
        complexity_pattern_absolute = self.__complexity_pattern("".join(self.ut_repr_absolute()), min_pattern_length=2)
        complexity_pattern = min(complexity_pattern_relative, complexity_pattern_absolute)

        dict = {"Note Values": complexity_note_values,
                "Pattern": complexity_pattern,
                "Note Classes": complexity_note_classes,
                "Concurrent Notes": complexity_concurrent_notes, }

        primary_complexity = complexity_note_values
        secondary_complexity = [complexity_pattern, complexity_concurrent_notes, complexity_note_classes]

        if primary_complexity == Complexity.EASY:
            # Easy or Medium
            if sum(map(lambda c: c.value,
                       secondary_complexity)) <= 2 * Complexity.MEDIUM.value or complexity_pattern == Complexity.EASY:
                return Complexity.EASY
            else:
                return Complexity.MEDIUM
        elif primary_complexity == Complexity.HARD:
            # Medium or Hard
            if sum(map(lambda c: c.value,
                       secondary_complexity)) > 2 * Complexity.MEDIUM.value or complexity_pattern == Complexity.HARD:
                return Complexity.HARD
            # if complexity_pattern == Complexity.HARD or \
            #         (complexity_pattern > Complexity.EASY
            #          and sum(map(lambda c: c <= Complexity.EASY, secondary_complexity)) <= 1) or \
            #         (sum(map(lambda c: c <= Complexity.MEDIUM, secondary_complexity)) <= 2):
            #     return Complexity.HARD
            else:
                return Complexity.MEDIUM
        else:
            if all(complexity < Complexity.HARD for complexity in secondary_complexity) and sum(
                    map(lambda c: c.value, secondary_complexity)) < 3 * Complexity.MEDIUM.value:
                return Complexity.EASY
            elif all(complexity > Complexity.EASY for complexity in secondary_complexity) and sum(
                    map(lambda c: c.value, secondary_complexity)) > 3 * Complexity.MEDIUM.value:
                return Complexity.HARD
            return Complexity.MEDIUM

    def __complexity_note_values(self):
        """
        Complexity analysis based on the average wait time. A shorter wait time implies lower note values, thus constituting
        a more difficult song.
        """
        time = 0
        occurrences = 0
        wait_buffer = 0

        for element in self.elements:
            if element.message_type == MessageType.wait:
                wait_buffer += element.value
            elif element.message_type == MessageType.play or element.message_type == MessageType.stop:
                if wait_buffer > 0:
                    time += wait_buffer
                    occurrences += 1
                    wait_buffer = 0
        average_wait_time = time / occurrences if occurrences > 0 else internal_ticks * self.denominator

        if average_wait_time >= internal_ticks * 18 / 24:
            return Complexity.EASY
        elif average_wait_time >= internal_ticks * 12 / 24:
            return Complexity.MEDIUM
        else:
            return Complexity.HARD

    def __complexity_note_classes(self):
        """
        Complexity analysis based on the amount of different note classes. A higher amount of unique note classes constitutes
        a higher difficulty rating.
        """
        classes = set()

        for element in self.elements:
            if element.message_type == MessageType.play:
                classes.add(element.value)

        amount_classes = len(classes)

        if amount_classes <= 4:
            return Complexity.EASY
        elif amount_classes <= 8:
            return Complexity.MEDIUM
        else:
            return Complexity.HARD

    def __complexity_concurrent_notes(self):
        """
        Complexity analysis based on the average amount of notes played at the same time, where more notes constitute a
        higher complexity rating.
        """
        notes = 0
        occurrences = 0
        last_element = None

        for element in self.elements:
            if element.message_type == MessageType.play:
                notes += 1
                if last_element is None or not last_element.message_type == MessageType.play:
                    occurrences += 1
            last_element = element

        average_concurrent_notes = notes / occurrences if occurrences else notes

        if average_concurrent_notes <= 2:
            return Complexity.EASY
        elif average_concurrent_notes <= 3.5:
            return Complexity.MEDIUM
        else:
            return Complexity.HARD

    def __complexity_pattern(self, representation: str, min_pattern_length: int = 2):
        original_representation = representation
        if self.ut_repr_count(original_representation) <= min_pattern_length:
            # Check if pattern can exist
            return Complexity.EASY

        original_regex = r"(?P<pattern>(?:[-+.]\d+[-+.]){len})(?:[-+.]\d+[-+.])*(?:(?P=pattern)(?:[-+.]\d+[-+.])*){pat}"
        regex = original_regex.format(len="{" + str(min_pattern_length) + "}", pat="{" + str(1) + "}")
        results = []

        iteration_index = 0
        iteration_representation = original_representation

        while re.compile(regex).search(iteration_representation):
            # While loop to check for multiple patterns
            group, amount = self.ut_pattern_recognition(original_regex, iteration_representation, min_pattern_length)
            results.append((iteration_representation, group, amount))

            # Search for other patterns
            iteration_representation = iteration_representation.replace(results[iteration_index][1], "")
            iteration_index += 1

        coverage = 1 - self.ut_repr_count(iteration_representation) / self.ut_repr_count(original_representation)

        if coverage >= 0.85 and len(results) <= 3:
            return Complexity.EASY
        elif coverage >= 0.6:
            # Easy or Medium
            if len(results) <= 3 and all(i[2] <= 5 for i in results):
                return Complexity.EASY
            else:
                return Complexity.MEDIUM
        elif coverage >= 0.25:
            # Medium or Hard
            if len(results) <= 4 and all(i[2] <= 7 for i in results):
                return Complexity.EASY
            else:
                return Complexity.MEDIUM
        else:
            return Complexity.HARD

    # Utility Functions

    @staticmethod
    def ut_pattern_recognition(regex_template: str, representation: str, min_pattern_length: int = 2) -> (str, int):
        sr = SequenceRelative
        representation_length = SequenceRelative.ut_repr_count(representation)
        pattern_length = min_pattern_length
        pattern_amount = 1
        pattern_coverage = -math.inf
        iteration_regex = regex_template.format(len="{" + str(pattern_length) + "}",
                                                pat="{" + str(pattern_amount) + "}")

        result = None

        while re.compile(iteration_regex).search(representation):
            # Increase length of pattern
            while re.compile(iteration_regex).search(representation):
                # Increase occurrences of pattern
                match = re.compile(iteration_regex).search(representation)
                local_group = match.groupdict().get("pattern")
                local_group_length = sr.ut_repr_count(local_group)
                local_pattern_span = local_group_length * (pattern_amount + 1)

                local_coverage = local_pattern_span / representation_length

                inherent_pattern = sr.ut_pattern_recognition(regex_template, local_group)
                if local_coverage >= pattern_coverage and (
                        inherent_pattern is None or sr.ut_repr_count(inherent_pattern[0]) *
                        inherent_pattern[1] < sr.ut_repr_count(local_group)):
                    # Found better fitting pattern
                    pattern_coverage = local_coverage
                    result = (local_group, pattern_amount + 1)

                pattern_amount += 1
                iteration_regex = regex_template.format(len="{" + str(pattern_length) + "}",
                                                        pat="{" + str(pattern_amount) + "}")

            pattern_length += 1
            pattern_amount = 1
            iteration_regex = regex_template.format(len="{" + str(pattern_length) + "}",
                                                    pat="{" + str(pattern_amount) + "}")

        return result

    @staticmethod
    def ut_rating_adjust(value: float):
        return min(5, max(1, value))

    @staticmethod
    def ut_minmax(value: float, minval: float = 1, maxval: float = 5):
        return min(maxval, max(minval, value))

    def ut_notes_first(self, steps: int) -> list:
        first_notes = list()

        for element in self.elements:
            if element.message_type == MessageType.wait and len(first_notes) >= steps:
                return first_notes
            if element.message_type == MessageType.play:
                first_notes.append(Note.from_note_value(element.value % 12))

        return first_notes

    def ut_notes_last(self, steps: int) -> list:
        last_notes = list()

        elements = self.elements.copy()
        elements.reverse()
        for element in elements:
            if element.message_type == MessageType.wait and len(last_notes) >= steps:
                return last_notes
            if element.message_type == MessageType.play:
                last_notes.append(Note.from_note_value(element.value % 12))

        return last_notes

    def ut_repr_relative(self) -> list[str]:
        relative_representation = []
        last_value = -1
        string = "{0:+}."

        for element in self.elements:
            if element.message_type == MessageType.play:
                if last_value != -1:
                    relative_representation.append(string.format(element.value - last_value))
                last_value = element.value

        return relative_representation

    def ut_repr_absolute(self) -> list[str]:
        absolute_representation = []
        string = ".{num}."

        for element in self.elements:
            if element.message_type == MessageType.play:
                absolute_representation.append(string.format(num=element.value))

        return absolute_representation

    @staticmethod
    def ut_repr_count(representation: str):
        return (representation.count("+") + representation.count("-") + representation.count(".")) / 2

    @staticmethod
    def ut_calc_rating_weight(rating: float, base: float = 1, ceiling: float = 5, factor_base: float = 1,
                              factor: float = 2):
        """
        Returns a scaling factor based on the distance of the value to the ceiling from the base. Values closer to the ceiling
        are scaled by a higher faction. This process is linear.
        """
        vector_length = ceiling - base
        adjusted_rating = rating - base
        relation = adjusted_rating / vector_length
        return factor_base + (factor - factor_base) * relation
        pass

    @staticmethod
    def ut_calc_weighted_sum(values: list, weights: list):
        weight_sum = 0
        result = 0
        for weight in weights:
            weight_sum += weight
        for i, value in enumerate(values):
            result += value * (weights[i] / weight_sum)
        return result

    @staticmethod
    def ut_generate_wait_message(wait_buffer: int) -> list:
        generated_messages = []
        if wait_buffer > 0:
            if wait_buffer > internal_ticks:
                for j in range(0, int(wait_buffer // internal_ticks)):
                    generated_messages.append((Element(MessageType.wait, internal_ticks, std_velocity)))
                if wait_buffer % internal_ticks > 0:
                    generated_messages.append(
                        Element(MessageType.wait, wait_buffer % internal_ticks, std_velocity))
            else:
                generated_messages.append(Element(MessageType.wait, wait_buffer, std_velocity))
        return generated_messages
