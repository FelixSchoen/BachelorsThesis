from __future__ import annotations
from src.MusicElements import *
from src.Utility import *
from mido import MidiFile, MetaMessage, MidiTrack


class Composition(Persistable):

    def __init__(self, right_hand: SequenceRelative, left_hand: SequenceRelative, numerator=4, denominator=4,
                 final_complexity=3):
        self.right_hand = right_hand
        self.left_hand = left_hand
        self.numerator = numerator
        self.denominator = denominator
        self.final_complexity = final_complexity

    # Conversion Related Functions

    @staticmethod
    def from_midi_file(midi_file: MidiFile) -> list[Composition]:
        sequences = []

        for i in range(1, len(midi_file.tracks)):
            seq = SequenceRelative.from_midi_track(midi_file.tracks[i])
            if "right" not in seq.name and "left" not in seq.name:
                continue
            sequences.append(seq)

        # Gather all tracks containing information
        right_hand_sequences = []
        left_hand_sequences = []

        for seq in sequences:
            if "right" in seq.name:
                right_hand_sequences.append(seq)
            elif "left" in seq.name:
                left_hand_sequences.append(seq)

        abs_seq_right = right_hand_sequences.pop(0).to_absolute_sequence()
        abs_seq_left = left_hand_sequences.pop(0).to_absolute_sequence()

        # Check if there are more than one tracks for a given hand
        if len(right_hand_sequences) > 0:
            for seq in right_hand_sequences:
                abs_seq_right.merge(seq.to_absolute_sequence())
        if len(left_hand_sequences) > 0:
            for seq in left_hand_sequences:
                abs_seq_left.merge(seq.to_absolute_sequence())

        rel_seq_right = abs_seq_right.quantize().to_relative_sequence().adjust()
        rel_seq_left = abs_seq_left.quantize().to_relative_sequence().adjust()

        # Split timings
        timings = Composition.get_split_timing(midi_file.tracks[0])

        timing = timings.pop(0)
        last_timing = timing

        tail_right = rel_seq_right
        seq_right = tail_right
        tail_left = rel_seq_left
        seq_left = tail_left

        compositions = []

        numerator = 4
        denominator = 4

        for i, timing in enumerate(timings):
            if tail_right is not None and tail_left is not None:
                right_tuple = tail_right.split(timing[0])
                left_tuple = tail_left.split(timing[0])

                seq_right = right_tuple[0]
                seq_right.numerator = numerator
                seq_right.denominator = denominator
                tail_right = right_tuple[1]

                seq_left = left_tuple[0]
                seq_left.numerator = numerator
                seq_left.denominator = denominator
                tail_left = left_tuple[1]

                numerator = timing[1]
                denominator = timing[2]

                composition = Composition(seq_right, seq_left, seq_right.numerator, seq_right.denominator)
                compositions.append(composition.preprocess())

                if i == len(timings) - 1:
                    # Last element
                    last_timing = timing
                    seq_right = right_tuple[1]
                    seq_left = left_tuple[1]

        if seq_right is not None and seq_left is not None:
            seq_right.numerator = last_timing[1]
            seq_right.denominator = last_timing[2]
            seq_left.numerator = last_timing[1]
            seq_left.denominator = last_timing[2]

            composition = Composition(seq_right, seq_left, seq_right.numerator, seq_right.denominator)
            compositions.append(composition.preprocess())

        return compositions

    def to_midi_file(self) -> MidiFile:
        midi_file = MidiFile()

        track_meta = MidiTrack()
        track_meta.append(MetaMessage("time_signature", numerator=self.numerator, denominator=self.denominator))
        track_right = self.right_hand.to_midi_track()
        track_right.insert(0, MetaMessage("track_name", name="Right Hand\x00", time=0))
        track_left = self.left_hand.to_midi_track()
        track_left.insert(0, MetaMessage("track_name", name="Left Hand\x00", time=0))

        midi_file.tracks.append(track_meta)
        midi_file.tracks.append(track_right)
        midi_file.tracks.append(track_left)

        return midi_file

    # Transformative Functions

    def preprocess(self) -> Composition:
        """
        Quantizes and adjusts the sequences of this composition
        """
        self.right_hand = self.right_hand.to_absolute_sequence().quantize().to_relative_sequence().adjust()
        self.left_hand = self.left_hand.to_absolute_sequence().quantize().to_relative_sequence().adjust()

        return self

    def transpose(self, steps: int) -> Composition:
        self.right_hand.transpose(steps)
        self.left_hand.transpose(steps)
        return self

    @staticmethod
    def stitch(compositions: list[Composition]) -> Composition:
        numerator = compositions[0].numerator
        denominator = compositions[0].denominator
        composition = Composition(SequenceRelative(numerator, denominator), SequenceRelative(numerator, denominator))
        composition.numerator = numerator
        composition.denominator = denominator

        for composition_to_stitch in compositions:
            composition.right_hand.stitch(composition_to_stitch.right_hand)
            composition.left_hand.stitch(composition_to_stitch.left_hand)

        return composition

    def split_to_bars(self) -> list[Composition]:
        compositions = []
        right_sequences = self.right_hand.split_to_bars()
        left_sequences = self.left_hand.split_to_bars()

        i = 0
        while i < max(len(right_sequences), len(left_sequences)):
            composition = Composition(right_hand=SequenceRelative(self.numerator, self.denominator), left_hand=SequenceRelative(self.numerator, self.denominator),
                                      numerator=self.numerator, denominator=self.denominator)
            if i < len(right_sequences):
                composition.right_hand = right_sequences[i]
            if i < len(left_sequences):
                composition.left_hand = left_sequences[i]
            if len(composition.right_hand.elements) == 0:
                composition.right_hand.elements = SequenceRelative.ut_generate_wait_message(self.numerator/self.denominator * 4 * internal_ticks)
            if len(composition.left_hand.elements) == 0:
                composition.left_hand.elements = SequenceRelative.ut_generate_wait_message(self.numerator/self.denominator * 4 * internal_ticks)
            compositions.append(composition)
            i += 1

        return compositions

    @staticmethod
    def stitch_to_equal_difficulty_classes(compositions: list[Composition], track_identifier: int) -> list[Composition]:
        classes = []
        pivot = None
        pivot_complexity = 0

        for composition in compositions:
            if len(classes) == 0:
                classes.append([composition])
                pivot_complexity = composition.get_track(track_identifier).complexity()
            else:
                decider_complexity = composition.get_track(track_identifier).complexity()
                if ComplexityRating.in_same_category(pivot_complexity, decider_complexity):
                    classes[-1].append(composition)
                else:
                    pivot_complexity = composition.get_track(track_identifier).complexity()
                    classes.append([composition])

        stitched_compositions = []

        for equal_complexity_list in classes:
            final_composition = Composition.stitch(equal_complexity_list)
            final_composition.final_complexity = Composition.get_average_complexity_class(equal_complexity_list,
                                                                                          track_identifier)
            stitched_compositions.append(final_composition)

        return stitched_compositions

    # Informative Functions

    @staticmethod
    def get_split_timing(midi_track: MidiTrack, modifier: float = internal_ticks / external_ticks) -> list[tuple]:
        timings = []

        value = 0
        for message in midi_track:
            value += message.time
            if message.type == "time_signature":
                timings.append((value * modifier, message.numerator, message.denominator))
                value = 0

        return timings

    def get_track(self, track_identifier: int):
        switcher = {
            RIGHT_HAND: self.right_hand,
            LEFT_HAND: self.left_hand
        }
        return switcher.get(track_identifier)

    @staticmethod
    def get_average_complexity_class(compositions: list[Composition], track_identifier: int) -> int:
        sum = 0
        amount = 0
        for composition in compositions:
            amount += 1
            track = composition.get_track(track_identifier)
            sum += track.complexity()
        complexity = ComplexityRating.get_complexity_rating(sum / amount)
        return complexity

    def to_neuron_representation(self, track_identifier: int) -> list[int]:
        track = self.get_track(track_identifier)
        return track.to_neuron_representation()

    def __str__(self) -> str:
        return "(Numerator: " + str(self.numerator) + ", Denominator: " + str(
            self.denominator) + "\nRight Hand: " + str(self.right_hand) + "\nLeft Hand: " + str(self.left_hand) + ")"
