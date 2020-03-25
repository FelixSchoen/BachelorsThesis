from __future__ import annotations
from src.MusicElements import *
from mido import MidiFile, MetaMessage, MidiTrack


class Composition:

    def __init__(self, right_hand: SequenceRelative, left_hand: SequenceRelative, numerator=4, denominator=4):
        self.right_hand = right_hand
        self.left_hand = left_hand
        self.numerator = numerator
        self.denominator = denominator
        self.complexity

    @staticmethod
    def from_midi_file(midi_file: MidiFile) -> list[Composition]:
        sequences = []

        for i in range(1, len(midi_file.tracks)):
            seq = SequenceRelative.from_midi_track(midi_file.tracks[i])
            if "right" not in seq.name and "left" not in seq.name:
                break
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
        rel_seq_right.numerator = timing[0]
        rel_seq_left.numerator = timing[0]
        rel_seq_right.denominator = timing[1]
        rel_seq_left.denominator = timing[1]

        tail_right = rel_seq_right
        tail_left = rel_seq_left

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
                compositions.append(composition)

                if i == len(timings) - 1:
                    # Last element
                    seq_right = right_tuple[1]
                    seq_left = left_tuple[1]

                    if seq_right is not None and seq_left is not None:
                        seq_right.numerator = timing[1]
                        seq_right.denominator = timing[2]
                        seq_left.numerator = timing[1]
                        seq_left.denominator = timing[2]

                        composition = Composition(seq_right, seq_left, seq_right.numerator, seq_right.denominator)
                        compositions.append(composition)

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

    def transpose(self, steps: int):
        self.right_hand.transpose(steps)
        self.left_hand.transpose(steps)

    def __str__(self) -> str:
        return "(Numerator: " + str(self.numerator) + ", Denominator: " + str(
            self.denominator) + "\nRight Hand: " + str(self.right_hand) + "\nLeft Hand: " + str(self.left_hand) + ")"
