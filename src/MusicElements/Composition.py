from __future__ import annotations
from src.MusicElements import *
from mido import MidiFile, MetaMessage


class Composition:

    def __init__(self, right_hand: SequenceRelative, left_hand: SequenceRelative, numerator=4, denominator=4):
        self.right_hand = right_hand
        self.left_hand = left_hand
        self.numerator = numerator
        self.denominator = denominator

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

        composition = Composition(rel_seq_right, rel_seq_left, rel_seq_right.numerator, rel_seq_right.denominator)
        return [composition]

    def to_midi_file(self) -> MidiFile:
        midi_file = MidiFile()
        track_right = self.right_hand.to_midi_track()
        track_right.insert(0, MetaMessage("track_name", name="Right Hand\x00", time=0))
        track_left = self.left_hand.to_midi_track()
        track_left.insert(0, MetaMessage("track_name", name="Left Hand\x00", time=0))
        midi_file.tracks.append(track_right)
        midi_file.tracks.append(track_left)

        return midi_file

    def __str__(self) -> str:
        return "(Numerator: " + str(self.numerator) + ", Denominator: " + str(
            self.denominator) + "\nRight Hand: " + str(self.right_hand) + "\nLeft Hand: " + str(self.left_hand) + ")"
