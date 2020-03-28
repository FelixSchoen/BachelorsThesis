import unittest
import os
from src.MusicElements import *
from mido import MidiFile


class SequenceRelativeSuite(unittest.TestCase):
    midi_file = MidiFile()
    sequence = SequenceRelative()

    def setUp(self):
        SequenceRelativeSuite.midi_file = MidiFile("res/beethoven_op27_mo3.mid")
        SequenceRelativeSuite.sequence = SequenceRelative.from_midi_track(self.midi_file.tracks[1])

    def test_sequence_creation(self):
        SequenceRelative.from_midi_track(self.midi_file.tracks[1])

    def test_sequence_conversion_to_track(self):
        self.sequence.to_midi_track()

    def test_sequence_conversion_to_absolute(self):
        self.sequence.to_absolute_sequence()

    def test_sequence_adjust(self):
        self.sequence.adjust()

    def test_sequence_transpose(self):
        self.sequence.transpose(3)

    def test_sequence_stitch(self):
        self.sequence.stitch(self.sequence)

    def test_sequence_split(self):
        self.sequence.split(internal_ticks)

    def test_sequence_split_to_bars(self):
        self.sequence.split_to_bars()

    def test_sequence_is_empty(self):
        self.sequence.is_empty()

    def test_sequence_complexity(self):
        self.sequence.split_to_bars()[0].complexity()


if __name__ == '__main__':
    unittest.main()
