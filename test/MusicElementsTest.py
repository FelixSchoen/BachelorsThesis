import unittest
from src.MusicElements import *
from mido import MidiFile


class SequenceTest(unittest.TestCase):
    midi_file = None
    sequence = None

    def setUp(self):
        SequenceTest.midi_file = MidiFile("res/beethoven_op27_mo3.mid")
        SequenceTest.sequence = SequenceRelative.from_midi_track(self.midi_file.tracks[1])

    def test_sequence_creation(self):
        print(SequenceRelative.from_midi_track(self.midi_file.tracks[2]))

    def test_sequence_conversion_toAbsolute(self):
        seq = SequenceRelative.from_midi_track(self.midi_file.tracks[2])
        print(seq.to_absolute_sequence())

    def test_sequence_adjust(self):
        print(self.sequence.adjust())

    def test_sequence_quantize(self):
        print(self.sequence.to_absolute_sequence().quantize())

    def test_quantize_values(self):
        start = [0.05]
        expected = [0]
        self.assertEqual(SequenceAbsolute.quantize_value(start[0]), expected[0])

    def test_sequence_convert_back(self):
        print(self.sequence)
        print(self.sequence.to_absolute_sequence().to_relative_sequence())

    def test_merge(self):
        seq1 = SequenceRelative.from_midi_track(self.midi_file.tracks[1])
        seq2 = SequenceRelative.from_midi_track(self.midi_file.tracks[2])
        seq = seq1.to_absolute_sequence().merge(seq2.to_absolute_sequence()).quantize().to_relative_sequence().adjust()
        print(seq)

    def test_create_composition(self):
        compositions = Composition.from_midi_file(self.midi_file)
        print(compositions[0])

    def test_composition_to_midi_file(self):
        compositions = Composition.from_midi_file(self.midi_file)
        midi_file = compositions[0].to_midi_file()
        midi_file.save("generated.mid")


if __name__ == '__main__':
    unittest.main()
