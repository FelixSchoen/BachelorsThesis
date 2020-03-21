import unittest
from src.MusicElements import *
from mido import MidiFile


class SequenceTest(unittest.TestCase):
    midi_file = None
    sequence = None

    def setUp(self):
        SequenceTest.midi_file = MidiFile("res/beethoven_op27.mid")
        SequenceTest.sequence = SequenceRelative.from_midi_track(self.midi_file.tracks[2])

    def test_sequence_creation(self):
        print(seq=SequenceRelative.from_midi_track(self.midi_file.tracks[2]))

    def test_sequence_conversion_toAbsolute(self):
        seq = SequenceRelative.from_midi_track(self.midi_file.tracks[2])
        print(seq.to_absolute_sequence())

    def test_sequence_adjust(self):
        print(self.sequence.adjust())

    def test_sequence_quantize(self):
        print(self.sequence.to_absolute_sequence().quantize())

    def test_sequence_convert_back(self):
        print(self.sequence)
        print(self.sequence.to_absolute_sequence().to_relative_sequence())

    def test_merge(self):
        seq1 = SequenceRelative.from_midi_track(self.midi_file.tracks[1])
        seq2 = SequenceRelative.from_midi_track(self.midi_file.tracks[2])
        seq = seq1.to_absolute_sequence().merge(seq2.to_absolute_sequence()).quantize().to_relative_sequence().adjust()
        midifile = MidiFile()
        midifile.tracks.append(seq.to_midi_track())
        midifile.save("../out/merged.mid")


if __name__ == '__main__':
    unittest.main()
