import unittest
import os
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

    def test_sequence_split(self):
        seq_tuple = self.sequence.to_absolute_sequence().quantize().to_relative_sequence().split(96)
        print(seq_tuple[0])

    def test_composition_get_timings(self):
        print(Composition.get_split_timing(self.midi_file.tracks[0]))

    def test_composition_split(self):
        composition = Composition.from_midi_file(self.midi_file)
        print(composition)

    def test_split_bars(self):
        sequences = self.sequence.to_absolute_sequence().quantize().to_relative_sequence().adjust().split_bars()
        print(sequences)

    def test_to_file(self):
        filename = "..\out/1.pkl"
        self.sequence.to_file(filename)
        os.remove(filename)

    def test_from_file(self):
        filename = "..\out/1.pkl"
        self.sequence.to_file(filename)
        seq = SequenceRelative.from_file(filename)
        print(seq)
        os.remove(filename)

    def test_split_then_stitch(self):
        split = self.sequence.to_absolute_sequence().quantize().to_relative_sequence().split_bars()
        seq = SequenceRelative.stitch(split)
        print(seq)
        self.save_seq_to_file("..\out/stitched.mid", seq)

    def test_quantize(self):
        for i in range(0, 50):
            print(str(i / 2) + ": " + str(SequenceAbsolute.quantize_value(i / 2)))

    @staticmethod
    def save_seq_to_file(filename: str, seq: SequenceRelative):
        midifile = MidiFile()
        track = seq.to_midi_track()
        midifile.tracks.append(track)
        midifile.save(filename)


if __name__ == '__main__':
    unittest.main()
