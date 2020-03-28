import unittest
import os
from src.MusicElements import *
from mido import MidiFile


class SequenceTest(unittest.TestCase):
    midi_file = MidiFile
    midi_fileEasy = MidiFile
    sequence = SequenceRelative
    sequenceEasy = SequenceRelative

    def setUp(self):
        SequenceTest.midi_file = MidiFile("res/beethoven_op27_mo3.mid")
        SequenceTest.midi_fileEasy = MidiFile("res/beethoven_op27_mo1.mid")
        SequenceTest.sequence = SequenceRelative.from_midi_track(self.midi_file.tracks[1])
        SequenceTest.sequenceEasy = SequenceRelative.from_midi_track(self.midi_fileEasy.tracks[2])

    def test_sequence_creation(self):
        print(SequenceRelative.from_midi_track(self.midi_file.tracks[2]))

    def test_sequence_conversion_toAbsolute(self):
        seq = SequenceRelative.from_midi_track(self.midi_file.tracks[2])
        print(seq.to_absolute_sequence())

    def test_sequence_adjust(self):
        print(self.sequence.to_absolute_sequence().quantize().to_relative_sequence().adjust())

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
        sequences = self.sequence.to_absolute_sequence().quantize().to_relative_sequence().adjust().split_to_bars()
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

    def test_relative_representation(self):
        print(self.sequence.ut_repr_relative())

    def test_calc_weighted_rating(self):
        print(SequenceRelative.ut_calc_rating_weight(1, base=5, ceiling=1))

    def test_composition_to_equal_difficulty_classes(self):
        composition = Composition.from_midi_file(self.midi_fileEasy)
        compositions = composition[0].split_to_bars()
        equal_complexity = Composition.stitch_to_equal_difficulty_classes(compositions, Composition.LEFT_HAND)
        i = 0
        for comp in equal_complexity:
            i += 1
            comp.to_midi_file().save("..\out/compl/" + str(i) + "-complxty" + str(comp.final_complexity) + ".mid")

    def test_test(self):
        compositions = Composition.from_midi_file(self.midi_fileEasy)
        bars = compositions[0].split_to_bars()
        i = 0
        for bar in bars:
            i += 1
            bar.to_midi_file().save("..\out/compl/" + str(i) + ".mid")

    @staticmethod
    def save_seq_to_file(filename: str, seq: SequenceRelative):
        midifile = MidiFile()
        track = seq.to_midi_track()
        midifile.tracks.append(track)
        midifile.save(filename)


if __name__ == '__main__':
    unittest.main()
