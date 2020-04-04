import unittest
import os
from colorama import *
from concurrent.futures import ThreadPoolExecutor
from src.MusicElements import *
from src.Utility import *
from mido import MidiFile
from time import sleep


class SequenceRelativeSuite(unittest.TestCase):
    midi_file = MidiFile()
    sequence = SequenceRelative()

    def setUp(self):
        SequenceRelativeSuite.midi_file = MidiFile("res/beethoven_op27_mo3.mid")
        SequenceRelativeSuite.sequence = SequenceRelative.from_midi_track(self.midi_file.tracks[1])

    def test_sequence_from_midi_track(self):
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


class SequenceAbsoluteSuite(unittest.TestCase):
    midi_file = MidiFile()
    composition = Composition(None, None)

    def test_quantize(self):
        print(SequenceAbsolute.quantize_value(1))

    def test_cutoff(self):
        cut_file = MidiFile("res/cutoff.mid")
        cut_seq = SequenceRelative.from_midi_track(cut_file.tracks[0])
        cut_seq.to_absolute_sequence().cutoff(24).to_relative_sequence().adjust()

        mid = MidiFile()
        mid.tracks.append(cut_seq.to_midi_track())
        mid.save("out.mid")


class CompositionSuite(unittest.TestCase):
    midi_file = MidiFile()
    composition = Composition(None, None)

    def setUp(self):
        CompositionSuite.midi_file = MidiFile("res/beethoven_op27_mo3.mid")
        CompositionSuite.composition = Composition.from_midi_file(self.midi_file)[0]

    def test_composition_from_midi_file(self):
        Composition.from_midi_file(self.midi_file)

    def test_composition_transpose(self):
        self.composition.transpose(3)

    def test_composition_stitch(self):
        Composition.stitch([self.composition, self.composition])

    def test_composition_split_to_bars(self):
        self.composition.split_to_bars()

    def test_composition_stitch_equal_complexity(self):
        comps = Composition.stitch_equal_complexity(self.composition.split_to_bars(), Constants.RIGHT_HAND)
        for i, comp in enumerate(comps):
            comp.to_midi_file().save("./out/num" + str(i) + "compl" + str(comp.final_complexity) + ".mid")

    def test_composition_get_split_timing(self):
        Composition.get_split_timing(self.midi_file.tracks[0])

    def test_composition_get_track(self):
        self.composition.get_track(Constants.RIGHT_HAND)

    def test_composition_to_neuron_representation(self):
        self.composition.to_neuron_representation(Constants.RIGHT_HAND)


class MetaDataSuite(unittest.TestCase):

    def test_element_to_neural_representation(self):
        for i in range(0, 88 * 2 + 24):
            print(Element.from_neuron_representation(Element.from_neuron_representation(i).to_neuron_representation()))


if __name__ == '__main__':
    unittest.main()
