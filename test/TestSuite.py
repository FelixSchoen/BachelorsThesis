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

    @staticmethod
    def calculate_complexity(name, midi_file, easy, medium, hard):
        try:
            compositions = Composition.from_midi_file(midi_file)
            for composition in compositions:
                if composition.numerator / composition.denominator != 4 / 4:
                    continue
                bars = composition.split_to_bars()
                for i, bar in enumerate(bars):
                    complexity = bar.right_hand.complexity()
                    if complexity == Complexity.EASY:
                        easy.append(bar.right_hand)
                    elif complexity == Complexity.MEDIUM:
                        medium.append(bar.right_hand)
                    else:
                        hard.append(bar.right_hand)
        except Exception as e:
            print(e)
            print(Fore.RED + "Exception " + str(name) + Style.RESET_ALL)

    def test_mest(self):
        seq = SequenceRelative()
        seq.elements.append(Element(MessageType.play, 24, 64))
        seq.elements.append(Element(MessageType.wait, 24, 64))
        seq.elements.append(Element(MessageType.stop, 24, 64))
        seq.elements.append(Element(MessageType.wait, 22, 64))

        print(seq)
        seq.adjust()
        print(seq)

    def test_test(self):
        easy = []
        medium = []
        hard = []

        executor = ThreadPoolExecutor()

        directories = []
        for (dirpath, dirnames, filenames) in os.walk("../res/midi"):
            for name in filenames:
                directories.append(dirpath + "/" + name)
        # directories = Util.util_remove_elements(directories, 1)

        for filepath in directories:
            midi_file = MidiFile(filepath)
            executor.submit(SequenceRelativeSuite.calculate_complexity, filepath, midi_file, easy, medium, hard)

        print("Easy: {easy}, Medium: {medium}, Hard: {hard}".format(easy=len(easy), medium=len(medium),
                                                                    hard=len(hard)))


class SequenceAbsoluteSuite(unittest.TestCase):
    midi_file = MidiFile()
    composition = Composition(None, None)

    def test_quantize(self):
        print(SequenceAbsolute.quantize_value(1))


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

    def test_composition_stitch_to_equal_difficulty_classes(self):
        Composition.stitch_to_equal_difficulty_classes(self.composition.split_to_bars(), Composition.RIGHT_HAND)

    def test_composition_get_split_timing(self):
        Composition.get_split_timing(self.midi_file.tracks[0])

    def test_composition_get_track(self):
        self.composition.get_track(Composition.RIGHT_HAND)

    def test_composition_get_average_complexity_class(self):
        Composition.get_average_complexity_class(
            [self.composition.split_to_bars()[0], self.composition.split_to_bars()[1]], Composition.RIGHT_HAND)

    def test_composition_to_neuron_representation(self):
        self.composition.to_neuron_representation(Composition.RIGHT_HAND)


class MetaDataSuite(unittest.TestCase):

    def test_element_to_neural_representation(self):
        for i in range(0, 88 * 2 + 24):
            print(Element.from_neuron_representation(Element.from_neuron_representation(i).to_neuron_representation()))


if __name__ == '__main__':
    unittest.main()
