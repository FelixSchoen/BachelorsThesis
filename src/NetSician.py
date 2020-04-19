from mido import MidiFile
from os import walk
from src.MusicElements import *
from src.Utility import *
from src.GenerationNetwork import *
from concurrent.futures import ThreadPoolExecutor


def judge_difficulty_stitch_and_persist(name, midi_file):
    try:
        compositions = Composition.from_midi_file(midi_file)

        for composition in compositions:
            if composition.numerator / composition.denominator != 4 / 4:
                # Only allow compositions of 4/4 time signature
                continue

            equal_classes = Composition.stitch_equal_complexity(composition.split_to_bars(), Constants.RIGHT_HAND)

            for i, equal_class in enumerate(equal_classes):
                if equal_class.final_complexity == Complexity.EASY:
                    equal_class.to_midi_file().save("../out/lib/easy/" + name + "-" + str(i) + ".mid")
                elif equal_class.final_complexity == Complexity.MEDIUM:
                    equal_class.to_midi_file().save("../out/lib/medium/" + name + "-" + str(i) + ".mid")
                else:
                    equal_class.to_midi_file().save("../out/lib/hard/" + name + "-" + str(i) + ".mid")

            print("Loaded " + name)

    except Exception as e:
        print(name, e)


def load_midi_files_and_persist():
    print("Started MIDI File conversion")
    print()

    executor = ThreadPoolExecutor()

    directories = []
    for (dirpath, dirnames, filenames) in walk("../res/midi"):
        for name in filenames:
            directories.append((dirpath + "/" + name, name))

    for pairing in directories:
        midi_file = MidiFile(pairing[0])
        executor.submit(judge_difficulty_stitch_and_persist, pairing[1][:-4], midi_file)


def count_complexity_classes():
    print("Started Complexity Count")
    print()

    countRightEasy = 0
    countRightMedium = 0
    countRightHard = 0
    countLeftEasy = 0
    countLeftMedium = 0
    countLeftHard = 0

    directories = []
    for (dirpath, dirnames, filenames) in walk("../res/midi"):
        for name in filenames:
            directories.append((dirpath + "/" + name, name))

    for pairing in directories:
        midi_file = MidiFile(pairing[0])
        try:
            compositions = Composition.from_midi_file(midi_file)

            for composition in compositions:
                if composition.numerator / composition.denominator != 4 / 4:
                    # Only allow compositions of 4/4 time signature
                    continue

                for bar in composition.split_to_bars():
                    complexity = bar.right_hand.complexity()

                    if complexity == Complexity.EASY:
                        countRightEasy += 1
                    elif complexity == Complexity.MEDIUM:
                        countRightMedium += 1
                    else:
                        countRightHard += 1

                    complexity = bar.left_hand.complexity()

                    if complexity == Complexity.EASY:
                        countLeftEasy += 1
                    elif complexity == Complexity.MEDIUM:
                        countLeftMedium += 1
                    else:
                        countLeftHard += 1

                print("Judged " + pairing[1][:-4])

        except Exception as e:
            print(name, e)

    print()
    print("Results:")
    print("Right Hand: Easy: {easy}, Medium: {medium}, Hard: {hard}".format(easy=countRightEasy, medium=countRightMedium, hard=countRightHard))
    print("Left Hand: Easy: {easy}, Medium: {medium}, Hard: {hard}".format(easy=countLeftEasy, medium=countLeftMedium, hard=countLeftHard))


if __name__ == '__main__':
    print("Begin Demo")
    print()

    print("Read MIDI File to Composition")
    file = MidiFile("../res/demo/beethoven_op27_mo1.mid")
    comp = Composition.from_midi_file(file)[0]
    print("Composition: {comp}".format(comp=str(comp)))
    print()

    print("Split to bars")
    bars = comp.split_to_bars()
    print("First bar: {bar}".format(bar=str(bars[0])))
    print()

    print("Judge Complexity")
    complexity = bars[0].right_hand.complexity()
    print("Complexity: {complexity}".format(complexity=str(complexity)))
    print()

    # Uncomment line in order to generate stitched complexity classes
    # load_midi_files_and_persist()

    # Calculate amount of each complexity class for each hand
    count_complexity_classes()
