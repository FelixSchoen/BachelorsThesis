from mido import MidiFile, MetaMessage

import Music as mu
from Music import Sequence, MessageType, Element


def main():
    print("Lel")

def test():
    seq = mu.Sequence.from_midi_file(MidiFile("res/stop_track.mid"))
    for i, sequence in enumerate(seq.split(seq.numerator, seq.denominator)):
        if i > 4: break
        print(sequence)
        sequence.to_midi_file().save("out/"+str(i)+".mid")


if __name__ == '__main__':
    # main()
    # test()
    test()
