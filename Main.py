from mido import MidiFile, MetaMessage

import Music as mu
from Music import Musical, MessageType, Element


def main():
    #midiFile = MidiFile("res/bach_op27_r.mid")
    midiFile = MidiFile("res/bach_op27_r.mid")

    bach = Musical.fromMidiFile(midiFile)
    created = bach.toMidiFile()

    created.save("res/created.mid")

def test():
    midifile = MidiFile("res/bach_op27_r.mid")
    mu.printMidiFile(midifile, 10)
    print("")
    musical = Musical.fromMidiFile(midifile)
    created = musical.toMidiFile()
    created.save("res/generated.mid")
    print("")
    mu.printMidiFile(created, 10)



if __name__ == '__main__':
    #main()
    test()
