from mido import MidiFile, MetaMessage
from enum import Enum


class MessageType(Enum):
    play = 0
    stop = 1
    wait = 2

    def __str__(self):
        if self.value == 0:
            return "p"
        elif self.value == 1:
            return "s"
        else:
            return "w"


class Musical:

    def __init__(self):
        self.elements = []

    @staticmethod
    def fromMidiFile(midifile: MidiFile):
        musical = Musical()

        # Parse midi
        for i, track in enumerate(midifile.tracks):
            if i != 0: break
            for message in track:
                if isinstance(message, MetaMessage): continue

                if message.type == "note_on" or message.type == "note_off":
                    if (message.time != 0):
                        # Generate Wait Message
                        musical.elements.append(Element(MessageType.wait, message.time))

                if message.type == "note_on":
                    musical.elements.append(Element(MessageType.play, message.note))
                elif message.type == "note_off":
                    musical.elements.append(Element(MessageType.stop, message.note))
                else:
                    continue

        print(musical.elements)
        print("Size: " + len(musical.elements).__str__())

        return musical


class Element:

    def __init__(self, messagetype: MessageType, value: int):
        super().__init__()
        self.message = messagetype
        self.value = value

    def __str__(self) -> str:
        return "(" + self.message.__str__() + self.value.__str__() + ")"

    def __repr__(self) -> str:
        return self.__str__()
