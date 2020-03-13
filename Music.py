from mido import MidiFile, MetaMessage, MidiTrack, Message
from enum import Enum


def printMidiFile(midifile: MidiFile, amount=-1):
    print("Start Midi File")
    print("Ticks per beat: " + midifile.ticks_per_beat.__str__() + ", Type: " + midifile.type.__str__())
    for j, track in enumerate(midifile.tracks):
        for i, message in enumerate(track):
            if i >= amount != -1: break
            print(message)
    print("End Midi File")


def printMetaMidiFile(midifile: MidiFile):
    print("Ticks per beat: " + midifile.ticks_per_beat.__str__() + ", Type: " + midifile.type.__str__())


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

            wait_buffer = 0

            for message in track:
                if isinstance(message, MetaMessage): continue

                if message.type == "note_on" or message.type == "note_off" or message.type == "control_change":
                    wait_buffer += message.time
                    if wait_buffer != 0 and message.type != "control_change":
                        # Generate Wait Message
                        musical.elements.append(Element(MessageType.wait, wait_buffer, message.velocity))
                        wait_buffer = 0

                if message.type == "note_on":
                    musical.elements.append(Element(MessageType.play, message.note, message.velocity))
                elif message.type == "note_off":
                    musical.elements.append(Element(MessageType.stop, message.note, message.velocity))
                elif message.type == "control_change":
                    continue
                else:
                    print(message.type)

        print(musical.elements)
        print("Size: " + len(musical.elements).__str__())

        return musical

    def toMidiFile(self):
        midi_file = MidiFile()
        track = MidiTrack()
        midi_file.ticks_per_beat = 96
        midi_file.type = 0
        midi_file.tracks.append(track)

        track.append(MetaMessage("track_name", name="Generated\x00", time=0))
        track.append(MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=36,
                                 notated_32nd_notes_per_beat=8, time=0))
        track.append(MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=36,
                                 notated_32nd_notes_per_beat=8, time=0))

        wait_buffer = 0
        active_notes = set()

        for element in self.elements:
            if element.message == MessageType.wait:
                wait_buffer += element.value
            elif element.message == MessageType.play:
                if element.value in active_notes: continue
                track.append(Message("note_on", note=element.value, velocity=element.velocity, time=wait_buffer))
                active_notes.add(element.value)
                wait_buffer = 0
            elif element.message == MessageType.stop:
                if not element.value in active_notes: continue
                track.append(Message("note_off", note=element.value, velocity=element.velocity, time=wait_buffer))
                active_notes.remove(element.value)
                wait_buffer = 0

        track.append(MetaMessage("end_of_track", time=0))

        return midi_file


class Element:

    def __init__(self, messagetype: MessageType, value: int, velocity: int):
        super().__init__()
        self.message = messagetype
        self.value = value
        self.velocity = velocity

    def __str__(self) -> str:
        return "(" + self.message.__str__() + self.value.__str__() + ")"

    def __repr__(self) -> str:
        return self.__str__()
