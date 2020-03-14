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


class Note(Enum):
    c = 0
    c_s = 1
    d = 2
    d_s = 3
    e = 4
    f = 5
    f_s = 6
    g = -5
    g_s = -4
    a = -3
    a_s = -2
    b = -1
    b_b = a
    e_b = d_s
    a_b = g_s
    d_b = c_s
    g_b = f_s
    c_b = b
    f_b = e


class Scale(Enum):
    cmaj_amin = {Note.c, Note.d, Note.e, Note.f, Note.g, Note.a, Note.b}
    gmaj_emin = {Note.g, Note.a, Note.b, Note.c, Note.d, Note.e, Note.f_s}
    dmaj_bmin = {Note.d, Note.e, Note.f_s, Note.g, Note.a, Note.b, Note.c_s}
    amaj_fsmin = {Note.a, Note.b, Note.c_s, Note.d, Note.e, Note.f_s, Note.g_s}
    emaj_csmin = {Note.e, Note.f_s, Note.g_s, Note.a, Note.b, Note.c_s, Note.d_s}
    bmaj_gsmin = {Note.b, Note.c_s, Note.d_s, Note.e, Note.f_s, Note.g_s, Note.a_s}
    fmaj_dmin = {Note.f, Note.g, Note.a, Note.b_b, Note.c, Note.d, Note.e}
    bbmaj_gmin = {Note.b_b, Note.c, Note.d, Note.e_b, Note.f, Note.g, Note.a}
    ebmaj_cmin = {Note.e_b, Note.f, Note.g, Note.a_b, Note.b_b, Note.c, Note.d}
    abmaj_fmin = {Note.a_b, Note.b_b, Note.c, Note.d_b, Note.e_b, Note.f, Note.g}
    dbmaj_bbmin = {Note.d_b, Note.e_b, Note.f, Note.g_b, Note.a_b, Note.b_b, Note.c}
    fsmaj_dsmin = {Note.f_s, Note.g_s, Note.a_s, Note.b, Note.c_s, Note.d_s, Note.f}  # Special


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


class Sequence:

    def __init__(self):
        self.elements = []

    @staticmethod
    def fromMidiFile(midifile: MidiFile):
        sequence = Sequence()

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
                        sequence.elements.append(Element(MessageType.wait, wait_buffer, message.velocity))
                        wait_buffer = 0

                if message.type == "note_on":
                    sequence.elements.append(Element(MessageType.play, message.note, message.velocity))
                elif message.type == "note_off":
                    sequence.elements.append(Element(MessageType.stop, message.note, message.velocity))
                elif message.type == "control_change":
                    continue
                else:
                    print(message.type)

        print(sequence.elements)
        print("Size: " + len(sequence.elements).__str__())

        return sequence

    def toMidiTrack(self):
        track = MidiTrack()

        track.append(MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=36,
                                 notated_32nd_notes_per_beat=8, time=0))
        track.append(MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=36,
                                 notated_32nd_notes_per_beat=8, time=0))

        wait_buffer = 0
        active_notes = set()

        for element in self.elements:
            if element.message_type == MessageType.wait:
                wait_buffer += element.value
            elif element.message_type == MessageType.play:
                if element.value in active_notes: continue
                track.append(Message("note_on", note=element.value, velocity=element.velocity, time=wait_buffer))
                active_notes.add(element.value)
                wait_buffer = 0
            elif element.message_type == MessageType.stop:
                if not element.value in active_notes: continue
                track.append(Message("note_off", note=element.value, velocity=element.velocity, time=wait_buffer))
                active_notes.remove(element.value)
                wait_buffer = 0

        track.append(MetaMessage("end_of_track", time=0))

        return track


class Musical:

    def __init__(self, right_hand: Sequence, left_hand: Sequence):
        self.right_hand = right_hand
        self.left_hand = left_hand

    @staticmethod
    def fromMidiFiles(right_hand: MidiFile, left_hand: MidiFile):
        musical = Musical(Sequence.fromMidiFile(right_hand), Sequence.fromMidiFile(left_hand))
        return musical

    def toMidiFile(self):
        midi_file = MidiFile()
        tpb = 96
        midi_file.ticks_per_beat = tpb
        midi_file.type = 1

        # Check if there are empty bars at the beginning
        wait_right = 0
        wait_right_velocity = 64
        if self.right_hand.elements[0].message_type == MessageType.wait:
            wait_right = self.right_hand.elements[0].value
            wait_right_velocity = self.right_hand.elements[0].velocity
        wait_left = 0
        wait_left_velocity = 64
        if self.left_hand.elements[0].message_type == MessageType.wait:
            wait_left = self.left_hand.elements[0].value
            wait_left_velocity = self.left_hand.elements[0].velocity

        if wait_right > 0 and wait_left > 0:
            while wait_right - 4 * tpb >= 0 and wait_left - 4 * tpb >= 0:
                wait_right -= 4 * tpb
                wait_left -= 4 * tpb
            self.right_hand.elements.pop(0)
            self.right_hand.elements.append(Element(MessageType.wait, wait_right, wait_right_velocity))
            self.left_hand.elements.pop(0)
            self.right_hand.elements.append(Element(MessageType.wait, wait_left, wait_left_velocity))

        right_track = self.right_hand.toMidiTrack()
        right_track.insert(0, MetaMessage("track_name", name="Right Hand\x00", time=0))
        left_track = self.left_hand.toMidiTrack()
        left_track.insert(0, MetaMessage("track_name", name="Left Hand\x00", time=0))
        midi_file.tracks.append(right_track)
        midi_file.tracks.append(left_track)
        return midi_file


class Element:

    def __init__(self, messagetype: MessageType, value: int, velocity: int):
        super().__init__()
        self.message_type = messagetype
        self.value = value
        self.velocity = velocity

    def __str__(self) -> str:
        return "(" + self.message_type.__str__() + self.value.__str__() + ")"

    def __repr__(self) -> str:
        return self.__str__()
