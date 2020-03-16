from mido import MidiFile, MetaMessage, MidiTrack, Message
from enum import Enum
import numpy as np


def print_midi_file(midi_file: MidiFile, amount=-1) -> None:
    print("Start Midi File")
    print("Ticks per beat: " + str(midi_file.ticks_per_beat) + ", Type: " + str(midi_file.type))
    for j, track in enumerate(midi_file.tracks):
        for i, message in enumerate(track):
            if i >= amount != -1:
                break
            print(message)
    print("End Midi File")


def print_meta_midi_file(midi_file: MidiFile) -> None:
    print("Ticks per beat: " + str(midi_file.ticks_per_beat) + ", Type: " + str(midi_file.type))


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

    @staticmethod
    def from_note_value(note_value: int):
        switcher = {
            0: Note.c,
            1: Note.c_s,
            2: Note.d,
            3: Note.d_s,
            4: Note.e,
            5: Note.f,
            6: Note.f_s,
            7: Note.g,
            8: Note.g_s,
            9: Note.a,
            10: Note.a_s,
            11: Note.b
        }
        return switcher.get(note_value, -1)


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

    def __init__(self, numerator=4, denominator=4):
        self.elements = []
        self.numerator = numerator
        self.denominator = denominator

    @staticmethod
    def from_midi_file(midi_file: MidiFile):
        sequence = Sequence()
        numerator = 4
        denominator = 4

        # Parse midi
        for i, track in enumerate(midi_file.tracks):
            if i != 0:
                break

            wait_buffer = 0

            for message in track:
                if isinstance(message, MetaMessage):
                    if message.type == "time_signature":
                        numerator = message.numerator
                        denominator = message.denominator
                        continue
                    else:
                        continue

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
                    print("Unknown message: " + message.type)

        sequence.numerator = numerator
        sequence.denominator = denominator
        return sequence

    def to_midi_track(self) -> MidiTrack:
        track = MidiTrack()

        track.append(
            MetaMessage("time_signature", numerator=self.numerator, denominator=self.denominator, clocks_per_click=36,
                        notated_32nd_notes_per_beat=8, time=0))

        wait_buffer = 0
        active_notes = set()

        for element in self.elements:
            if element.message_type == MessageType.wait:
                wait_buffer += element.value
            elif element.message_type == MessageType.play:
                if element.value in active_notes:
                    continue
                track.append(Message("note_on", note=element.value, velocity=element.velocity, time=wait_buffer))
                active_notes.add(element.value)
                wait_buffer = 0
            elif element.message_type == MessageType.stop:
                if element.value not in active_notes:
                    continue
                track.append(Message("note_off", note=element.value, velocity=element.velocity, time=wait_buffer))
                active_notes.remove(element.value)
                wait_buffer = 0

        track.append(MetaMessage("end_of_track", time=0))

        return track

    def to_midi_file(self) -> MidiFile:
        midi_file = MidiFile()
        tpb = Musical.ticks_per_beat
        midi_file.ticks_per_beat = tpb
        midi_file.type = 1
        midi_file.tracks.append(self.to_midi_track())
        return midi_file

    def detect_scale(self) -> list:
        mismatch = dict()

        for scale in Scale:
            mismatch[scale] = 0

        for element in self.elements:
            if element.message_type != MessageType.play:
                continue
            note_value = element.value % 12
            note = Note.from_note_value(note_value)
            for scale in Scale:
                if note not in scale.value:
                    mismatch[scale] += 1

        return sorted(mismatch.items(), key=lambda item: item[1])[0][0]

    def transpose(self, steps: int):
        for i, element in enumerate(self.elements):
            if element.message_type != MessageType.play and element.message_type != MessageType.stop:
                continue
            self.elements.pop(i)
            element.value += steps
            while element.value < 21:
                element.value += 12
            while element.value > 108:
                element.value -= 12
            self.elements.insert(i, element)

    def split(self, numerator: int, denominator: int) -> list:
        max_duration = Musical.ticks_per_beat * (numerator / denominator)

        # Sequence to append
        seq = Sequence(numerator, denominator)
        # Generated Sequences
        seq_list = list()
        # Initial queue, instructions of the original sequence are in here
        initial_queue = self.elements.copy()
        # Carry queue, elements that need to be carried to the next bar contained in here
        carry_queue = list()

        # Duration of all elements in current iteration
        duration = 0
        # Values of notes that are open, used to close these at the end of an iteration
        open_notes = set()
        # Determines if current bar is done
        flag_end = False
        # Determines if carry queue should be used
        flag_carry = False

        print("Debug Initial: " + str(initial_queue))

        while len(initial_queue) != 0 or len(carry_queue) != 0:
            if len(carry_queue) != 0 and flag_carry:
                element = carry_queue.pop(0)
            else:
                flag_carry = False
                element = initial_queue.pop(0)

            if element.message_type == MessageType.play:
                if duration < max_duration:
                    # Can play note in this bar
                    open_notes.add(WrappedElement(element))
                    seq.elements.append(element)
                else:
                    # Need to carry note to next bar
                    carry_queue.append(element)
            elif element.message_type == MessageType.stop:
                open_notes.remove(WrappedElement(element))
                seq.elements.append(element)
            elif element.message_type == MessageType.wait:
                if duration + element.value <= max_duration:
                    # Wait message fits in its entirety
                    duration += element.value
                    seq.elements.append(element)
                else:
                    # Wait does not fit
                    fit_duration = max_duration - duration
                    remainder_duration = element.value - fit_duration
                    duration += fit_duration
                    if fit_duration > 0:
                        seq.elements.append(Element(MessageType.wait, int(fit_duration), element.velocity))
                    carry_queue.append(Element(MessageType.wait, int(remainder_duration), element.velocity))
                    flag_end = True

            # Process ending of bar, add to list, close open notes
            if flag_end:
                flag_carry = True
                for open_note in open_notes:
                    carry_queue.insert(0, open_note.element)
                    seq.elements.append(Element(MessageType.stop, open_note.element.value, Musical.std_velocity))
                seq_list.append(seq)
                seq = Sequence(numerator, denominator)
                duration = 0
                flag_end = False

        # Manually append last sequence, because it never triggers end detection
        seq_list.append(seq)
        return seq_list

    def __str__(self) -> str:
        return "(Numerator: " + str(self.numerator) + ", Denominator: " + str(
            self.denominator) + ", Elements: " + str(self.elements) + ")"


class Musical:
    std_velocity = 64
    ticks_per_beat = 96

    def __init__(self, right_hand: Sequence, left_hand: Sequence, numerator=4, denominator=4):
        self.right_hand = right_hand
        self.left_hand = left_hand
        self.numerator = numerator
        self.denominator = denominator

    @staticmethod
    def from_midi_file(right_hand: MidiFile, left_hand: MidiFile):
        seqr = Sequence.from_midi_file(right_hand)
        seql = Sequence.from_midi_file(left_hand)
        numerator = np.lcm(seqr.numerator, seql.numerator)
        denominator = np.lcm(seqr.denominator, seql.denominator)
        musical = Musical(seqr, seql, numerator, denominator)
        return musical

    def to_midi_file(self):
        midi_file = MidiFile()
        tpb = Musical.ticks_per_beat
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

        right_track = self.right_hand.to_midi_track()
        right_track.insert(0, MetaMessage("track_name", name="Right Hand\x00", time=0))
        left_track = self.left_hand.to_midi_track()
        left_track.insert(0, MetaMessage("track_name", name="Left Hand\x00", time=0))
        midi_file.tracks.append(right_track)
        midi_file.tracks.append(left_track)
        return midi_file

    def detect_scale(self):
        return self.right_hand.detect_scale(), self.left_hand.detect_scale()

    def transpose(self, steps: int):
        self.right_hand.transpose(steps)
        self.left_hand.transpose(steps)


class Element:

    def __init__(self, messagetype: MessageType, value: int, velocity: int):
        super().__init__()
        self.message_type = messagetype
        self.value = value
        self.velocity = velocity

    def __str__(self) -> str:
        return "(" + str(self.message_type) + str(self.value) + ")"

    def __repr__(self) -> str:
        return self.__str__()


class WrappedElement:
    """
    A wrapper for the Element class, implementing a custom comparator,
    only comparing the value of the contained element.
    """

    def __init__(self, element: Element):
        self.element = element

    def __eq__(self, other):
        return self.element.value == other.element.value

    def __hash__(self):
        return hash(self.element.value)
