from __future__ import annotations
from enum import Enum
from src.Utility import Exceptions, Constants


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
    cmaj_amin = [Note.c, Note.d, Note.e, Note.f, Note.g, Note.a, Note.b]
    gmaj_emin = [Note.g, Note.a, Note.b, Note.c, Note.d, Note.e, Note.f_s]
    dmaj_bmin = [Note.d, Note.e, Note.f_s, Note.g, Note.a, Note.b, Note.c_s]
    amaj_fsmin = [Note.a, Note.b, Note.c_s, Note.d, Note.e, Note.f_s, Note.g_s]
    emaj_csmin = [Note.e, Note.f_s, Note.g_s, Note.a, Note.b, Note.c_s, Note.d_s]
    bmaj_gsmin = [Note.b, Note.c_s, Note.d_s, Note.e, Note.f_s, Note.g_s, Note.a_s]
    fmaj_dmin = [Note.f, Note.g, Note.a, Note.b_b, Note.c, Note.d, Note.e]
    bbmaj_gmin = [Note.b_b, Note.c, Note.d, Note.e_b, Note.f, Note.g, Note.a]
    ebmaj_cmin = [Note.e_b, Note.f, Note.g, Note.a_b, Note.b_b, Note.c, Note.d]
    abmaj_fmin = [Note.a_b, Note.b_b, Note.c, Note.d_b, Note.e_b, Note.f, Note.g]
    dbmaj_bbmin = [Note.d_b, Note.e_b, Note.f, Note.g_b, Note.a_b, Note.b_b, Note.c]
    fsmaj_dsmin = [Note.f_s, Note.g_s, Note.a_s, Note.b, Note.c_s, Note.d_s, Note.f]  # Special


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


class ComplexityRating:

    @staticmethod
    def get_complexity_rating(value: float) -> int:
        if value < 5 / 3:
            return 1
        elif value < 2.5:
            return 2
        elif value < 3.5:
            return 3
        elif value < 13 / 3:
            return 4
        else:
            return 5

    @staticmethod
    def in_same_category(this_difficulty: float, that_difficulty: float, tolerance: float = 0.15) -> bool:
        if ComplexityRating.get_complexity_rating(this_difficulty) == ComplexityRating.get_complexity_rating(
                that_difficulty) or \
                ComplexityRating.get_complexity_rating(this_difficulty) == ComplexityRating.get_complexity_rating(
            that_difficulty + tolerance) or \
                ComplexityRating.get_complexity_rating(this_difficulty) == ComplexityRating.get_complexity_rating(
            that_difficulty - tolerance):
            return True
        return False


class Element:

    def __init__(self, messagetype: MessageType, value: float, velocity: int):
        super().__init__()
        self.message_type = messagetype
        self.value = value
        self.velocity = velocity

    def to_neuron_representation(self, padding: bool = True):
        add = 0
        if padding:
            add = 1

        if self.message_type == MessageType.stop:
            return self.value - 21 + add
        elif self.message_type == MessageType.play:
            return self.value - 21 + 88 + add
        elif self.message_type == MessageType.wait:
            return self.value - 1 + 88 * 2 + add

    @staticmethod
    def from_neuron_representation(value: int, padding: bool = True) -> Element:
        add = 0
        if padding:
            add = 1

        if 0 <= value - add < 88:
            return Element(messagetype=MessageType.stop, value=value + 21 - add, velocity=Constants.std_velocity)
        elif 88 <= value - add < 176:
            return Element(messagetype=MessageType.play, value=value + 21 - 88 - add, velocity=Constants.std_velocity)
        elif 176 <= value - add < 200:
            return Element(messagetype=MessageType.wait, value=value + 1 - 2 * 88 - add,
                           velocity=Constants.std_velocity)
        else:
            raise Exceptions.InvalidRepresentation

    def __str__(self) -> str:
        return "(" + str(self.message_type) + str(self.value) + ")"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other):
        return self.to_neuron_representation() < other.to_neuron_representation()


class WrappedElement:
    """
    A wrapper for the Element class, implementing a custom comparator,
    only comparing the value of the contained element.
    """

    def __init__(self, element: Element):
        self.element = element

    def __eq__(self, other):
        return self.element.value == other.element.value

    def __lt__(self, other):
        return self.element.value < other.element.value

    def __hash__(self):
        return hash(self.element.value)
