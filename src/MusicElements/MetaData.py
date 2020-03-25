from enum import Enum


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


class Element:

    def __init__(self, messagetype: MessageType, value: float, velocity: int):
        super().__init__()
        self.message_type = messagetype
        self.value = value
        self.velocity = velocity

    def to_neuron_representation(self):
        if self.message_type == MessageType.play:
            return self.value - 21
        elif self.message_type == MessageType.stop:
            return self.value - 21 + 88
        elif self.message_type == MessageType.wait:
            switcher = {
                2: 0,
                3: 1,
                4: 2,
                6: 3,
                8: 4,
                9: 5,
                10: 6,
                12: 7,
                14: 8,
                15: 9,
                16: 10,
                18: 11,
                20: 12,
                21: 13,
                22: 14,
                24: 15
            }

            raw_val = switcher.get(self.value)
            return raw_val + 88 * 2

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
