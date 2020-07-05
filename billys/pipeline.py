from enum import Enum


class Step(Enum):
    INIT = 0
    DEWARP = 1
    CONTRAST = 2
    ORC = 3
    FEAT_PREPROC = 4
    TRAIN_CLASSIFIER = 5

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not ((self < other) and (self == other))

    def __ge__(self, other):
        return not (self < other)

    def __ne__(self, other):
        return not (self == other)

    def __int__(self):
        return self.value

    def __hash__(self):
        return self.value
