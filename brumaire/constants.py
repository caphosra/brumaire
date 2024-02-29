import numpy as np
import numpy.typing as npt

type NDIntArray = npt.NDArray[np.int64]
type NDFloatArray = npt.NDArray[np.float32]
type NDBoolArray = npt.NDArray[np.bool_]


class Suit:
    CLUB = 0
    DIAMOND = 1
    HEART = 2
    SPADE = 3
    JOKER = 4


class CardStatus:
    UNKNOWN = 0
    IN_HAND = 1
    PLAYED = 2


class Role:
    UNKNOWN = 0
    NAPOLEON = 1
    ADJUTANT = 2
    ALLY = 3


class AdjStrategy:
    LENGTH = 8

    ALMIGHTY = 0
    MAIN_JACK = 1
    SUB_JACK = 2
    PARTNER = 3
    TRUMP_TWO = 4
    FLIPPED_TWO = 5
    TRUMP_MAXIMUM = 6
    RANDOM = 7


RWD_WINS_TRICK = 0.02
RWD_NAPOLEON_WINS = 2.0
RWD_ALLY_WINS = 1.0

DECL_INPUT_SIZE = 20 + 3
