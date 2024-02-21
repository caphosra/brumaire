import numpy as np
import numpy.typing as npt

type NDIntArray = npt.NDArray[np.int64]
type NDFloatArray = npt.NDArray[np.float32]
type NDBoolArray = npt.NDArray[np.bool_]

SUIT_CLUB = 0
SUIT_DIAMOND = 1
SUIT_HEART = 2
SUIT_SPADE = 3
SUIT_JOKER = 4

CARD_UNKNOWN = 0
CARD_IN_HAND = 1
CARD_TRICKED = 2

ROLE_UNKNOWN = 0
ROLE_NAPOLEON = 1
ROLE_ADJUTANT = 2
ROLE_ALLY = 3

ADJ_ALMIGHTY = 0
ADJ_MAIN_JACK = 1
ADJ_SUB_JACK = 2
ADJ_PARTNER = 3
ADJ_TRUMP_TWO = 4
ADJ_FLIPPED_TWO = 5
ADJ_TRUMP_MAXIMUM = 6
ADJ_RANDOM = 7

RWD_WINS_TRICK = 0.02
RWD_NAPOLEON_WINS = 2.0
RWD_ALLY_WINS = 1.0
