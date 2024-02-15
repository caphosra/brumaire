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

REQ_DECL_S = 0
REQ_DECL_N = 1
REQ_ADJ_S = 2
REQ_ADJ_N = 3
REQ_DISC = 4
REQ_TRICK = 5

RWD_WINS_TRICK = 0.02

RWD_NAPOLEON_WINS = 2
RWD_ALLY_WINS = 1

from . import agent
from . import board
from . import session
