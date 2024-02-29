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

    @staticmethod
    def to_str(suit: int) -> str:
        if suit == Suit.CLUB:
            return "CLUB"
        elif suit == Suit.DIAMOND:
            return "DIAMOND"
        elif suit == Suit.HEART:
            return "HEART"
        elif suit == Suit.SPADE:
            return "SPADE"
        else:
            return "JOKER"


class CardStatus:
    UNKNOWN = 0
    IN_HAND = 1
    PLAYED = 2


class Role:
    UNKNOWN = 0
    NAPOLEON = 1
    ADJUTANT = 2
    ALLY = 3

    @staticmethod
    def to_str(role: int) -> str:
        match role:
            case Role.UNKNOWN:
                return "Unknown"
            case Role.ADJUTANT:
                return "Adjutant"
            case Role.NAPOLEON:
                return "Napoleon"
            case Role.ALLY:
                return "Ally"
            case _:
                raise "An invalid role is detected."


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

    @staticmethod
    def to_str(st: int) -> str:
        match st:
            case AdjStrategy.ALMIGHTY:
                return "Almighty"
            case AdjStrategy.MAIN_JACK:
                return "Main Jack"
            case AdjStrategy.SUB_JACK:
                return "Sub Jack"
            case AdjStrategy.PARTNER:
                return "Partner"
            case AdjStrategy.TRUMP_TWO:
                return "Trump Two"
            case AdjStrategy.FLIPPED_TWO:
                return "Flipped Two"
            case AdjStrategy.TRUMP_MAXIMUM:
                return "Trump Maximum"
            case AdjStrategy.RANDOM:
                return "Random"
            case _:
                raise "An unknown strategy is detected."


RWD_WINS_TRICK = 0.02
RWD_NAPOLEON_WINS = 2.0
RWD_ALLY_WINS = 1.0

DECL_INPUT_SIZE = 20 + 3
