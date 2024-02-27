import numpy as np

from brumaire.constants import (
    NDIntArray,
    ROLE_UNKNOWN,
    ROLE_ADJUTANT,
    ROLE_NAPOLEON,
    ROLE_ALLY,
    ADJ_ALMIGHTY,
    ADJ_MAIN_JACK,
    ADJ_SUB_JACK,
    ADJ_PARTNER,
    ADJ_TRUMP_MAXIMUM,
    ADJ_TRUMP_TWO,
    ADJ_FLIPPED_TWO,
    ADJ_RANDOM,
    SUIT_HEART,
    SUIT_SPADE,
)


def role_to_str(role: int) -> str:
    if role == ROLE_UNKNOWN:
        return "Unknown"
    elif role == ROLE_ADJUTANT:
        return "Adjutant"
    elif role == ROLE_NAPOLEON:
        return "Napoleon"
    elif role == ROLE_ALLY:
        return "Ally"
    else:
        raise "An invalid role is detected."


#
# There are two styles to store the declarations in this program.
# One is a "card-oriented" style, in which the record holds exact suits and numbers of adjutant cards.
# The other is a "strategy-oriented" style, in which the record holds characters of adjutant cards, such as being an almighty.
# A "card-oriented" style is used to log the game while a "strategy-oriented" style is used the agent to make a decision.
#


def convert_to_card_oriented(decl: NDIntArray, strongest: NDIntArray) -> NDIntArray:
    """
    Converts the declarations in a "strategy-oriented" into those in a "card-oriented" style.
    """

    board_num = decl.shape[0]
    converted = np.zeros((board_num, 4))
    converted[:, 0] = decl[:, 0]
    converted[:, 1] = decl[:, 1]
    adj_card = decl[:, 2]

    converted[:, 2] = np.random.randint(4, size=(board_num,))
    converted[:, 3] = np.random.randint(13, size=(board_num,))

    converted[adj_card == ADJ_ALMIGHTY, 2] = SUIT_SPADE
    converted[adj_card == ADJ_ALMIGHTY, 3] = 14 - 2

    converted[adj_card == ADJ_MAIN_JACK, 2] = converted[adj_card == ADJ_MAIN_JACK, 0]
    converted[adj_card == ADJ_MAIN_JACK, 3] = 11 - 2

    converted[adj_card == ADJ_SUB_JACK, 2] = 3 - converted[adj_card == ADJ_SUB_JACK, 0]
    converted[adj_card == ADJ_SUB_JACK, 3] = 11 - 2

    converted[adj_card == ADJ_PARTNER, 2] = SUIT_HEART
    converted[adj_card == ADJ_PARTNER, 3] = 12 - 2

    converted[adj_card == ADJ_TRUMP_TWO, 2] = converted[adj_card == ADJ_TRUMP_TWO, 0]
    converted[adj_card == ADJ_TRUMP_TWO, 3] = 2 - 2

    converted[adj_card == ADJ_FLIPPED_TWO, 2] = (
        3 - converted[adj_card == ADJ_FLIPPED_TWO, 0]
    )
    converted[adj_card == ADJ_FLIPPED_TWO, 3] = 2 - 2

    for idx in range(board_num):
        if adj_card[idx] == ADJ_TRUMP_MAXIMUM:
            strongest_card = strongest[idx, decl[idx, 0]]
            converted[idx, 2] = strongest_card // 13
            converted[idx, 3] = strongest_card % 13

    return converted


def convert_to_strategy_oriented(decl: NDIntArray, strongest: NDIntArray) -> NDIntArray:
    """
    Converts the declarations in a "card-oriented" into those in a "strategy-oriented" style.
    """

    board_num = decl.shape[0]
    converted = np.zeros((board_num, 3), dtype=int)
    converted[:, 0] = decl[:, 0]
    converted[:, 1] = decl[:, 1]

    converted[:, 2] = ADJ_RANDOM

    for idx in range(board_num):
        if decl[idx, 2] == strongest[idx, decl[idx, 0]]:
            # It can be ADJ_RANDOM.
            # However, we will ignore this because the possibility of "false-positive" is too low.
            converted[:, 2] = ADJ_TRUMP_MAXIMUM

    converted[
        (decl[:, 2] == (3 - decl[:, 0])) & (decl[:, 3] == 2 - 2), 2
    ] = ADJ_FLIPPED_TWO
    converted[(decl[:, 2] == decl[:, 0]) & (decl[:, 3] == 2 - 2), 2] = ADJ_TRUMP_TWO
    converted[(decl[:, 2] == SUIT_HEART) & (decl[:, 3] == 12 - 2), 2] = ADJ_PARTNER
    converted[
        (decl[:, 2] == (3 - decl[:, 0])) & (decl[:, 3] == 11 - 2), 2
    ] = ADJ_SUB_JACK
    converted[(decl[:, 2] == decl[:, 0]) & (decl[:, 3] == 11 - 2), 2] = ADJ_MAIN_JACK
    converted[(decl[:, 2] == SUIT_SPADE) & (decl[:, 3] == 14 - 2), 2] = ADJ_ALMIGHTY

    return converted
