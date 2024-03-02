import numpy as np

from brumaire.constants import NDIntArray, NDFloatArray, AdjStrategy, Suit


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

    converted[adj_card == AdjStrategy.ALMIGHTY, 2] = Suit.SPADE
    converted[adj_card == AdjStrategy.ALMIGHTY, 3] = 14 - 2

    converted[adj_card == AdjStrategy.MAIN_JACK, 2] = converted[
        adj_card == AdjStrategy.MAIN_JACK, 0
    ]
    converted[adj_card == AdjStrategy.MAIN_JACK, 3] = 11 - 2

    converted[adj_card == AdjStrategy.SUB_JACK, 2] = (
        3 - converted[adj_card == AdjStrategy.SUB_JACK, 0]
    )
    converted[adj_card == AdjStrategy.SUB_JACK, 3] = 11 - 2

    converted[adj_card == AdjStrategy.PARTNER, 2] = Suit.HEART
    converted[adj_card == AdjStrategy.PARTNER, 3] = 12 - 2

    converted[adj_card == AdjStrategy.TRUMP_TWO, 2] = converted[
        adj_card == AdjStrategy.TRUMP_TWO, 0
    ]
    converted[adj_card == AdjStrategy.TRUMP_TWO, 3] = 2 - 2

    converted[adj_card == AdjStrategy.FLIPPED_TWO, 2] = (
        3 - converted[adj_card == AdjStrategy.FLIPPED_TWO, 0]
    )
    converted[adj_card == AdjStrategy.FLIPPED_TWO, 3] = 2 - 2

    for idx in range(board_num):
        if adj_card[idx] == AdjStrategy.TRUMP_MAXIMUM:
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

    converted[:, 2] = AdjStrategy.RANDOM

    for idx in range(board_num):
        if decl[idx, 2] == strongest[idx, decl[idx, 0]]:
            # It can be AdjStrategy.RANDOM.
            # However, we will ignore this because the possibility of "false-positive" is too low.
            converted[:, 2] = AdjStrategy.TRUMP_MAXIMUM

    converted[
        (decl[:, 2] == (3 - decl[:, 0])) & (decl[:, 3] == 2 - 2), 2
    ] = AdjStrategy.FLIPPED_TWO
    converted[
        (decl[:, 2] == decl[:, 0]) & (decl[:, 3] == 2 - 2), 2
    ] = AdjStrategy.TRUMP_TWO
    converted[
        (decl[:, 2] == Suit.HEART) & (decl[:, 3] == 12 - 2), 2
    ] = AdjStrategy.PARTNER
    converted[
        (decl[:, 2] == (3 - decl[:, 0])) & (decl[:, 3] == 11 - 2), 2
    ] = AdjStrategy.SUB_JACK
    converted[
        (decl[:, 2] == decl[:, 0]) & (decl[:, 3] == 11 - 2), 2
    ] = AdjStrategy.MAIN_JACK
    converted[
        (decl[:, 2] == Suit.SPADE) & (decl[:, 3] == 14 - 2), 2
    ] = AdjStrategy.ALMIGHTY

    return converted


def convert_strategy_oriented_to_input(
    decl_input: NDFloatArray, decl: NDIntArray
) -> NDFloatArray:
    size = decl.shape[0]

    assert decl_input.shape == (size, 60)
    assert decl.shape == (size, 3)

    normalized_decl = np.zeros((size, 4 + 1 + AdjStrategy.LENGTH))
    normalized_decl[:, 0:4] = np.eye(4)[decl[:, 0]]
    normalized_decl[:, 4] = (decl[:, 1] - 12) / 8
    normalized_decl[:, 4 + 1 : 4 + 1 + AdjStrategy.LENGTH] = np.eye(AdjStrategy.LENGTH)[
        decl[:, 2]
    ]

    inputs = np.concatenate((decl_input, normalized_decl), axis=1)
    return inputs
