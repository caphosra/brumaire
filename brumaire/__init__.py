import numpy as np
from typing import Self

SUIT_SPADE = 0
SUIT_HEART = 1
SUIT_DIAMOND = 2
SUIT_CLUB = 3
SUIT_JOKER = 4

CARD_UNKNOWN = 0
CARD_IN_HAND = 1
CARD_TRICKED = 2

ROLE_UNKNOWN = 0
ROLE_NAPOLEON = 1
ROLE_ADJUTANT = 2
ROLE_ALLY = 3

REQ_DEP = 0
REQ_ADJ_S = 1
REQ_ADJ_N = 2
REQ_DISC = 3
REQ_TRICK = 4

class BoardData:
    cards: np.ndarray
    taken: np.ndarray
    roles: np.ndarray
    declaration: np.ndarray

    def __init__(self, cards: np.ndarray, taken: np.ndarray, roles: np.ndarray, declaration: np.ndarray) -> Self:
        assert cards.shape == (4, 54)
        assert taken.shape == (5,)
        assert roles.shape == (5,)
        assert declaration.shape == (2,)

        self.cards = cards
        self.taken = taken
        self.roles = roles
        self.declaration = declaration

    def get_card_status(self, suit: int, num: int) -> np.ndarray:
        return self.cards[:, suit * 13 + num]

    def change_perspective(self, player: int) -> Self:
        cards = self.cards.copy()
        taken = self.taken.copy()
        roles = self.roles.copy()
        declaration = self.declaration.copy()

        # Roll the lists to make the player first.
        taken = np.roll(taken, -player)
        roles = np.roll(roles, -player)

        # Hide the role information if the adjutant card has not been public.
        is_role_unknown = np.any((cards[0] == CARD_IN_HAND) & (cards[1] != player) & (cards[3] == 1))
        if is_role_unknown:
            player_role = roles[0]
            roles[roles == ROLE_ADJUTANT] = ROLE_UNKNOWN
            roles[roles == ROLE_ALLY] = ROLE_UNKNOWN
            roles[0] = player_role

        # Update owners of cards.
        card_known = cards[0] != CARD_UNKNOWN
        cards[1, card_known] = (cards[1, card_known] - player) % 5

        # Mark unknown the cards which the others have.
        assert CARD_UNKNOWN == 0
        others_own = (cards[0] == CARD_IN_HAND) & (cards[1] != 0)
        cards[0, others_own] = CARD_UNKNOWN
        cards[1, others_own] = 0
        cards[2, others_own] = 0

        return BoardData(cards, taken, roles, declaration)

def generate_board() -> BoardData:
    cards = np.zeros((4, 54))
    taken = np.zeros(5)
    roles = np.zeros(5)
    declaration = np.zeros(2)

    owners = np.concatenate((np.repeat(np.arange(5), 10), np.array([5, 5, 5, 5])))
    np.random.shuffle(owners)
    cards[0] = CARD_IN_HAND
    cards[1] = owners

    for player in range(5):
        cards[2, cards[1] == player] = np.arange(10)

    cards[:, cards[1] == 5] = np.array([CARD_UNKNOWN, 0, 0, 0])

    return BoardData(cards, taken, roles, declaration)
