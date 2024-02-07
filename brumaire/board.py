import numpy as np
import numpy.typing as npt
from typing import Self

from . import *

type NDIntArray = npt.NDArray[np.int64]

class BoardData:
    board_num: int
    cards: NDIntArray
    taken: NDIntArray
    roles: NDIntArray
    decl: NDIntArray
    lead: NDIntArray

    """
    [<a player who tricked the lead>, <a suit of the lead>]
    """

    def __init__(self, board_num: int, cards: NDIntArray, taken: NDIntArray, roles: NDIntArray, decl: NDIntArray, lead: NDIntArray) -> Self:
        assert cards.shape == (board_num, 54, 4)
        assert taken.shape == (board_num, 5)
        assert roles.shape == (board_num, 5)
        assert decl.shape == (board_num, 2)
        assert lead.shape == (board_num, 2)

        self.board_num = board_num
        self.cards = cards
        self.taken = taken
        self.roles = roles
        self.decl = decl
        self.lead = lead

    def get_card_status(self, suit: int, num: int) -> NDIntArray:
        return self.cards[:, :, suit * 13 + num]

    def get_napoleon(self) -> NDIntArray:
        """
        Find the player who is a napoleon.
        You should not call this method before a napoleon is determined.
        """

        return np.argwhere(self.roles == ROLE_NAPOLEON)[:, 1]

    def change_perspective(self, player: int) -> Self:
        cards = self.cards.copy()
        taken = self.taken.copy()
        roles = self.roles.copy()
        declaration = self.decl.copy()
        lead = self.lead.copy()

        # Roll the lists to make the player first.
        taken = np.roll(taken, -player, axis=1)
        roles = np.roll(roles, -player, axis=1)

        # Change the first player.
        lead[0] = (lead[0] - player) % 5

        # Hide the role information if the adjutant card has not been public.
        is_role_unknown = np.any((cards[:, :, 0] == CARD_IN_HAND) & (cards[:, :, 1] != player) & (cards[:, :, 3] == 1), axis=1)
        is_role_unknown = np.repeat(np.reshape(is_role_unknown, (-1, 1)), 5, axis=1)

        assert np.shape(is_role_unknown) == (self.board_num, 5)

        player_role = np.copy(roles[:, 0])
        roles[is_role_unknown & (roles == ROLE_ADJUTANT)] = ROLE_UNKNOWN
        roles[is_role_unknown & (roles == ROLE_ALLY)] = ROLE_UNKNOWN
        roles[:, 0] = player_role

        # Update owners of cards.
        card_known = cards[:, :, 0] != CARD_UNKNOWN
        cards[card_known, 1] = (cards[card_known, 1] - player) % 5

        # Mark unknown the cards which the others have.
        assert CARD_UNKNOWN == 0
        others_own = (cards[:, :, 0] == CARD_IN_HAND) & (cards[:, :, 1] != 0)
        cards[others_own] = np.repeat(np.array([[CARD_UNKNOWN, 0, 0, 0]]), len(cards[others_own]), axis=0)

        return BoardData(self.board_num, cards, taken, roles, declaration, lead)

    def hand(self, player: int) -> NDIntArray:
        return (self.cards[:, 0] == CARD_IN_HAND) & (self.cards[:, 1] == player)

    def get_hand_filter(self, player: int) -> np.ndarray:
        lead_suit = self.lead[:, 1]
        is_trump: bool = self.decl[:, 0] == lead_suit
        hand: np.ndarray = self.cards[:, 2, self.hand(player)]

        if lead_suit != SUIT_JOKER:
            possible_cards_index = np.arange(lead_suit * 13, (lead_suit + 1) * 13)
            if is_trump:
                possible_cards_index = np.concatenate((possible_cards_index, np.array([SUIT_JOKER * 13, SUIT_JOKER * 13 + 1])))
            possible_cards = self.cards[:, possible_cards_index]
            restricted_hand = possible_cards[2, (possible_cards[0] == CARD_IN_HAND) & (possible_cards[1] == player)]
            if len(restricted_hand) > 0:
                hand = restricted_hand

        return np.sum(np.eye(10)[hand.astype(np.int64)], axis=0)

    def get_trick_winner(self, cards: np.ndarray, first_player: int) -> int:
        trump = self.decl[0].astype(np.int64)
        lead_suit = self.lead[1].astype(np.int64)

        # SPADE A
        almighty_card = SUIT_SPADE * 13 + (1 - 2 + 13)
        # HEART Q
        partner_card = SUIT_HEART * 13 + (12 - 2)
        # TRUMP J
        main_jack = trump * 13 + (11 - 2)
        # Flipped TRUMP J
        if trump == SUIT_SPADE:
            sub_jack = SUIT_CLUB * 13 + (11 - 2)
        elif trump == SUIT_HEART:
            sub_jack = SUIT_DIAMOND * 13 + (11 - 2)
        elif trump == SUIT_DIAMOND:
            sub_jack = SUIT_HEART * 13 + (11 - 2)
        else:
            assert trump == SUIT_CLUB
            sub_jack = SUIT_SPADE * 13 + (11 - 2)

        if np.any(cards == almighty_card) and np.any(cards == partner_card):
            return np.where(cards == partner_card)[0][0]
        elif np.any(cards == almighty_card):
            return np.where(cards == almighty_card)[0][0]
        elif np.any(cards == main_jack):
            return np.where(cards == main_jack)[0][0]

        suit_list = cards.copy() // 13
        suit_list[suit_list == SUIT_JOKER] = trump
        if np.all(suit_list == suit_list[0]):
            if np.any(cards == suit_list[0] * 13 + (2 - 2)):
                return np.where(cards == suit_list[0] * 13 + (2 - 2))[0][0]

        if np.any(cards == sub_jack):
            return np.where(cards == sub_jack)[0][0]

        if np.any((trump * 13 <= cards) & (cards < (trump + 1) * 13)):
            strong = np.amax(cards[(trump * 13 <= cards) & (cards < (trump + 1) * 13)])
            return np.where(cards == strong)[0][0]

        if cards[first_player] // 13 == SUIT_JOKER:
            return first_player

        assert np.any((lead_suit * 13 <= cards) & (cards < (lead_suit + 1) * 13))

        strong = np.amax(cards[(lead_suit * 13 <= cards) & (cards < (lead_suit + 1) * 13)])
        return np.where(cards == strong)[0][0]

def generate_board(board_num: int) -> BoardData:
    cards = np.zeros((board_num, 54, 4))
    taken = np.zeros((board_num, 5))
    roles = np.zeros((board_num, 5))
    decl = np.repeat(np.array([[SUIT_SPADE, 12]]), board_num, axis=0)
    lead = np.repeat(np.array([[0, SUIT_JOKER]]), board_num, axis=0)

    cards[:, :, 0] = CARD_IN_HAND

    for idx in range(board_num):
        # Shuffle the numbers.
        owners = np.concatenate((np.repeat(np.arange(5), 10), np.array([5, 5, 5, 5])))
        np.random.shuffle(owners)
        cards[idx, :, 1] = owners

        # Index the cards.
        for player in range(5):
            cards[idx, cards[idx, :, 1] == player, 2] = np.arange(10)

        # Reset parameters of cards which no one holds.
        cards[idx, cards[idx, :, 1] == 5] = np.array([CARD_UNKNOWN, 0, 0, 0])

    return BoardData(board_num, cards, taken, roles, decl, lead)
