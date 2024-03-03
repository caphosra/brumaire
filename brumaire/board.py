from __future__ import annotations
import numpy as np
from typing import Tuple
import itertools

from brumaire.constants import (
    CardStatus,
    NDBoolArray,
    NDFloatArray,
    NDIntArray,
    Role,
    Suit,
)


class BoardData:
    VEC_SIZE = 54 * 4 + 5 + 5 + 2 + 2

    TRICK_INPUT_SIZE = 54 * 4 + 5 + 5 + 2 + 2

    board_num: int
    cards: NDIntArray
    taken: NDIntArray
    roles: NDIntArray
    decl: NDIntArray
    lead: NDIntArray

    suit_transform: NDIntArray

    """
    [<a player who tricked the lead>, <a suit of the lead>]
    """

    def __init__(
        self,
        board_num: int,
        cards: NDIntArray,
        taken: NDIntArray,
        roles: NDIntArray,
        decl: NDIntArray,
        lead: NDIntArray,
    ) -> None:
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

        self.suit_transform = np.zeros((5, 54), dtype=bool)
        for suit in range(4):
            self.suit_transform[suit, (suit * 13) : ((suit + 1) * 13)] = True
        self.suit_transform[Suit.JOKER, :] = True

    @staticmethod
    def generate(board_num: int) -> BoardData:
        """
        Generates a board with cards shuffled.
        """

        cards = np.zeros((board_num, 54, 4), dtype=int)
        taken = np.zeros((board_num, 5), dtype=int)
        roles = np.zeros((board_num, 5), dtype=int)
        decl = np.repeat(np.array([[Suit.SPADE, 12]], dtype=int), board_num, axis=0)
        lead = np.repeat(np.array([[0, Suit.JOKER]], dtype=int), board_num, axis=0)

        cards[:, :, 0] = CardStatus.IN_HAND

        for idx in range(board_num):
            # Shuffle the numbers.
            owners = np.concatenate(
                (np.repeat(np.arange(5), 10), np.array([5, 5, 5, 5]))
            )
            np.random.shuffle(owners)
            cards[idx, :, 1] = owners

            # Index the cards.
            for player in range(5):
                cards[idx, cards[idx, :, 1] == player, 2] = np.arange(10)

            # Reset parameters of cards which no one holds.
            cards[idx, cards[idx, :, 1] == 5] = np.array([CardStatus.UNKNOWN, 0, 0, 0])

        return BoardData(board_num, cards, taken, roles, decl, lead)

    def to_vector(self) -> NDFloatArray:
        """
        Converts this into a float vector.
        """

        cards = self.cards.copy().astype(float)
        cards[:, :, 0] /= 2.0
        cards[:, :, 1] /= 4.0
        cards[:, :, 2] /= 50.0
        cards = cards.reshape((self.board_num, 54 * 4))

        taken = self.taken.copy().astype(float)
        taken /= 20.0

        roles = self.roles.copy().astype(float)
        roles /= 3.0

        decl = self.decl.copy().astype(float)
        decl[:, 0] /= 3.0
        decl[:, 1] = (decl[:, 1] - 12.0) / 8.0

        lead = self.lead.copy().astype(float)
        lead[:, 0] /= 4.0
        lead[:, 1] /= 4.0

        return np.concatenate((cards, taken, roles, decl, lead), axis=1)

    @staticmethod
    def from_vector(vec: NDFloatArray) -> BoardData:
        """
        An invert method of `to_vector`.
        """

        assert vec.shape[1] == BoardData.VEC_SIZE

        board_num = vec.shape[0]

        cards = vec[:, 0 : 54 * 4].reshape((board_num, 54, 4)).copy()
        cards[:, :, 0] *= 2
        cards[:, :, 1] *= 4
        cards[:, :, 2] *= 50
        cards = cards.astype(int)

        taken = vec[:, 54 * 4 : 54 * 4 + 5].copy()
        taken *= 20
        taken = taken.astype(int)

        roles = vec[:, 54 * 4 + 5 : 54 * 4 + 5 + 5].copy()
        roles *= 3
        roles = roles.astype(int)

        decl = vec[:, 54 * 4 + 5 + 5 : 54 * 4 + 5 + 5 + 2].copy()
        decl[:, 0] *= 3
        decl[:, 1] = decl[:, 1] * 8 + 12
        decl = decl.astype(int)

        lead = vec[:, 54 * 4 + 5 + 5 + 2 : 54 * 4 + 5 + 5 + 2 + 2].copy()
        lead[:, 0] *= 4.0
        lead[:, 1] *= 4.0
        lead = lead.astype(int)

        return BoardData(board_num, cards, taken, roles, decl, lead)

    def get_napoleon(self) -> NDIntArray:
        """
        Find the player who is a napoleon.
        You should not call this method before a napoleon is determined.
        """

        return np.argwhere(self.roles == Role.NAPOLEON)[:, 1]

    def is_adj_revealed(self) -> NDBoolArray:
        return self.cards[self.cards[:, :, 3] == 1, 0] == CardStatus.PLAYED

    def slice_boards(self, board_filter: NDBoolArray) -> BoardData:
        board_num = np.sum(board_filter)

        cards = self.cards[board_filter]
        taken = self.taken[board_filter]
        roles = self.roles[board_filter]
        declaration = self.decl[board_filter]
        lead = self.lead[board_filter]

        return BoardData(board_num, cards, taken, roles, declaration, lead)

    def change_perspective(self, players: NDIntArray) -> BoardData:
        assert players.shape == (self.board_num,)

        cards = self.cards.copy()
        taken = self.taken.copy()
        roles = self.roles.copy()
        declaration = self.decl.copy()
        lead = self.lead.copy()

        # Roll the lists to make the player first.
        for idx in range(self.board_num):
            taken[idx] = np.roll(taken[idx], -players[idx])
            roles[idx] = np.roll(roles[idx], -players[idx])

        # Change the first player.
        lead[:, 0] = (lead[:, 0] - players) % 5

        # Hide the role information if the adjutant card has not been public.
        is_role_unknown = np.any(
            (cards[:, :, 0] == CardStatus.IN_HAND)
            & ((cards[:, :, 1].T == players).T)
            & (cards[:, :, 3] == 1),
            axis=1,
        )
        is_role_unknown = np.repeat(np.reshape(is_role_unknown, (-1, 1)), 5, axis=1)

        assert np.shape(is_role_unknown) == (self.board_num, 5)

        player_role = np.copy(roles[:, 0])
        roles[is_role_unknown & (roles == Role.ADJUTANT)] = Role.UNKNOWN
        roles[is_role_unknown & (roles == Role.ALLY)] = Role.UNKNOWN
        roles[:, 0] = player_role

        # Update owners of cards.
        for idx in range(self.board_num):
            card_known = cards[idx, :, 0] != CardStatus.UNKNOWN
            cards[idx, card_known, 1] = (cards[idx, card_known, 1] - players[idx]) % 5

        # Mark unknown the cards which the others have.
        assert CardStatus.UNKNOWN == 0
        others_own = (cards[:, :, 0] == CardStatus.IN_HAND) & (cards[:, :, 1] != 0)
        cards[others_own] = np.repeat(
            np.array([[CardStatus.UNKNOWN, 0, 0, 0]]), len(cards[others_own]), axis=0
        )

        return BoardData(self.board_num, cards, taken, roles, declaration, lead)

    def change_perspective_to_one(self, player: int) -> BoardData:
        return self.change_perspective(np.repeat(player, self.board_num))

    def get_suits_map(self, suits: NDIntArray) -> NDBoolArray:
        return self.suit_transform[suits]

    def get_hand(self, idx: int, player: int) -> NDBoolArray:
        return (self.cards[idx, :, 0] == CardStatus.IN_HAND) & (
            self.cards[idx, :, 1] == player
        )

    def get_hands(self, players: NDIntArray) -> NDBoolArray:
        """
        Get hands of players across boards.
        """

        assert players.shape == (self.board_num,)

        return (self.cards[:, :, 0] == CardStatus.IN_HAND) & (
            self.cards[:, :, 1].T == players
        ).T

    def get_players_hands(self, player: int) -> NDBoolArray:
        return self.get_hands(np.ones(self.board_num, dtype=np.int64) * player)

    def get_hand_filter(self, player: int) -> NDBoolArray:
        lead_suit = self.lead[:, 1]
        is_trump: NDBoolArray = self.decl[:, 0] == lead_suit

        suits_map = self.get_suits_map(lead_suit)
        suits_map[is_trump, 52:54] = True

        hands = self.get_players_hands(player)
        possible_cards = suits_map & hands

        nothing = ~np.any(possible_cards, axis=1)
        possible_cards[nothing] = hands[nothing]

        assert np.all(self.cards[possible_cards, 0] == CardStatus.IN_HAND)
        assert np.all(self.cards[possible_cards, 1] == player)

        return possible_cards

    def get_trick_winner(self, cards: NDIntArray) -> NDIntArray:
        trump = self.decl[:, 0]
        lead_suit = self.lead[:, 1]
        scores = cards.copy()
        suits = cards // 13

        # SPADE A
        almighty_card = Suit.SPADE * 13 + (1 - 2 + 13)
        # HEART Q
        partner_card = Suit.HEART * 13 + (12 - 2)
        # TRUMP J
        main_jack = trump * 13 + (11 - 2)
        # Flipped TRUMP J
        sub_jack = np.zeros(self.board_num)
        sub_jack[trump == Suit.SPADE] = Suit.CLUB * 13 + (11 - 2)
        sub_jack[trump == Suit.HEART] = Suit.DIAMOND * 13 + (11 - 2)
        sub_jack[trump == Suit.DIAMOND] = Suit.HEART * 13 + (11 - 2)
        sub_jack[trump == Suit.CLUB] = Suit.SPADE * 13 + (11 - 2)
        # LEAD 2
        lead_two = lead_suit * 13 + (2 - 2)

        # Score the cards to determine the winners.
        LEAD_BONUS = 100
        TRUMP_BONUS = LEAD_BONUS * 3
        SUB_JACK_BONUS = TRUMP_BONUS * 3
        SAME_TWO_BONUS = SUB_JACK_BONUS * 3
        MAIN_JACK_BONUS = SAME_TWO_BONUS * 3
        ALMIGHTY_BONUS = MAIN_JACK_BONUS * 3
        PARTNER_CARD_BONUS = ALMIGHTY_BONUS * 3

        for idx in range(self.board_num):
            scores[idx, suits[idx] == lead_suit[idx]] += LEAD_BONUS
            if scores[idx, 0] == Suit.JOKER:
                scores[idx, 0] = LEAD_BONUS - 1

            scores[idx, suits[idx] == trump[idx]] += TRUMP_BONUS
            scores[idx, cards[idx] == sub_jack[idx]] += SUB_JACK_BONUS

            suits[idx, suits[idx] == Suit.JOKER] = trump[idx]
            if np.all(suits[idx] == lead_suit[idx]):
                scores[idx, cards[idx] == lead_two[idx]] += SAME_TWO_BONUS

            scores[idx, cards[idx] == main_jack[idx]] += MAIN_JACK_BONUS
            scores[idx, cards[idx] == almighty_card] += ALMIGHTY_BONUS

            if np.any(cards[idx] == almighty_card) and np.any(
                cards[idx] == partner_card
            ):
                scores[idx, cards[idx] == partner_card] += PARTNER_CARD_BONUS

        return np.argmax(scores, axis=1)

    def get_strongest_for_each_suits(self) -> NDIntArray:
        """
        Get a strongest card in the players hand.
        If there are no cards that meets the conditions, it will return a random card.

        This method works properly only after calling `change_perspective`.
        """

        strongest = np.random.randint(52, size=(self.board_num, 4))
        for idx in range(self.board_num):
            for suit in range(4):
                suits_card = self.cards[idx, suit * 13 : (suit + 1) * 13]
                players_cards = np.argwhere(
                    (suits_card[:, 0] == CardStatus.IN_HAND) & (suits_card[:, 1] == 0)
                )
                if players_cards.size != 0:
                    strongest_card = players_cards[-1][0]
                    strongest[idx, suit] = suit * 13 + strongest_card
        return strongest

    def get_adj_card(self, idx: int) -> Tuple[int, int]:
        adj_card = np.argwhere(self.cards[idx, :, 3] == 1)[0][0]
        return adj_card // 13, adj_card % 13 + 2

    def convert_to_decl_input(self, player: int) -> NDFloatArray:
        decl_input = np.zeros((self.board_num, 60))

        player_owns = np.argwhere(self.get_players_hands(player))
        for idx in range(self.board_num):
            cards = player_owns[player_owns[:, 0] == idx, 1]
            decl_input[idx, 0:50] = np.eye(5)[cards // 13].flatten()
            decl_input[idx, 50:] = (cards % 13) / 12

        return decl_input

    def to_trick_input(self) -> NDFloatArray:
        trick_input = self.to_vector()
        return trick_input

    def reindex_hands(self, players: NDIntArray, card_num: int = 10) -> None:
        hands = self.get_hands(players)
        self.cards[hands, 2] = np.repeat(
            np.arange(card_num)[None, :], self.board_num, axis=0
        ).flatten()

    def enumerate_discard_patterns(self) -> Tuple[NDIntArray, NDFloatArray]:
        all_indexes = list(range(14))

        comb: NDIntArray = np.array(
            list(itertools.combinations(all_indexes, 4)), dtype=int
        )
        comb_len = comb.shape[0]

        vec_lists = np.zeros((comb_len, self.board_num, self.VEC_SIZE))

        for pattern in range(comb_len):
            cards = self.cards.copy()
            for disc in range(4):
                disc_cards = (
                    (self.cards[:, :, 0] == CardStatus.IN_HAND)
                    & (self.cards[:, :, 1] == 0)
                    & (self.cards[:, :, 2] == comb[pattern, disc])
                )
                cards[disc_cards, 0] = CardStatus.PLAYED
                cards[disc_cards, 2] = -1
            board = BoardData(
                self.board_num, cards, self.taken, self.roles, self.decl, self.lead
            )

            players = np.zeros(self.board_num, dtype=int)
            board.reindex_hands(players)

            vec_lists[pattern] = board.to_vector()

        return comb, vec_lists
