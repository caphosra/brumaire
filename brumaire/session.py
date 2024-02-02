import numpy as np
from typing import List

from . import *
from brumaire.agent import AgentBase
from brumaire.board import BoardData

def generate_board() -> BoardData:
    cards = np.zeros((4, 54))
    taken = np.zeros(5)
    roles = np.zeros(5)
    decl = np.array([SUIT_SPADE, 12])
    lead = np.array([0, SUIT_JOKER])

    owners = np.concatenate((np.repeat(np.arange(5), 10), np.array([5, 5, 5, 5])))
    np.random.shuffle(owners)
    cards[0] = CARD_IN_HAND
    cards[1] = owners

    for player in range(5):
        cards[2, cards[1] == player] = np.arange(10)

    cards[:, cards[1] == 5] = np.array([CARD_UNKNOWN, 0, 0, 0])

    return BoardData(cards, taken, roles, decl, lead)

def suit_to_str(suit: int) -> str:
    if suit == SUIT_CLUB:
        return "CLUB"
    elif suit == SUIT_DIAMOND:
        return "DIAMOND"
    elif suit == SUIT_HEART:
        return "HEART"
    elif suit == SUIT_SPADE:
        return "SPADE"
    else:
        return "JOKER"

def card_to_str(card: np.ndarray, convert_number: bool = True) -> str:
    assert card.shape == (2,)

    if card[0] == SUIT_JOKER:
        return "JOKER"
    else:
        suit = suit_to_str(card[0])
        if convert_number:
            num = card[1].astype(np.int64) + 2
            if num == 11:
                num_text = "J"
            elif num == 12:
                num_text = "Q"
            elif num == 13:
                num_text = "K"
            elif num == 14:
                num_text = "A"
            else:
                num_text = str(num)
        else:
            num_text = str(card[1].astype(np.int64))
        return f"{suit} {num_text}"

class Game:
    board: BoardData
    agents: List[AgentBase]

    def __init__(self, agents: List[AgentBase]) -> None:
        assert len(agents) == 5

        self.agents = agents

        self.init_board()

    def init_board(self) -> None:
        self.board = generate_board()

    def extract_choice(self, array: np.ndarray, filter_arr: np.ndarray) -> int:
        filtered = np.maximum(array, 0.) * filter_arr
        sum = np.sum(filtered)
        if sum == 0.:
            possibility = filter_arr / np.sum(filter_arr)
        else:
            possibility = filtered / sum

        return np.random.choice(10, 1, p=possibility)[0]

    def get_declaration(self, current_decl: np.ndarray, suit: int) -> np.ndarray:
        if current_decl[0] < suit:
            return np.array([suit, current_decl[1]])
        else:
            return np.array([suit, current_decl[1] + 1])

    def decide_napoleon(self) -> None:
        first_player = np.random.randint(0, 5)

        print(f"player{first_player} is a first player.")

        declarations = np.zeros((5, 4))
        for player in range(5):
            board = self.board.change_perspective(player)

            declarations[player] = self.agents[player].declare_goal(board)

            print(f"player{player} declared {card_to_str(declarations[player, [0, 1]], False)}.")

        rolled_declarations = np.roll(declarations, -first_player, axis=0)

        declaration_indexes = (rolled_declarations[:, 1] * 4 + rolled_declarations[:, 0]) * 5 - np.arange(5)

        if np.amax(declaration_indexes) < (13 * 4 + SUIT_CLUB) * 5 - 4:
            # All of the players want to fold.
            # Choose a napoleon randomly.
            napoleon = np.random.randint(0, 5)
            declarations[napoleon, 1] = 13
            self.board.decl = declarations[napoleon, [0, 1]]
        else:
            napoleon = np.argmax(declaration_indexes)
            second = np.sort(declaration_indexes)[-2]

            # Reduce the number of cards.
            declaration_red = (declaration_indexes[napoleon] - second) // 5 // 4
            napoleon = (first_player + napoleon) % 5
            declarations[napoleon, 1] -= declaration_red
            declarations[napoleon, 1] = max(13, declarations[napoleon, 1])

            self.board.decl = declarations[napoleon, [0, 1]]

        print(f"player{napoleon} is a napoleon for {card_to_str(declarations[napoleon, [0, 1]], False)}.")

        self.board.roles[napoleon] = ROLE_NAPOLEON
        self.board.lead = np.array([napoleon, SUIT_JOKER])

        adj_declaration_card = declarations[napoleon, [2, 3]].astype(np.int64)
        adjutant_card_text = card_to_str(adj_declaration_card)
        print(f"The adjutant card is {adjutant_card_text}.")

        adjutant_card = self.board.cards[:, adj_declaration_card[0] * 13 + adj_declaration_card[1]]
        adjutant_card[3] = 1.
        if adjutant_card[0] == CARD_IN_HAND and adjutant_card[1] != napoleon:
            self.board.roles[adjutant_card[1].astype(np.int64)] = ROLE_ADJUTANT
        self.board.roles[self.board.roles == ROLE_UNKNOWN] = ROLE_ALLY

    def discard_additional_cards(self) -> None:
        napoleon = self.board.get_napoleon()

        self.board.cards[1, self.board.cards[0] == CARD_UNKNOWN] = napoleon
        self.board.cards[0, self.board.cards[0] == CARD_UNKNOWN] = CARD_IN_HAND

        napoleons_hand = self.board.hand(napoleon)

        self.board.cards[2, napoleons_hand] = np.arange(14)

        board = self.board.change_perspective(napoleon)

        four_hots_decision = self.agents[napoleon].discard(board)
        decision = np.where(four_hots_decision > 0.)[0]

        cards_to_be_discarded = napoleons_hand & np.isin(self.board.cards[2], decision)

        cards = np.argwhere(cards_to_be_discarded)[:, 0]
        for card in cards:
            card_text = card_to_str(np.array([card // 13, card % 13]))
            print(f"player{napoleon} discards {card_text}.")

        self.board.cards[0, cards_to_be_discarded] = CARD_TRICKED
        self.board.cards[1, cards_to_be_discarded] = napoleon
        self.board.cards[2, cards_to_be_discarded] = -1

        napoleons_hand = self.board.hand(napoleon)
        self.board.cards[2, napoleons_hand] = np.arange(10)

    def trick(self, turn_num: int) -> None:
        first_player: int = self.board.lead[0].astype(np.int64)

        cards_tricked = np.zeros(5)

        for player_index in range(5):
            player = (first_player + player_index) % 5

            board = self.board.change_perspective(player)

            hand_filter = self.board.get_hand_filter(player)

            decision = self.agents[player].put_card(board)
            num = self.extract_choice(decision, hand_filter)

            selected_card = self.board.hand(player) & (self.board.cards[2] == num)

            card = np.argwhere(selected_card)[0][0]
            card_text = card_to_str(np.array([card // 13, card % 13]))
            print(f"player{player} tricks {card_text}.")

            cards_tricked[player] = card
            self.board.cards[np.array([0, 1, 2]), selected_card] = np.array([CARD_TRICKED, player, turn_num * 5 + player_index])

            if player_index == 0:
                suit = card // 13
                if suit == SUIT_JOKER:
                    suit = self.board.decl[0]
                self.board.lead[1] = suit

        winner = self.board.get_trick_winner(cards_tricked, first_player)
        print(f"player{winner} wins the trick.")

        self.board.lead = np.array([winner, SUIT_JOKER])
        self.board.taken[winner] += np.count_nonzero(cards_tricked % 13 >= (10 - 2))

    def game(self) -> None:
        self.decide_napoleon()
        self.discard_additional_cards()
        for i in range(10):
            self.trick(i)
        taken_by_napoleon = np.sum(self.board.taken[(self.board.roles == ROLE_NAPOLEON) | (self.board.roles == ROLE_ADJUTANT)])
        if taken_by_napoleon >= self.board.decl[1]:
            print("The napoleon and their adjutant win.")
        else:
            print("The allies win.")
