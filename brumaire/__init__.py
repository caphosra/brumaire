import numpy as np
from typing import Self, List

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

REQ_DEP = 0
REQ_ADJ_S = 1
REQ_ADJ_N = 2
REQ_DISC = 3
REQ_TRICK = 4

RES_SIZE = 14

class BoardData:
    cards: np.ndarray
    taken: np.ndarray
    roles: np.ndarray
    decl: np.ndarray
    lead: np.ndarray
    """
    [<a player who tricked the lead>, <a suit of the lead>]
    """

    def __init__(self, cards: np.ndarray, taken: np.ndarray, roles: np.ndarray, decl: np.ndarray, lead: np.ndarray) -> Self:
        assert cards.shape == (4, 54)
        assert taken.shape == (5,)
        assert roles.shape == (5,)
        assert decl.shape == (2,)
        assert lead.shape == (2,)

        self.cards = cards
        self.taken = taken
        self.roles = roles
        self.decl = decl
        self.lead = lead

    def get_card_status(self, suit: int, num: int) -> np.ndarray:
        return self.cards[:, suit * 13 + num]

    def get_napoleon(self) -> int:
        """
        Find the player who is a napoleon.
        You should not call this method before a napoleon is determined.
        """

        return np.argwhere(self.roles == ROLE_NAPOLEON).reshape(1)[0]

    def change_perspective(self, player: int) -> Self:
        cards = self.cards.copy()
        taken = self.taken.copy()
        roles = self.roles.copy()
        declaration = self.decl.copy()
        lead = self.lead.copy()

        # Roll the lists to make the player first.
        taken = np.roll(taken, -player)
        roles = np.roll(roles, -player)

        # Change the first player.
        lead[0] = (lead[0] - player) % 5

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

        return BoardData(cards, taken, roles, declaration, lead)

    def hand(self, player: int) -> np.ndarray:
        return (self.cards[0] == CARD_IN_HAND) & (self.cards[1] == player)

    def get_hand_filter(self, player: int) -> np.ndarray:
        lead_suit = self.lead[1].astype(np.int64)
        is_trump: bool = self.decl[0].astype(np.int64) == lead_suit
        hand: np.ndarray = self.cards[2, self.hand(player)]

        if lead_suit != SUIT_JOKER:
            possible_cards_index = np.arange(lead_suit * 13, (lead_suit + 1) * 13)
            if is_trump:
                possible_cards_index = np.concatenate((possible_cards_index, np.array([SUIT_JOKER * 13, SUIT_JOKER * 13 + 1])))
            possible_cards = self.cards[:, possible_cards_index]
            restricted_hand = possible_cards[2, (possible_cards[0] == CARD_IN_HAND) & (possible_cards[1] == player)]
            if len(restricted_hand) > 0:
                hand = restricted_hand

        return np.sum(np.eye(RES_SIZE)[hand.astype(np.int64)], axis=0)

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

class AgentBase:
    def make_decision(self, board: BoardData, req: int, selected_suit: int = SUIT_JOKER) -> np.ndarray:
        pass

    def tell_reward(self, reward: float):
        pass

class RandomAgent(AgentBase):
    decl_p: float

    def __init__(self, decl_p: float = .05) -> None:
        super().__init__()

        assert 0 <= decl_p * 4 <= 1
        self.decl_p = decl_p

    def make_decision(self, board: BoardData, req: int, selected_suit: int = SUIT_JOKER) -> np.ndarray:
        if req == REQ_DEP:
            choice = np.random.choice(5, 1, p=[self.decl_p, self.decl_p, self.decl_p, self.decl_p, 1 - self.decl_p * 4])[0]
            return np.eye(RES_SIZE)[choice]
        elif req == REQ_ADJ_S:
            return np.eye(RES_SIZE)[np.random.randint(0, 4)]
        elif req == REQ_ADJ_N:
            return np.eye(RES_SIZE)[np.random.randint(0, 13)]
        elif req == REQ_DISC:
            return np.eye(RES_SIZE)[np.random.randint(0, 14)]
        elif req == REQ_TRICK:
            return np.eye(RES_SIZE)[np.random.randint(0, 10)]
        else:
            raise Exception("An unknown request is given.")

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

        return np.random.choice(RES_SIZE, 1, p=possibility)[0]

    def get_declaration(self, current_decl: np.ndarray, suit: int) -> np.ndarray:
        if current_decl[0] < suit:
            return np.array([suit, current_decl[1]])
        else:
            return np.array([suit, current_decl[1] + 1])

    def decide_napoleon(self) -> None:
        declared = False
        current_decl = np.array([SUIT_SPADE, 12])
        remained_players = np.ones(5, dtype=np.int64)
        current_player = np.random.randint(0, 5)
        while (declared and np.count_nonzero(remained_players) >= 2) \
                or (not declared and np.count_nonzero(remained_players) >= 1):
            if remained_players[current_player]:
                self.board.decl = current_decl
                board = self.board.change_perspective(current_player)
                decision = self.agents[current_player].make_decision(board, REQ_DEP)
                selected_suit = self.extract_choice(decision, np.concatenate((np.ones(5), np.zeros(RES_SIZE - 5))))
                if selected_suit == 4:
                    print(f"player{current_player} refused.")
                    remained_players[current_player] = 0
                else:
                    new_decl = self.get_declaration(current_decl, selected_suit)
                    if new_decl[1] > 20:
                        print(f"player{current_player} refused because the goal is more than 20.")
                        remained_players[current_player] = 0
                    else:
                        new_decl_str = card_to_str(new_decl, convert_number=False)
                        print(f"player{current_player} declared {new_decl_str}.")
                        current_decl = new_decl
                        declared = True
            current_player += 1
            current_player %= 5

        # If no one declared their goal, force someone to do it.
        # The player who have to declare the goal is selected randomly.
        if not declared:
            selected_player = np.random.randint(0, 5)
            self.board.decl = np.array([SUIT_SPADE, 12])
            board = self.board.change_perspective(selected_player)

            decision = self.agents[selected_player].make_decision(board, REQ_DEP)
            selected_suit = self.extract_choice(decision, np.concatenate((np.ones(4), np.zeros(RES_SIZE - 4))))

            current_decl = np.array([selected_suit, 13])
            remained_players[selected_player] = 1.

        napoleon = remained_players.argmax()
        self.board.decl = current_decl
        self.board.roles[napoleon] = ROLE_NAPOLEON
        self.board.lead = np.array([napoleon, SUIT_JOKER])

    def decide_adjutant(self) -> None:
        napoleon = self.board.get_napoleon()
        board = self.board.change_perspective(napoleon)

        decision = self.agents[napoleon].make_decision(board, REQ_ADJ_S)
        selected_suit = self.extract_choice(decision, np.concatenate((np.ones(4), np.zeros(RES_SIZE - 4))))

        decision = self.agents[napoleon].make_decision(board, REQ_ADJ_N, selected_suit)
        num = self.extract_choice(decision, np.concatenate((np.ones(13), np.zeros(RES_SIZE - 13))))

        adjutant_card_text = card_to_str(np.array([selected_suit, num]))
        print(f"The adjutant card is {adjutant_card_text}.")

        adjutant_card = self.board.cards[:, selected_suit * 13 + num]
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

        for _ in range(4):
            board = self.board.change_perspective(napoleon)

            hand_filter = self.board.get_hand_filter(napoleon)

            decision = self.agents[napoleon].make_decision(board, REQ_DISC)
            num = self.extract_choice(decision, hand_filter)

            card = np.argwhere(napoleons_hand & (self.board.cards[2] == num))[0][0]
            card_text = card_to_str(np.array([card // 13, card % 13]))
            print(f"player{napoleon} discards {card_text}.")

            self.board.cards[np.array([0, 1, 2]), napoleons_hand & (self.board.cards[2] == num)] = np.array([CARD_TRICKED, napoleon, -1])

    def trick(self, turn_num: int) -> None:
        first_player: int = self.board.lead[0].astype(np.int64)

        cards_tricked = np.zeros(5)

        for player_index in range(5):
            player = (first_player + player_index) % 5

            board = self.board.change_perspective(player)

            hand_filter = self.board.get_hand_filter(player)

            decision = self.agents[player].make_decision(board, REQ_TRICK)
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
        self.decide_adjutant()
        self.discard_additional_cards()
        for i in range(10):
            self.trick(i)
        taken_by_napoleon = np.sum(self.board.taken[(self.board.roles == ROLE_NAPOLEON) | (self.board.roles == ROLE_ADJUTANT)])
        if taken_by_napoleon >= self.board.decl[1]:
            print("The napoleon and their adjutant win.")
        else:
            print("The allies win.")
