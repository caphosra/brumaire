import numpy as np
from typing import List, Callable

from . import *
from brumaire.agent import AgentBase
from brumaire.board import BoardData, generate_board
from brumaire.record import Recorder

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
    board_num: int
    agents: List[AgentBase]
    log_enabled: bool
    logs: List[List[str]]
    recorder: Recorder

    def __init__(self, board_num: int, agents: List[AgentBase], log_enabled: bool = False) -> None:
        assert len(agents) == 5

        self.board_num = board_num
        self.agents = agents
        self.log_enabled = log_enabled

        if log_enabled:
            self.clear_logs()

        self.init_board()
        self.init_recorder()

    def clear_logs(self) -> None:
        assert self.log_enabled

        self.logs = list(map(lambda _: list(), range(self.board_num)))

    def init_board(self) -> None:
        self.board = generate_board(self.board_num)

    def init_recorder(self) -> None:
        self.recorder = Recorder(self.board_num)

    def log(self, log_func: Callable[[int], str]) -> None:
        if self.log_enabled:
            for idx in range(self.board_num):
                message = log_func(idx)
                self.logs[idx].append(message)

    def get_declaration(self, current_decl: np.ndarray, suit: int) -> np.ndarray:
        if current_decl[0] < suit:
            return np.array([suit, current_decl[1]])
        else:
            return np.array([suit, current_decl[1] + 1])

    def decide_napoleon(self) -> None:
        first_player = np.random.randint(5, size=self.board_num)

        self.log(lambda idx: f"player{first_player[idx]} is a first player.")

        declarations = np.zeros((self.board_num, 5, 4))
        for player in range(5):
            board = self.board.change_perspective_to_one(player)

            declarations[:, player] = self.agents[player].declare_goal(board)

            self.log(lambda idx: f"player{player} declared {card_to_str(declarations[idx, player, [0, 1]], False)}.")

            self.recorder.first_boards[player] = board.to_vector()
            self.recorder.declarations[player] = declarations[:, player, :]

        # Select a napoleon randomly.
        # The result will be dropped if at least one person wants to be.
        random_napoleon = np.amax(declarations[:, :, 1], axis=1) < 13

        self.log(
            lambda idx: "The napoleon will be selected randomly." if random_napoleon[idx] else "There is a valid goal."
        )

        randomly_selected_napoleon = \
            np.repeat(random_napoleon[:, None], 5, axis=1) \
            & (np.eye(5)[np.random.randint(5, size=self.board_num)] == 1)

        # Rewrite the declaration of randomly-selected player
        # only if all of the players want to fold.
        declarations[randomly_selected_napoleon, 1] = 13

        # Calculate a parameter to prioritize the goals.
        rolled_declarations = declarations.copy()
        for idx in range(self.board_num):
            rolled_declarations[idx] = np.roll(declarations[idx], -first_player[idx], axis=0)

        declaration_indexes = \
            (rolled_declarations[:, :, 1] * 4 + rolled_declarations[:, :, 0]) * 5 \
            - np.repeat(np.arange(5)[None, :], self.board_num, axis=0)

        napoleon = np.argmax(declaration_indexes, axis=1)
        sorted_indexes = np.sort(declaration_indexes, axis=1)
        first = sorted_indexes[:, -1]
        second = sorted_indexes[:, -2]

        # Reduce the number of cards.
        declaration_red = (first - second) // 5 // 4
        napoleon = (first_player + napoleon) % 5

        for idx in range(self.board_num):
            declarations[idx, napoleon[idx], 1] -= declaration_red[idx]
            declarations[idx, napoleon[idx], 1] = max(13, declarations[idx, napoleon[idx], 1])

            self.board.decl[idx] = declarations[idx, napoleon[idx], [0, 1]]

            self.board.roles[idx, napoleon[idx]] = ROLE_NAPOLEON
            self.board.lead[idx] = np.array([napoleon[idx], SUIT_JOKER])

            adj_declaration_card = declarations[idx, napoleon[idx], [2, 3]].astype(np.int64)

            adjutant_card = self.board.cards[idx, adj_declaration_card[0] * 13 + adj_declaration_card[1]]
            adjutant_card[3] = 1.
            if adjutant_card[0] == CARD_IN_HAND and adjutant_card[1] != napoleon[idx]:
                self.board.roles[idx, adjutant_card[1].astype(np.int64)] = ROLE_ADJUTANT
            self.board.roles[idx, self.board.roles[idx] == ROLE_UNKNOWN] = ROLE_ALLY

        self.log(lambda idx: f"player{napoleon[idx]} is a napoleon for {card_to_str(declarations[idx, napoleon[idx], [0, 1]], False)}.")
        self.log(lambda idx: f"The adjutant card is {card_to_str(declarations[idx, napoleon[idx], [2, 3]])}.")

    def discard_additional_cards(self) -> None:
        napoleons = self.board.get_napoleon()

        # Give 4 additional cards.
        for idx in range(self.board_num):
            self.board.cards[idx, self.board.cards[idx, :, 0] == CARD_UNKNOWN, 1] = napoleons[idx]
        self.board.cards[self.board.cards[:, :, 0] == CARD_UNKNOWN, 0] = CARD_IN_HAND

        # Reindex the napoleons hands.
        napoleons_hands = self.board.get_hands(napoleons)
        self.board.cards[napoleons_hands, 2] = np.repeat(np.arange(14)[None, :], self.board_num, axis=0).flatten()

        # Have the agents decide cards to be discarded.
        # This operation is conducted for each agent, not for each board.
        to_be_discarded = np.zeros((self.board_num, 14), dtype=bool)
        for player in range(5):
            napoleons_board = self.board.slice_boards(napoleons == player)
            napoleons_board = napoleons_board.change_perspective_to_one(player)

            four_hots_decision = self.agents[player].discard(napoleons_board)
            to_be_discarded[napoleons == player] = four_hots_decision > 0.

        # Extend range of the list from the hands to all of the cards.
        cards_to_be_discarded = np.zeros((self.board_num, 54), dtype=bool)
        for idx in range(self.board_num):
            cards_to_be_discarded[idx, self.board.cards[idx, : , 1] == napoleons[idx]] = to_be_discarded[idx]

        # Write the consequence of discarding cards to the log.
        if self.log_enabled:
            discarded_cards_list = np.argwhere(cards_to_be_discarded)
            discarded = np.zeros((self.board_num, 4))
            for idx in range(self.board_num):
                discarded[idx] = discarded_cards_list[discarded_cards_list[:, 0] == idx, 1]
            for card_idx in range(4):
                self.log(lambda idx: f"player{napoleons[idx]} discards {card_to_str(np.array([discarded[idx, card_idx] // 13, discarded[idx, card_idx] % 13]))}.")

        # Reflect the consequence to the boards.
        for idx in range(self.board_num):
            self.board.cards[idx, cards_to_be_discarded[idx], 0:3] = np.array([CARD_TRICKED, napoleons[idx], -1])

        # Reindex the napoleons hands.
        napoleons_hands = self.board.get_hands(napoleons)
        self.board.cards[napoleons_hands, 2] = np.repeat(np.arange(10)[None, :], self.board_num, axis=0).flatten()

    def trick(self, turn_num: int) -> None:
        self.log(lambda _: f"--- TRICK: {turn_num} ---")

        first_players = self.board.lead[:, 0]
        cards_tricked = np.zeros((self.board_num, 5), dtype=int)

        for player_index in range(5):
            players = (first_players + player_index) % 5

            # Have agents decide their cards to put.
            decisions = np.zeros((self.board_num, 54), dtype=bool)
            for player in range(5):
                if not np.any(players == player):
                    continue

                players_boards = self.board.slice_boards(players == player)
                players_boards = players_boards.change_perspective_to_one(player)

                hand_filter = players_boards.get_hand_filter(0)
                agent_decision = self.agents[player].put_card(players_boards, hand_filter)
                agent_decision = agent_decision * hand_filter

                assert np.all(np.sum(agent_decision, axis=1) == 1)

                decisions[players == player] = agent_decision

                # Record the decisions
                self.recorder.boards[player, players == player, turn_num] = players_boards.to_vector()
                self.recorder.hand_filters[player, players == player, turn_num] = hand_filter
                self.recorder.decisions[player, players == player, turn_num] = agent_decision

            # Mark the cards as already put.
            self.board.cards[decisions, 0] = CARD_TRICKED
            self.board.cards[decisions, 2] = turn_num * 5 + player_index

            # Record the cards chosen.
            decisions = decisions.astype(int)
            decisions_arg = np.argmax(decisions, axis=1)
            cards_tricked[:, player_index] = decisions_arg

            # Write cards to the log.
            self.log(lambda idx: f"player{players[idx]} puts {card_to_str(np.array([decisions_arg[idx] // 13, decisions_arg[idx] % 13]))}.")

            # Note a suit of the lead.
            # Refer the declaration if the lead is a joker.
            if player_index == 0:
                suits = decisions_arg // 13
                suits[suits == SUIT_JOKER] = self.board.decl[suits == SUIT_JOKER, 0]
                self.board.lead[:, 1] = suits

        # Calculate the winners.
        winners = (self.board.get_trick_winner(cards_tricked) + first_players) % 5

        self.log(lambda idx: f"player{winners[idx]} wins the trick.")

        # Initialize info of lead.
        self.board.lead[:, 0] = winners
        self.board.lead[:, 1] = SUIT_JOKER

        # Add scores.
        taken = np.count_nonzero(cards_tricked % 13 >= (10 - 2), axis=1)
        for idx in range(self.board_num):
            self.board.taken[idx, winners[idx]] += taken[idx]

        # Give rewards to the agents.
        # Since some hidden info can be retrieved from the rewards, these are not reported to them.
        rewards = np.zeros((self.board_num, 5))
        for idx in range(self.board_num):
            winners_role = self.board.roles[idx, winners[idx]]
            if winners_role == ROLE_NAPOLEON or winners_role == ROLE_ADJUTANT:
                rewards[idx, self.board.roles[idx] == ROLE_ALLY] -= taken[idx] * RWD_WINS_TRICK
                rewards[idx, self.board.roles[idx] == ROLE_NAPOLEON] += taken[idx] * RWD_WINS_TRICK
                rewards[idx, self.board.roles[idx] == ROLE_ADJUTANT] += taken[idx] * RWD_WINS_TRICK
            else:
                rewards[idx, self.board.roles[idx] == ROLE_ALLY] += taken[idx] * RWD_WINS_TRICK
                rewards[idx, self.board.roles[idx] == ROLE_NAPOLEON] -= taken[idx] * RWD_WINS_TRICK
                rewards[idx, self.board.roles[idx] == ROLE_ADJUTANT] -= taken[idx] * RWD_WINS_TRICK
        self.recorder.rewards[:, :, turn_num] = rewards.T

    def check_result(self) -> None:
        winners = np.zeros((5, self.board_num))
        for idx in range(self.board_num):
            napoleon_team = (self.board.roles[idx] == ROLE_NAPOLEON) | (self.board.roles[idx] == ROLE_ADJUTANT)

            taken = self.board.taken[idx, napoleon_team]
            if np.sum(taken) >= self.board.decl[idx, 1]:
                winners[napoleon_team, idx] = 1
                self.recorder.rewards[napoleon_team, idx, 9] += RWD_NAPOLEON_WINS
            else:
                winners[~napoleon_team, idx] = 1
                self.recorder.rewards[~napoleon_team, idx, 9] += RWD_ALLY_WINS

        self.log(
            lambda idx: f"player{", player".join(map(lambda x: str(x), list(np.argwhere(winners[:, idx] == 1)[:, 0])))} win(s)."
        )

        self.recorder.winners = winners

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
