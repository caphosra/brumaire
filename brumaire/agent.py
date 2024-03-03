import numpy as np

from brumaire.board import BoardData
from brumaire.constants import NDIntArray, Suit
from brumaire.controller import BrumaireController
from brumaire.utils import convert_to_card_oriented


class AgentBase:
    def declare_goal(self, board: BoardData) -> NDIntArray:
        """
        Declare the goal. It will returns a ndarray shaped (4,)
        """
        return np.repeat(
            np.array([[Suit.SPADE, 12, Suit.SPADE, 14 - 2]], dtype=int),
            board.board_num,
            axis=0,
        )

    def discard(self, board: BoardData) -> NDIntArray:
        decision = np.zeros((board.board_num, 14), dtype=int)
        decision[:, [0, 1, 2, 3]] = 1
        return decision

    def play_card(self, board: BoardData) -> NDIntArray:
        hand_filter = board.get_hand_filter(0)
        decision = np.zeros((board.board_num, 54), dtype=int)
        for idx in range(board.board_num):
            decision[idx, np.argmax(hand_filter[idx])] = 1
        return decision


class RandomAgent(AgentBase):
    decl_p: float

    def __init__(self) -> None:
        super().__init__()

    def declare_goal(self, board: BoardData) -> NDIntArray:
        return np.random.randint(
            [0, 12, 0, 0], [4, 14, 4, 13], size=(board.board_num, 4)
        )

    def discard(self, board: BoardData) -> NDIntArray:
        decision = np.zeros((board.board_num, 14), dtype=int)
        choice = np.random.choice(np.arange(14), 4, replace=False)
        decision[:, choice] = 1
        return decision

    def play_card(self, board: BoardData) -> NDIntArray:
        hand_filter = board.get_hand_filter(0)
        decision = np.zeros((board.board_num, 54), dtype=int)
        for idx in range(board.board_num):
            possibilities = hand_filter[idx].astype(int) / np.sum(hand_filter[idx])
            decided = np.random.choice(54, 1, p=possibilities)[0]
            decision[idx, decided] = 1
        return decision


class BrumaireAgent(RandomAgent):
    controller: BrumaireController
    epsilon: float

    def __init__(self, controller: BrumaireController, epsilon: float = 0.0) -> None:
        super().__init__()

        self.controller = controller
        self.epsilon = epsilon

    def declare_goal(self, board: BoardData) -> NDIntArray:
        samples = np.random.rand(board.board_num)
        strongest = board.get_strongest_for_each_suits()
        decl_input = board.convert_to_decl_input(0)

        decl = np.zeros((board.board_num, 4), dtype=int)
        decl[samples > self.epsilon] = self.controller.decl_goal(
            decl_input[samples > self.epsilon], strongest[samples > self.epsilon]
        )
        random_num = np.count_nonzero(samples <= self.epsilon)
        decl[samples <= self.epsilon] = convert_to_card_oriented(
            np.random.randint([0, 12, 0], [4, 14, 8], size=(random_num, 3)),
            strongest[samples <= self.epsilon],
        )
        return decl

    def discard(self, board: BoardData) -> NDIntArray:
        comb, vec = board.enumerate_discard_patterns()

        boards = BoardData.from_vector(vec.reshape((-1, BoardData.VEC_SIZE)))
        hand_filters = boards.get_hand_filter(0).reshape(
            (vec.shape[0], vec.shape[1], 54)
        )

        evaluated = np.zeros((vec.shape[0], vec.shape[1]))

        for pattern in range(vec.shape[0]):
            best_rewards = self.controller.estimate_best_reward(
                vec[pattern], hand_filters[pattern]
            )
            evaluated[pattern] = best_rewards

        discarded = np.sum(
            np.eye(14)[comb[np.argmax(evaluated, axis=0)]], axis=1
        ).astype(int)

        return discarded

    def play_card(self, board: BoardData) -> NDIntArray:
        hand_filter = board.get_hand_filter(0)

        samples = np.random.rand(board.board_num)
        trick_input = board.to_trick_input()

        selected = np.zeros((board.board_num, 54), dtype=int)
        selected[samples > self.epsilon] = self.controller.make_decision(
            trick_input[samples > self.epsilon], hand_filter[samples > self.epsilon]
        )
        selected[samples <= self.epsilon] = super().play_card(
            board.slice_boards(samples <= self.epsilon)
        )
        return selected
