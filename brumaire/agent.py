import numpy as np
from numpy import ndarray

from brumaire.board import BoardData
from brumaire.constants import NDIntArray, Suit
from brumaire.controller import BrumaireController
from brumaire.utils import convert_to_card_oriented


class AgentBase:
    def declare_goal(self, board: BoardData) -> np.ndarray:
        """
        Declare the goal. It will returns a ndarray shaped (4,)
        """
        return np.repeat(
            np.array([[Suit.SPADE, 12, Suit.SPADE, 14 - 2]]), board.board_num, axis=0
        )

    def discard(self, board: BoardData) -> np.ndarray:
        decision = np.zeros((board.board_num, 14))
        decision[:, [0, 1, 2, 3]] = 1
        return decision

    def put_card(self, board: BoardData, hand_filter: NDIntArray) -> np.ndarray:
        assert board.board_num == hand_filter.shape[0]

        decision = np.zeros((board.board_num, 54))
        for idx in range(board.board_num):
            decision[idx, np.argmax(hand_filter[idx])] = 1
        return decision

    def tell_reward(self, reward: float):
        pass


class RandomAgent(AgentBase):
    decl_p: float

    def __init__(self) -> None:
        super().__init__()

    def declare_goal(self, board: BoardData) -> ndarray:
        return np.random.randint(
            [0, 12, 0, 0], [4, 14, 4, 13], size=(board.board_num, 4)
        )

    def discard(self, board: BoardData) -> np.ndarray:
        decision = np.zeros((board.board_num, 14))
        choice = np.random.choice(np.arange(14), 4, replace=False)
        decision[:, choice] = 1.0
        return decision

    def put_card(self, board: BoardData, hand_filter: NDIntArray) -> np.ndarray:
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

    def declare_goal(self, board: BoardData) -> ndarray:
        samples = np.random.rand(board.board_num)
        strongest = board.get_strongest_for_each_suits()
        decl_input = board.convert_to_decl_input(0)

        decl = np.zeros((board.board_num, 4))
        decl[samples > self.epsilon] = self.controller.decl_goal(
            decl_input[samples > self.epsilon], strongest[samples > self.epsilon]
        )
        random_num = np.count_nonzero(samples <= self.epsilon)
        decl[samples <= self.epsilon] = convert_to_card_oriented(
            np.random.randint([0, 12, 0], [4, 14, 8], size=(random_num, 3)),
            strongest[samples <= self.epsilon],
        )
        return decl

    def discard(self, board: BoardData) -> np.ndarray:
        return super().discard(board)

    def put_card(self, board: BoardData, hand_filter: NDIntArray) -> np.ndarray:
        samples = np.random.rand(board.board_num)
        board_vec = board.to_vector()

        selected = np.zeros((board.board_num, 54))
        selected[samples > self.epsilon] = self.controller.make_decision(
            board_vec[samples > self.epsilon], hand_filter[samples > self.epsilon]
        )
        selected[samples <= self.epsilon] = super().put_card(
            board.slice_boards(samples <= self.epsilon),
            hand_filter[samples <= self.epsilon],
        )
        return selected
