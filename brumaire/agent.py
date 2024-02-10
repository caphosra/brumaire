import numpy as np
from numpy import ndarray

from brumaire import *
from brumaire.board import BoardData

class AgentBase:
    def declare_goal(self, board: BoardData) -> np.ndarray:
        """
        Declare the goal. It will returns a ndarray shaped (4,)
        """
        return np.repeat(np.array([[SUIT_SPADE, 12, SUIT_SPADE, 14 - 2]]), board.board_num, axis=0)

    def discard(self, board: BoardData) -> np.ndarray:
        decision = np.zeros((board.board_num, 14))
        decision[:, [0, 1, 2, 3]] = 1
        return decision

    def put_card(self, board: BoardData) -> np.ndarray:
        decision = np.zeros((board.board_num, 10))
        decision[:, 0] = 1
        return decision

    def tell_reward(self, reward: float):
        pass

class RandomAgent(AgentBase):
    decl_p: float

    def __init__(self) -> None:
        super().__init__()

    def declare_goal(self, board: BoardData) -> ndarray:
        return np.random.randint([0, 12, 0, 0], [4, 14, 4, 13], size=(board.board_num, 4))

    def discard(self, board: BoardData) -> np.ndarray:
        decision = np.zeros((board.board_num, 14))
        choice = np.random.choice(np.arange(14), 4, replace=False)
        decision[:, choice] = 1.
        return decision

    def put_card(self, board: BoardData) -> np.ndarray:
        return np.repeat((np.eye(10)[np.random.randint(0, 10)])[None, :], board.board_num, axis=0)
