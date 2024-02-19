from __future__ import annotations
from typing import Tuple

from . import *
from brumaire.board import BOARD_VEC_SIZE

TURN = 10

class Recorder:
    first_boards: NDFloatArray
    """
    shape: `(5, board_num, BOARD_VEC_SIZE)`
    """

    declarations: NDIntArray
    """
    shape: `(5, board_num, 4)`
    """

    boards: NDFloatArray
    """
    shape: `(5, board_num, TURN, BOARD_VEC_SIZE)`
    """

    hand_filters: NDIntArray
    """
    shape: `(5, board_num, TURN, 54)`
    """

    decisions: NDIntArray
    """
    shape: `(5, board_num, TURN, 54)`
    """

    rewards: NDFloatArray
    """
    shape: `(5, board_num, TURN)`
    """

    winners: NDIntArray
    """
    shape: `(5, board_num)`
    """

    _board_num: int

    def __init__(self, board_num: int) -> None:
        self._board_num = board_num

        self.first_boards = np.zeros((5, board_num, BOARD_VEC_SIZE))
        self.declarations = np.zeros((5, board_num, 4))

        self.boards = np.zeros((5, board_num, TURN, BOARD_VEC_SIZE))
        self.hand_filters = np.zeros((5, board_num, TURN, 54))
        self.decisions = np.zeros((5, board_num, TURN, 54))
        self.rewards = np.zeros((5, board_num, TURN))

        self.winners = np.zeros((5, board_num))

    def get_data_size(self) -> int:
        return 5 * self._board_num

    def filter_by_board(self, board_filter: NDIntArray) -> Recorder:
        new_recorder = Recorder(board_filter.shape[0])
        new_recorder.first_boards = self.first_boards[:, board_filter]
        new_recorder.declarations = self.declarations[:, board_filter]
        new_recorder.boards = self.boards[:, board_filter]
        new_recorder.hand_filters = self.hand_filters[:, board_filter]
        new_recorder.decisions = self.decisions[:, board_filter]
        new_recorder.rewards = self.rewards[:, board_filter]
        return new_recorder

    def gen_batch(self, batch_size: int, test_size: int) -> Tuple[Recorder, Recorder]:
        assert batch_size + test_size <= self._board_num

        choice = np.random.choice(self._board_num, batch_size + test_size, replace=False)
        batch_choice = choice[0:batch_size]
        test_choice = choice[batch_size:]

        return self.filter_by_board(batch_choice), self.filter_by_board(test_choice)
