from __future__ import annotations
import numpy as np
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter

from brumaire.board import BOARD_VEC_SIZE, board_from_vector
from brumaire.constants import (
    NDFloatArray,
    NDIntArray,
    ROLE_ADJUTANT,
    ROLE_ALLY,
    ROLE_NAPOLEON,
)

TURN = 10


class Recorder:
    first_boards: NDFloatArray
    """
    shape: `(5, board_num, BOARD_VEC_SIZE)`
    """

    strongest: NDIntArray
    """
    shape: `(5, board_num, 4)`
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
        self.strongest = np.zeros((5, board_num, 4), dtype=int)
        self.declarations = np.zeros((5, board_num, 4), dtype=int)

        self.boards = np.zeros((5, board_num, TURN, BOARD_VEC_SIZE))
        self.hand_filters = np.zeros((5, board_num, TURN, 54), dtype=int)
        self.decisions = np.zeros((5, board_num, TURN, 54), dtype=int)
        self.rewards = np.zeros((5, board_num, TURN))

        self.winners = np.zeros((5, board_num))

    def get_data_size(self) -> int:
        return self._board_num

    def filter_by_board(self, board_filter: NDIntArray) -> Recorder:
        new_recorder = Recorder(board_filter.shape[0])
        new_recorder.first_boards = self.first_boards[:, board_filter]
        new_recorder.strongest = self.strongest[:, board_filter]
        new_recorder.declarations = self.declarations[:, board_filter]
        new_recorder.boards = self.boards[:, board_filter]
        new_recorder.hand_filters = self.hand_filters[:, board_filter]
        new_recorder.decisions = self.decisions[:, board_filter]
        new_recorder.rewards = self.rewards[:, board_filter]
        new_recorder.winners = self.winners[:, board_filter]
        return new_recorder

    def gen_batch(self, batch_size: int, test_size: int) -> Tuple[Recorder, Recorder]:
        assert batch_size + test_size <= self._board_num

        choice = np.random.choice(
            self._board_num, batch_size + test_size, replace=False
        )
        batch_choice = choice[0:batch_size]
        test_choice = choice[batch_size:]

        return self.filter_by_board(batch_choice), self.filter_by_board(test_choice)

    def avg_reward(self, player: int) -> float:
        return np.sum(self.rewards[player]) / self._board_num

    def win_rate(self, player: int) -> float:
        return np.sum(self.winners[player]) / self._board_num

    def total_win_rate(self) -> float:
        return np.sum(self.winners) / self._board_num / 5

    def fold_rate(self, player: int) -> float:
        return np.sum(self.declarations[player, :, 1] == 12) / self._board_num

    def win_as_role_rate(self, player: int, role: int, default: float = 0.5) -> float:
        board = board_from_vector(self.boards[player, :, 0])
        board_count = np.sum(board.roles[:, 0] == role)
        win_count = np.sum(self.winners[player, board.roles[:, 0] == role])

        if board_count == 0:
            return default
        else:
            return win_count / board_count

    def write_eval_result(
        self, player: int, writer: SummaryWriter, step: int = 0
    ) -> None:
        reward = self.avg_reward(player)
        win_rate = self.win_rate(player)
        total_win_rate = self.total_win_rate()
        fold_rate = self.fold_rate(player)
        win_as_napoleon_rate = self.win_as_role_rate(player, ROLE_NAPOLEON)
        win_as_adjutant_rate = self.win_as_role_rate(player, ROLE_ADJUTANT)
        win_as_ally_rate = self.win_as_role_rate(player, ROLE_ALLY)

        writer.add_scalar("eval/reward", reward, step)
        writer.add_scalar("eval/win rate", win_rate, step)
        writer.add_scalar("eval/win rate diff", win_rate - total_win_rate, step)
        writer.add_scalar("eval/fold rate", fold_rate, step)
        writer.add_scalar("eval/napoleon/win rate", win_as_napoleon_rate, step)
        writer.add_scalar("eval/adjutant/win rate", win_as_adjutant_rate, step)
        writer.add_scalar("eval/ally/win rate", win_as_ally_rate, step)
