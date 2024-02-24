from __future__ import annotations
import numpy as np
import torch

from brumaire.board import BOARD_VEC_SIZE


class TrainableExp:
    """
    Stores the experiences in a ready-for-train style.
    """

    data_num: int
    """
    The number of experiences this record holds.
    """

    p: torch.Tensor
    """
    Possibilities of chosen.

    shape: `(data_num,)`
    """

    boards: torch.Tensor
    """
    shape: `(data_num, BOARD_VEC_SIZE)`
    """

    decisions: torch.Tensor
    """
    shape: `(data_num, 54)`
    """

    estimated_rewards: torch.Tensor
    """
    shape: `(data_num,)`
    """

    first_boards: torch.Tensor
    """
    shape: `(data_num, BOARD_VEC_SIZE)`
    """

    declarations: torch.Tensor
    """
    shape: `(data_num, 3)`
    """

    def __init__(
        self,
        data_num: int,
        p: torch.Tensor,
        boards: torch.Tensor,
        decisions: torch.Tensor,
        estimated_rewards: torch.Tensor,
        first_boards: torch.Tensor,
        declarations: torch.Tensor,
        skip_p_validation: bool = False,
    ) -> None:
        assert p.shape == (data_num,)
        assert boards.shape == (data_num, BOARD_VEC_SIZE)
        assert decisions.shape == (data_num, 54)
        assert estimated_rewards.shape == (data_num,)
        assert first_boards.shape == (data_num, BOARD_VEC_SIZE)
        assert declarations.shape == (data_num, 3)

        self.data_num = data_num
        self.boards = boards
        self.decisions = decisions
        self.estimated_rewards = estimated_rewards
        self.first_boards = first_boards
        self.declarations = declarations

        if not skip_p_validation:
            p = p / torch.sum(p)
        self.p = p

    def _filter_exp(self, choice: torch.Tensor) -> TrainableExp:
        assert choice.ndim == 1

        data_num = choice.shape[0]
        p = self.p[choice]
        boards = self.boards[choice]
        decisions = self.decisions[choice]
        estimated_rewards = self.estimated_rewards[choice]
        first_boards = self.first_boards[choice]
        declarations = self.declarations[choice]
        return TrainableExp(
            data_num,
            p,
            boards,
            decisions,
            estimated_rewards,
            first_boards,
            declarations,
        )

    def merge(self, exp: TrainableExp) -> TrainableExp:
        data_num = self.data_num + exp.data_num
        p = torch.cat((self.p * self.data_num, exp.p * exp.data_num)) / (
            self.data_num + exp.data_num
        )
        boards = torch.cat((self.boards, exp.boards))
        decisions = torch.cat((self.decisions, exp.decisions))
        estimated_rewards = torch.cat(
            (self.estimated_rewards, exp.estimated_rewards)
        )
        first_boards = torch.cat((self.first_boards, exp.first_boards))
        declarations = torch.cat((self.declarations, exp.declarations))
        return TrainableExp(
            data_num,
            p,
            boards,
            decisions,
            estimated_rewards,
            first_boards,
            declarations,
            skip_p_validation=True,
        )

    def gen_batch(self, batch_size: int) -> TrainableExp:
        assert batch_size <= self.data_num

        choice = np.random.choice(self.data_num, batch_size, replace=False, p=self.p)
        choice = torch.tensor(choice, dtype=torch.int64)
        return self._filter_exp(choice)
