from __future__ import annotations
from typing import Any, Tuple
import numpy as np
import torch

from brumaire.board import BOARD_VEC_SIZE
from brumaire.constants import NDFloatArray, NDIntArray
from brumaire.record import Recorder
from brumaire.model import BrumaireTrickModel
from brumaire.utils import convert_to_strategy_oriented


class ExperienceDB:
    decl_size: int
    trick_size: int

    first_boards: NDFloatArray
    """
    shape: `(decl_size, BOARD_VEC_SIZE)`
    """

    decl: NDIntArray
    """
    shape: `(decl_size, 3)`
    """

    total_rewards: NDFloatArray
    """
    shape: `(decl_size,)`
    """

    boards: NDFloatArray
    """
    shape: `(trick_size, BOARD_VEC_SIZE)`
    """

    decisions: NDIntArray
    """
    shape: `(trick_size, 54)`
    """

    estimated_rewards: NDFloatArray
    """
    shape: `(trick_size,)`
    """

    def __init__(self) -> None:
        self.decl_size = 0
        self.trick_size = 0

        self.first_boards = np.zeros((0, BOARD_VEC_SIZE), dtype=float)
        self.decl = np.zeros((0, 3), dtype=int)
        self.total_rewards = np.array([], dtype=float)
        self.boards = np.zeros((0, BOARD_VEC_SIZE), dtype=float)
        self.decisions = np.zeros((0, 54), dtype=int)
        self.estimated_rewards = np.array([], dtype=float)

    def import_from_record(
        self,
        player: int,
        recorder: Recorder,
        trick_model: BrumaireTrickModel,
        gamma: float,
        device: Any,
    ) -> None:
        self.decl_size += recorder.get_data_size()
        self.trick_size += recorder.get_data_size() * 10

        decl = convert_to_strategy_oriented(
            recorder.declarations[player], recorder.strongest[player]
        )
        total_rewards = np.sum(recorder.rewards[player], axis=1)
        estimated_rewards = self._estimate_rewards(
            player, recorder, trick_model, gamma, device
        )

        self.first_boards = np.concatenate(
            ((self.first_boards, recorder.first_boards[player]))
        )
        self.decl = np.concatenate((self.decl, decl))
        self.total_rewards = np.concatenate((self.total_rewards, total_rewards))

        self.boards = np.concatenate(
            (self.boards, recorder.boards[player].reshape((-1, BOARD_VEC_SIZE)))
        )
        self.decisions = np.concatenate(
            (self.decisions, recorder.decisions[player].reshape((-1, 54)))
        )
        self.estimated_rewards = np.concatenate(
            (self.estimated_rewards, estimated_rewards)
        )

    def _estimate_rewards(
        self,
        player: int,
        recorder: Recorder,
        trick_model: BrumaireTrickModel,
        gamma: float,
        device: Any,
    ) -> NDFloatArray:
        boards = recorder.boards[player]
        boards = torch.tensor(boards, dtype=torch.float32, device=device)

        rewards = recorder.rewards[player]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        hand_filters = recorder.hand_filters[player]
        hand_filters = torch.tensor(hand_filters, dtype=torch.float32, device=device)

        hand_filters[hand_filters == 0] = -torch.inf
        hand_filters[hand_filters == 1] = 0

        estimations = torch.zeros((recorder.get_data_size(), 10), device=device)
        for turn in range(10):
            estimations[:, turn] = rewards[:, turn]
            if turn < 10 - 1:
                next_boards = boards[:, turn + 1, :]
                next_hand_filters = hand_filters[:, turn + 1, :]

                with torch.no_grad():
                    evaluated: torch.Tensor = (
                        trick_model(next_boards) + next_hand_filters
                    )
                    estimations[:, turn] += evaluated.max(dim=1)[0] * gamma

        estimations_numpy: NDFloatArray = estimations.cpu().numpy()
        return estimations_numpy.flatten()

    def gen_decl_batch(
        self, size: int, device: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen = np.random.choice(self.decl_size, size, replace=False)

        first_boards = torch.tensor(
            self.first_boards[chosen], dtype=torch.float32, device=device
        )

        decl = self.decl[chosen]
        decl_arg = np.reshape(
            decl[:, 0] * 16 + np.minimum(decl[:, 1] - 12, 1) * 8 + decl[:, 2], (-1, 1)
        )
        decl_arg = torch.tensor(decl_arg, dtype=torch.int64, device=device)

        total_rewards = torch.tensor(
            self.total_rewards[chosen].reshape((-1, 1)),
            dtype=torch.float32,
            device=device,
        )

        return first_boards, decl_arg, total_rewards

    def gen_trick_batch(
        self, size: int, device: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen = np.random.choice(self.trick_size, size, replace=False)

        boards = torch.tensor(self.boards[chosen], dtype=torch.float32, device=device)

        decisions = self.decisions[chosen]
        decisions_arg = np.reshape(np.argmax(decisions, axis=1), (-1, 1))
        decisions_arg = torch.tensor(decisions_arg, dtype=torch.int64, device=device)

        estimated_rewards = torch.tensor(
            self.estimated_rewards[chosen].reshape((-1, 1)),
            dtype=torch.float32,
            device=device,
        )

        return boards, decisions_arg, estimated_rewards
