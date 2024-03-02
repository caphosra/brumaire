from __future__ import annotations
from typing import Any, Tuple
import numpy as np
import torch

from brumaire.board import BoardData
from brumaire.constants import (
    NDFloatArray,
    NDIntArray,
    DECL_INPUT_SIZE,
)
from brumaire.record import Recorder
from brumaire.model import BrumaireTrickModel
from brumaire.utils import (
    convert_to_strategy_oriented,
    convert_strategy_oriented_to_input,
)


class ExperienceDB:
    decl_size: int
    trick_size: int

    decl_input: NDFloatArray
    """
    shape: `(decl_size, DECL_INPUT_SIZE)`
    """

    results: NDIntArray
    """
    shape: `(decl_size, 2)`
    """

    trick_input: NDFloatArray
    """
    shape: `(trick_size, Board.TRICK_INPUT_SIZE)`
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

        self.decl_input = np.zeros((0, DECL_INPUT_SIZE), dtype=float)
        self.results = np.zeros((0, 2), dtype=float)
        self.trick_input = np.zeros((0, BoardData.TRICK_INPUT_SIZE), dtype=float)
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

        decl_input = self._get_decl_input(player, recorder)
        self.decl_input = np.concatenate((self.decl_input, decl_input))

        new_decl_size = recorder.get_data_size()
        results = np.zeros((new_decl_size, 2))
        for idx in range(new_decl_size):
            results[idx, recorder.winners[player, idx]] = 1
        self.results = np.concatenate((self.results, results))

        board = BoardData.from_vector(
            recorder.boards[player].reshape((-1, BoardData.VEC_SIZE))
        )
        trick_input = board.to_trick_input()

        estimated_rewards = self._estimate_rewards(
            player, trick_input, recorder, trick_model, gamma, device
        )

        self.trick_input = np.concatenate((self.trick_input, trick_input))
        self.decisions = np.concatenate(
            (self.decisions, recorder.decisions[player].reshape((-1, 54)))
        )
        self.estimated_rewards = np.concatenate(
            (self.estimated_rewards, estimated_rewards)
        )

    def _get_decl_input(self, player: int, recorder: Recorder):
        size = recorder.get_data_size()

        decl = convert_to_strategy_oriented(
            recorder.declarations[player], recorder.strongest[player]
        )

        board = BoardData.from_vector(recorder.first_boards[player])
        decl_input = board.convert_to_decl_input(player)
        decl_input = convert_strategy_oriented_to_input(decl_input, decl)

        assert decl_input.shape == (size, DECL_INPUT_SIZE)

        return decl_input

    def _estimate_rewards(
        self,
        player: int,
        trick_input: NDFloatArray,
        recorder: Recorder,
        trick_model: BrumaireTrickModel,
        gamma: float,
        device: Any,
    ) -> NDFloatArray:
        size = recorder.get_data_size()

        trick_input = trick_input.reshape((size, 10, BoardData.TRICK_INPUT_SIZE))
        trick_input = torch.tensor(trick_input, dtype=torch.float32, device=device)

        rewards = recorder.rewards[player]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        hand_filters = recorder.hand_filters[player]
        hand_filters = torch.tensor(hand_filters, dtype=torch.float32, device=device)

        hand_filters[hand_filters == 0] = -torch.inf
        hand_filters[hand_filters == 1] = 0

        estimations = torch.zeros((size, 10), device=device)
        for turn in range(10):
            estimations[:, turn] = rewards[:, turn]
            if turn < 10 - 1:
                next_boards = trick_input[:, turn + 1, :]
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chosen = np.random.choice(self.decl_size, size, replace=False)

        decl_input = torch.tensor(
            self.decl_input[chosen], dtype=torch.float32, device=device
        )

        results = torch.tensor(
            self.results[chosen],
            dtype=torch.float32,
            device=device,
        )

        return decl_input, results

    def gen_trick_batch(
        self, size: int, device: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chosen = np.random.choice(self.trick_size, size, replace=False)

        trick_input = torch.tensor(
            self.trick_input[chosen], dtype=torch.float32, device=device
        )

        decisions = self.decisions[chosen]
        decisions_arg = np.reshape(np.argmax(decisions, axis=1), (-1, 1))
        decisions_arg = torch.tensor(decisions_arg, dtype=torch.int64, device=device)

        estimated_rewards = torch.tensor(
            self.estimated_rewards[chosen].reshape((-1, 1)),
            dtype=torch.float32,
            device=device,
        )

        return trick_input, decisions_arg, estimated_rewards
