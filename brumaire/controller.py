from __future__ import annotations
import numpy as np
from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os

from brumaire.constants import NDIntArray, NDFloatArray, AdjStrategy
from brumaire.model import BrumaireDeclModel, BrumaireTrickModel, BrumaireHParams
from brumaire.utils import convert_to_card_oriented, convert_strategy_oriented_to_input
from brumaire.exp import ExperienceDB


class BrumaireController:
    decl_model: BrumaireDeclModel
    trick_model: BrumaireTrickModel

    h_params: BrumaireHParams

    decl_optimizer: torch.optim.Optimizer
    trick_optimizer: torch.optim.Optimizer

    writer: SummaryWriter | None
    decl_global_step: int
    trick_global_step: int
    device: Any

    def __init__(
        self,
        h_params: BrumaireHParams,
        device,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.decl_model = BrumaireDeclModel(h_params, device)

        self.trick_model = BrumaireTrickModel(h_params, device)

        self.decl_optimizer = torch.optim.AdamW(
            self.decl_model.parameters(), lr=h_params.decl_ita, amsgrad=True
        )
        self.trick_optimizer = torch.optim.AdamW(
            self.trick_model.parameters(), lr=h_params.trick_ita, amsgrad=True
        )

        self.writer = writer
        self.decl_global_step = 0
        self.trick_global_step = 0
        self.device = device
        self.h_params = h_params

    def copy_from_other(self, agent: BrumaireController):
        self.decl_model.load_state_dict(agent.decl_model.state_dict())
        self.trick_model.load_state_dict(agent.trick_model.state_dict())

    def save(self, dir_path: str) -> None:
        torch.save(
            self.decl_model.state_dict(), os.path.join(dir_path, "decl_model_data")
        )
        torch.save(
            self.trick_model.state_dict(), os.path.join(dir_path, "trick_model_data")
        )

    def load(self, dir_path: str) -> None:
        state = torch.load(os.path.join(dir_path, "decl_model_data"))
        self.decl_model.load_state_dict(state)

        state = torch.load(os.path.join(dir_path, "trick_model_data"))
        self.trick_model.load_state_dict(state)

    def estimate_win_p(self, decl_input: NDFloatArray) -> NDFloatArray:
        """
        Calculates the win probabilities for each declaration.
        """
        size = decl_input.shape[0]

        win_p = torch.zeros((size, 4, 14 - 12, AdjStrategy.LENGTH))

        for suit in range(4):
            for num in range(12, 14):
                for strategy in range(AdjStrategy.LENGTH):
                    decl = np.repeat(np.array([[suit, num, strategy]]), size, axis=0)
                    inputs = convert_strategy_oriented_to_input(decl_input, decl)

                    inputs_tensor = torch.tensor(
                        inputs, dtype=torch.float32, device=self.device
                    )

                    self.decl_model.eval()
                    with torch.no_grad():
                        evaluated: torch.Tensor = self.decl_model(inputs_tensor)
                        win_p[:, suit, num - 12, strategy] = F.softmax(
                            evaluated, dim=1
                        )[:, 1]

        win_p_numpy = win_p.cpu().numpy()
        return win_p_numpy

    def decl_goal(self, decl_input: NDFloatArray, strongest: NDIntArray) -> NDIntArray:
        size = decl_input.shape[0]

        win_p = self.estimate_win_p(decl_input)
        win_p = np.reshape(win_p, (size, 4 * (14 - 12) * AdjStrategy.LENGTH))
        chosen = np.argmax(win_p, axis=1)

        decl = np.zeros((size, 3), dtype=int)
        decl[:, 0] = chosen // ((14 - 12) * AdjStrategy.LENGTH)
        decl[:, 1] = (
            chosen % ((14 - 12) * AdjStrategy.LENGTH) // AdjStrategy.LENGTH + 12
        )
        decl[:, 2] = chosen % AdjStrategy.LENGTH

        return convert_to_card_oriented(decl, strongest)

    def estimate_rewards(
        self, trick_input: NDFloatArray, hand_index: NDIntArray
    ) -> NDFloatArray:
        trick_input = torch.tensor(trick_input, dtype=torch.float32, device=self.device)
        hand_index = torch.tensor(hand_index, dtype=torch.float32, device=self.device)

        hand_index[hand_index == 0] = -torch.inf
        hand_index[hand_index == 1] = 0

        self.trick_model.eval()
        with torch.no_grad():
            evaluated: torch.Tensor = self.trick_model(trick_input) + hand_index
            evaluated_numpy = evaluated.cpu().numpy()

        return evaluated_numpy

    def get_best_action(
        self, trick_input: NDFloatArray, hand_index: NDIntArray
    ) -> NDIntArray:
        evaluated = self.estimate_rewards(trick_input, hand_index)
        return evaluated.argmax(axis=1)

    def estimate_best_reward(
        self, trick_input: NDFloatArray, hand_index: NDIntArray
    ) -> NDFloatArray:
        evaluated = self.estimate_rewards(trick_input, hand_index)
        return evaluated.max(axis=1)

    def train_decl(
        self,
        train_db: ExperienceDB,
        test_db: ExperienceDB,
        train_size: int,
        test_size: int,
        epoch: int = 100,
    ):
        self.decl_model.train()

        for _ in range(epoch):
            (
                train_decl_input,
                train_results,
            ) = train_db.gen_decl_batch(train_size, self.device)
            (
                test_decl_input,
                test_results,
            ) = test_db.gen_decl_batch(test_size, self.device)

            #
            # Training
            #
            evaluated: torch.Tensor = self.decl_model(train_decl_input)

            criterion = torch.nn.CrossEntropyLoss()
            loss: torch.Tensor = criterion(evaluated, train_results)

            self.decl_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(
                self.decl_model.parameters(), self.h_params.decl_clip_grad
            )
            self.decl_optimizer.step()

            if self.writer:
                self.writer.add_scalar(
                    "loss/decl-train", loss.item(), self.decl_global_step
                )

            #
            # Test
            #
            with torch.no_grad():
                evaluated: torch.Tensor = self.decl_model(test_decl_input)

                criterion = torch.nn.CrossEntropyLoss()
                loss: torch.Tensor = criterion(evaluated, test_results)

                if self.writer:
                    self.writer.add_scalar(
                        "loss/decl-test", loss.item(), self.decl_global_step
                    )

            self.decl_global_step += 1

    def train_trick(
        self,
        train_db: ExperienceDB,
        test_db: ExperienceDB,
        train_size: int,
        test_size: int,
        epoch: int = 100,
    ):
        self.trick_model.train()

        for _ in range(epoch):
            (
                train_trick_input,
                train_arg_decisions,
                train_estimated_rewards,
            ) = train_db.gen_trick_batch(train_size, self.device)
            (
                test_trick_input,
                test_arg_decisions,
                test_estimated_rewards,
            ) = test_db.gen_trick_batch(test_size, self.device)

            #
            # Training
            #
            evaluated: torch.Tensor = self.trick_model(train_trick_input)
            evaluated = evaluated.gather(1, train_arg_decisions)

            criterion = torch.nn.SmoothL1Loss()
            loss: torch.Tensor = criterion(evaluated, train_estimated_rewards)

            self.trick_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(
                self.trick_model.parameters(), self.h_params.trick_clip_grad
            )
            self.trick_optimizer.step()

            if self.writer:
                self.writer.add_scalar(
                    "loss/train", loss.item(), self.trick_global_step
                )

            #
            # Test
            #
            with torch.no_grad():
                evaluated: torch.Tensor = self.trick_model(test_trick_input)
                evaluated = evaluated.gather(1, test_arg_decisions)

                criterion = torch.nn.SmoothL1Loss()
                loss: torch.Tensor = criterion(evaluated, test_estimated_rewards)

                if self.writer:
                    self.writer.add_scalar(
                        "loss/test", loss.item(), self.trick_global_step
                    )

            self.trick_global_step += 1
