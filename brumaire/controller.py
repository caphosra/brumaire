from __future__ import annotations
import numpy as np
from typing import Any
import torch
from torch.utils.tensorboard import SummaryWriter
import os

from brumaire.constants import (
    NDIntArray,
    NDFloatArray,
)
from brumaire.model import BrumaireDeclModel, BrumaireTrickModel, BrumaireHParams
from brumaire.utils import convert_to_card_oriented
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

    def decl_goal(self, board_vec: NDFloatArray, strongest: NDIntArray) -> NDIntArray:
        board_vec = torch.tensor(board_vec, dtype=torch.float32, device=self.device)

        self.decl_model.eval()
        with torch.no_grad():
            evaluated: torch.Tensor = self.decl_model(board_vec)
            evaluated = evaluated.argmax(dim=1).cpu().numpy().astype(int)

        decl = np.zeros((board_vec.shape[0], 3), dtype=int)
        decl[:, 0] = evaluated // 16
        decl[:, 1] = (evaluated % 16) // 8 + 12
        decl[:, 2] = (evaluated % 16) % 8
        return convert_to_card_oriented(decl, strongest)

    def make_decision(
        self, board_vec: NDFloatArray, hand_filter: NDIntArray
    ) -> NDIntArray:
        board_vec = torch.tensor(board_vec, dtype=torch.float32, device=self.device)
        hand_filter = torch.tensor(hand_filter, dtype=torch.float32, device=self.device)

        hand_filter[hand_filter == 0] = -torch.inf
        hand_filter[hand_filter == 1] = 0

        self.trick_model.eval()
        with torch.no_grad():
            evaluated: torch.Tensor = self.trick_model(board_vec) + hand_filter
            evaluated = evaluated.argmax(dim=1).cpu().numpy().astype(int)

        return np.eye(54)[evaluated]

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
                train_first_boards,
                train_arg_decl,
                train_total_rewards,
            ) = train_db.gen_decl_batch(train_size, self.device)
            (
                test_first_boards,
                test_arg_decl,
                test_total_rewards,
            ) = test_db.gen_decl_batch(test_size, self.device)

            #
            # Training
            #
            evaluated: torch.Tensor = self.decl_model(train_first_boards)
            evaluated = evaluated.gather(1, train_arg_decl)

            criterion = torch.nn.SmoothL1Loss()
            loss: torch.Tensor = criterion(evaluated, train_total_rewards)

            self.decl_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(
                self.decl_model.parameters(), self.h_params.decl_clip_grad
            )
            self.trick_optimizer.step()

            if self.writer:
                self.writer.add_scalar(
                    "loss/decl-train", loss.item(), self.decl_global_step
                )

            #
            # Test
            #
            with torch.no_grad():
                evaluated: torch.Tensor = self.decl_model(test_first_boards)
                evaluated = evaluated.gather(1, test_arg_decl)

                criterion = torch.nn.SmoothL1Loss()
                loss: torch.Tensor = criterion(evaluated, test_total_rewards)

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
                train_boards,
                train_arg_decisions,
                train_estimated_rewards,
            ) = train_db.gen_trick_batch(train_size, self.device)
            (
                test_boards,
                test_arg_decisions,
                test_estimated_rewards,
            ) = test_db.gen_trick_batch(test_size, self.device)

            #
            # Training
            #
            evaluated: torch.Tensor = self.trick_model(train_boards)
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
                evaluated: torch.Tensor = self.trick_model(test_boards)
                evaluated = evaluated.gather(1, test_arg_decisions)

                criterion = torch.nn.SmoothL1Loss()
                loss: torch.Tensor = criterion(evaluated, test_estimated_rewards)

                if self.writer:
                    self.writer.add_scalar(
                        "loss/test", loss.item(), self.trick_global_step
                    )

            self.trick_global_step += 1
