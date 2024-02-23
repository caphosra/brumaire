from __future__ import annotations
import numpy as np
from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
import os

from brumaire.board import BOARD_VEC_SIZE
from brumaire.constants import (
    NDIntArray,
    NDFloatArray,
    SUIT_SPADE,
    SUIT_HEART,
    ADJ_ALMIGHTY,
    ADJ_MAIN_JACK,
    ADJ_SUB_JACK,
    ADJ_PARTNER,
    ADJ_TRUMP_TWO,
    ADJ_FLIPPED_TWO,
    ADJ_TRUMP_MAXIMUM,
    ADJ_RANDOM,
)
from brumaire.record import Recorder


class BrumaireHParams:
    decl_l1_node: int
    decl_l2_node: int
    decl_l3_node: int

    decl_ita: float = 0.0
    decl_clip_grad: float = 0.0

    l1_node: int
    l2_node: int
    l3_node: int

    ita: float = 0.0
    gamma: float = 0.0
    clip_grad: float = 0.0

    def write_summary(self, writer: SummaryWriter):
        exp, ssi, sei = hparams(
            {
                "decl/l1 node": self.decl_l1_node,
                "decl/l2 node": self.decl_l2_node,
                "decl/l3 node": self.decl_l3_node,
                "decl/ita": self.decl_ita,
                "decl/clip grad": self.decl_clip_grad,
                "l1 node": self.l1_node,
                "l2 node": self.l2_node,
                "l3 node": self.l3_node,
                "ita": self.ita,
                "gamma": self.gamma,
                "clip grad": self.clip_grad,
            },
            {},
        )

        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)


class AvantBrumaireModel(torch.nn.Module):
    layer1: torch.nn.Linear
    dropout_layer1: torch.nn.Dropout
    layer2: torch.nn.Linear
    dropout_layer2: torch.nn.Dropout
    layer3: torch.nn.Linear
    dropout_layer3: torch.nn.Dropout
    layer4: torch.nn.Linear

    def __init__(self, h_param: BrumaireHParams, device) -> None:
        super(AvantBrumaireModel, self).__init__()

        self.layer1 = torch.nn.Linear(BOARD_VEC_SIZE, h_param.decl_l1_node, device=device)
        self.dropout_layer1 = torch.nn.Dropout()
        self.layer2 = torch.nn.Linear(h_param.decl_l1_node, h_param.decl_l2_node, device=device)
        self.dropout_layer2 = torch.nn.Dropout()
        self.layer3 = torch.nn.Linear(h_param.decl_l2_node, h_param.decl_l3_node, device=device)
        self.dropout_layer3 = torch.nn.Dropout()
        self.layer4 = torch.nn.Linear(h_param.decl_l3_node, 4 * 8 * 2, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.dropout_layer1(self.layer1(x)))
        x = F.leaky_relu(self.dropout_layer2(self.layer2(x)))
        x = F.leaky_relu(self.dropout_layer3(self.layer3(x)))
        return self.layer4(x)


class BrumaireModel(torch.nn.Module):
    layer1: torch.nn.Linear
    dropout_layer1: torch.nn.Dropout
    layer2: torch.nn.Linear
    dropout_layer2: torch.nn.Dropout
    layer3: torch.nn.Linear
    dropout_layer3: torch.nn.Dropout
    layer4: torch.nn.Linear

    def __init__(self, h_param: BrumaireHParams, device) -> None:
        super(BrumaireModel, self).__init__()

        self.layer1 = torch.nn.Linear(
            BOARD_VEC_SIZE, h_param.l1_node, device=device
        )
        self.dropout_layer1 = torch.nn.Dropout()
        self.layer2 = torch.nn.Linear(
            h_param.l1_node, h_param.l2_node, device=device
        )
        self.dropout_layer2 = torch.nn.Dropout()
        self.layer3 = torch.nn.Linear(
            h_param.l2_node, h_param.l3_node, device=device
        )
        self.dropout_layer3 = torch.nn.Dropout()
        self.layer4 = torch.nn.Linear(h_param.l3_node, 54, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.dropout_layer1(self.layer1(x)))
        x = F.leaky_relu(self.dropout_layer2(self.layer2(x)))
        x = F.leaky_relu(self.dropout_layer3(self.layer3(x)))
        return self.layer4(x)


class BrumaireController:
    decl_model: AvantBrumaireModel

    model: BrumaireModel
    target: BrumaireModel

    h_params: BrumaireHParams

    decl_optimizer: torch.optim.Optimizer
    optimizer: torch.optim.Optimizer

    writer: SummaryWriter | None
    decl_global_step: int
    global_step: int
    device: Any

    def __init__(
        self,
        h_params: BrumaireHParams,
        device,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.decl_model = AvantBrumaireModel(h_params, device)

        self.model = BrumaireModel(h_params, device)
        self.target = BrumaireModel(h_params, device)

        self.decl_optimizer = torch.optim.AdamW(
            self.decl_model.parameters(), lr=h_params.decl_ita, amsgrad=True
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=h_params.ita, amsgrad=True
        )

        self.writer = writer
        self.decl_global_step = 0
        self.global_step = 0
        self.device = device
        self.h_params = h_params

    def copy_from_other(self, agent: BrumaireController):
        self.decl_model.load_state_dict(agent.decl_model.state_dict())
        self.model.load_state_dict(agent.model.state_dict())

    def save(self, dir_path: str) -> None:
        torch.save(self.decl_model.state_dict(), os.path.join(dir_path, "decl_model_data"))
        torch.save(self.model.state_dict(), os.path.join(dir_path, "model_data"))

    def load(self, dir_path: str) -> None:
        state = torch.load(os.path.join(dir_path, "decl_model_data"))
        self.decl_model.load_state_dict(state)

        state = torch.load(os.path.join(dir_path, "model_data"))
        self.model.load_state_dict(state)

    def convert_to_card_oriented(
        self, decl: NDIntArray, strongest: NDIntArray
    ) -> NDIntArray:
        """
        Converts the decl from `(board_num, 3)` to `(board_num, 4)`.
        """

        board_num = decl.shape[0]
        converted = np.zeros((board_num, 4))
        converted[:, 0] = decl[:, 0]
        converted[:, 1] = decl[:, 1]
        adj_card = decl[:, 2]

        converted[:, 2] = np.random.randint(4, size=(board_num,))
        converted[:, 3] = np.random.randint(13, size=(board_num,))

        converted[adj_card == ADJ_ALMIGHTY, 2] = SUIT_SPADE
        converted[adj_card == ADJ_ALMIGHTY, 3] = 14 - 2

        converted[adj_card == ADJ_MAIN_JACK, 2] = converted[
            adj_card == ADJ_MAIN_JACK, 0
        ]
        converted[adj_card == ADJ_MAIN_JACK, 3] = 11 - 2

        converted[adj_card == ADJ_SUB_JACK, 2] = (
            3 - converted[adj_card == ADJ_SUB_JACK, 0]
        )
        converted[adj_card == ADJ_SUB_JACK, 3] = 11 - 2

        converted[adj_card == ADJ_PARTNER, 2] = SUIT_HEART
        converted[adj_card == ADJ_PARTNER, 3] = 12 - 2

        converted[adj_card == ADJ_TRUMP_TWO, 2] = converted[
            adj_card == ADJ_TRUMP_TWO, 0
        ]
        converted[adj_card == ADJ_TRUMP_TWO, 3] = 2 - 2

        converted[adj_card == ADJ_FLIPPED_TWO, 2] = (
            3 - converted[adj_card == ADJ_FLIPPED_TWO, 0]
        )
        converted[adj_card == ADJ_FLIPPED_TWO, 3] = 2 - 2

        for idx in range(board_num):
            if adj_card[idx] == ADJ_TRUMP_MAXIMUM:
                strongest_card = strongest[idx, decl[idx, 0]]
                converted[idx, 2] = strongest_card // 13
                converted[idx, 3] = strongest_card % 13

        return converted

    def convert_to_adj_type_oriented(
        self, decl: NDIntArray, strongest: NDIntArray
    ) -> NDIntArray:
        """
        Converts the decl from `(board_num, 4)` to `(board_num, 3)`.
        """

        board_num = decl.shape[0]
        converted = np.zeros((board_num, 3), dtype=int)
        converted[:, 0] = decl[:, 0]
        converted[:, 1] = decl[:, 1]

        converted[:, 2] = ADJ_RANDOM

        for idx in range(board_num):
            if decl[idx, 2] == strongest[idx, decl[idx, 0]]:
                # It can be ADJ_RANDOM.
                # However, we will ignore this because the possibility of "false-positive" is too low.
                converted[:, 2] = ADJ_TRUMP_MAXIMUM

        converted[
            (decl[:, 2] == (3 - decl[:, 0])) & (decl[:, 3] == 2 - 2), 2
        ] = ADJ_FLIPPED_TWO
        converted[(decl[:, 2] == decl[:, 0]) & (decl[:, 3] == 2 - 2), 2] = ADJ_TRUMP_TWO
        converted[(decl[:, 2] == SUIT_HEART) & (decl[:, 3] == 12 - 2), 2] = ADJ_PARTNER
        converted[
            (decl[:, 2] == (3 - decl[:, 0])) & (decl[:, 3] == 11 - 2), 2
        ] = ADJ_SUB_JACK
        converted[
            (decl[:, 2] == decl[:, 0]) & (decl[:, 3] == 11 - 2), 2
        ] = ADJ_MAIN_JACK
        converted[(decl[:, 2] == SUIT_SPADE) & (decl[:, 3] == 14 - 2), 2] = ADJ_ALMIGHTY

        return converted

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
        return self.convert_to_card_oriented(decl, strongest)

    def make_decision(
        self, board_vec: NDFloatArray, hand_filter: NDIntArray
    ) -> NDIntArray:
        board_vec = torch.tensor(board_vec, dtype=torch.float32, device=self.device)
        hand_filter = torch.tensor(hand_filter, dtype=torch.float32, device=self.device)

        hand_filter[hand_filter == 0] = -torch.inf
        hand_filter[hand_filter == 1] = 0

        self.model.eval()
        with torch.no_grad():
            evaluated: torch.Tensor = self.model(board_vec) + hand_filter
            evaluated = evaluated.argmax(dim=1).cpu().numpy().astype(int)

        return np.eye(54)[evaluated]

    def copy_target(self) -> None:
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

    def estimate_Q_value(self, recorder: Recorder, gamma: float) -> torch.Tensor:
        boards = recorder.boards[0]
        boards = torch.tensor(boards, dtype=torch.float32, device=self.device)

        rewards = recorder.rewards[0]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        hand_filters = recorder.hand_filters[0]
        hand_filters = torch.tensor(
            hand_filters, dtype=torch.float32, device=self.device
        )

        hand_filters[hand_filters == 0] = -torch.inf
        hand_filters[hand_filters == 1] = 0

        estimations = torch.zeros((recorder.get_data_size(), 10), device=self.device)
        for turn in range(10):
            estimations[:, turn] = rewards[:, turn]
            if turn < 10 - 1:
                next_boards = boards[:, turn + 1, :]
                next_hand_filters = hand_filters[:, turn + 1, :]

                with torch.no_grad():
                    evaluated: torch.Tensor = (
                        self.target(next_boards) + next_hand_filters
                    )
                    estimations[:, turn] += evaluated.max(dim=1)[0] * gamma

        return estimations

    def train_decl(
        self,
        recorder: Recorder,
        batch_size: int,
        test_size: int,
        epoch: int = 100,
    ):
        batch_data, test_data = recorder.gen_batch(batch_size, test_size)

        batch_first_boards = batch_data.first_boards[0]
        batch_first_boards = torch.tensor(
            batch_first_boards, dtype=torch.float32, device=self.device
        )

        test_first_boards = test_data.first_boards[0]
        test_first_boards = torch.tensor(
            test_first_boards, dtype=torch.float32, device=self.device
        )

        batch_decl = self.convert_to_adj_type_oriented(
            batch_data.declarations[0], batch_data.strongest[0]
        )
        batch_arg_decl = (
            batch_decl[:, 0] * 16
            + np.minimum(batch_decl[:, 1] - 12, 1) * 8
            + batch_decl[:, 2]
        )
        batch_decl = torch.tensor(
            batch_arg_decl, dtype=torch.int64, device=self.device
        ).reshape((-1, 1))

        test_decl = self.convert_to_adj_type_oriented(
            test_data.declarations[0], test_data.strongest[0]
        )
        test_arg_decl = (
            test_decl[:, 0] * 16
            + np.minimum(test_decl[:, 1] - 12, 1) * 8
            + test_decl[:, 2]
        )
        test_decl = torch.tensor(
            test_arg_decl, dtype=torch.int64, device=self.device
        ).reshape((-1, 1))

        batch_rewards = np.sum(batch_data.rewards[0], axis=1)
        batch_rewards = torch.tensor(
            batch_rewards, dtype=torch.float32, device=self.device
        ).reshape((-1, 1))

        test_rewards = np.sum(test_data.rewards[0], axis=1)
        test_rewards = torch.tensor(
            test_rewards, dtype=torch.float32, device=self.device
        ).reshape((-1, 1))

        self.decl_model.train()

        for _ in range(epoch):
            self.copy_target()

            #
            # Training
            #
            evaluated: torch.Tensor = self.decl_model(batch_first_boards)
            evaluated = evaluated.gather(1, batch_decl)

            criterion = torch.nn.SmoothL1Loss()
            loss: torch.Tensor = criterion(evaluated, batch_rewards)

            self.decl_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(
                self.decl_model.parameters(), self.h_params.decl_clip_grad
            )
            self.optimizer.step()

            if self.writer:
                self.writer.add_scalar(
                    "loss/decl-train", loss.item(), self.decl_global_step
                )

            #
            # Test
            #
            with torch.no_grad():
                evaluated: torch.Tensor = self.decl_model(test_first_boards)
                evaluated = evaluated.gather(1, test_decl)

                criterion = torch.nn.SmoothL1Loss()
                loss: torch.Tensor = criterion(evaluated, test_rewards)

                if self.writer:
                    self.writer.add_scalar(
                        "loss/decl-test", loss.item(), self.decl_global_step
                    )

            self.decl_global_step += 1

    def train(
        self,
        recorder: Recorder,
        batch_size: int,
        test_size: int,
        epoch: int = 100,
    ):
        batch_data, test_data = recorder.gen_batch(batch_size, test_size)

        batch_boards = batch_data.boards[0].reshape((-1, BOARD_VEC_SIZE))
        batch_boards = torch.tensor(
            batch_boards, dtype=torch.float32, device=self.device
        )

        test_boards = test_data.boards[0].reshape((-1, BOARD_VEC_SIZE))
        test_boards = torch.tensor(test_boards, dtype=torch.float32, device=self.device)

        batch_decisions = batch_data.decisions[0].reshape((-1, 54))
        batch_decisions = torch.tensor(
            np.argmax(batch_decisions, axis=1)[:, None],
            dtype=torch.int64,
            device=self.device,
        )

        test_decisions = test_data.decisions[0].reshape((-1, 54))
        test_decisions = torch.tensor(
            np.argmax(test_decisions, axis=1)[:, None],
            dtype=torch.int64,
            device=self.device,
        )

        self.model.train()

        for _ in range(epoch):
            self.copy_target()

            #
            # Training
            #
            estimations = self.estimate_Q_value(
                batch_data, self.h_params.gamma
            ).reshape((-1, 1))

            evaluated: torch.Tensor = self.model(batch_boards)
            evaluated = evaluated.gather(1, batch_decisions)

            criterion = torch.nn.SmoothL1Loss()
            loss: torch.Tensor = criterion(evaluated, estimations)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.h_params.clip_grad
            )
            self.optimizer.step()

            if self.writer:
                self.writer.add_scalar("loss/train", loss.item(), self.global_step)

            #
            # Test
            #
            estimations = self.estimate_Q_value(test_data, self.h_params.gamma).reshape(
                (-1, 1)
            )

            with torch.no_grad():
                evaluated: torch.Tensor = self.model(test_boards)
                evaluated = evaluated.gather(1, test_decisions)

                criterion = torch.nn.SmoothL1Loss()
                loss: torch.Tensor = criterion(evaluated, estimations)

                if self.writer:
                    self.writer.add_scalar("loss/test", loss.item(), self.global_step)

            self.global_step += 1
