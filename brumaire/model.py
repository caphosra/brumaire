import numpy as np
from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from brumaire.board import BOARD_VEC_SIZE
from brumaire.constants import NDIntArray, NDFloatArray
from brumaire.record import Recorder


class BrumaireHParams:
    linear1_node_num: int
    linear2_node_num: int
    linear3_node_num: int
    ita: float
    gamma: float
    clip_grad: float

    def write_summary(self, writer: SummaryWriter):
        writer.add_hparams(
            {
                "linear1 node num": self.linear1_node_num,
                "linear2 node num": self.linear2_node_num,
                "linear3 node num": self.linear3_node_num,
                "ita": self.ita,
                "gamma": self.gamma,
                "clip grad": self.clip_grad,
            },
            {},
        )


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
            BOARD_VEC_SIZE, h_param.linear1_node_num, device=device
        )
        self.dropout_layer1 = torch.nn.Dropout()
        self.layer2 = torch.nn.Linear(
            h_param.linear1_node_num, h_param.linear2_node_num, device=device
        )
        self.dropout_layer2 = torch.nn.Dropout()
        self.layer3 = torch.nn.Linear(
            h_param.linear2_node_num, h_param.linear3_node_num, device=device
        )
        self.dropout_layer3 = torch.nn.Dropout()
        self.layer4 = torch.nn.Linear(h_param.linear3_node_num, 54, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.dropout_layer1(self.layer1(x)))
        x = F.leaky_relu(self.dropout_layer2(self.layer2(x)))
        x = F.leaky_relu(self.dropout_layer3(self.layer3(x)))
        return self.layer4(x)


class BrumaireController:
    model: BrumaireModel
    target: BrumaireModel
    h_params: BrumaireHParams
    optimizer: torch.optim.Optimizer
    writer: SummaryWriter | None
    global_step: int
    device: Any

    def __init__(
        self,
        h_params: BrumaireHParams,
        device,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.model = BrumaireModel(h_params, device)
        self.target = BrumaireModel(h_params, device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=h_params.ita, amsgrad=True
        )
        self.writer = writer
        self.global_step = 0
        self.device = device
        self.h_params = h_params

    def save(self, dir_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(dir_path, "model_data"))

    def load(self, dir_path: str) -> None:
        state = torch.load(os.path.join(dir_path, "model_data"))
        self.model.load_state_dict(state)

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
        boards = recorder.boards.reshape((-1, 10, BOARD_VEC_SIZE))
        boards = torch.tensor(boards, dtype=torch.float32, device=self.device)

        rewards = recorder.rewards.reshape((-1, 10))
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        hand_filters = recorder.hand_filters.reshape((-1, 10, 54))
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

    def train(
        self,
        recorder: Recorder,
        batch_size: int,
        test_size: int,
        epoch: int = 100,
    ):
        batch_data, test_data = recorder.gen_batch(batch_size, test_size)

        batch_boards = batch_data.boards.reshape((-1, BOARD_VEC_SIZE))
        batch_boards = torch.tensor(
            batch_boards, dtype=torch.float32, device=self.device
        )

        test_boards = test_data.boards.reshape((-1, BOARD_VEC_SIZE))
        test_boards = torch.tensor(test_boards, dtype=torch.float32, device=self.device)

        batch_decisions = batch_data.decisions.reshape((-1, 54))
        batch_decisions = torch.tensor(
            np.argmax(batch_decisions, axis=1)[:, None],
            dtype=torch.int64,
            device=self.device,
        )

        test_decisions = test_data.decisions.reshape((-1, 54))
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
