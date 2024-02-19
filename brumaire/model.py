from typing import Any
import torch
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm

from . import *
from brumaire.board import BOARD_VEC_SIZE
from brumaire.record import Recorder

LINEAR1_NODE_NUM = 5000
LINEAR2_NODE_NUM = 2000
LINEAR3_NODE_NUM = 1000


class BrumaireModel(torch.nn.Module):
    layer1: torch.nn.Linear
    layer2: torch.nn.Linear
    layer3: torch.nn.Linear
    layer4: torch.nn.Linear

    def __init__(self, device) -> None:
        super(BrumaireModel, self).__init__()

        self.layer1 = torch.nn.Linear(BOARD_VEC_SIZE, LINEAR1_NODE_NUM, device=device)
        self.layer2 = torch.nn.Linear(LINEAR1_NODE_NUM, LINEAR2_NODE_NUM, device=device)
        self.layer3 = torch.nn.Linear(LINEAR2_NODE_NUM, LINEAR3_NODE_NUM, device=device)
        self.layer4 = torch.nn.Linear(LINEAR3_NODE_NUM, 54, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class BrumaireController:
    model: BrumaireModel
    target: BrumaireModel
    optimizer: torch.optim.Optimizer
    device: Any

    def __init__(self, device, ita: float = 0.001) -> None:
        self.model = BrumaireModel(device)
        self.target = BrumaireModel(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=ita, amsgrad=True
        )
        self.device = device

    def make_decision(
        self, board_vec: NDFloatArray, hand_filter: NDIntArray
    ) -> NDIntArray:
        board_vec = torch.tensor(board_vec, dtype=torch.float32, device=self.device)
        hand_filter = torch.tensor(hand_filter, dtype=torch.float32, device=self.device)

        hand_filter[hand_filter == 0] = -torch.inf
        hand_filter[hand_filter == 1] = 0

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
        gamma: float = 0.99,
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

        for _ in tqdm(range(epoch)):
            self.copy_target()

            #
            # Training
            #
            estimations = self.estimate_Q_value(batch_data, gamma).reshape((-1, 1))

            evaluated: torch.Tensor = self.model(batch_boards)
            evaluated = evaluated.gather(1, batch_decisions)

            criterion = torch.nn.SmoothL1Loss()
            loss: torch.Tensor = criterion(evaluated, estimations)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), 10.0)
            self.optimizer.step()

            print(f"train loss: {loss.item()}")

            #
            # Test
            #
            estimations = self.estimate_Q_value(test_data, gamma).reshape((-1, 1))

            with torch.no_grad():
                evaluated: torch.Tensor = self.model(test_boards)
                evaluated = evaluated.gather(1, test_decisions)

                criterion = torch.nn.SmoothL1Loss()
                loss: torch.Tensor = criterion(evaluated, estimations)
                print(f"test loss: {loss.item()}")
