from typing import Any
import torch
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm

from . import *
from brumaire.board import BOARD_VEC_SIZE
from brumaire.record import Recorder

LINEAR1_NODE_NUM = 1000
LINEAR2_NODE_NUM = 1000
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
    optimizer: torch.optim.Optimizer
    device: Any

    def __init__(self, device, ita: float=0.001) -> None:
        self.model = BrumaireModel(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=ita, amsgrad=True)
        self.device = device

    def make_decision(self, board_vec: NDFloatArray, hand_filter: NDIntArray) -> NDIntArray:
        board_vec = torch.tensor(board_vec, dtype=torch.float32, device=self.device)
        hand_filter = torch.tensor(hand_filter, dtype=torch.float32, device=self.device)

        hand_filter[hand_filter == 0] = -torch.inf
        hand_filter[hand_filter == 1] = 0

        with torch.no_grad():
            evaluated: torch.Tensor = self.model(board_vec) + hand_filter
            evaluated = evaluated.argmax(dim=1).cpu().numpy().astype(int)

        return np.eye(54)[evaluated]

    def train(self, recorder: Recorder, epoch:int = 100, gamma: float = 0.99):
        boards = recorder.boards.reshape((-1, 10, BOARD_VEC_SIZE))
        boards = torch.tensor(boards, dtype=torch.float32, device=self.device)

        rewards = recorder.rewards.reshape((-1, 10))
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        hand_filters = recorder.hand_filters.reshape((-1, 10, 54))
        hand_filters = torch.tensor(hand_filters, dtype=torch.float32, device=self.device)

        hand_filters[hand_filters == 0] = -torch.inf
        hand_filters[hand_filters == 1] = 0

        decisions = recorder.decisions.reshape((-1, 10, 54))

        for _ in tqdm(range(epoch)):
            policy = BrumaireModel(self.device)
            policy.load_state_dict(self.model.state_dict())

            for turn in range(9, -1, -1):
                expectations = rewards[:, turn]
                if turn < 10 - 1:
                    next_boards = boards[:, turn + 1, :]
                    next_hand_filters = hand_filters[:, turn + 1, :]

                    with torch.no_grad():
                        evaluated: torch.Tensor = policy(next_boards) + next_hand_filters
                        expectations += evaluated.max(dim=1)[0] * gamma

                expectations = expectations.reshape((-1, 1))

                boards_tensor = boards[:, turn, :]
                decisions_tensor = torch.tensor(np.argmax(decisions[:, turn, :], axis=1)[:, None], dtype=torch.int64, device=self.device)

                evaluated: torch.Tensor = self.model(boards_tensor)
                evaluated = evaluated.gather(1, decisions_tensor)

                criterion = torch.nn.SmoothL1Loss()
                loss: torch.Tensor = criterion(evaluated, expectations)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_value_(self.model.parameters(), 10.)
                self.optimizer.step()
