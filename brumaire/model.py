from typing import Any
import torch
import torch.nn.functional as F

from . import *
from brumaire.board import BOARD_VEC_SIZE
from brumaire.record import Recorder

LINEAR1_NODE_NUM = 100
LINEAR2_NODE_NUM = 100

class BrumaireModel(torch.nn.Module):
    layer1: torch.nn.Linear
    layer2: torch.nn.Linear
    layer3: torch.nn.Linear

    def __init__(self, device) -> None:
        super(BrumaireModel, self).__init__()

        self.layer1 = torch.nn.Linear(BOARD_VEC_SIZE, LINEAR1_NODE_NUM, device=device)
        self.layer2 = torch.nn.Linear(LINEAR1_NODE_NUM, LINEAR2_NODE_NUM, device=device)
        self.layer3 = torch.nn.Linear(LINEAR2_NODE_NUM, 54, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class BrumaireController:
    model: BrumaireModel
    device: Any

    def __init__(self, device) -> None:
        self.model = BrumaireModel(device)
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
