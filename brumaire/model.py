from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from brumaire.board import BOARD_VEC_SIZE
from brumaire.constants import DECL_INPUT_SIZE


class BrumaireHParams:
    decl_l1_node: int
    decl_l2_node: int
    decl_l3_node: int

    decl_ita: float = 0.0
    decl_clip_grad: float = 0.0

    trick_l1_node: int
    trick_l2_node: int
    trick_l3_node: int

    gamma: float = 0.0
    trick_ita: float = 0.0
    trick_clip_grad: float = 0.0

    def write_summary(self, writer: SummaryWriter):
        exp, ssi, sei = hparams(
            {
                "decl l1 node": self.decl_l1_node,
                "decl l2 node": self.decl_l2_node,
                "decl l3 node": self.decl_l3_node,
                "decl ita": self.decl_ita,
                "decl clip grad": self.decl_clip_grad,
                "trick l1 node": self.trick_l1_node,
                "trick l2 node": self.trick_l2_node,
                "trick l3 node": self.trick_l3_node,
                "trick ita": self.trick_ita,
                "trick clip grad": self.trick_clip_grad,
                "gamma": self.gamma,
            },
            {},
        )

        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)


class BrumaireDeclModel(torch.nn.Module):
    layer1: torch.nn.Linear
    dropout_layer1: torch.nn.Dropout
    layer2: torch.nn.Linear
    dropout_layer2: torch.nn.Dropout
    layer3: torch.nn.Linear
    dropout_layer3: torch.nn.Dropout
    layer4: torch.nn.Linear

    def __init__(self, h_param: BrumaireHParams, device) -> None:
        super(BrumaireDeclModel, self).__init__()

        self.layer1 = torch.nn.Linear(
            DECL_INPUT_SIZE, h_param.decl_l1_node, device=device
        )
        self.dropout_layer1 = torch.nn.Dropout()
        self.layer2 = torch.nn.Linear(
            h_param.decl_l1_node, h_param.decl_l2_node, device=device
        )
        self.dropout_layer2 = torch.nn.Dropout()
        self.layer3 = torch.nn.Linear(
            h_param.decl_l2_node, h_param.decl_l3_node, device=device
        )
        self.dropout_layer3 = torch.nn.Dropout()
        self.layer4 = torch.nn.Linear(h_param.decl_l3_node, 2, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.dropout_layer1(self.layer1(x)))
        x = F.leaky_relu(self.dropout_layer2(self.layer2(x)))
        x = F.leaky_relu(self.dropout_layer3(self.layer3(x)))
        return self.layer4(x)


class BrumaireTrickModel(torch.nn.Module):
    layer1: torch.nn.Linear
    dropout_layer1: torch.nn.Dropout
    layer2: torch.nn.Linear
    dropout_layer2: torch.nn.Dropout
    layer3: torch.nn.Linear
    dropout_layer3: torch.nn.Dropout
    layer4: torch.nn.Linear

    def __init__(self, h_param: BrumaireHParams, device) -> None:
        super(BrumaireTrickModel, self).__init__()

        self.layer1 = torch.nn.Linear(
            BOARD_VEC_SIZE, h_param.trick_l1_node, device=device
        )
        self.dropout_layer1 = torch.nn.Dropout()
        self.layer2 = torch.nn.Linear(
            h_param.trick_l1_node, h_param.trick_l2_node, device=device
        )
        self.dropout_layer2 = torch.nn.Dropout()
        self.layer3 = torch.nn.Linear(
            h_param.trick_l2_node, h_param.trick_l3_node, device=device
        )
        self.dropout_layer3 = torch.nn.Dropout()
        self.layer4 = torch.nn.Linear(h_param.trick_l3_node, 54, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.dropout_layer1(self.layer1(x)))
        x = F.leaky_relu(self.dropout_layer2(self.layer2(x)))
        x = F.leaky_relu(self.dropout_layer3(self.layer3(x)))
        return self.layer4(x)
