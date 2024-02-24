from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from brumaire.board import BOARD_VEC_SIZE


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

        self.layer1 = torch.nn.Linear(
            BOARD_VEC_SIZE, h_param.decl_l1_node, device=device
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

        self.layer1 = torch.nn.Linear(BOARD_VEC_SIZE, h_param.l1_node, device=device)
        self.dropout_layer1 = torch.nn.Dropout()
        self.layer2 = torch.nn.Linear(h_param.l1_node, h_param.l2_node, device=device)
        self.dropout_layer2 = torch.nn.Dropout()
        self.layer3 = torch.nn.Linear(h_param.l2_node, h_param.l3_node, device=device)
        self.dropout_layer3 = torch.nn.Dropout()
        self.layer4 = torch.nn.Linear(h_param.l3_node, 54, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.dropout_layer1(self.layer1(x)))
        x = F.leaky_relu(self.dropout_layer2(self.layer2(x)))
        x = F.leaky_relu(self.dropout_layer3(self.layer3(x)))
        return self.layer4(x)
