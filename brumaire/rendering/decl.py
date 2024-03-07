from typing import Tuple
from PIL import Image

from brumaire.rendering.base import (
    ComponentBase,
    TableComponent,
    SuitComponent,
    TextComponent,
    NothingComponent,
    MarginComponent,
    VComponent,
)
from brumaire.rendering.hand import HandComponent
from brumaire.board import BoardData
from brumaire.controller import BrumaireController
from brumaire.constants import AdjStrategy


class DeclTableComponent(ComponentBase):
    DECL_MARGIN = 10
    CELL_SIZE = 100, 50
    SUIT_SIZE = 40
    FONT_SIZE = 10
    FONT_COLOR = (10, 10, 10)

    child: ComponentBase

    def __init__(
        self, idx: int, board: BoardData, player: int, controller: BrumaireController
    ) -> None:
        decl_input = board.convert_to_decl_input(player)
        win_p = controller.estimate_win_p(decl_input)

        def child_fun(x, y):
            if x == 0 and y == 0:
                return NothingComponent(DeclTableComponent.CELL_SIZE)
            if x == 0 and y != 0:
                return SuitComponent(
                    DeclTableComponent.CELL_SIZE, DeclTableComponent.SUIT_SIZE, y - 1
                )
            if x != 0 and y == 0:
                return TextComponent(
                    DeclTableComponent.CELL_SIZE,
                    AdjStrategy.to_str(x - 1),
                    DeclTableComponent.FONT_SIZE,
                    DeclTableComponent.FONT_COLOR,
                )
            p_12 = win_p[idx, y - 1, 0, x - 1]
            p_13 = win_p[idx, y - 1, 1, x - 1]
            return TextComponent(
                DeclTableComponent.CELL_SIZE,
                f"{p_12:.3f} / {p_13:.3f}",
                DeclTableComponent.FONT_SIZE,
                DeclTableComponent.FONT_COLOR,
            )

        self.child = MarginComponent(
            DeclTableComponent.DECL_MARGIN,
            VComponent(
                [
                    HandComponent(idx, board, player),
                    TableComponent(AdjStrategy.LENGTH + 1, 4 + 1, child_fun),
                ]
            ),
        )

    def get_size(self) -> Tuple[int, int]:
        return self.child.get_size()

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self.child.draw(image, pos)
