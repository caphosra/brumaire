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
    HComponent,
    CardComponent,
)
from brumaire.rendering.hand import HandComponent
from brumaire.board import BoardData
from brumaire.controller import BrumaireController
from brumaire.constants import AdjStrategy


class DeclEvalTableComponent(ComponentBase):
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
                return NothingComponent(DeclEvalTableComponent.CELL_SIZE)
            if x == 0 and y != 0:
                return SuitComponent(
                    DeclEvalTableComponent.CELL_SIZE,
                    DeclEvalTableComponent.SUIT_SIZE,
                    y - 1,
                )
            if x != 0 and y == 0:
                return TextComponent(
                    DeclEvalTableComponent.CELL_SIZE,
                    AdjStrategy.to_str(x - 1),
                    DeclEvalTableComponent.FONT_SIZE,
                    DeclEvalTableComponent.FONT_COLOR,
                )
            p_12 = win_p[idx, y - 1, 0, x - 1]
            p_13 = win_p[idx, y - 1, 1, x - 1]
            return TextComponent(
                DeclEvalTableComponent.CELL_SIZE,
                f"{p_12:.3f} / {p_13:.3f}",
                DeclEvalTableComponent.FONT_SIZE,
                DeclEvalTableComponent.FONT_COLOR,
            )

        self.child = MarginComponent(
            DeclEvalTableComponent.DECL_MARGIN,
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


class DeclTableComponent(ComponentBase):
    MARGIN = 10

    DECL_SUIT_BOX_SIZE = 40, 50
    DECL_SUIT_SIZE = 30
    DECL_TEXT_SIZE = 40, 50
    DECL_FONT_SIZE = 20

    NAPOLEON_TEXT_HEIGHT = 20
    NAPOLEON_FONT_SIZE = 10

    ADJ_FONT_SIZE = 10
    ADJ_TEXT_HEIGHT = 20
    ADJ_CARD_SIZE = 40, 50

    FONT_COLOR = 10, 10, 10

    child: ComponentBase

    def __init__(self, idx: int, board: BoardData) -> None:
        suit = board.decl[idx, 0]
        num = board.decl[idx, 1]

        decl_component = MarginComponent(
            DeclTableComponent.MARGIN,
            HComponent(
                [
                    SuitComponent(
                        DeclTableComponent.DECL_SUIT_BOX_SIZE,
                        DeclTableComponent.DECL_SUIT_SIZE,
                        suit,
                    ),
                    TextComponent(
                        DeclTableComponent.DECL_TEXT_SIZE,
                        str(num),
                        DeclTableComponent.DECL_FONT_SIZE,
                        DeclTableComponent.FONT_COLOR,
                    ),
                ]
            ),
        )

        napoleon = board.get_napoleon()[idx]
        width, _ = decl_component.get_size()
        decl_component = VComponent(
            [
                TextComponent(
                    (width, DeclTableComponent.NAPOLEON_TEXT_HEIGHT),
                    f"pl. {napoleon}",
                    DeclTableComponent.NAPOLEON_FONT_SIZE,
                    DeclTableComponent.FONT_COLOR,
                ),
                decl_component,
            ]
        )

        adj_suit, adj_num = board.get_adj_card(idx)
        card_component = MarginComponent(
            DeclTableComponent.MARGIN,
            CardComponent(DeclTableComponent.ADJ_CARD_SIZE, adj_suit, adj_num),
        )

        width, _ = card_component.get_size()
        card_component = VComponent(
            [
                TextComponent(
                    (width, DeclTableComponent.ADJ_TEXT_HEIGHT),
                    "Adj.",
                    DeclTableComponent.ADJ_FONT_SIZE,
                    DeclTableComponent.FONT_COLOR,
                ),
                card_component,
            ]
        )

        self.child = MarginComponent(
            DeclTableComponent.MARGIN, HComponent([decl_component, card_component])
        )

    def get_size(self) -> Tuple[int, int]:
        return self.child.get_size()

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self.child.draw(image, pos)
