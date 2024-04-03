from typing import Tuple, List
from PIL import Image

from brumaire.controller import BrumaireController
from brumaire.board import BoardData
from brumaire.rendering.base import (
    ComponentBase,
    MarginComponent,
    HComponent,
    VComponent,
    TextComponent,
    TableComponent,
)
from brumaire.rendering.history import HistoryComponent
from brumaire.rendering.decl import DeclTableComponent
from brumaire.rendering.hand import HandEvalComponent, HandComponent
from brumaire.constants import Role


class RoleComponent(ComponentBase):
    SIZE = 40, 20
    FONT_SIZE = 10

    child: ComponentBase

    @staticmethod
    def role_to_color(role: Role) -> Tuple[int, int, int]:
        match role:
            case Role.UNKNOWN:
                return 10, 10, 10
            case Role.ADJUTANT:
                return 100, 10, 100
            case Role.NAPOLEON:
                return 240, 10, 10
            case Role.ALLY:
                return 10, 10, 240
            case _:
                raise "An invalid role is detected."

    def __init__(self, idx: int, board: BoardData, player: int) -> None:
        role = board.roles[idx, player]
        self.child = TextComponent(
            RoleComponent.SIZE,
            Role.to_str(role),
            RoleComponent.FONT_SIZE,
            RoleComponent.role_to_color(role),
        )

    def get_size(self) -> Tuple[int, int]:
        return self.child.get_size()

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self.child.draw(image, pos)


class PlayerComponent(ComponentBase):
    MARGIN = 5
    COLLECTED_SIZE = 40, 20
    FONT_SIZE = 15
    FONT_COLOR = 10, 10, 10

    child: ComponentBase

    def __init__(self, idx: int, board: BoardData, player: int) -> None:
        collected = board.taken[idx, player]

        self.child = MarginComponent(
            PlayerComponent.MARGIN,
            VComponent(
                [
                    RoleComponent(idx, board, player),
                    TextComponent(
                        PlayerComponent.COLLECTED_SIZE,
                        str(collected),
                        PlayerComponent.FONT_SIZE,
                        PlayerComponent.FONT_COLOR,
                    ),
                ]
            ),
        )

    def get_size(self) -> Tuple[int, int]:
        return self.child.get_size()

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self.child.draw(image, pos)


class PlayerInfoComponent(ComponentBase):
    MARGIN = 10
    TEXT_HEIGHT = 20
    FONT_SIZE = 10
    FONT_COLOR = 10, 10, 10

    child: ComponentBase

    def __init__(self, idx: int, board: BoardData) -> None:
        children: List[ComponentBase] = list()
        for player_idx in range(5):
            component = PlayerComponent(idx, board, player_idx)
            children.append(component)

        width, _ = children[0].get_size()

        def children_fun(x, y):
            if y == 0:
                return TextComponent(
                    (width, PlayerInfoComponent.TEXT_HEIGHT),
                    f"pl. {x}",
                    PlayerInfoComponent.FONT_SIZE,
                    PlayerInfoComponent.FONT_COLOR,
                )
            else:
                return children[x]

        self.child = MarginComponent(
            PlayerInfoComponent.MARGIN, TableComponent(5, 2, children_fun)
        )

    def get_size(self) -> Tuple[int, int]:
        return self.child.get_size()

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self.child.draw(image, pos)


class BoardComponent(ComponentBase):
    MARGIN = 5

    hand_component: ComponentBase
    child: ComponentBase

    def __init__(
        self, idx: int, board: BoardData, controller: BrumaireController | None = None
    ) -> None:
        if controller is None:
            self.hand_component = HandComponent(idx, board, 0)
        else:
            self.hand_component = HandEvalComponent(idx, board, controller)

        self.child = MarginComponent(
            BoardComponent.MARGIN,
            VComponent(
                [
                    HComponent(
                        [
                            HistoryComponent(idx, board),
                            VComponent(
                                [
                                    DeclTableComponent(idx, board),
                                    PlayerInfoComponent(idx, board),
                                ]
                            ),
                        ]
                    ),
                    self.hand_component,
                ]
            ),
        )

    def get_size(self) -> Tuple[int, int]:
        return self.child.get_size()

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self.child.draw(image, pos)
