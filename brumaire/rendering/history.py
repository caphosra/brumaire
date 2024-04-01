from typing import Tuple
from PIL import Image
import numpy as np

from brumaire.constants import CardStatus
from brumaire.board import BoardData
from brumaire.rendering.base import (
    ComponentBase,
    CardComponent,
    NothingComponent,
    MarginComponent,
    HComponent,
    VComponent,
    TextComponent,
)


class HistoryComponent(ComponentBase):
    HAND_MARGIN = 5
    CARD_SIZE = (40, 50)
    TEXT_SIZE = (40, 20)
    FONT_SIZE = 10
    FONT_COLOR = (10, 10, 10)

    child: ComponentBase

    def __init__(self, idx: int, board: BoardData) -> None:
        discarded_cards = list()
        discarded = np.argwhere(
            (board.cards[idx, :, 0] == CardStatus.PLAYED)
            & (board.cards[idx, :, 2] == -1)
        )[:, 0]
        discarded_len = discarded.shape[0]
        for discarded_idx in range(4):
            if discarded_idx < discarded_len:
                suit = discarded[discarded_idx] // 13
                num = discarded[discarded_idx] % 13 + 2
                component = CardComponent(HistoryComponent.CARD_SIZE, suit, num)
            else:
                component = NothingComponent(HistoryComponent.CARD_SIZE)
            component = MarginComponent(HistoryComponent.HAND_MARGIN, component)
            discarded_cards.append(component)
        discarded_component = VComponent(discarded_cards)

        played = list(
            np.argwhere(
                (board.cards[idx, :, 0] == CardStatus.PLAYED)
                & (board.cards[idx, :, 2] != -1)
            )[:, 0]
        )
        played = sorted(played, key=lambda c: board.cards[idx, c, 2])
        played_len = len(played)

        played_turn = list()
        for turn in range(10):
            played_components = list()
            for played_idx in range(5):
                card_num = turn * 5 + played_idx
                if card_num < played_len:
                    player = board.cards[idx, played[card_num], 1]
                    suit = played[card_num] // 13
                    num = played[card_num] % 13 + 2
                    component = VComponent(
                        [
                            TextComponent(
                                HistoryComponent.TEXT_SIZE,
                                f"pl.{player}",
                                HistoryComponent.FONT_SIZE,
                                HistoryComponent.FONT_COLOR,
                            ),
                            CardComponent(HistoryComponent.CARD_SIZE, suit, num),
                        ]
                    )
                else:
                    component = VComponent(
                        [
                            NothingComponent(HistoryComponent.TEXT_SIZE),
                            NothingComponent(HistoryComponent.CARD_SIZE),
                        ]
                    )
                component = MarginComponent(HistoryComponent.HAND_MARGIN, component)
                played_components.append(component)
            played_turn.append(VComponent(played_components))
        played_component = HComponent(played_turn)

        self.child = HComponent([discarded_component, played_component])

    def get_size(self) -> Tuple[int, int]:
        return self.child.get_size()

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self.child.draw(image, pos)
