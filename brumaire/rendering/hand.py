from typing import Tuple
from PIL import Image
import numpy as np

from brumaire.board import BoardData
from brumaire.rendering.base import (
    ComponentBase,
    CardComponent,
    NothingComponent,
    MarginComponent,
    HComponent,
)


class HandComponent(ComponentBase):
    HAND_MARGIN = 5
    CARD_WIDTH = 40
    CARD_HEIGHT = 50

    child: ComponentBase

    def __init__(
        self, idx: int, board: BoardData, player: int, hand_size: int = 10
    ) -> None:
        hands = np.argwhere(board.get_hand(idx, player))[:, 0]
        length = hands.shape[0]
        children = list()

        for idx in range(hand_size):
            if idx < length:
                suit = hands[idx] // 13
                num = hands[idx] % 13 + 2
                child = CardComponent(
                    (HandComponent.CARD_WIDTH, HandComponent.CARD_HEIGHT), suit, num
                )
            else:
                child = NothingComponent(
                    (HandComponent.CARD_WIDTH, HandComponent.CARD_HEIGHT)
                )

            child = MarginComponent(HandComponent.HAND_MARGIN, child)
            children.append(child)

        self.child = HComponent(children)

    def get_size(self) -> Tuple[int, int]:
        return self.child.get_size()

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self.child.draw(image, pos)
