from typing import Tuple
from PIL import Image
import numpy as np

from brumaire.board import BoardData
from brumaire.constants import CardStatus
from brumaire.controller import BrumaireController
from brumaire.rendering.base import (
    ComponentBase,
    CardComponent,
    NothingComponent,
    MarginComponent,
    HComponent,
    VComponent,
    TextComponent,
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


class HandEvalComponent(ComponentBase):
    HAND_MARGIN = 5
    CARD_SIZE = (40, 50)
    TEXT_SIZE = (40, 20)
    FONT_SIZE = 10
    FONT_COLOR = (10, 10, 10)

    child: ComponentBase

    def __init__(
        self, idx: int, board: BoardData, controller: BrumaireController
    ) -> None:
        trick_input = board.to_trick_input()
        hand_index = board.get_filtered_hand_index(0)
        rewards = controller.estimate_rewards(trick_input, hand_index)

        children = list()

        for hand_idx in range(10):
            card = np.argwhere(
                (board.cards[idx, :, 0] == CardStatus.IN_HAND)
                & (board.cards[idx, :, 1] == 0)
                & (board.cards[idx, :, 2] == hand_idx)
            )[:, 0]
            assert card.shape[0] <= 1

            if card.shape[0] == 1:
                suit = card[0] // 13
                num = card[0] % 13 + 2
                reward = rewards[idx, hand_idx]
                child = VComponent(
                    [
                        TextComponent(
                            HandEvalComponent.TEXT_SIZE,
                            f"{reward:.2f}",
                            HandEvalComponent.FONT_SIZE,
                            HandEvalComponent.FONT_COLOR,
                        ),
                        CardComponent(HandEvalComponent.CARD_SIZE, suit, num),
                    ]
                )
            else:
                assert rewards[idx, hand_idx] == -np.inf

                child = VComponent(
                    [
                        NothingComponent(HandEvalComponent.TEXT_SIZE),
                        NothingComponent(HandEvalComponent.CARD_SIZE),
                    ]
                )

            child = MarginComponent(HandComponent.HAND_MARGIN, child)
            children.append(child)

        self.child = HComponent(children)

    def get_size(self) -> Tuple[int, int]:
        return self.child.get_size()

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self.child.draw(image, pos)
