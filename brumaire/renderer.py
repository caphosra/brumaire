from __future__ import annotations
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from brumaire.board import BoardData
from brumaire.constants import (
    Suit,
    Role,
    CardStatus,
    AdjStrategy,
)
from brumaire.record import Recorder
from brumaire.controller import BrumaireController

IMAGE_DIRECTORY = "./img"
FONT_FILE = "./fonts/NotoSansJP-Regular.ttf"

BOARD_MARGIN = 10
CELL_WIDTH = 50
CELL_HEIGHT = 50
BOARD_SUIT_SIZE = 30
BOARD_WIDTH = CELL_WIDTH * 14 + BOARD_MARGIN * 2
BOARD_HEIGHT = CELL_HEIGHT * 6 + BOARD_MARGIN * 2

INFO_MARGIN = 10
INFO_CELL_WIDTH = 50
INFO_CELL_HEIGHT = 50
INFO_WIDTH = INFO_CELL_WIDTH * 5 + INFO_MARGIN * 2
INFO_HEIGHT = INFO_CELL_HEIGHT * 2 + INFO_MARGIN * 2

DECL_TABLE_MARGIN = 10
DECL_TABLE_CELL_WIDTH = 50
DECL_TABLE_CELL_HEIGHT = 50
DECL_TABLE_SUIT_SIZE = 20
DECL_TABLE_WIDTH = DECL_TABLE_CELL_WIDTH * 4 + DECL_TABLE_MARGIN * 2
DECL_TABLE_HEIGHT = DECL_TABLE_CELL_HEIGHT * 2 + DECL_TABLE_MARGIN * 2


class Renderer:
    suit_images: List[Image.Image]
    font: ImageFont.FreeTypeFont
    small_font: ImageFont.FreeTypeFont

    def __init__(self) -> None:
        self.suit_images = [
            Image.open(os.path.join(IMAGE_DIRECTORY, "club.png")),
            Image.open(os.path.join(IMAGE_DIRECTORY, "diamond.png")),
            Image.open(os.path.join(IMAGE_DIRECTORY, "heart.png")),
            Image.open(os.path.join(IMAGE_DIRECTORY, "spade.png")),
        ]
        self.font = ImageFont.truetype(FONT_FILE, 20)
        self.small_font = ImageFont.truetype(FONT_FILE, 10)

    def __enter__(self) -> Renderer:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def num_to_rich(self, num: int) -> str:
        return (
            "A"
            if num == 14
            else (
                "K"
                if num == 13
                else ("Q" if num == 12 else "J" if num == 11 else str(num))
            )
        )

    def draw_suit(
        self,
        image: Image.Image,
        pos: Tuple[int, int],
        rect: Tuple[int, int],
        size: int,
        suit: int,
    ) -> None:
        left, top = pos
        width, height = rect
        if suit == Suit.JOKER:
            draw = ImageDraw.Draw(image)
            draw.text(
                (
                    left + width // 2,
                    top + height // 2,
                ),
                text="J",
                anchor="mm",
                font=self.font,
                fill=(10, 10, 10),
            )
        else:
            margin_left = (width - size) // 2
            margin_top = (height - size) // 2
            suit_image = self.suit_images[suit].resize((size, size))
            image.paste(
                suit_image,
                (left + margin_left, top + margin_top),
                mask=suit_image,
            )

    def draw_card(
        self,
        image: Image.Image,
        pos: Tuple[int, int],
        rect: Tuple[int, int],
        suit: int,
        num: int,
    ) -> Image:
        SUIT_RATIO = 0.8

        left, top = pos
        width, height = rect
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (left, top, left + width, top + height), outline=(10, 10, 10), width=1
        )
        if suit != Suit.JOKER:
            draw.text(
                (
                    left + width // 2,
                    top + height * 3 // 4,
                ),
                text=self.num_to_rich(num + 2),
                anchor="mm",
                font=self.font,
                fill=(10, 10, 10),
            )
        self.draw_suit(
            image,
            (left + width // 2 - height // 4, top),
            (height // 2, height // 2),
            int(height // 2 * SUIT_RATIO),
            suit,
        )

        return image

    def role_color(self, role: int) -> str:
        if role == Role.UNKNOWN:
            return (10, 10, 10)
        elif role == Role.NAPOLEON:
            return (250, 10, 10)
        elif role == Role.ADJUTANT:
            return (250, 10, 250)
        elif role == Role.ALLY:
            return (10, 10, 250)
        else:
            raise "An invalid role is passed."

    def _draw_table(
        self,
        image: Image.Image,
        pos: Tuple[int, int],
        cell_size: Tuple[int, int],
        cell_num: Tuple[int, int],
    ) -> None:
        draw = ImageDraw.Draw(image)

        left, top = pos
        cell_width, cell_height = cell_size
        cell_x, cell_y = cell_num

        for x in range(cell_x + 1):
            draw.line(
                (
                    left + cell_width * x,
                    top,
                    left + cell_width * x,
                    top + cell_height * cell_y,
                ),
                fill=(10, 10, 10),
                width=1,
            )
        for y in range(cell_y + 1):
            draw.line(
                (
                    left,
                    top + cell_height * y,
                    left + cell_width * cell_x,
                    top + cell_height * y,
                ),
                fill=(10, 10, 10),
                width=1,
            )

    def _draw_hand(
        self,
        image: Image.Image,
        pos: Tuple[int, int],
        idx: int,
        board: BoardData,
        player: int,
    ) -> Tuple[int, int]:
        HAND_MARGIN = 5
        CARD_WIDTH = 40
        CARD_HEIGHT = 50

        left, top = pos
        hands = np.argwhere(board.get_hand(idx, player))[:, 0]
        for idx in range(hands.shape[0]):
            suit = hands[idx] // 13
            num = hands[idx] % 13
            image = self.draw_card(
                image,
                (
                    left + HAND_MARGIN + (HAND_MARGIN + CARD_WIDTH) * idx,
                    top + HAND_MARGIN,
                ),
                (CARD_WIDTH, CARD_HEIGHT),
                suit,
                num,
            )

        width = HAND_MARGIN * 2 + CARD_WIDTH * hands.shape[0]
        height = HAND_MARGIN * 2 + CARD_HEIGHT
        return width, height

    def draw_decl_table(
        self,
        image: Image.Image,
        idx: int,
        board: BoardData,
        player: int,
        controller: BrumaireController,
    ) -> None:
        DECL_MARGIN = 10
        DECL_CELL_HEIGHT = 50
        DECL_CELL_WIDTH = 100
        DECL_SUIT_RATIO = 0.8

        draw = ImageDraw.Draw(image)

        _, hand_height = self._draw_hand(
            image, (DECL_MARGIN, DECL_MARGIN), idx, board, player
        )

        self._draw_table(
            image,
            (DECL_MARGIN, hand_height + DECL_MARGIN),
            (DECL_CELL_WIDTH, DECL_CELL_HEIGHT),
            (AdjStrategy.LENGTH + 1, 4 + 1),
        )

        decl_input = board.convert_to_decl_input(player)
        win_p = controller.estimate_win_p(decl_input)

        for suit in range(4):
            left = DECL_MARGIN
            top = hand_height + DECL_MARGIN + DECL_CELL_HEIGHT * (suit + 1)
            size = int(DECL_CELL_HEIGHT * DECL_SUIT_RATIO)
            self.draw_suit(
                image, (left, top), (DECL_CELL_WIDTH, DECL_CELL_HEIGHT), size, suit
            )

        for st in range(AdjStrategy.LENGTH):
            left = DECL_MARGIN + DECL_CELL_WIDTH * (st + 1)
            top = hand_height + DECL_MARGIN
            size = int(DECL_CELL_HEIGHT * DECL_SUIT_RATIO)
            draw.text(
                (
                    left + DECL_CELL_WIDTH // 2,
                    top + DECL_CELL_HEIGHT // 2,
                ),
                text=AdjStrategy.to_str(st),
                anchor="mm",
                font=self.small_font,
                fill=(10, 10, 10),
            )

        for st in range(AdjStrategy.LENGTH):
            for suit in range(4):
                left = DECL_MARGIN + DECL_CELL_WIDTH * (st + 1)
                top = DECL_MARGIN + hand_height + DECL_CELL_HEIGHT * (suit + 1)
                p_12 = win_p[idx, suit, 0, st]
                p_13 = win_p[idx, suit, 1, st]
                draw.text(
                    (
                        left + DECL_CELL_WIDTH // 2,
                        top + DECL_CELL_HEIGHT // 2,
                    ),
                    text=f"{p_12:.3f} / {p_13:.3f}",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 10),
                )

    def _draw_player_info(
        self, image: Image.Image, top: int, left: int, idx: int, board: BoardData
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)

        for player_idx in range(3):
            draw.line(
                (
                    left + INFO_MARGIN,
                    top + INFO_MARGIN + INFO_CELL_HEIGHT * player_idx,
                    left + INFO_MARGIN + INFO_CELL_WIDTH * 5,
                    top + INFO_MARGIN + INFO_CELL_HEIGHT * player_idx,
                ),
                fill=(10, 10, 10),
                width=1,
            )
        for player_idx in range(6):
            draw.line(
                (
                    left + INFO_MARGIN + INFO_CELL_WIDTH * player_idx,
                    top + INFO_MARGIN,
                    left + INFO_MARGIN + INFO_CELL_WIDTH * player_idx,
                    top + INFO_MARGIN + INFO_CELL_HEIGHT * 2,
                ),
                fill=(10, 10, 10),
                width=1,
            )

        for player_idx in range(5):
            draw.text(
                (
                    left
                    + INFO_MARGIN
                    + INFO_CELL_WIDTH * player_idx
                    + INFO_CELL_WIDTH // 2,
                    top + INFO_MARGIN + INFO_CELL_HEIGHT // 2,
                ),
                text=f"{player_idx}",
                anchor="mm",
                font=self.font,
                fill=(10, 10, 10),
            )
            role = board.roles[idx, player_idx]
            draw.text(
                (
                    left
                    + INFO_MARGIN
                    + INFO_CELL_WIDTH * player_idx
                    + INFO_CELL_WIDTH // 2,
                    top + INFO_MARGIN + INFO_CELL_HEIGHT + INFO_CELL_HEIGHT // 3,
                ),
                text=Role.to_str(role),
                anchor="mm",
                font=self.small_font,
                fill=self.role_color(role),
            )
            draw.text(
                (
                    left
                    + INFO_MARGIN
                    + INFO_CELL_WIDTH * player_idx
                    + INFO_CELL_WIDTH // 2,
                    top + INFO_MARGIN + INFO_CELL_HEIGHT + INFO_CELL_HEIGHT * 2 // 3,
                ),
                text=str(int(board.taken[idx, player_idx])),
                anchor="mm",
                font=self.small_font,
                fill=(10, 10, 10),
            )

        return image

    def _draw_decl(
        self, image: Image.Image, top: int, left: int, idx: int, board: BoardData
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)

        for i in range(3):
            draw.line(
                (
                    left + DECL_TABLE_MARGIN,
                    top + DECL_TABLE_MARGIN + DECL_TABLE_CELL_HEIGHT * i,
                    left + DECL_TABLE_MARGIN + DECL_TABLE_CELL_WIDTH * 4,
                    top + DECL_TABLE_MARGIN + DECL_TABLE_CELL_HEIGHT * i,
                ),
                fill=(10, 10, 10),
                width=1,
            )

        for i in range(3):
            draw.line(
                (
                    left + DECL_TABLE_MARGIN + DECL_TABLE_CELL_WIDTH * 2 * i,
                    top + DECL_TABLE_MARGIN,
                    left + DECL_TABLE_MARGIN + DECL_TABLE_CELL_WIDTH * 2 * i,
                    top + DECL_TABLE_MARGIN + DECL_TABLE_CELL_HEIGHT * 2,
                ),
                fill=(10, 10, 10),
                width=1,
            )

        draw.text(
            (
                left + DECL_TABLE_MARGIN + DECL_TABLE_CELL_WIDTH,
                top + DECL_TABLE_MARGIN + DECL_TABLE_CELL_HEIGHT // 2,
            ),
            text="Decl.",
            anchor="mm",
            font=self.font,
            fill=(10, 10, 10),
        )
        draw.text(
            (
                left + DECL_TABLE_MARGIN + DECL_TABLE_CELL_WIDTH * 3,
                top + DECL_TABLE_MARGIN + DECL_TABLE_CELL_HEIGHT // 2,
            ),
            text="Adj.",
            anchor="mm",
            font=self.font,
            fill=(10, 10, 10),
        )
        self.draw_suit(
            image,
            (
                left + DECL_TABLE_MARGIN,
                top + DECL_TABLE_MARGIN + DECL_TABLE_CELL_HEIGHT,
            ),
            (DECL_TABLE_CELL_WIDTH, DECL_TABLE_CELL_HEIGHT),
            DECL_TABLE_SUIT_SIZE,
            suit=board.decl[idx, 0],
        )
        draw.text(
            (
                left + DECL_TABLE_MARGIN + DECL_TABLE_CELL_WIDTH * 3 // 2,
                top + DECL_TABLE_MARGIN + DECL_TABLE_CELL_HEIGHT * 3 // 2,
            ),
            text=f"{board.decl[idx, 1]}",
            anchor="mm",
            font=self.font,
            fill=(10, 10, 10),
        )
        suit, num = board.get_adj_card(idx)
        self.draw_suit(
            image,
            (
                left + DECL_TABLE_MARGIN + DECL_TABLE_CELL_WIDTH * 2,
                top + DECL_TABLE_MARGIN + DECL_TABLE_CELL_HEIGHT,
            ),
            (DECL_TABLE_CELL_WIDTH, DECL_TABLE_CELL_HEIGHT),
            DECL_TABLE_SUIT_SIZE,
            suit,
        )
        draw.text(
            (
                left + DECL_TABLE_MARGIN + DECL_TABLE_CELL_WIDTH * 7 // 2,
                top + DECL_TABLE_MARGIN + DECL_TABLE_CELL_HEIGHT * 3 // 2,
            ),
            text=self.num_to_rich(num),
            anchor="mm",
            font=self.font,
            fill=(10, 10, 10),
        )

        return image

    def _draw_board_table(
        self, image: Image.Image, top: int, left: int, idx: int, board: BoardData
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)

        for i in range(7):
            draw.line(
                (
                    left + BOARD_MARGIN,
                    top + BOARD_MARGIN + CELL_HEIGHT * i,
                    left + BOARD_MARGIN + CELL_WIDTH * (14 if i != 6 else 3),
                    top + BOARD_MARGIN + CELL_HEIGHT * i,
                ),
                fill=(10, 10, 10),
                width=1,
            )
        for i in range(15):
            draw.line(
                (
                    left + BOARD_MARGIN + CELL_WIDTH * i,
                    top + BOARD_MARGIN,
                    left + BOARD_MARGIN + CELL_WIDTH * i,
                    top + BOARD_MARGIN + CELL_HEIGHT * (5 if i >= 4 else 6),
                ),
                fill=(10, 10, 10),
                width=1,
            )

        for i in range(13):
            draw.text(
                (
                    left + BOARD_MARGIN + CELL_WIDTH * (i + 1) + CELL_WIDTH // 2,
                    top + BOARD_MARGIN + CELL_HEIGHT // 2,
                ),
                text=f"{self.num_to_rich(i + 2)}",
                anchor="mm",
                font=self.font,
                fill=(10, 10, 10),
            )

        for suit in range(4):
            self.draw_suit(
                image,
                (left + BOARD_MARGIN, top + BOARD_MARGIN + CELL_HEIGHT * (suit + 1)),
                (CELL_WIDTH, CELL_HEIGHT),
                BOARD_SUIT_SIZE,
                suit,
            )

        draw.text(
            (
                left + BOARD_MARGIN + CELL_WIDTH // 2,
                top + BOARD_MARGIN + CELL_HEIGHT * 5 + CELL_HEIGHT // 2,
            ),
            text="J",
            anchor="mm",
            font=self.font,
            fill=(10, 10, 10),
        )

        for i in range(54):
            status = board.cards[idx, i, 0]
            if status == CardStatus.UNKNOWN:
                draw.text(
                    (
                        left
                        + BOARD_MARGIN
                        + CELL_WIDTH * (i % 13 + 1)
                        + CELL_WIDTH // 2,
                        top
                        + BOARD_MARGIN
                        + CELL_HEIGHT * (i // 13 + 1)
                        + CELL_HEIGHT // 2,
                    ),
                    text="?",
                    anchor="mm",
                    font=self.font,
                    fill=(10, 10, 10),
                )
            elif status == CardStatus.PLAYED:
                draw.text(
                    (
                        left
                        + BOARD_MARGIN
                        + CELL_WIDTH * (i % 13 + 1)
                        + CELL_WIDTH // 2,
                        top
                        + BOARD_MARGIN
                        + CELL_HEIGHT * (i // 13 + 1)
                        + CELL_HEIGHT // 4,
                    ),
                    text="PLAYED",
                    anchor="mm",
                    font=self.small_font,
                    fill=(240, 10, 10),
                )
                draw.text(
                    (
                        left
                        + BOARD_MARGIN
                        + CELL_WIDTH * (i % 13 + 1)
                        + CELL_WIDTH // 2,
                        top
                        + BOARD_MARGIN
                        + CELL_HEIGHT * (i // 13 + 1)
                        + CELL_HEIGHT // 2,
                    ),
                    text=f"pl. {int(board.cards[idx, i, 1])}",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 10),
                )
                turn = int(board.cards[idx, i, 2])
                draw.text(
                    (
                        left
                        + BOARD_MARGIN
                        + CELL_WIDTH * (i % 13 + 1)
                        + CELL_WIDTH // 2,
                        top
                        + BOARD_MARGIN
                        + CELL_HEIGHT * (i // 13 + 1)
                        + CELL_HEIGHT * 3 // 4,
                    ),
                    text=f"{turn}",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 10),
                )
            elif status == CardStatus.IN_HAND:
                draw.text(
                    (
                        left
                        + BOARD_MARGIN
                        + CELL_WIDTH * (i % 13 + 1)
                        + CELL_WIDTH // 2,
                        top
                        + BOARD_MARGIN
                        + CELL_HEIGHT * (i // 13 + 1)
                        + CELL_HEIGHT // 4,
                    ),
                    text="IN HAND",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 240),
                )
                draw.text(
                    (
                        left
                        + BOARD_MARGIN
                        + CELL_WIDTH * (i % 13 + 1)
                        + CELL_WIDTH // 2,
                        top
                        + BOARD_MARGIN
                        + CELL_HEIGHT * (i // 13 + 1)
                        + CELL_HEIGHT // 2,
                    ),
                    text=f"pl. {int(board.cards[idx, i, 1])}",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 10),
                )
                draw.text(
                    (
                        left
                        + BOARD_MARGIN
                        + CELL_WIDTH * (i % 13 + 1)
                        + CELL_WIDTH // 2,
                        top
                        + BOARD_MARGIN
                        + CELL_HEIGHT * (i // 13 + 1)
                        + CELL_HEIGHT * 3 // 4,
                    ),
                    text=f"No. {int(board.cards[idx, i, 2])}",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 10),
                )

        return image

    def render_board(self, idx: int, board: BoardData) -> Image.Image:
        image = Image.new(
            "RGB",
            size=(BOARD_WIDTH, BOARD_HEIGHT + INFO_HEIGHT),
            color=(255, 255, 255),
        )

        image = self._draw_board_table(image, 0, 0, idx, board)
        image = self._draw_player_info(image, BOARD_HEIGHT, 0, idx, board)
        image = self._draw_decl(image, BOARD_HEIGHT, INFO_WIDTH, idx, board)

        return image

    def render_decl(
        self, idx: int, board: BoardData, player: int, controller: BrumaireController
    ) -> Image.Image:
        image = Image.new(
            "RGB",
            size=(920, 330),
            color=(255, 255, 255),
        )
        self.draw_decl_table(image, idx, board, player, controller)
        return image

    def write_decl(
        self,
        writer: SummaryWriter,
        recorder: Recorder,
        controller: BrumaireController,
        player: int,
        num: int,
        step: int = 0,
    ) -> None:
        size = recorder.get_data_size()

        assert num <= size

        chosen = list(np.random.choice(size, num, replace=False))

        images = np.zeros((num, 3, 330, 920))

        with Renderer() as r:
            for idx, board_idx in enumerate(chosen):
                board_vec = recorder.first_boards[player, board_idx].reshape((1, -1))
                board = BoardData.from_vector(board_vec)

                image = r.render_decl(0, board, 0, controller)
                images[idx] = np.transpose(np.array(image), (2, 0, 1)) / 255

        writer.add_images("decl", images, step)

    def close(self):
        for suit_image in self.suit_images:
            suit_image.close()
