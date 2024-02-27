from __future__ import annotations
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
import os

from brumaire.board import BoardData
from brumaire.constants import (
    CARD_IN_HAND,
    CARD_TRICKED,
    CARD_UNKNOWN,
    ROLE_UNKNOWN,
    ROLE_NAPOLEON,
    ROLE_ADJUTANT,
    ROLE_ALLY,
)
from brumaire.utils import role_to_str

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
        margin_left = (width - size) // 2
        margin_top = (height - size) // 2
        suit_image = self.suit_images[suit].resize((size, size))
        image.paste(
            suit_image,
            (left + margin_left, top + margin_top),
            mask=suit_image,
        )

    def role_color(self, role: int) -> str:
        if role == ROLE_UNKNOWN:
            return (10, 10, 10)
        elif role == ROLE_NAPOLEON:
            return (250, 10, 10)
        elif role == ROLE_ADJUTANT:
            return (250, 10, 250)
        elif role == ROLE_ALLY:
            return (10, 10, 250)
        else:
            raise "An invalid role is passed."

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
                text=role_to_str(role),
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
            if status == CARD_UNKNOWN:
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
            elif status == CARD_TRICKED:
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
                    text="TRICKED",
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
            elif status == CARD_IN_HAND:
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

    def close(self):
        for suit_image in self.suit_images:
            suit_image.close()
