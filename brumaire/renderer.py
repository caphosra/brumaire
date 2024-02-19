from typing import List
from PIL import Image, ImageDraw, ImageFont
import os

from . import *
from brumaire.board import BoardData

IMAGE_DIRECTORY = "./img"
FONT_FILE = "./fonts/NotoSansJP-Regular.ttf"

BOARD_MARGIN = 10
CELL_WIDTH = 50
CELL_HEIGHT = 50


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

    def render_board(self, idx: int, board: BoardData) -> List[Image.Image]:
        image = Image.new(
            "RGB",
            size=(CELL_HEIGHT * 14 + BOARD_MARGIN * 2, 500),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(image)

        for i in range(7):
            draw.line(
                (
                    BOARD_MARGIN,
                    BOARD_MARGIN + CELL_HEIGHT * i,
                    BOARD_MARGIN + CELL_WIDTH * (14 if i != 6 else 3),
                    BOARD_MARGIN + CELL_HEIGHT * i,
                ),
                fill=(10, 10, 10),
                width=1,
            )
        for i in range(15):
            draw.line(
                (
                    BOARD_MARGIN + CELL_WIDTH * i,
                    BOARD_MARGIN,
                    BOARD_MARGIN + CELL_WIDTH * i,
                    BOARD_MARGIN + CELL_HEIGHT * (5 if i >= 4 else 6),
                ),
                fill=(10, 10, 10),
                width=1,
            )

        for i in range(13):
            draw.text(
                (
                    BOARD_MARGIN + CELL_WIDTH * (i + 1) + CELL_WIDTH // 2,
                    BOARD_MARGIN + CELL_HEIGHT // 2,
                ),
                text=f"{self.num_to_rich(i + 2)}",
                anchor="mm",
                font=self.font,
                fill=(10, 10, 10),
            )

        for suit in range(4):
            suit_image = self.suit_images[suit].resize((CELL_WIDTH, CELL_HEIGHT))
            image.paste(
                suit_image,
                (BOARD_MARGIN, BOARD_MARGIN + CELL_HEIGHT * (suit + 1)),
                mask=suit_image,
            )

        draw.text(
            (
                BOARD_MARGIN + CELL_WIDTH // 2,
                BOARD_MARGIN + CELL_HEIGHT * 5 + CELL_HEIGHT // 2,
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
                        BOARD_MARGIN + CELL_WIDTH * (i % 13 + 1) + CELL_WIDTH // 2,
                        BOARD_MARGIN + CELL_HEIGHT * (i // 13 + 1) + CELL_HEIGHT // 2,
                    ),
                    text="?",
                    anchor="mm",
                    font=self.font,
                    fill=(10, 10, 10),
                )
            elif status == CARD_TRICKED:
                draw.text(
                    (
                        BOARD_MARGIN + CELL_WIDTH * (i % 13 + 1) + CELL_WIDTH // 2,
                        BOARD_MARGIN + CELL_HEIGHT * (i // 13 + 1) + CELL_HEIGHT // 4,
                    ),
                    text="TRICKED",
                    anchor="mm",
                    font=self.small_font,
                    fill=(240, 10, 10),
                )
                draw.text(
                    (
                        BOARD_MARGIN + CELL_WIDTH * (i % 13 + 1) + CELL_WIDTH // 2,
                        BOARD_MARGIN + CELL_HEIGHT * (i // 13 + 1) + CELL_HEIGHT // 2,
                    ),
                    text=f"pl. {int(board.cards[idx, i, 1])}",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 10),
                )
                turn = int(board.cards[idx, i, 2])
                draw.text(
                    (
                        BOARD_MARGIN + CELL_WIDTH * (i % 13 + 1) + CELL_WIDTH // 2,
                        BOARD_MARGIN
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
                        BOARD_MARGIN + CELL_WIDTH * (i % 13 + 1) + CELL_WIDTH // 2,
                        BOARD_MARGIN + CELL_HEIGHT * (i // 13 + 1) + CELL_HEIGHT // 4,
                    ),
                    text="IN HAND",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 240),
                )
                draw.text(
                    (
                        BOARD_MARGIN + CELL_WIDTH * (i % 13 + 1) + CELL_WIDTH // 2,
                        BOARD_MARGIN + CELL_HEIGHT * (i // 13 + 1) + CELL_HEIGHT // 2,
                    ),
                    text=f"pl. {int(board.cards[idx, i, 1])}",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 10),
                )
                draw.text(
                    (
                        BOARD_MARGIN + CELL_WIDTH * (i % 13 + 1) + CELL_WIDTH // 2,
                        BOARD_MARGIN
                        + CELL_HEIGHT * (i // 13 + 1)
                        + CELL_HEIGHT * 3 // 4,
                    ),
                    text=f"No. {int(board.cards[idx, i, 2])}",
                    anchor="mm",
                    font=self.small_font,
                    fill=(10, 10, 10),
                )

        return image

    def close(self):
        for suit_image in self.suit_images:
            suit_image.close()
