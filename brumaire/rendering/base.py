from typing import List, Tuple, Dict, Any, Callable
from PIL import Image, ImageDraw, ImageFont
import os

from brumaire.constants import Suit
from brumaire.utils import num_to_str


class ComponentBase:
    def get_size(self) -> Tuple[int, int]:
        raise "Not implemented."

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        raise "Not implemented."

    def render(self) -> Image.Image:
        image = Image.new(
            "RGB",
            size=self.get_size(),
            color=(255, 255, 255),
        )
        self.draw(image, (0, 0))
        return image


class NothingComponent(ComponentBase):
    """
    Renders nothing.
    """

    size: Tuple[int, int] | None

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def get_size(self) -> Tuple[int, int]:
        return self.size

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        pass


class HComponent(ComponentBase):
    """
    Renders components horizontally.
    """

    children: List[ComponentBase]
    size: Tuple[int, int]

    def __init__(self, children: List[ComponentBase]) -> None:
        self.children = children

        width = sum(map(lambda c: c.get_size()[0], self.children))
        height = max(map(lambda c: c.get_size()[1], self.children))
        self.size = width, height

    def get_size(self) -> Tuple[int, int]:
        return self.size

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        x, y = pos
        for r in self.children:
            r.draw(image, (x, y))
            x += r.get_size()[0]


class VComponent(ComponentBase):
    """
    Renders components vertically.
    """

    children: List[ComponentBase]
    size: Tuple[int, int]

    def __init__(self, children: List[ComponentBase]) -> None:
        self.children = children

        width = max(map(lambda c: c.get_size()[0], self.children))
        height = sum(map(lambda c: c.get_size()[1], self.children))
        self.size = width, height

    def get_size(self) -> Tuple[int, int]:
        return self.size

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        x, y = pos
        for r in self.children:
            r.draw(image, (x, y))
            y += r.get_size()[1]


class MarginComponent(ComponentBase):
    """
    Add a margin to the component.
    """

    margin: int
    child: ComponentBase

    def __init__(self, margin: int, child: ComponentBase) -> None:
        self.margin = margin
        self.child = child

    def get_size(self) -> Tuple[int, int]:
        child_w, child_h = self.child.get_size()
        return child_w + self.margin * 2, child_h + self.margin * 2

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        x, y = pos
        self.child.draw(image, (x + self.margin, y + self.margin))


class TextComponent(ComponentBase):
    """
    Draw a text.
    """

    FONTS: Dict[int, ImageFont.FreeTypeFont] = {}
    FONT_FILE: str = "./fonts/NotoSansJP-Regular.ttf"

    size: Tuple[int, int]
    text: str
    font_size: int
    color: Any

    def __init__(
        self, size: Tuple[int, int], text: str, font_size: int, color: Any
    ) -> None:
        self.size = size
        self.text = text
        self.font_size = font_size
        self.color = color

    def get_size(self) -> Tuple[int, int]:
        return self.size

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        x, y = pos
        width, height = self.size
        if self.font_size not in TextComponent.FONTS:
            TextComponent.FONTS[self.font_size] = ImageFont.truetype(
                TextComponent.FONT_FILE, self.font_size
            )
        font = TextComponent.FONTS[self.font_size]
        draw = ImageDraw.Draw(image)
        draw.text(
            (
                x + width // 2,
                y + height // 2,
            ),
            text=self.text,
            anchor="mm",
            font=font,
            fill=self.color,
        )


class SuitComponent(ComponentBase):
    """
    Draw a suit.
    """

    SUIT_IMAGES: List[Image.Image] | None = None
    IMAGE_DIR = "./img"
    JOKER_FONT_SIZE = 20
    JOKER_COLOR = (10, 10, 10)

    size: Tuple[int, int]
    suit_size: int
    suit: int

    def __init__(self, size: Tuple[int, int], suit_size: int, suit: int) -> None:
        if not SuitComponent.SUIT_IMAGES:
            SuitComponent.SUIT_IMAGES = [
                Image.open(os.path.join(SuitComponent.IMAGE_DIR, "club.png")),
                Image.open(os.path.join(SuitComponent.IMAGE_DIR, "diamond.png")),
                Image.open(os.path.join(SuitComponent.IMAGE_DIR, "heart.png")),
                Image.open(os.path.join(SuitComponent.IMAGE_DIR, "spade.png")),
            ]

        self.size = size
        self.suit_size = suit_size
        self.suit = suit

    def get_size(self) -> Tuple[int, int]:
        return self.size

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        assert SuitComponent.SUIT_IMAGES

        x, y = pos
        width, height = self.size
        if self.suit == Suit.JOKER:
            TextComponent(
                (width, height),
                "J",
                SuitComponent.JOKER_FONT_SIZE,
                SuitComponent.JOKER_COLOR,
            ).draw(image, (x, y))
        else:
            margin_left = (width - self.suit_size) // 2
            margin_top = (height - self.suit_size) // 2
            suit_image = SuitComponent.SUIT_IMAGES[self.suit].resize(
                (self.suit_size, self.suit_size)
            )
            image.paste(
                suit_image,
                (x + margin_left, y + margin_top),
                mask=suit_image,
            )


class CardComponent(ComponentBase):
    """
    Draw a card.
    """

    SUIT_RATIO = 0.8
    OUTLINE_COLOR = (10, 10, 10)
    OUTLINE_WIDTH = 1
    FONT_COLOR = (10, 10, 10)
    FONT_SIZE = 20

    size: Tuple[int, int]
    suit: int
    num: int

    def __init__(self, size: Tuple[int, int], suit: int, num: int) -> None:
        self.size = size
        self.suit = suit
        self.num = num

    def get_size(self) -> Tuple[int, int]:
        return self.size

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        x, y = pos
        w, h = self.size

        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (x, y, x + w, y + h),
            outline=CardComponent.OUTLINE_COLOR,
            width=CardComponent.OUTLINE_WIDTH,
        )

        num_text = num_to_str(self.num)
        num_component = (
            TextComponent(
                (w, h // 2), num_text, CardComponent.FONT_SIZE, CardComponent.FONT_COLOR
            )
            if self.suit != Suit.JOKER
            else NothingComponent((w, h // 2))
        )

        VComponent(
            [
                SuitComponent(
                    (w, h // 2), int(h // 2 * CardComponent.SUIT_RATIO), self.suit
                ),
                num_component,
            ]
        ).draw(image, (x, y))


class TableComponent(ComponentBase):
    """
    Draw a table.
    """

    OUTLINE_COLOR = (10, 10, 10)
    OUTLINE_WIDTH = 1

    size: Tuple[int, int]
    cell_x: int
    cell_y: int
    children: List[List[ComponentBase]]

    def __init__(
        self,
        cell_x: int,
        cell_y: int,
        children_fun: Callable[[int, int], ComponentBase],
    ) -> None:
        self.cell_x = cell_x
        self.cell_y = cell_y
        self.children = list()
        for x in range(cell_x):
            x_list = list()
            for y in range(cell_y):
                component = children_fun(x, y)
                x_list.append(component)
            self.children.append(x_list)

        width_sum = 0
        for x in range(cell_x):
            width, _ = self.children[x][0].get_size()
            for y in range(1, cell_y):
                current_width, _ = self.children[x][y].get_size()
                assert width == current_width
            width_sum += width

        height_sum = 0
        for y in range(cell_y):
            _, height = self.children[0][y].get_size()
            for x in range(1, cell_x):
                _, current_height = self.children[x][y].get_size()
                assert height == current_height
            height_sum += height

        self.size = width_sum, height_sum

    def get_size(self) -> Tuple[int, int]:
       return self.size

    def _draw_lines(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        x, y = pos
        w, h = self.size
        draw = ImageDraw.Draw(image)

        width_sum = 0
        for x_idx in range(self.cell_x + 1):
            draw.line(
                (
                    x + width_sum,
                    y,
                    x + width_sum,
                    y + h,
                ),
                fill=TableComponent.OUTLINE_COLOR,
                width=TableComponent.OUTLINE_WIDTH,
            )
            if x_idx < self.cell_x:
                width, _ = self.children[x_idx][0].get_size()
                width_sum += width

        height_sum = 0
        for y_idx in range(self.cell_y + 1):
            draw.line(
                (
                    x,
                    y + height_sum,
                    x + w,
                    y + height_sum,
                ),
                fill=TableComponent.OUTLINE_COLOR,
                width=TableComponent.OUTLINE_WIDTH,
            )
            if y_idx < self.cell_y:
                _, height = self.children[0][y_idx].get_size()
                height_sum += height

    def draw(self, image: Image.Image, pos: Tuple[int, int]) -> None:
        self._draw_lines(image, pos)

        x, y = pos
        width_sum = 0
        for x_idx in range(self.cell_x):
            height_sum = 0
            for y_idx in range(self.cell_y):
                self.children[x_idx][y_idx].draw(image, (x + width_sum, y + height_sum))
                _, height = self.children[x_idx][y_idx].get_size()
                height_sum += height
            width, _ = self.children[x_idx][0].get_size()
            width_sum += width
