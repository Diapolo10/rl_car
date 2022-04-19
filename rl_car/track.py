import arcade
from pathlib import Path
from typing import Union
from PIL import Image  # type: ignore

from config_file import (  # type: ignore
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
)

FilePath = Union[str, Path]


def hitbox_from_image(image_path: FilePath, hit_box_detail: float = 4.5) -> arcade.PointList:
    """Generates a valid hitbox from a given image file"""

    with Image.open(image_path) as image:
        hitbox = list(arcade.calculate_hit_box_points_detailed(image, hit_box_detail=hit_box_detail))

    # Fills in the "doubles" needed by arcade.draw_lines in order to get full boundaries
    # (otherwise it draws every other segment)
    hitbox.extend(hitbox[:1])
    for idx in range(len(hitbox)-3, 0, -1):
        hitbox.insert(idx, hitbox[idx])

    return hitbox


def align_hitbox(hitbox: arcade.PointList, x_orig=WINDOW_WIDTH//2, y_orig=WINDOW_HEIGHT//2) -> arcade.PointList:
    """Aligns a hitbox to a new origo, defaulting to the centre of the window"""

    return [
        (x+x_orig, y+y_orig)
        for x, y, *_ in hitbox
    ]
