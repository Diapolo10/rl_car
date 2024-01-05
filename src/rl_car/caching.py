"""Implements caching for hitboxes"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from PIL import Image

from rl_car.config import CACHE_FILE
from rl_car.track import (
    align_hitbox,
    hitbox_from_image,
)

if TYPE_CHECKING:
    from arcade import PointList

    from rl_car.config import FilePath


def get_hitbox_from_cache(image_file: FilePath) -> PointList:
    """Get hitbox for a given image file from the cache"""

    status = "No saved hitbox for the given imagefile."

    if not CACHE_FILE.exists():
        CACHE_FILE.write_text('{}')

    cache = json.loads(CACHE_FILE.read_text())

    with Image.open(image_file) as image:
        image_hash = hashlib.sha256(image.tobytes()).hexdigest()

    if image_hash not in cache:
        hitbox = align_hitbox(hitbox_from_image(image_file))
        add_hitbox_to_cache(image_file, hitbox, image_hash)
    else:
        hitbox = cache[image_hash]
        status = "Hitbox data fetched from cache."

    print(status)  # TODO: Use logging instead

    return hitbox


def add_hitbox_to_cache(image_file: FilePath, hitbox: PointList, image_hash: str | None = None):
    """Add hitbox for a image file to the cache"""

    status = "Successfully added generated hitbox to cache."

    if image_hash is None:
        with Image.open(image_file) as image:
            image_hash = hashlib.sha256(image.tobytes()).hexdigest()

    cache = json.loads(CACHE_FILE.read_text())
    cache[image_hash] = hitbox

    try:
        CACHE_FILE.write_text(json.dumps(cache))
    except TypeError as err:
        status = f"Failed to add JSON-incompatible type to cache, check hitbox data --\n{err}"

    print(status)  # TODO: Use logging instead
