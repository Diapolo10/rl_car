"""Implements vehicles"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import arcade

from rl_car.config import (
    DRAG,
    MAX_SPEED,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

if TYPE_CHECKING:
    from rl_car.config import FilePath


class Player(arcade.Sprite):
    """ Player class """

    def __init__(self, filename: FilePath, scale: float, angle=0) -> None:
        """Set up the car"""

        # Call the parent Sprite constructor
        super().__init__(str(filename), scale, angle)

        # Info on where we are going.
        # Angle comes automatically from the parent class.
        self.thrust: float = 0
        self.speed: float = 0
        self.max_speed: float = MAX_SPEED
        self.drag: float = DRAG


    def update(self):  # NOTE: Might need to switch to on_update
        """Update position"""

        if self.speed > 0:
            self.speed = max(self.speed-self.drag, 0)

        if self.speed < 0:
            self.speed = min(self.speed+self.drag, 0)

        self.speed += self.thrust
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        if self.speed < -self.max_speed:
            self.speed = -self.max_speed

        # NOTE: The Arcade API changes swapped angles to clockwise, changes may be needed
        self.change_x = -math.sin(math.radians(self.angle)) * self.speed
        self.change_y = math.cos(math.radians(self.angle)) * self.speed

        self.center_x += self.change_x
        self.center_y += self.change_y

        # Check to see if we hit the screen edge
        if self.left < 0:
            self.left = 0

            self.change_x = 0  # Zero x speed

        elif self.right > WINDOW_WIDTH - 1:
            self.right = WINDOW_WIDTH - 1

            self.change_x = 0

        if self.bottom < 0:
            self.bottom = 0

            self.change_y = 0

        elif self.top > WINDOW_HEIGHT - 1:
            self.top = WINDOW_HEIGHT - 1

            self.change_y = 0

        super().update()
