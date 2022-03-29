"""Implement the game"""

import math
from pathlib import Path
from typing import Optional, Union

import arcade

from config_file import (  # type: ignore
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    DRAG,
    MAX_SPEED,
    FRICTION,
    SPRITE_SCALING,
    LASER_SCALED_LENGTH,
    LASER_ANGLE,
)

ROOT_DIR = Path(__file__).parent
IMAGE_DIR = ROOT_DIR / 'images'
CAR_SPRITE = IMAGE_DIR / 'car.png'
# TRACK_SPRITE = IMAGE_DIR / 'track.png'
# TRACK_TP_SPRITE = IMAGE_DIR / 'track_tp.png'
TRACK_BARE_SPRITE = IMAGE_DIR / 'track_bare.png'
TRACK_BORDER_SPRITE = IMAGE_DIR / 'track_border.png'

FilePath = Union[str, Path]


class Player(arcade.Sprite):
    """ Player class """

    def __init__(self, filename: FilePath, scale: float, angle=0):
        """Set up the car"""

        # Call the parent Sprite constructor
        super().__init__(str(filename), scale, angle)

        # Info on where we are going.
        # Angle comes in automatically from the parent class.
        self.thrust: float = 0
        self.speed: float = 0
        self.max_speed: float = MAX_SPEED
        self.drag: float = DRAG


    def update(self):
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


class MyGame(arcade.Window):
    """Main application class"""

    def __init__(self, width: int, height: int, title: str):
        """Initialiser"""

        super().__init__(width, height, title, vsync=True)

        # Variables that will hold sprite lists
        self.player_list: Optional[arcade.SpriteList] = None
        self.track_list: Optional[arcade.SpriteList] = None

        # Set up the player info
        self.player_sprite: Optional[Player] = None

        # Set up track info
        self.track_sprite: Optional[arcade.Sprite] = None
        self.track_border_sprite: Optional[arcade.Sprite] = None

        # Track the current state of what key is pressed
        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        # Set the background color
        arcade.set_background_color(arcade.color.AMAZON)


    def setup(self):
        """Set up the game and initialize the variables"""

        # Sprite lists
        self.player_list = arcade.SpriteList()
        self.track_list = arcade.SpriteList(is_static=True)

        # Set up the player
        self.player_sprite = Player(CAR_SPRITE, SPRITE_SCALING)
        self.player_sprite.center_x = WINDOW_WIDTH // 10  # 50
        self.player_sprite.center_y = WINDOW_HEIGHT // 10  # 50
        self.player_list.append(self.player_sprite)

        # Set up the track
        self.track_sprite = arcade.Sprite(
            TRACK_BARE_SPRITE,
            center_x=WINDOW_WIDTH//2,
            center_y=WINDOW_HEIGHT//2
        )
        self.track_border_sprite = arcade.Sprite(
            TRACK_BORDER_SPRITE,
            center_x=WINDOW_WIDTH//2,
            center_y=WINDOW_HEIGHT//2
        )
        self.track_list.extend((self.track_sprite, self.track_border_sprite))


    def on_draw(self):
        """Render the screen"""

        # This command has to happen before we start drawing
        self.clear()

        # Draw all the sprites.
        self.track_list.draw()
        self.player_list.draw()

        # Player origin
        orig_x = self.player_sprite.center_x
        orig_y = self.player_sprite.center_y
        angle = self.player_sprite.angle
        line_length = LASER_SCALED_LENGTH

        laser_lines: arcade.PointList = []

        for offset in range(0, 360, LASER_ANGLE):
            laser_lines.append((orig_x, orig_y))
            laser_lines.append(
                (
                    orig_x - line_length * math.sin(math.radians(angle+offset)),
                    orig_y + line_length * math.cos(math.radians(angle+offset))
                )
            )

        arcade.draw_lines(laser_lines, arcade.color.RED)

        # Display speed
        arcade.draw_text(f"X Speed: {self.player_sprite.change_x:6.3f}", 10, 90, arcade.color.BLACK)
        arcade.draw_text(f"Y Speed: {self.player_sprite.change_y:6.3f}", 10, 70, arcade.color.BLACK)
        arcade.draw_text(f"Angle: {self.player_sprite.angle:6.3f}", 10, 50, arcade.color.BLACK)


    def on_update(self, delta_time: float):
        """Movement and game logic"""

        if self.player_sprite is None:
            return

        # Add some friction
        if self.player_sprite.change_x > FRICTION:
            self.player_sprite.change_x -= FRICTION
        elif self.player_sprite.change_x < -FRICTION:
            self.player_sprite.change_x += FRICTION
        else:
            self.player_sprite.change_x = 0

        if self.player_sprite.change_y > FRICTION:
            self.player_sprite.change_y -= FRICTION
        elif self.player_sprite.change_y < -FRICTION:
            self.player_sprite.change_y += FRICTION
        else:
            self.player_sprite.change_y = 0

        if self.player_sprite.change_x > MAX_SPEED:
            self.player_sprite.change_x = MAX_SPEED
        elif self.player_sprite.change_x < -MAX_SPEED:
            self.player_sprite.change_x = -MAX_SPEED
        if self.player_sprite.change_y > MAX_SPEED:
            self.player_sprite.change_y = MAX_SPEED
        elif self.player_sprite.change_y < -MAX_SPEED:
            self.player_sprite.change_y = -MAX_SPEED

        if self.player_list is not None:
            self.player_list.update()


    def on_key_press(self, symbol: int, modifiers: int):
        """Called whenever a key is pressed"""

        if self.player_sprite is None:
            return

        if symbol == arcade.key.LEFT:
            self.player_sprite.change_angle = 5
            self.left_pressed = True
        if symbol == arcade.key.RIGHT:
            self.player_sprite.change_angle = -5
            self.right_pressed = True
        if symbol == arcade.key.UP:
            self.player_sprite.thrust = 0.15
            self.up_pressed = True
        elif symbol == arcade.key.DOWN:
            self.player_sprite.thrust = -.2
            self.down_pressed = True


    def on_key_release(self, symbol: int, modifiers: int):
        """Called when the user releases a key"""

        if self.player_sprite is None:
            return

        if symbol == arcade.key.LEFT:
            self.player_sprite.change_angle = 0
        if symbol == arcade.key.RIGHT:
            self.player_sprite.change_angle = 0
        if symbol == arcade.key.UP:
            self.player_sprite.thrust = 0
        elif symbol == arcade.key.DOWN:
            self.player_sprite.thrust = 0


def main():
    """Main function"""

    window = MyGame(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
