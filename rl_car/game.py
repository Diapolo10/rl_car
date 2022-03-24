"""Implement the game"""

import math
import time
from pathlib import Path

import arcade

from .config import (  # type: ignore
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_TITLE,
    MAX_SPEED,
    DRAG,
    MAX_SPEED,
    ACCELERATION_RATE,
    FRICTION,
    SPRITE_SCALING,
)

ROOT_DIR = Path(__file__).parent
IMAGE_DIR = ROOT_DIR / 'images'
CAR_SPRITE = IMAGE_DIR / 'car.png'
TRACK_SPRITE = IMAGE_DIR / 'track.png'

class Player(arcade.Sprite):
    """ Player class """

    def __init__(self, filename, scale, angle=0):
        """Set up the car"""

        # Call the parent Sprite constructor
        super().__init__(filename, scale, angle)

        # Info on where we are going.
        # Angle comes in automatically from the parent class.
        self.thrust = 0
        self.speed = 0
        self.max_speed = MAX_SPEED
        self.drag = DRAG


    def update(self):
        """Update position"""

        if self.speed > 0:
            self.speed -= self.drag
            if self.speed < 0:
                self.speed = 0

        if self.speed < 0:
            self.speed += self.drag
            if self.speed > 0:
                self.speed = 0

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

        """ Call the parent class. """
        super().update()

class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title):
        """
        Initializer
        """

        # Call the parent class initializer
        super().__init__(width, height, title, vsync=True)

        # Variables that will hold sprite lists
        self.player_list = None

        # Set up the player info
        self.player_sprite = None

        # Track the current state of what key is pressed
        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        # Set the background color
        arcade.set_background_color(arcade.color.AMAZON)

    def setup(self):
        """ Set up the game and initialize the variables. """

        # Sprite lists
        self.player_list = arcade.SpriteList()

        # Set up the player
        self.player_sprite = Player(CAR_SPRITE, SPRITE_SCALING)
        self.player_sprite.center_x = 50
        self.player_sprite.center_y = 50
        self.player_list.append(self.player_sprite)

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        self.clear()

        # Draw all the sprites.
        self.player_list.draw()

        # Display speed
        arcade.draw_text(f"X Speed: {self.player_sprite.change_x:6.3f}", 10, 50, arcade.color.BLACK)
        arcade.draw_text(f"Y Speed: {self.player_sprite.change_y:6.3f}", 10, 70, arcade.color.BLACK)

    def on_update(self, delta_time):
        """ Movement and game logic """

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



        # Apply acceleration based on the keys pressed

        # if self.up_pressed and not self.down_pressed:

        #     self.player_sprite.change_y += ACCELERATION_RATE

        # elif self.down_pressed and not self.up_pressed:

        #     self.player_sprite.change_y += -ACCELERATION_RATE

        # if self.left_pressed and not self.right_pressed:

        #     self.player_sprite.change_x += -ACCELERATION_RATE

        # elif self.right_pressed and not self.left_pressed:

        #     self.player_sprite.change_x += ACCELERATION_RATE



        if self.player_sprite.change_x > MAX_SPEED:

            self.player_sprite.change_x = MAX_SPEED

        elif self.player_sprite.change_x < -MAX_SPEED:

            self.player_sprite.change_x = -MAX_SPEED

        if self.player_sprite.change_y > MAX_SPEED:

            self.player_sprite.change_y = MAX_SPEED

        elif self.player_sprite.change_y < -MAX_SPEED:

            self.player_sprite.change_y = -MAX_SPEED



        # Call update to move the sprite

        # If using a physics engine, call update on it instead of the sprite

        # list.

        self.player_list.update()



    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """

        if key == arcade.key.LEFT:
            self.player_sprite.change_angle = 3
        elif key == arcade.key.RIGHT:
            self.player_sprite.change_angle = -3
        elif key == arcade.key.UP:
            self.player_sprite.thrust = 0.15
        elif key == arcade.key.DOWN:
            self.player_sprite.thrust = -.2

        # if key == arcade.key.UP:
        #     self.up_pressed = True
        # elif key == arcade.key.DOWN:
        #     self.down_pressed = True
        # elif key == arcade.key.LEFT:
        #     self.left_pressed = True
        # elif key == arcade.key.RIGHT:
        #     self.right_pressed = True

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key. """

        if key == arcade.key.LEFT:
            self.player_sprite.change_angle = 0
        elif key == arcade.key.RIGHT:
            self.player_sprite.change_angle = 0
        elif key == arcade.key.UP:
            self.player_sprite.thrust = 0
        elif key == arcade.key.DOWN:
            self.player_sprite.thrust = 0

        # if key == arcade.key.UP:
        #     self.up_pressed = False
        # elif key == arcade.key.DOWN:
        #     self.down_pressed = False
        # elif key == arcade.key.LEFT:
        #     self.left_pressed = False
        # elif key == arcade.key.RIGHT:
        #     self.right_pressed = False


def main():
    """ Main function """
    window = MyGame(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
