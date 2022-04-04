"""Implement the game"""

import math
from pathlib import Path
from typing import Optional, Union

import arcade
from PIL import Image  # type: ignore
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString  # type: ignore
from shapely.ops import nearest_points

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
TRACK_TP_SPRITE = IMAGE_DIR / 'track_tp.png'
TRACK_BARE_SPRITE = IMAGE_DIR / 'track_bare.png'
TRACK_BORDER_SPRITE = IMAGE_DIR / 'track_border.png'
TRACK_BORDER_INNER_SPRITE = IMAGE_DIR / 'track_border_inner.png'
TRACK_BORDER_OUTER_SPRITE = IMAGE_DIR / 'track_border_outer.png'

FilePath = Union[str, Path]


class Player(arcade.Sprite):
    """ Player class """

    def __init__(self, filename: FilePath, scale: float, angle=0):
        """Set up the car"""

        # Call the parent Sprite constructor
        super().__init__(str(filename), scale, angle)

        # Info on where we are going.
        # Angle comes automatically from the parent class.
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
        self.track_border_inner_sprite: Optional[arcade.Sprite] = None
        self.track_border_outer_sprite: Optional[arcade.Sprite] = None
        self.track_inner_hitbox: arcade.PointList = []
        self.track_outer_hitbox: arcade.PointList = []

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
        self.player_sprite.center_x = WINDOW_WIDTH // 7  # 50
        self.player_sprite.center_y = WINDOW_HEIGHT // 2  # 50
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
        self.track_border_inner_sprite = arcade.Sprite(
            TRACK_BORDER_INNER_SPRITE,
            center_x=WINDOW_WIDTH//2,
            center_y=WINDOW_HEIGHT//2
        )
        self.track_border_outer_sprite = arcade.Sprite(
            TRACK_BORDER_OUTER_SPRITE,
            center_x=WINDOW_WIDTH//2,
            center_y=WINDOW_HEIGHT//2
        )
        # TODO: Cache hitboxes using sprite hashes to hasten load times
        self.track_inner_hitbox = align_hitbox(hitbox_from_image(TRACK_BORDER_INNER_SPRITE))
        self.track_outer_hitbox = align_hitbox(hitbox_from_image(TRACK_TP_SPRITE))
        self.track_inner_linestring = LineString(self.track_inner_hitbox)
        self.track_outer_linestring = LineString(self.track_outer_hitbox)

        self.track_list.extend((self.track_sprite, self.track_border_sprite))


    def on_draw(self):
        """Render the screen"""

        # This command has to happen before we start drawing
        self.clear()

        # Draw all the sprites.
        self.track_list.draw()
        self.player_list.draw()
        # self.player_sprite.draw_hit_box()
        # self.track_border_sprite.draw_hit_box()

        # Player origin
        orig_x = self.player_sprite.center_x
        orig_y = self.player_sprite.center_y
        angle = self.player_sprite.angle
        line_length = LASER_SCALED_LENGTH

        laser_lines: arcade.PointList = []
        # line_collision_points: arcade.PointList = []

        for offset in range(0, 360, LASER_ANGLE):
            laser_lines.append((orig_x, orig_y))
            laser_lines.append(
                (
                    orig_x - line_length * math.sin(math.radians(angle+offset)),
                    orig_y + line_length * math.cos(math.radians(angle+offset))
                )
            )

        ## OVER CONSTRUCTION ##

        hitbox = MultiLineString(
            (
                LineString(self.track_inner_hitbox),
                LineString(self.track_outer_hitbox)
            )
        )

        for a, b in zip(laser_lines[::2], laser_lines[1::2]):
            line = LineString((a, b))
            x, y = b[0], b[1]
            colour = arcade.color.WHITE
            if line.intersects(hitbox):
                point = line.intersection(hitbox)
                if isinstance(point, Point):
                    x, y = point.x, point.y
                elif isinstance(point, LineString):
                    (x, y), *_ = point.coords
                elif isinstance(point, MultiPoint):
                    points = point.geoms
                    # dests = nearest_points(Point(orig_x, orig_y), points)
                    x, y = points[0].x, points[0].y
                # x, y = point[0], point[1]
                colour = arcade.color.BLUE
            # elif line.intersects(self.track_outer_linestring):
            #     point = line.intersection(self.track_inner_linestring)
            #     if isinstance(point, Point):
            #         x, y = point.x, point.y
            #     # print(point)
            #     # x, y = point[0], point[1]
            #     colour = arcade.color.BLUE

            arcade.draw_circle_filled(x, y, radius=5, color=colour)

        ## UNDER CONSTRUCTION ##

        arcade.draw_lines(laser_lines, arcade.color.RED)
        arcade.draw_lines(self.track_inner_hitbox, arcade.color.YELLOW)
        arcade.draw_lines(self.track_outer_hitbox, arcade.color.GREEN)

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


def main():
    """Main function"""

    window = MyGame(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
