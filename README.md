# RL-Car

Let's teach an AI to drive - using reinforcement learning.

## Development

First, clone or download the repository to your device. Ensure that it has Python installed between versions 3.6 and 3.8. As of writing, the `keras` dependency doesn't work on Python 3.9 and above. The rest of the codebase expects at least 3.6.

1. If you do not have Poetry installed, run `pip install poetry`
2. Navigate to the repository
3. Run `poetry install` too install all required dependencies and to create a virtual environment for the project
4. Run `poetry shell` to enter the created virtual environment
5. If need be, direct your IDE of choice to use the created virtual environment as the interpreter; on VS Code, this is done by doing:
   1. Press `Ctrl+Shift+P`
   2. Type in "python", choose `Python: Select Interpreter`
   3. If the virtual environment isn't listed, select `Enter interpreter path...` and navigate to the Python executable inside said virtual environment, otherwise click it on the list

A quick primer on the project structure:

- `.github` contains GitHub Actions scripts that, among other things, auto-run linters
- `docs` contains documentation - _if we had any..._
- `rl_car` contains all files relevant to the actual project, top level being code. At present all the code is in `rl_car/game.py`, but this may change as the codebase gets split. Assets are stored in `rl_car/images`.
- `tests` contains unit tests.
- `pyproject.toml` contains all project configuration and metadata, including linter settings, dependencies, and info about the project
- `poetry.lock` stores and locks all dependency versions, delete it if you run into issues and run `poetry update`
- `CHANGELOG.md` logs changes to the project between versions - at present it's updated _by hand_

To run linters and unit tests manually, on Windows you may read the `Makefile` and run the associated commands manually. If you're using Linux or Mac OS, it shhould be as simple as running the appropriate `make` commands.

Stuff to-do:

- Finish implementing the simulation
- Implement NN using TensorFlow or Pytorch
- Train NN
- ???
- Profit

## Troubleshooting

If you run into an `ImportError` message like this one

```text
Traceback (most recent call last):
  File "d:/github/rl_car/rl_car/game.py", line 7, in <module>
    import arcade
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\arcade\__init__.py", line 102, in <module>
    from .camera import Camera
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\arcade\camera.py", line 7, in <module>
    from pyglet.math import Mat4, Vec2, Vec3
ImportError: cannot import name 'Vec2' from 'pyglet.math' (C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\pyglet\math.py)
```

you'll want to run `pip install pyglet==2.0a2`. The problem is caused by current builds of Arcade using an alpha release of `pyglet` and Poetry prioritising version `2.0.dev12` over it. Personally I can't tell which party is at fault here.

If you run into

```text
Traceback (most recent call last):
  File "d:/github/rl_car/rl_car/game.py", line 351, in <module>
    main()
  File "d:/github/rl_car/rl_car/game.py", line 347, in main
    arcade.run()
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\arcade\window_commands.py", line 323, in run
    pyglet.app.run()
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\pyglet\app\__init__.py", line 107, in run
    event_loop.run(interval)
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\pyglet\app\base.py", line 184, in run
    timeout = self.idle()
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\pyglet\app\base.py", line 245, in idle
    self.clock.call_scheduled_functions(dt)
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\pyglet\clock.py", line 277, in call_scheduled_functions
    item.func(now - item.last_ts, *item.args, **item.kwargs)
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\pyglet\app\base.py", line 154, in _redraw_windows
    window.dispatch_event('on_draw')
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\pyglet\window\__init__.py", line 1329, in dispatch_event
    super().dispatch_event(*args)
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\pyglet\event.py", line 422, in dispatch_event
    if getattr(self, event_type)(*args):
  File "d:/github/rl_car/rl_car/game.py", line 221, in on_draw
    dests = nearest_points(Point(orig_x, orig_y), points)
  File "C:\Users\laril\AppData\Local\pypoetry\Cache\virtualenvs\rl-car-usInqqGI-py3.8\lib\site-packages\shapely\ops.py", line 333, in nearest_points
    seq = lgeos.methods['nearest_points'](g1._geom, g2._geom)
OSError: exception: access violation reading 0x0000000000000000
```

then comment out the line `dests = nearest_points(Point(orig_x, orig_y), points)` - it is currently unknown what's causing the problem here.
