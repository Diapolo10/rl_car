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
