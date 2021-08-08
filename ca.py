from pathlib import Path
from itertools import product
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def build_rule(n):
    """Build the nth rule with as a mapping (dictionary):
             triple(0/1, 0/1, 0/1) -> new state of cell
                     ^    ^    ^
    state of:      left  cell right
                 neighbour  neighbour
    """
    combos = product((0, 1), repeat=3)
    rule_str = f"{bin(n)[-1:1:-1]:0<8}"
    return {state: int(bit) for state, bit in zip(combos, rule_str)}


def print_states(states, on="*", off=" "):
    """Print state evolution on console (on for state 1, off for state 0)"""
    print("\n".join("".join(on if s else off for s in row) for row in states))


def evolve(rule_n, start, iterations, folder=None):
    """Calculate the evolution of a row of cellular automata given in start
     under the rule with number rule_n over iterations many steps and saves
     the result as an image
    """

    # Setup: Matrix for recording the states. First row is the given start
    # state.
    states = np.zeros((iterations + 1, len(start)), np.uint8)
    states[0] = start

    # Build the rule
    rule = build_rule(rule_n)

    # Calculate the evolution (cells on the the edges always stay 0)
    m = states.shape[1] - 1
    for t in range(1, states.shape[0]):
        states[t, 1:m] = [
            rule[tuple(states[t - 1, i - 1 : i + 2])] for i in range(1, m)
        ]

    # Save result
    if folder:
        path = Path(folder)
        if not path.exists():
            path.mkdir(exist_ok=True)
        plt.imsave(
            path / f"ca_{rule_n:0>3}.png", states, cmap=plt.get_cmap("Blues")
        )

    return states


def evolve_all(
    *, rules=range(1, 257), start=None, iterations=100, folder=None
):
    """Plot the evolution with given start state for all rule numbers given
    (default = all)
    """
    if __name__ == "__main__":  # Only for starting the Pool-processes

        # Create folder if it doesn't exist
        if folder is None:
            folder = Path.cwd() / "plots"
        path = Path(folder)
        if not path.exists():
            path.mkdir(exist_ok=True)

        # If no start state is given, use one with all zeroes except for the
        # cell in the middle
        if start is None:
            start = np.zeros(iterations, dtype=np.uint8)
            m, r = divmod(iterations, 2)
            start[m + r] = 1

        # Use multiprocessing Pool to speed up the calculations (which are
        # independent)
        args = ((n, start.copy(), iterations, path) for n in rules)
        with Pool(12) as p:
            p.starmap(evolve, args, chunksize=4)


def evolve_animated(
        rule_n, start=None, iterations=100, save=False, folder=None
):

    # If no start state is given, use one with all zeroes except for the cell
    # in the middle
    if start is None:
        start = np.zeros(iterations, dtype=np.uint8)
        m, r = divmod(iterations, 2)
        start[m + r] = 1

    # Create folder if it doesn't exist
    if save:
        if folder is None:
            folder = Path.cwd() / "plots"
        path = Path(folder)
        if not path.exists():
            path.mkdir(exist_ok=True)

    states = evolve(rule_n, start, iterations * 5, folder=None)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.axis("off")

    def run(i):
        ax.clear()
        ax.axis("off")
        if i < iterations:
            frame = np.zeros((iterations, len(start)), dtype=np.int8)
            frame[0 : i + 1, :] = states[0 : i + 1, :]
        else:
            frame = states[i - iterations + 1 : i + 1, :]
        img = ax.imshow(frame)

        return img,

    ani = FuncAnimation(
        fig, run, frames=iterations * 4, interval=1, repeat=True, blit=True
    )
    if save:
        file_path = folder / f"ca_{rule_n:3>0}.gif"
        ani.save(file_path)
    plt.show()


evolve_animated(105, save=True)