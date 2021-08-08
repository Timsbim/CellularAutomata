from pathlib import Path
from itertools import product, starmap
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt


def build_rule(n):
    """Build the rule with number no as a mapping (dictionary):
                 triple(0/1, 0/1, 0/1) -> new state of cell
                         ^    ^    ^
        state of:      left  cell right
                     neighbour  neighbour
    """
    combos = product((0, 1), repeat=3)
    rule_str = f"{bin(n)[-1:1:-1]:0<8}"
    return {state: int(bit) for state, bit in zip(combos, rule_str)}


def print_states(states, on='*', off=' '):
    """Print state evolution on console (on for state 1, off for state 0)"""
    print('\n'.join(''.join(on if s else off for s in row) for row in states))


def evolve(rule_n, start_state, iterations, folder=None):
    """Calculate the evolution of a row of cellular automata given in
    start_state under the rule with number rule over iterations many steps
    and saves the result as an image
    """

    # Setup: Matrix for recording the states. First row is the given start
    # state
    states = np.zeros((iterations + 1, len(start_state)), np.uint8)
    states[0] = start_state

    # Build the rule
    rule = build_rule(rule_n)

    # Calculate the evolution (cells on the the edges always stay 0)
    m = states.shape[1] - 1
    for t in range(1, states.shape[0]):
        states[t, 1:m] = [
            rule[tuple(states[t-1, i-1:i+2])] for i in range(1, m)
        ]

    # Save result
    if folder:
        path = Path(folder)
        if not path.exists():
            path.mkdir(exist_ok=True)
        plt.imsave(
            path / f"ca_{rule_n:0>3}.png", states, cmap=plt.get_cmap('Blues')
        )


def evolve_all(*, start_state=None, iterations=100, folder=None):
    # Create folder before multiprocessing to avoid (very unlikely) collision
    if folder is None:
        folder = Path.cwd() / "plots"
    path = Path(folder)
    if not path.exists():
        path.mkdir(exist_ok=True)

    # If no start state is given, use one with all zeroes except for the cell
    # in the middle
    if start_state is None:
        start_state = np.zeros(iterations, dtype=np.uint8)
        start_state[iterations // 2] = 1

    # Use a pool for the calculations
    args = (
        (n, start_state.copy(), iterations, path) for n in range(1, 257)
    )
    with Pool(14) as p:
        p.starmap(evolve, args, chunksize=4)


if __name__ == '__main__':

    evolve_all()

