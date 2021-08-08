import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import product, starmap

from pprint import pprint
from time import perf_counter
from multiprocessing import Pool


def build_rule(no):
    """Build the rule with number no as a mapping (dictionary):
                 triple(0/1, 0/1, 0/1) -> new state of cell
                         ^    ^    ^
        state of:      left  cell right
                     neighbour  neighbour
    """
    combos = product((0, 1), repeat=3)
    rule_str = f"{bin(no)[-1:1:-1]:0<8}"
    return {state: int(bit) for state, bit in zip(combos, rule_str)}


def print_states(states, on='*', off=' '):
    """Print state evolvement on console (on for state 1, off for state 0)"""
    print('\n'.join(''.join(on if s else off for s in row) for row in states))


def evolve(start_state, rule_no, iterations, folder=None):
    """Calculate the evolution of a row of cellular automata given in
    start_state under the rule with number rule_no over iterations many steps
    and saves the result as an image
    """

    # Setup: Matrix for recording the states. First row is the given start
    # state
    states = np.zeros((len(start_state), iterations + 1), np.uint8)
    states[0] = start_state

    # Build the rule
    rule = build_rule(rule_no)

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
            path / f'ca_{rule_no}.png', states, cmap=plt.get_cmap('Blues')
        )

    return None


if __name__ == '__main__':

    size = 100
    iterations = size - 1
    start_state = np.zeros(size, dtype=np.uint8)
    start_state[size // 2] = 1

    start = perf_counter()
    results = [
        *starmap(
            evolve,
            (
                (start_state.copy(), no, iterations, 'tmp1')
                for no in range(1, 257)
            )
        )
    ]
    end = perf_counter()
    normal = end - start

    start = perf_counter()
    with Pool(14) as p:
        p.starmap(
              evolve,
              (
                  (start_state.copy(), no, iterations, 'tmp2')
                  for no in range(1, 257)
              ),
              chunksize=4
          )
    end = perf_counter()
    parallel = end - start

    print('Starmap (normal):', normal)
    print('Starmap (parallel):', parallel)
