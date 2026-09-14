# Fixed-Point Bundle Method for Variational Inequalities and Game Equilibria

This repository contains the experiment codes and results for the fixed-point bundle method for solving variational inequalities (VIs) and game equilibrium problems.

## Contents

- `vi_numpy.py`: implementation for the paper titled “A path-following framework on fiber bundle for variational inequalities” (available at [https://arxiv.org/abs/2606.00778] and [https://optimization-online.org/?p=34972]).
- `game_numpy.py`: implementation for the paper titled “The fixed-point bundle method over product-of-simplex domains arising from game equilibria” (available at [https://optimization-online.org/?p=36751]).
- `main.py`: experiment entry points.
- `data/`: experiment results.

## Operator Models

`main.py` implements two operator models:

- a neural network operator `F`;
- a payoff tensor operator `F`.

## Experiment Entry Points

- `test_vi` calls `vi_numpy.py` to solve a VI over a simplex domain with a neural network operator `F`.
- `test_game` calls `game_numpy.py` to solve a VI over a product-of-simplex domain with either a neural network operator `F` or a payoff tensor operator `F`. When the payoff tensor operator `F` is selected, the VI corresponds to a finite normal-form game.

## Usage

Invoke the experiment functions from `main.py`. See the function signatures in `main.py` for the required arguments.

Experiment outputs are stored in the `data/` folder.

## Citation

If you use this code, please cite the corresponding papers:

```bibtex
@misc{sun2026pathfollowingframeworkfiberbundle,
      title={A path-following framework on fiber bundle for variational inequalities}, 
      author={Hongbo Sun},
      year={2026},
      eprint={2606.00778},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2606.00778}, 
}
```
