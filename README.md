# Fixed-Point Bundle Method for Variational Inequalities and Game Equilibria

This repository contains the experiment code and results for the fixed-point bundle method for solving variational inequalities (VIs) and game equilibrium problems.

## Contents

- `vi_numpy.py`: implementation for the paper titled “A path-following framework on fiber bundle for variational inequalities” (available at [arXiv:2606.00778](https://arxiv.org/abs/2606.00778) and [Optimization Online](https://optimization-online.org/?p=34972)).
- `game_numpy.py`: implementation for the paper titled “The fixed-point bundle method over product-of-simplex domains arising from game equilibria” (available at [arXiv:2609.16158](https://arxiv.org/abs/2609.16158) and [Optimization Online](https://optimization-online.org/?p=36751)).
- `main.py`: experiment entry points.
- `data/`: experiment results.

## Operator Models

`main.py` implements two operator models:

- the neural network representation `nn_F` models general real-analytic VI operators;
- a payoff tensor representation `payoff_F` models the VI operator for general finite normal-form games.

## Experiment Entry Points

- `test_vi` calls `vi_numpy.py` to solve a VI over a simplex domain with a neural network operator.
- `test_game` calls `game_numpy.py` to solve a VI over a product-of-simplex domain with either a neural network operator or a payoff tensor operator.
- `group_test` runs `test_vi` or `test_game` in parallel using `multiprocessing`, where the number of worker processes is specified by `process_num`. It is intended for solving multiple VIs or games simultaneously.

## Algorithm Options

The solvers are based on a predictor-corrector framework. `vi_numpy.py` and `game_numpy.py` support two corrector options, which are selectable inside the `corrector_comp` functions. See the papers for a detailed discussion of the differences between the two correctors.

## Caveats

The current implementation uses a naive navigation of the fixed-point bundle, with the sole goal of finding at least one solution for each input VI instance.

- The current singularity-avoidance mechanism uses the simplest stateless or randomized strategy to produce a step along the fiber. For complex problem instances, adaptively determining the singularity-avoidance step based on the algorithm state could further improve performance.
- Some solutions cannot be reached by the current navigation. If convergence to a specific solution is required, the `decrease_only` option in `predictor_corrector` can help reach any solution; however, there is no convergence guarantee if `decrease_only=True` at every step.

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

@misc{sun2026fixedpointbundlemethodproductofsimplex,
      title={The fixed-point bundle method over product-of-simplex domains arising from game equilibria}, 
      author={Hongbo Sun},
      year={2026},
      eprint={2609.16158},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2609.16158}, 
}
```
