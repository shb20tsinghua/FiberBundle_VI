# Fixed-Point Bundle Method for Variational Inequalities and Game Equilibria

This repository contains the experiment code and results for the fixed-point bundle method for solving variational inequalities (VIs) and game equilibrium problems. The algorithm follows the predictor-corrector path-following framework. It takes VIs in the form of ${\rm VI}(F,\Delta_n)$ or ${\rm VI}(F,\Delta_n^m)$ and guarantees to find a solution for each input VI instance.

- `vi_numpy.py` (for ${\rm VI}(F,\Delta_n)$): implementation for the paper titled “A path-following framework on fiber bundle for variational inequalities” (available at [arXiv:2606.00778](https://arxiv.org/abs/2606.00778) and [Optimization Online](https://optimization-online.org/?p=34972)).
- `game_numpy.py` (for ${\rm VI}(F,\Delta_n^m)$): implementation for the paper titled “The fixed-point bundle method over product-of-simplex domains arising from game equilibria” (available at [arXiv:2609.16158](https://arxiv.org/abs/2609.16158) and [Optimization Online](https://optimization-online.org/?p=36751)).

## Usage

### Constructing a VI problem

The algorithm takes either ${\rm VI}(F,\Delta_n)$ or ${\rm VI}(F,\Delta_n^m)$ as an input VI. `main.py` implements two operator models for $F$:

- **Neural network model `nn_F`:**
  - It represents general real-analytic VI operators, which can have domain either $\Delta_n$ or $\Delta_n^m$.
  - The neural network has structure `[dim,*hidden_layers,dim]` with `hidden_layers` in specified shape and `dim=n` or `dim=m*n`.
  - Three activations are available: `tanh`, `sigmoid`, `linear`.
- **Payoff tensor model `payoff_F`:**
  - It represents the VI operator of general finite normal-form games, which has domain $\Delta_n^m$.
  - The operator is the gradient of a low-rank CP (canonical polyadic) representation of multilinear functions, such that $F_i(\pi)=\nabla_{\pi_i}f_i(\pi)$, where $f_i(\pi)=\sum_{r=1}^R\prod_{j=1}^m(U_{r,i,j}\pi_j)$ with `R` specified.

Each model contains a function `value_Jaco` evaluating the function value and the Jacobian matrix of $F$ at a given point.

### Calling the solver

`vi_numpy.py` holds the solver `VI_simplex` that takes an operator $F:\Delta_n\to\mathbb{R}^n$ as input, and `game_numpy.py` holds the solver `VI_simplex_product` that takes an operator on $F:\Delta_n^m\to\mathbb{R}^{m\times n}$ as input. Both solvers have a function `solve` to solve the input VI, which takes the following input parameters:

- `iter_init`: iteration starting point, where the path-following starts.
- `gap_tol`: required precision. ${\rm VI}(F,\Delta_n)$ is solved to ${\rm gap}(\sigma)=\sigma^\top F(\sigma)-\min_a F_a(\sigma)\leq{\rm gap_tol}$, and ${\rm VI}(F,\Delta_n^m)$ is solved to ${\rm gap}_i(\pi)=\pi_i^\top F_i(\pi)-\min_a F_{i,a}(\pi)\leq{\rm gap_tol}$.
- `efficient_corrector=True`: the solvers provide two correctors to choose from, one derived from the KKT conditions and one derived from the barrier problem, where the one derived from the barrier problem is more efficient.
- `maxnit=50000`: maximum iteration number for the total algorithm.
- `maxsubnit=10`: maximum iteration number for corrector subiteration.
- `verbose=0`: option for printing the iteration process. `0` for not printing, `1` for printing only after corrector complete, `2` for printing every corrector subiteration step.
- `record_file=None`: file that the iteration process is printed into. `None` for not printing.
- `print_preci=6`: precision of the prints.

### Solver outputs

The `solve` function in both solvers returns the following results:

- solution $\sigma\in\Delta_n$ or $\pi\in\Delta_n^m$ of the input ${\rm VI}(F,\Delta_n)$ or ${\rm VI}(F,\Delta_n^m)$, or `None` if the solver fails to converge
- iteration count
- gap function value ${\rm gap}(\sigma)$ or $\max_i{\rm gap}_i(\pi)$
- barrier parameter sum $\mathbf{1}^\top\mu$ or $\max_i\mathbf{1}^\top\tau_i$
- fixed-point bundle equation norm $\lVert G(\sigma,\mu)\rVert$ or $\lVert G(\pi,\tau)\rVert$
- singularity avoidance times

## Experiments

The experiments reported in the papers differ from those described below. The next version of the papers will adopt the settings and results presented here, because the current experiments have two issues:

- The uniform parameter initialization in the neural network causes the tanh activation to saturate, which reduces the complexity of the operator $F$. We have therefore switched to Xavier normal initialization $N(0,2/({\rm inputwidth}+{\rm outputwidth}))$.
- The neural network structure `[n,50,n]` and the low-rank CP representation `R=50` suffer from rank deficiency. We have therefore changed them to `[n,n,n]` and `R=m*n`.

### Solving ${\rm VI}(F,\Delta_n)$ with `vi_numpy.py`

**Setting:**

- **Model of $F$:** $F$ is a tanh-activated neural network with architecture `[n,n,n]` or `[256]*depth`. Its parameters are randomly generated using Xavier normal initialization $N(0,2/({\rm inputwidth}+{\rm outputwidth}))$.
- **Starting point:** randomly generated.
- **Required precision:** `gap_tol=1e-5`.
- **Correctors:** two correctors are used, one derived from the KKT conditions and one derived from the barrier problem.

**Results:** The following table reports the mean and median of the iteration count and the singularity avoidance times when solving the VIs. Each data point is computed from 200 instances, totaling 6400 instances. The algorithm converges to a solution in every instance.

<table>
  <caption>Table 1: Statistics (mean / median) solving VIs on simplex</caption>
  <thead>
    <tr>
      <th rowspan="2" valign="bottom">NN structure</th>
      <th colspan="2"><div align="center">KKT correcter</div></th>
      <th colspan="2"><div align="center">Barrier correcter</div></th>
    </tr>
    <tr>
      <th>Iteration count</th><th>Singularity avoidance times</th><th>Iteration count</th><th>Singularity avoidance times</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>[2]*3</td><td>280 / 291</td><td>0.1 / 0.0</td><td>294 / 309</td><td>0.1 / 0.0</td></tr>
    <tr><td>[4]*3</td><td>353 / 327</td><td>1.2 / 0.0</td><td>360 / 339</td><td>1.3 / 0.0</td></tr>
    <tr><td>[8]*3</td><td>386 / 334</td><td>2.8 / 0.0</td><td>377 / 338</td><td>2.0 / 0.0</td></tr>
    <tr><td>[16]*3</td><td>436 / 341</td><td>4.5 / 0.0</td><td>410 / 332</td><td>4.1 / 0.0</td></tr>
    <tr><td>[32]*3</td><td>554 / 359</td><td>10.7 / 0.0</td><td>709 / 335</td><td>10.8 / 0.0</td></tr>
    <tr><td>[64]*3</td><td>804 / 521</td><td>24.8 / 8.0</td><td>723 / 501</td><td>24.8 / 8.5</td></tr>
    <tr><td>[128]*3</td><td>1194 / 726</td><td>46.2 / 20.0</td><td>1009 / 656</td><td>43.6 / 20.5</td></tr>
    <tr><td>[256]*3</td><td>1062 / 648</td><td>40.4 / 18.5</td><td>954 / 598</td><td>41.1 / 17.0</td></tr>
    <tr><td>[512]*3</td><td>1062 / 634</td><td>40.3 / 17.5</td><td>927 / 579</td><td>39.8 / 17.5</td></tr>
    <tr><td>[1024]*3</td><td>948 / 553</td><td>35.0 / 10.0</td><td>892 / 452</td><td>38.3 / 8.5</td></tr>
    <tr><td>[256]*2</td><td>1650 / 981</td><td>72.9 / 34.5</td><td>1258 / 837</td><td>60.9 / 33.5</td></tr>
    <tr><td>[256]*3</td><td>1062 / 648</td><td>40.4 / 18.5</td><td>954 / 598</td><td>41.1 / 17.0</td></tr>
    <tr><td>[256]*4</td><td>902 / 571</td><td>31.1 / 12.0</td><td>870 / 512</td><td>35.5 / 11.5</td></tr>
    <tr><td>[256]*5</td><td>823 / 543</td><td>26.7 / 9.0</td><td>717 / 474</td><td>25.4 / 8.5</td></tr>
    <tr><td>[256]*6</td><td>685 / 442</td><td>19.1 / 5.0</td><td>605 / 406</td><td>18.2 / 4.5</td></tr>
    <tr><td>[256]*7</td><td>757 / 424</td><td>23.8 / 4.5</td><td>621 / 396</td><td>19.7 / 3.5</td></tr>
    <tr><td>[256]*8</td><td>717 / 422</td><td>21.0 / 3.5</td><td>584 / 376</td><td>16.8 / 3.0</td></tr>
  </tbody>
</table>

\* For tanh-activated neural network with Xavier normal initialization, function complexity actually decreases as depth increases.

### Solving ${\rm VI}(F,\Delta_n^m)$ with `game_numpy.py`

**Setting:**

- **Models of $F$:**
  - *Neural network model:* $F$ is a tanh-activated neural network with architecture `[m*n,m*n,m*n]`. Its parameters are randomly generated using Xavier normal initialization $N(0,2/({\rm inputwidth}+{\rm outputwidth}))$.
  - *Payoff tensor model:* $F$ is the gradient of a low-rank CP payoff function $f_i(\pi)=\sum_{r=1}^R\prod_{j=1}^m(U_{r,i,j}\pi_j)$ with $R=mn$. Its parameters $U_{r,i,j}$ are randomly generated from a normal distribution $N(1,1/m)$.
- **Starting point:** randomly generated.
- **Required precision:** `gap_tol=1e-5`.
- **Corrector:** only the corrector derived from the barrier problem (the efficient corrector) is used.

**Results:** The following table reports the mean and median of the iteration count and the number of singularity-avoidance steps when solving the VIs. Each data point is computed from 200 instances, totaling 5600 instances. The algorithm converges to a solution in every instance.

<table>
  <caption>Table 2: Statistics (mean / median) solving VIs on product-of-simplex</caption>
  <thead>
    <tr>
      <th rowspan="2" valign="bottom">m,n</th>
      <th colspan="2"><div align="center">Neural network model</div></th>
      <th colspan="2"><div align="center">Payoff tensor model</div></th>
    </tr>
    <tr>
      <th>Iteration count</th><th>Singularity avoidance times</th><th>Iteration count</th><th>Singularity avoidance times</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>8,2</td><td>369 / 350</td><td>0.8 / 0.0</td><td>360 / 353</td><td>0.2 / 0.0</td></tr>
    <tr><td>8,4</td><td>446 / 387</td><td>3.3 / 0.0</td><td>439 / 394</td><td>2.0 / 0.0</td></tr>
    <tr><td>8,8</td><td>618 / 517</td><td>10.2 / 5.0</td><td>548 / 442</td><td>5.3 / 0.0</td></tr>
    <tr><td>8,16</td><td>907 / 789</td><td>23.4 / 18.5</td><td>632 / 565</td><td>7.9 / 5.0</td></tr>
    <tr><td>8,32</td><td>2067 / 1344</td><td>75.0 / 44.0</td><td>823 / 702</td><td>14.7 / 10.0</td></tr>
    <tr><td>2,8</td><td>394 / 356</td><td>2.0 / 0.0</td><td>409 / 403</td><td>0.3 / 0.0</td></tr>
    <tr><td>4,8</td><td>435 / 375</td><td>3.4 / 0.0</td><td>454 / 412</td><td>2.0 / 0.0</td></tr>
    <tr><td>8,8</td><td>618 / 517</td><td>10.2 / 5.0</td><td>548 / 442</td><td>5.3 / 0.0</td></tr>
    <tr><td>16,8</td><td>1053 / 836</td><td>27.2 / 18.0</td><td>710 / 640</td><td>11.3 / 9.0</td></tr>
    <tr><td>32,8</td><td>2561 / 1823</td><td>85.4 / 58.5</td><td>1287 / 1133</td><td>31.0 / 25.0</td></tr>
    <tr><td>128,2</td><td>1142 / 955</td><td>30.7 / 23.0</td><td>942 / 754</td><td>20.1 / 12.5</td></tr>
    <tr><td>64,4</td><td>2616 / 1790</td><td>85.8 / 53.5</td><td>1544 / 1265</td><td>40.9 / 31.0</td></tr>
    <tr><td>32,8</td><td>2561 / 1823</td><td>85.4 / 58.5</td><td>1287 / 1133</td><td>31.0 / 25.0</td></tr>
    <tr><td>16,16</td><td>2128 / 1518</td><td>70.8 / 44.5</td><td>1054 / 929</td><td>23.3 / 19.0</td></tr>
    <tr><td>8,32</td><td>2067 / 1344</td><td>75.0 / 44.0</td><td>823 / 702</td><td>14.7 / 10.0</td></tr>
    <tr><td>4,64</td><td>1616 / 1084</td><td>62.1 / 37.5</td><td>619 / 533</td><td>6.0 / 1.0</td></tr>
    <tr><td>2,128</td><td>1203 / 825</td><td>48.6 / 30.0</td><td>532 / 477</td><td>2.5 / 0.0</td></tr>
  </tbody>
</table>

## Citation

If you use this code, please cite the corresponding papers:

```bibtex
@misc{paper1,
      title={A path-following framework on fiber bundle for variational inequalities}, 
      author={Hongbo Sun},
      year={2026},
      eprint={2606.00778},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2606.00778}, 
}

@misc{paper2,
      title={The fixed-point bundle method over product-of-simplex domains arising from game equilibria}, 
      author={Hongbo Sun},
      year={2026},
      eprint={2609.16158},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2609.16158}, 
}
```
