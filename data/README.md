# Experiment data

This folder contains the per-instance results behind the numerical experiments of the
repository. There are three experiment families, one per path-following variant:

| Directory                 | Iteration                                       | Log files       | Tables   |
| ------------------------- | ----------------------------------------------- | --------------- | -------- |
| `singlelevel_derivfree/`  | single-level, derivative-free (Broyden wrapper) | 32 VI + 28 game | 3.1, 3.2 |
| `singlelevel_derivbased/` | single-level, derivative-based                  | 32 VI + 28 game | 2.1, 2.2 |
| `twolevel_derivbased/`    | two-level, derivative-based                     | 32 VI + 28 game | 1.1, 1.2 |

All three families use the same settings: `gap_tol = 1e-5`, 200 instances per configuration
(seeds `0 ... 199`), and the algorithm converged on every instance (`success = True` in every row).
Tables 3.1 and 3.2 (single-level derivative-free) are the ones summarised as Tables 1 and 2 of the
repository [README](../README.md).

## Contents and sizes

Every family has the same two subfolders:

- `vi/` — $\mathrm{VI}(F,\Delta_n)$: 16 configurations × 2 correctors = 32 log files;
- `game/` — $\mathrm{VI}(F,\Delta_n^m)$: 14 `(m, n)` configurations × 2 operator models = 28 log files.

Every `.log` file has a header line followed by exactly 200 data rows, one per seed `0 ... 199`.
The rows are written in completion order, so the seeds are not necessarily sorted.
Each family therefore contains 32 + 28 = 60 log files and 60 × 200 = 12,000 data rows; the three
families together contain 36,000 rows, all with `success = True`.

## File naming

| Folder  | Pattern                                | Meaning                                                                                                |
| ------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `vi/`   | `kkt_<n>.log` / `barr_<n>.log`         | architecture `[n, n, n]`, KKT / barrier corrector, `n` ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}    |
| `vi/`   | `kkt_256_<i>.log` / `barr_256_<i>.log` | architecture `[256]*d` with depth `d = i + 2`; `i` ∈ {0, 2, 3, 4, 5, 6} gives `d` ∈ {2, 4, 5, 6, 7, 8} |
| `game/` | `nn_<m>_<n>.log`                       | neural-network operator, architecture `[m*n, m*n, m*n]`                                                |
| `game/` | `payoff_<m>_<n>.log`                   | payoff-tensor operator, CP rank `R = m*n`                                                              |

In `vi/`, only the neural-network operator is tested: `kkt` selects the KKT corrector
(`efficient_corrector=False`) and `barr` the barrier corrector (`efficient_corrector=True`).
In `game/`, only the barrier corrector is used: `nn` selects the neural-network operator and
`payoff` the payoff-tensor operator.

## Log format

Every per-configuration `.log` file has a header line followed by one row per instance. The header and
the row format are:

```text
|seed |success|  nit  |    gap     |barrier_sum |   G_norm   |avoid_n|
|  0  |True   |  165  |9.885889e-06|2.536627e-05|1.694000e-16|   0   |
```

| Column        | Meaning                                                                            |
| ------------- | ---------------------------------------------------------------------------------- |
| `seed`        | random seed of the instance (`0 ... 199`)                                          |
| `success`     | `True` if the solver reached the requested gap tolerance                           |
| `nit`         | total iteration count                                                              |
| `gap`         | final gap value: $\mathrm{gap}(\sigma)$ or $\max_i\mathrm{gap}_i(\pi)$             |
| `barrier_sum` | final barrier-parameter sum: $\mathbf{1}^\top\mu$ or $\max_i\mathbf{1}^\top\tau_i$ |
| `G_norm`      | norm of the fixed-point bundle equation residual                                   |
| `avoid_n`     | number of singularity-avoidance steps                                              |

## Tables

The tables below report `int(mean(nit))`, `int(median(nit))`, and the mean and median of `avoid_n` formatted to one decimal place.
Compared with the classical two-level iteration, the single-level iteration needs several times fewer iterations, and singularity avoidance is nearly absent. The derivative-free mode needs nearly the same number of iterations as the derivative-based mode, while requiring no Jacobian; this makes the single-level derivative-free implementation the most practical choice.

**Table 1.1.** Statistics (mean / median) solving $\mathrm{VI}(F,\Delta_n)$ with two-level derivative-based path-following iteration.

| Architecture | `n` or `d` | KKT: iterations | KKT: avoidance | Barrier: iterations | Barrier: avoidance |
| ------------ | ---------- | --------------- | -------------- | ------------------- | ------------------ |
| `[n, n, n]`  | 2          | 280 / 291       | 0.1 / 0.0      | 294 / 309           | 0.1 / 0.0          |
| `[n, n, n]`  | 4          | 353 / 327       | 1.2 / 0.0      | 360 / 339           | 1.3 / 0.0          |
| `[n, n, n]`  | 8          | 386 / 334       | 2.8 / 0.0      | 377 / 338           | 2.0 / 0.0          |
| `[n, n, n]`  | 16         | 436 / 341       | 4.5 / 0.0      | 410 / 332           | 4.1 / 0.0          |
| `[n, n, n]`  | 32         | 554 / 359       | 10.7 / 0.0     | 511 / 335           | 10.8 / 0.0         |
| `[n, n, n]`  | 64         | 804 / 521       | 24.8 / 8.0     | 723 / 501           | 24.8 / 8.5         |
| `[n, n, n]`  | 128        | 1194 / 726      | 46.2 / 20.0    | 1009 / 656          | 43.6 / 20.5        |
| `[n, n, n]`  | 256        | 1062 / 648      | 40.4 / 18.5    | 954 / 598           | 41.1 / 17.0        |
| `[n, n, n]`  | 512        | 1062 / 634      | 40.3 / 17.5    | 927 / 579           | 39.8 / 17.5        |
| `[n, n, n]`  | 1024       | 948 / 553       | 35.0 / 10.0    | 892 / 452           | 38.3 / 8.5         |
| `[256]*d`    | 2          | 1650 / 981      | 72.9 / 34.5    | 1258 / 837          | 60.9 / 33.5        |
| `[256]*d`    | 3          | 1062 / 648      | 40.4 / 18.5    | 954 / 598           | 41.1 / 17.0        |
| `[256]*d`    | 4          | 902 / 571       | 31.1 / 12.0    | 870 / 512           | 35.5 / 11.5        |
| `[256]*d`    | 5          | 823 / 543       | 26.7 / 9.0     | 717 / 474           | 25.4 / 8.5         |
| `[256]*d`    | 6          | 685 / 442       | 19.1 / 5.0     | 605 / 406           | 18.2 / 4.5         |
| `[256]*d`    | 7          | 757 / 424       | 23.8 / 4.5     | 621 / 396           | 19.7 / 3.5         |
| `[256]*d`    | 8          | 717 / 422       | 21.0 / 3.5     | 584 / 376           | 16.8 / 3.0         |

**Table 1.2.** Statistics (mean / median) solving $\mathrm{VI}(F,\Delta_n^m)$ with two-level derivative-based path-following iteration.

| `m`, `n` | NN: iterations | NN: avoidance | Payoff: iterations | Payoff: avoidance |
| -------- | -------------- | ------------- | ------------------ | ----------------- |
| 8, 2     | 369 / 350      | 0.8 / 0.0     | 360 / 353          | 0.2 / 0.0         |
| 8, 4     | 446 / 387      | 3.3 / 0.0     | 439 / 394          | 2.0 / 0.0         |
| 8, 8     | 618 / 517      | 10.2 / 5.0    | 548 / 442          | 5.3 / 0.0         |
| 8, 16    | 907 / 789      | 23.4 / 18.5   | 632 / 565          | 7.9 / 5.0         |
| 8, 32    | 2067 / 1344    | 75.0 / 44.0   | 823 / 702          | 14.7 / 10.0       |
| 2, 8     | 394 / 356      | 2.0 / 0.0     | 409 / 403          | 0.3 / 0.0         |
| 4, 8     | 435 / 375      | 3.4 / 0.0     | 454 / 412          | 2.0 / 0.0         |
| 8, 8     | 618 / 517      | 10.2 / 5.0    | 548 / 442          | 5.3 / 0.0         |
| 16, 8    | 1053 / 836     | 27.2 / 18.0   | 710 / 640          | 11.3 / 9.0        |
| 32, 8    | 2561 / 1823    | 85.4 / 58.5   | 1287 / 1133        | 31.0 / 25.0       |
| 2, 128   | 1203 / 825     | 48.6 / 30.0   | 532 / 477          | 2.5 / 0.0         |
| 4, 64    | 1616 / 1084    | 62.1 / 37.5   | 619 / 533          | 6.0 / 1.0         |
| 8, 32    | 2067 / 1344    | 75.0 / 44.0   | 823 / 702          | 14.7 / 10.0       |
| 16, 16   | 2128 / 1518    | 70.8 / 44.5   | 1054 / 929         | 23.3 / 19.0       |
| 32, 8    | 2561 / 1823    | 85.4 / 58.5   | 1287 / 1133        | 31.0 / 25.0       |
| 64, 4    | 2616 / 1790    | 85.8 / 53.5   | 1544 / 1265        | 40.9 / 31.0       |
| 128, 2   | 1142 / 955     | 30.7 / 23.0   | 942 / 754          | 20.1 / 12.5       |

**Table 2.1.** Statistics (mean / median) solving $\mathrm{VI}(F,\Delta_n)$ with single-level derivative-based path-following iteration.

| Architecture | `n` or `d` | KKT: iterations | KKT: avoidance | Barrier: iterations | Barrier: avoidance |
| ------------ | ---------- | --------------- | -------------- | ------------------- | ------------------ |
| `[n, n, n]`  | 2          | 163 / 166       | 0.0 / 0.0      | 163 / 166           | 0.0 / 0.0          |
| `[n, n, n]`  | 4          | 171 / 171       | 0.0 / 0.0      | 171 / 171           | 0.0 / 0.0          |
| `[n, n, n]`  | 8          | 179 / 173       | 0.1 / 0.0      | 176 / 173           | 0.1 / 0.0          |
| `[n, n, n]`  | 16         | 177 / 174       | 0.0 / 0.0      | 176 / 174           | 0.0 / 0.0          |
| `[n, n, n]`  | 32         | 183 / 176       | 0.1 / 0.0      | 181 / 175           | 0.0 / 0.0          |
| `[n, n, n]`  | 64         | 192 / 185       | 0.0 / 0.0      | 201 / 185           | 0.3 / 0.0          |
| `[n, n, n]`  | 128        | 219 / 208       | 0.1 / 0.0      | 220 / 208           | 0.0 / 0.0          |
| `[n, n, n]`  | 256        | 272 / 250       | 0.3 / 0.0      | 271 / 252           | 0.2 / 0.0          |
| `[n, n, n]`  | 512        | 359 / 337       | 0.8 / 1.0      | 357 / 322           | 0.9 / 1.0          |
| `[n, n, n]`  | 1024       | 606 / 557       | 1.6 / 1.0      | 440 / 400           | 1.2 / 1.0          |
| `[256]*d`    | 2          | 295 / 264       | 0.3 / 0.0      | 300 / 266           | 0.3 / 0.0          |
| `[256]*d`    | 3          | 272 / 250       | 0.3 / 0.0      | 271 / 252           | 0.2 / 0.0          |
| `[256]*d`    | 4          | 268 / 250       | 0.2 / 0.0      | 275 / 253           | 0.2 / 0.0          |
| `[256]*d`    | 5          | 263 / 248       | 0.2 / 0.0      | 266 / 248           | 0.2 / 0.0          |
| `[256]*d`    | 6          | 255 / 243       | 0.1 / 0.0      | 260 / 246           | 0.1 / 0.0          |
| `[256]*d`    | 7          | 260 / 240       | 0.2 / 0.0      | 259 / 243           | 0.1 / 0.0          |
| `[256]*d`    | 8          | 253 / 242       | 0.1 / 0.0      | 254 / 242           | 0.1 / 0.0          |

**Table 2.2.** Statistics (mean / median) solving $\mathrm{VI}(F,\Delta_n^m)$ with single-level derivative-based path-following iteration.

| `m`, `n` | NN: iterations | NN: avoidance | Payoff: iterations | Payoff: avoidance |
| -------- | -------------- | ------------- | ------------------ | ----------------- |
| 8,2      | 175 / 173      | 0.0 / 0.0     | 173 / 173          | 0.0 / 0.0         |
| 8,4      | 181 / 176      | 0.0 / 0.0     | 179 / 174          | 0.0 / 0.0         |
| 8,8      | 201 / 188      | 0.2 / 0.0     | 188 / 177          | 0.0 / 0.0         |
| 8,16     | 248 / 227      | 0.2 / 0.0     | 210 / 199          | 0.0 / 0.0         |
| 8,32     | 418 / 347      | 1.0 / 1.0     | 298 / 283          | 0.3 / 0.0         |
| 2,8      | 176 / 174      | 0.0 / 0.0     | 174 / 174          | 0.0 / 0.0         |
| 4,8      | 180 / 175      | 0.0 / 0.0     | 178 / 174          | 0.0 / 0.0         |
| 8,8      | 201 / 188      | 0.2 / 0.0     | 188 / 177          | 0.0 / 0.0         |
| 16,8     | 262 / 231      | 0.3 / 0.0     | 211 / 197          | 0.1 / 0.0         |
| 32,8     | 500 / 427      | 1.5 / 1.0     | 370 / 335          | 0.8 / 1.0         |
| 2,128    | 323 / 281      | 0.5 / 0.0     | 271 / 266          | 0.1 / 0.0         |
| 4,64     | 377 / 317      | 0.7 / 1.0     | 280 / 270          | 0.1 / 0.0         |
| 8,32     | 418 / 347      | 1.0 / 1.0     | 298 / 283          | 0.3 / 0.0         |
| 16,16    | 439 / 372      | 1.2 / 1.0     | 327 / 306          | 0.6 / 1.0         |
| 32,8     | 500 / 427      | 1.5 / 1.0     | 370 / 335          | 0.8 / 1.0         |
| 64,4     | 431 / 379      | 1.0 / 1.0     | 350 / 324          | 0.7 / 1.0         |
| 128,2    | 270 / 245      | 0.3 / 0.0     | 237 / 217          | 0.1 / 0.0         |

**Table 3.1.** Statistics (mean / median) solving $\mathrm{VI}(F,\Delta_n)$ with single-level derivative-free path-following iteration. This table is summarised as Table 1 of the repository README.

| Architecture | `n` or `d` | KKT: iterations | KKT: avoidance | Barrier: iterations | Barrier: avoidance |
| ------------ | ---------- | --------------- | -------------- | ------------------- | ------------------ |
| `[n, n, n]`  | 2          | 163 / 166       | 0.0 / 0.0      | 163 / 166           | 0.0 / 0.0          |
| `[n, n, n]`  | 4          | 171 / 171       | 0.0 / 0.0      | 171 / 171           | 0.0 / 0.0          |
| `[n, n, n]`  | 8          | 179 / 173       | 0.1 / 0.0      | 176 / 173           | 0.1 / 0.0          |
| `[n, n, n]`  | 16         | 176 / 174       | 0.0 / 0.0      | 176 / 174           | 0.0 / 0.0          |
| `[n, n, n]`  | 32         | 181 / 176       | 0.0 / 0.0      | 181 / 175           | 0.0 / 0.0          |
| `[n, n, n]`  | 64         | 192 / 185       | 0.0 / 0.0      | 203 / 185           | 0.2 / 0.0          |
| `[n, n, n]`  | 128        | 219 / 208       | 0.1 / 0.0      | 221 / 208           | 0.1 / 0.0          |
| `[n, n, n]`  | 256        | 272 / 250       | 0.3 / 0.0      | 273 / 252           | 0.4 / 0.0          |
| `[n, n, n]`  | 512        | 359 / 337       | 0.8 / 1.0      | 349 / 322           | 0.8 / 1.0          |
| `[n, n, n]`  | 1024       | 606 / 557       | 1.6 / 1.0      | 443 / 400           | 1.2 / 1.0          |
| `[256]*d`    | 2          | 294 / 264       | 0.3 / 0.0      | 298 / 267           | 0.3 / 0.0          |
| `[256]*d`    | 3          | 272 / 250       | 0.3 / 0.0      | 273 / 252           | 0.4 / 0.0          |
| `[256]*d`    | 4          | 268 / 250       | 0.2 / 0.0      | 276 / 253           | 0.3 / 0.0          |
| `[256]*d`    | 5          | 263 / 248       | 0.2 / 0.0      | 265 / 248           | 0.1 / 0.0          |
| `[256]*d`    | 6          | 255 / 243       | 0.1 / 0.0      | 258 / 246           | 0.1 / 0.0          |
| `[256]*d`    | 7          | 259 / 240       | 0.2 / 0.0      | 259 / 243           | 0.2 / 0.0          |
| `[256]*d`    | 8          | 253 / 242       | 0.1 / 0.0      | 255 / 242           | 0.1 / 0.0          |

**Table 3.2.** Statistics (mean / median) solving $\mathrm{VI}(F,\Delta_n^m)$ with single-level derivative-free path-following iteration. This table is summarised as Table 2 of the repository README.

| `m`, `n` | NN: iterations | NN: avoidance | Payoff: iterations | Payoff: avoidance |
| -------- | -------------- | ------------- | ------------------ | ----------------- |
| 8,2      | 175 / 173      | 0.0 / 0.0     | 173 / 173          | 0.0 / 0.0         |
| 8,4      | 181 / 176      | 0.0 / 0.0     | 179 / 174          | 0.0 / 0.0         |
| 8,8      | 198 / 188      | 0.0 / 0.0     | 187 / 178          | 0.0 / 0.0         |
| 8,16     | 251 / 228      | 0.2 / 0.0     | 210 / 201          | 0.0 / 0.0         |
| 8,32     | 409 / 342      | 1.0 / 1.0     | 310 / 289          | 0.5 / 0.0         |
| 2,8      | 176 / 174      | 0.0 / 0.0     | 174 / 174          | 0.0 / 0.0         |
| 4,8      | 181 / 175      | 0.0 / 0.0     | 178 / 174          | 0.0 / 0.0         |
| 8,8      | 198 / 188      | 0.0 / 0.0     | 187 / 178          | 0.0 / 0.0         |
| 16,8     | 264 / 235      | 0.4 / 0.0     | 218 / 203          | 0.1 / 0.0         |
| 32,8     | 494 / 427      | 1.4 / 1.0     | 374 / 333          | 0.8 / 1.0         |
| 2,128    | 320 / 280      | 0.5 / 0.0     | 271 / 266          | 0.1 / 0.0         |
| 4,64     | 371 / 313      | 0.9 / 1.0     | 283 / 270          | 0.2 / 0.0         |
| 8,32     | 409 / 342      | 1.0 / 1.0     | 310 / 289          | 0.5 / 0.0         |
| 16,16    | 469 / 386      | 1.2 / 1.0     | 344 / 314          | 0.7 / 1.0         |
| 32,8     | 494 / 427      | 1.4 / 1.0     | 374 / 333          | 0.8 / 1.0         |
| 64,4     | 438 / 393      | 1.0 / 1.0     | 369 / 339          | 0.8 / 1.0         |
| 128,2    | 297 / 248      | 0.8 / 0.0     | 251 / 225          | 0.2 / 0.0         |

## Reproducibility

All rows are determined by the random seed: the operator parameters and the starting point are drawn
after `np.random.seed(seed)`, and the singularity-avoidance steps draw from the same generator. The
logs are therefore reproducible on a given platform, library version, and BLAS thread count;
the driver pins the number of BLAS/OpenMP threads to one by default.
