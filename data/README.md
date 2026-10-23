# Experiment data

This folder contains the per-instance results behind Tables 1 and 2 of the
repository [README](../README.md).

| Directory    | Problem                                                                  | Files | Instances |
| ------------ | ------------------------------------------------------------------------ | ----- | --------- |
| `data_vi/`   | $\mathrm{VI}(F,\Delta_n)$ — neural-network operator                      | 32    | 6,400     |
| `data_game/` | $\mathrm{VI}(F,\Delta_n^m)$ — neural-network and payoff-tensor operators | 28    | 5,600     |

Every file holds exactly 200 rows, one per seed `0 ... 199`.

| Folder       | Pattern                                  | Meaning                                                                                             |
| ------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `data_vi/`   | `kkt_<n>.log` / `barr_<n>.log`           | architecture `[n, n, n]`, KKT / barrier corrector, `n` ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024} |
| `data_vi/`   | `kkt_256_d<d>.log` / `barr_256_d<d>.log` | architecture `[256]*d`, KKT / barrier corrector, `d` ∈ {2, 4, 5, 6, 7, 8}                           |
| `data_game/` | `nn_<m>_<n>.log`                         | neural-network operator, architecture `[m*n, m*n, m*n]`                                             |
| `data_game/` | `payoff_<m>_<n>.log`                     | payoff-tensor operator, rank `R = m*n`                                                              |


In `data_vi/`, `kkt` uses `efficient_corrector=False`; `barr` uses `efficient_corrector=True` (the default).
In `data_game/`, both use `efficient_corrector=True`.
