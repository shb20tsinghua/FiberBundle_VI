from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "data" / "record.log"

SEED = 2
GAP_TOL = 1e-10
VI_DIMENSION = ["1", "256"]
GAME_DIMENSION = ["16", "16"]
OUTPUT = REPO / "docs" / "overview.png"


def trace(seed: int, gap_tol: float, dimension: list[str]) -> str:
    command = [
        sys.executable, "main.py",
        "--dimension", *dimension,
        "--model", "nn_F",
        "--seed", str(seed),
        "--gap-tol", str(gap_tol),
        "--verbose", "1",
    ]
    previous = RECORD.read_text(encoding="utf-8") if RECORD.exists() else None
    try:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        return RECORD.read_text(encoding="utf-8")
    finally:
        if previous is None:
            RECORD.unlink(missing_ok=True)
        else:
            RECORD.write_text(previous, encoding="utf-8")


def panel(ax, text: str, title: str) -> None:
    rows = [line.split("|") for line in text.splitlines()[1:]]
    nits = np.array([int(row[1]) for row in rows])
    gaps = np.array([float(row[2]) for row in rows])
    ax.semilogy(nits, gaps, color="#1f77b4", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("iteration")
    ax.set_ylabel("gap")
    ax.grid(True, which="both", alpha=0.25)


def main() -> None:
    vi = trace(SEED, GAP_TOL, VI_DIMENSION)
    game = trace(SEED, GAP_TOL, GAME_DIMENSION)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=160)
    panel(axes[0], vi, "Simplex VI (dim=256)")
    panel(axes[1], game, "Product-of-simplex VI (dim=16×16)")
    fig.suptitle("Convergence of the path-following iteration", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight")
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
