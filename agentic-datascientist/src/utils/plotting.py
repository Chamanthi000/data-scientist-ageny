from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt

def save_hist(series, title: str, path: str | Path):
    path = Path(path)
    fig = plt.figure()
    series.hist()
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
