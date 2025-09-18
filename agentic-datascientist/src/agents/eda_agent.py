from __future__ import annotations
from pathlib import Path
import pandas as pd
from .dataset_agent import DatasetAgent
from ..utils.plotting import save_hist

class EDAAgent:
    def run(self, df: pd.DataFrame, outdir: str | Path) -> str:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # Save histograms for numeric columns
        for c in df.select_dtypes(include="number").columns:
            save_hist(df[c].dropna(), f"Distribution of {c}", outdir / f"hist_{c}.png")

        # Build a short markdown narrative
        da = DatasetAgent()
        summ = da.summarize(df)
        md = [f"# EDA Report", f"- Rows: {summ.n_rows}", f"- Cols: {summ.n_cols}", "## Columns:"]
        for c, meta in summ.columns.items():
            line = f"- **{c}** ({meta['dtype']}), nulls={meta['nulls']}, unique={meta['unique']}"
            md.append(line)
        report_md = "\n".join(md)
        (outdir / "eda.md").write_text(report_md)
        return report_md
