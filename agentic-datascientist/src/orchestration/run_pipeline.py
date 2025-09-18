from __future__ import annotations
import argparse
from pathlib import Path
from rich import print
from ..utils.io import read_csv, write_json, ensure_dir
from ..agents.dataset_agent import DatasetAgent
from ..agents.eda_agent import EDAAgent
from ..agents.model_agent import ModelAgent
from ..agents.report_agent import ReportAgent
from ..agents.planner import Planner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV dataset')
    parser.add_argument('--target', required=True, help='Target column (binary 0/1)')
    parser.add_argument('--outdir', default='examples/output', help='Output directory')
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    print(f"[bold cyan]Loading[/] {args.data}")
    df = read_csv(args.data)

    # 1) Plan
    plan = Planner().make_plan(args.target)
    write_json({"plan": [p.__dict__ for p in plan]}, Path(outdir) / "plan.json")

    # 2) EDA
    eda_md = EDAAgent().run(df, Path(outdir) / "eda")

    # 3) Modeling
    model_info = ModelAgent(target=args.target).train(df, outdir)

    # 4) Report
    report_path = ReportAgent().write(eda_md, model_info, outdir)
    print(f"[bold green]Done.[/] Report at: {report_path}")

if __name__ == '__main__':
    main()
