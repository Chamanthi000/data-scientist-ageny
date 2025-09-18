from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any

class ReportAgent:
    def write(self, eda_md: str, model_info: Dict[str, Any], outdir: str | Path) -> str:
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "report.md").write_text(eda_md + "\n\n## Modeling\n" + json.dumps(model_info, indent=2))
        (outdir / "metrics.json").write_text(json.dumps(model_info, indent=2))
        return str(outdir / "report.md")
