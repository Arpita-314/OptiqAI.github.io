# backend/decision_framework/summary_generator.py
from datetime import datetime
import json
import os

def generate_markdown_summary(title, decision_matrix: dict, recommendation: dict, outdir="runs"):
    os.makedirs(outdir, exist_ok=True)
    fname = f"{title.replace(' ','_')}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.md"
    path = os.path.join(outdir, fname)
    lines = []
    lines.append(f"# {title}\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")
    lines.append("## Decision Matrix\n")
    lines.append(json.dumps(decision_matrix, indent=2))
    lines.append("\n## Recommendation\n")
    lines.append(json.dumps(recommendation, indent=2))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
    return path
