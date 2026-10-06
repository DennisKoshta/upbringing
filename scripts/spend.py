"""Modal spend for this project: every app named upb-* since the budget start date."""
import datetime
import json
import subprocess
import sys

BUDGET, START = 100.0, "2026-10-06"

end = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()  # --end is required in practice
out = subprocess.run(["modal", "billing", "report", "--start", START, "--end", end, "--json"], capture_output=True, text=True, check=True).stdout
rows = [r for r in json.loads(out) if r["description"].startswith("upb-")]
by_app = {}
for r in rows:
    by_app[r["description"]] = by_app.get(r["description"], 0.0) + float(r["cost"])
for name, cost in sorted(by_app.items(), key=lambda kv: -kv[1]):
    print(f"  {name:<24} ${cost:7.2f}")
total = sum(by_app.values())
print(f"  {'total':<24} ${total:7.2f} of ${BUDGET:.0f} ({total / BUDGET:.0%})")
sys.exit(1 if total > BUDGET else 0)
