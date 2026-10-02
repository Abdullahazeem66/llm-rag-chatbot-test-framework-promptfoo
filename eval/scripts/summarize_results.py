"""Summarize the latest promptfoo result file per suite (JSON files saved with `promptfoo eval -o eval/results/<name>.json`).

  python eval/scripts/summarize_results.py              # latest run of every suite
  python eval/scripts/summarize_results.py --failures   # also list failing cases and reasons
  python eval/scripts/summarize_results.py --by category
"""
import argparse
import json
import re
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
STAMP = re.compile(r"-(\d{8}-\d{6})\.json$")


def suite_name(path: Path) -> str:
    match = STAMP.search(path.name)
    return (path.name[: match.start()] if match else path.stem).replace("__", "/")


def run_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def latest_results() -> dict[str, Path]:
    latest: dict[str, Path] = {}
    for path in RESULTS_DIR.glob("*.json"):
        suite = suite_name(path)
        if suite not in latest or path.stat().st_mtime > latest[suite].stat().st_mtime:
            latest[suite] = path
    return dict(sorted(latest.items()))


def column(row: dict) -> str:
    provider = (row.get("provider") or {}).get("label") or (row.get("provider") or {}).get("id", "?")
    prompt = (row.get("prompt") or {}).get("label", "")
    return f"{provider} × {prompt}" if prompt else provider


def matrix_table(rows: list[dict]) -> None:
    columns: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        columns[column(r)].append(r)
    names = sorted({n for r in rows for n in (r.get("namedScores") or {})})
    width = max(len(c) for c in columns) + 2
    print("  " + "column".ljust(width) + "pass   " + "  ".join(n[:12].rjust(12) for n in names) + "   cost$     p50ms")
    for col, rs in columns.items():
        passed = sum(1 for r in rs if r.get("success"))
        cells = []
        for n in names:
            vals = [r["namedScores"][n] for r in rs if n in (r.get("namedScores") or {})]
            cells.append(f"{sum(vals) / len(vals):.3f}".rjust(12) if vals else "-".rjust(12))
        cost = sum((r.get("cost") or 0) for r in rs)
        lat = sorted(r.get("latencyMs") or 0 for r in rs)
        print(f"  {col.ljust(width)}{passed:>2}/{len(rs):<3} " + "  ".join(cells) + f"   {cost:.4f}  {lat[len(lat) // 2]:>7}")


def summarize(path: Path, show_failures: bool, by: str | None) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]["results"]
    if len({column(r) for r in rows}) > 1:
        print(f"\n{suite_name(path)}  ({run_time(path)})")
        matrix_table(rows)
        if not show_failures:
            return
    passed = sum(1 for r in rows if r.get("success"))
    cost = sum((r.get("cost") or 0) for r in rows)
    metrics: dict[str, list[float]] = defaultdict(list)
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for name, value in (r.get("namedScores") or {}).items():
            metrics[name].append(value)
        if by:
            groups[str((r.get("testCase", {}).get("metadata") or {}).get(by))].append(bool(r.get("success")))

    print(f"\n{suite_name(path)}  ({run_time(path)})")
    print(f"  pass {passed}/{len(rows)} ({passed / max(len(rows), 1):.0%})   app cost ${cost:.4f}")
    for name, values in metrics.items():
        print(f"  {name:<18} {sum(values) / len(values):.3f}")
    for group, results in sorted(groups.items()):
        print(f"  [{by}={group}] pass {sum(results)}/{len(results)}")
    if show_failures:
        for r in rows:
            if r.get("success"):
                continue
            meta = r.get("testCase", {}).get("metadata") or {}
            reasons = [c.get("reason", "") for c in (r.get("gradingResult") or {}).get("componentResults", []) if not c.get("pass")]
            reason = " | ".join(reasons) or r.get("error") or ""
            print(f"    FAIL {meta.get('id', '?'):<10} {reason[:220]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suites", nargs="*", help="substring filter on suite names")
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--by", help="group pass rate by a test metadata key, e.g. category")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for suite, path in latest_results().items():
        if not args.suites or any(s in suite for s in args.suites):
            summarize(path, args.failures, args.by)


if __name__ == "__main__":
    main()
