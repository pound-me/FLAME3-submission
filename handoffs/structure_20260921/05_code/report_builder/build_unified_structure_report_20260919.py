"""Build the unified FLAME3 structure-screen report from frozen Stage 3 JSON files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

RESEARCH_ARMS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "R1", "R2", "R3", "B1")
EVALUATED_ARMS = {"S1", "S3", "S4", "S6", "R1", "R2", "R3", "B1"}
SECOND_BATCH = {"S2", "S5", "S7", "S8"}
SEEDS = (200, 201, 202)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def mean_sd(values):
    return statistics.fmean(values), statistics.stdev(values)


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_mean_sd(values, digits=4):
    if values is None:
        return "N/A"
    mean, sd = mean_sd(values)
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def source_for_arm(arm, first, append):
    if arm in append["decisions"]:
        return (
            append["append_clean_epoch26_30"][arm],
            append["decisions"][arm],
            append["efficiency_static"][arm],
        )
    if arm in first["decisions"]:
        return first["clean_windows"][arm], first["decisions"][arm], first["efficiency_static"][arm]
    return None


def stage0_costs(payload):
    return {row["arm"]: row for row in payload["rows"]}


def build_rows(first, append, costs):
    rows = []
    for arm in RESEARCH_ARMS:
        source = source_for_arm(arm, first, append)
        if source is None:
            cost = costs.get(arm, {})
            rows.append({
                "arm": arm,
                "run_status": "NOT_RUN_SECOND_BATCH_NOT_AUTHORIZED",
                "clean_smoke_mean": None,
                "clean_smoke_sd": None,
                "clean_smoke_delta_mean": None,
                "precision_axis": "NOT_EVALUATED",
                "clean_guards_3_of_3": None,
                "sigma005_drop_mean": None,
                "sigma005_drop_sd": None,
                "robust_axis": "NOT_EVALUATED",
                "fire_full134_mean_record_only": None,
                "deploy_parameters": cost.get("estimated_deploy_parameters"),
                "gmacs_proxy": cost.get("estimated_total_gmacs_proxy"),
                "gmacs_change_percent": cost.get("proxy_change_percent"),
                "formal_latency": "NOT_RUN",
                "overall_decision": "NOT_RUN_NOT_A_FAILURE",
            })
            continue
        clean, decision, efficiency = source
        smoke = [clean[str(seed)]["smoke_iou_A001_A047"] for seed in SEEDS]
        fire = [clean[str(seed)]["fire_heat_iou_full134_record_only"] for seed in SEEDS]
        delta = [row["clean_smoke_delta"] for row in decision["seeds"]]
        drops = [row["candidate_sigma005_drop"] for row in decision["seeds"]]
        rows.append({
            "arm": arm,
            "run_status": "COMPLETE_3_SEEDS_30_EPOCHS",
            "clean_smoke_mean": statistics.fmean(smoke),
            "clean_smoke_sd": statistics.stdev(smoke),
            "clean_smoke_delta_mean": statistics.fmean(delta),
            "precision_axis": "PASS" if decision["precision_axis_pass"] else "NO_DISTINGUISHABLE_EFFECT",
            "clean_guards_3_of_3": decision["clean_guards_3_of_3"],
            "sigma005_drop_mean": statistics.fmean(drops),
            "sigma005_drop_sd": statistics.stdev(drops),
            "robust_axis": decision["robust_axis_decision"],
            "fire_full134_mean_record_only": statistics.fmean(fire),
            "deploy_parameters": efficiency["deploy_parameters"],
            "gmacs_proxy": efficiency["gmacs_proxy"],
            "gmacs_change_percent": efficiency["gmacs_change_percent"],
            "formal_latency": decision["formal_latency_status"],
            "overall_decision": decision["overall_decision"],
        })
    return rows


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effect_table(attribution, metric):
    rows = []
    for condition in attribution["conditions"]:
        record = attribution["aggregate"][condition][metric]
        rows.append({
            "condition": condition,
            "R1_minus_B0": record["R1_minus_B0"],
            "R2_minus_B0": record["R2_minus_B0"],
            "interaction": record["interaction"],
        })
    return rows


def effect_cell(record):
    return f"{record['mean']:+.4f} +/- {record['sample_sd']:.4f}"


def report_markdown(rows, first, append, attribution, evidence):
    lines = [
        "# FLAME3 Network Structure Screen: Unified Stage 3 Report",
        "",
        "Date: 2026-09-19",
        "",
        "## Scope",
        "",
        "The preregistered twelve research arms are S1-S8, R1-R3 and B1. E1 is an engineering control and is reported separately. S2, S5, S7 and S8 were not authorized for Stage 2 and remain not run; no values are imputed.",
        "",
        "All evaluated runs use seeds 200/201/202 and the fixed epoch26-30 window. Fire/Heat values are record-only. Formal latency remains deferred.",
        "",
        "## Twelve-arm table",
        "",
        "| Arm | Status | Clean Smoke mean +/- SD | Mean delta vs v1.1 | Precision | Guards 3/3 | Drop@0.05 mean +/- SD | Robust | GMACs change | Fire full134 record-only | Overall |",
        "|---|---|---:|---:|---|---|---:|---|---:|---:|---|",
    ]
    for row in rows:
        smoke = (
            "N/A" if row["clean_smoke_mean"] is None
            else f"{row['clean_smoke_mean']:.4f} +/- {row['clean_smoke_sd']:.4f}"
        )
        drop = (
            "N/A" if row["sigma005_drop_mean"] is None
            else f"{row['sigma005_drop_mean']:.4f} +/- {row['sigma005_drop_sd']:.4f}"
        )
        lines.append(
            f"| {row['arm']} | {row['run_status']} | {smoke} | {fmt(row['clean_smoke_delta_mean'])} | "
            f"{row['precision_axis']} | {row['clean_guards_3_of_3']} | {drop} | {row['robust_axis']} | "
            f"{fmt(row['gmacs_change_percent'], 3)}% | {fmt(row['fire_full134_mean_record_only'])} | {row['overall_decision']} |"
        )
    lines += [
        "",
        "## E1 engineering control",
        "",
    ]
    e1 = first["decisions"]["E1"]
    e1_clean = first["clean_windows"]["E1"]
    e1_smoke = [e1_clean[str(seed)]["smoke_iou_A001_A047"] for seed in SEEDS]
    lines.append(
        f"E1 clean Smoke IoU is {fmt_mean_sd(e1_smoke)}. Static GMACs change is "
        f"{e1['efficiency']['gmacs_change_percent']:.3f}%. It passes only the engineering-efficiency control; it is not a research-arm precision win."
    )
    lines += [
        "",
        "## R1/R2/R3/v1.1 2x2 attribution: Smoke IoU",
        "",
        "| Condition | R1-B0 | R2-B0 | Interaction |",
        "|---|---:|---:|---:|",
    ]
    for row in effect_table(attribution, "smoke_iou_A001_A047"):
        lines.append(
            f"| {row['condition']} | {effect_cell(row['R1_minus_B0'])} | "
            f"{effect_cell(row['R2_minus_B0'])} | {effect_cell(row['interaction'])} |"
        )
    lines += [
        "",
        "These effects are descriptive, not formal, because the retained B0 perturbation checkpoint is v1.1 epoch100 while R1/R2/R3 use epoch26-30. Same-seed pairing is preserved, but the symmetric baseline window is unavailable.",
        "",
        "## Record-only attribution: S",
        "",
        "| Condition | R1-B0 | R2-B0 | Interaction |",
        "|---|---:|---:|---:|",
    ]
    for row in effect_table(attribution, "selection_score_S_record_only"):
        lines.append(
            f"| {row['condition']} | {effect_cell(row['R1_minus_B0'])} | "
            f"{effect_cell(row['R2_minus_B0'])} | {effect_cell(row['interaction'])} |"
        )
    lines += [
        "",
        "## Record-only attribution: Fire/Heat A001-A047",
        "",
        "| Condition | R1-B0 | R2-B0 | Interaction |",
        "|---|---:|---:|---:|",
    ]
    for row in effect_table(attribution, "fire_heat_iou_A001_A047_record_only"):
        lines.append(
            f"| {row['condition']} | {effect_cell(row['R1_minus_B0'])} | "
            f"{effect_cell(row['R2_minus_B0'])} | {effect_cell(row['interaction'])} |"
        )
    lines += [
        "",
        "## Frozen decisions",
        "",
        "- No evaluated research arm passes the clean Smoke precision rule.",
        "- The formal robustness rule remains not determinable because v1.1 epoch26-30 perturbation checkpoints do not exist.",
        "- E1 is retained only as an engineering-efficiency control.",
        "- Fire/Heat, S and mIoU do not trigger any positive decision.",
        "- The second batch was not started.",
        "",
        "## Evidence",
        "",
    ]
    for label, path, digest in evidence:
        lines.append(f"- {label}: `{path}`; SHA256 `{digest}`")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-summary", type=Path, required=True)
    parser.add_argument("--append-summary", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--stage0-cost", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    first = read(args.first_summary)
    append = read(args.append_summary)
    attribution = read(args.attribution)
    costs = stage0_costs(read(args.stage0_cost))
    rows = build_rows(first, append, costs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "FLAME3_STRUCTURE_12_ARM_THREE_AXIS_TABLE_20260919.csv", rows)
    payload = {
        "date": "2026-09-19",
        "research_arms": list(RESEARCH_ARMS),
        "engineering_control": "E1",
        "rows": rows,
        "attribution_2x2": attribution,
        "Fire_used_for_decision": False,
        "second_batch_started": False,
    }
    json_path = args.output_dir / "FLAME3_STRUCTURE_12_ARM_THREE_AXIS_REPORT_20260919.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    evidence = [
        ("First-batch Stage 3 summary", args.first_summary, sha(args.first_summary)),
        ("S1/R2/R3 appended Stage 3 summary", args.append_summary, sha(args.append_summary)),
        ("R1/R2/R3/v1.1 attribution", args.attribution, sha(args.attribution)),
        ("Stage 0 static cost table", args.stage0_cost, sha(args.stage0_cost)),
    ]
    report = report_markdown(rows, first, append, attribution, evidence)
    report_path = args.output_dir / "FLAME3_STRUCTURE_12_ARM_THREE_AXIS_REPORT_20260919.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "report": str(report_path.resolve()), "report_sha256": sha(report_path),
        "json": str(json_path.resolve()), "json_sha256": sha(json_path),
    }, indent=2))


if __name__ == "__main__":
    main()
