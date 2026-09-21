"""Evaluate only the new B0 window and append, never replace, historical results."""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import run_baseline as training
import reference_stage3 as evaluator
from replay_policy import recovery_check, paired_row, interaction
from stage3_protocol import CONDITIONS, EPOCHS, SEEDS, metric_view, average_views, read_json, read_jsonl, write_json, write_csv, sha256

ROOT = Path(__file__).resolve().parent


def expected_clean(unit):
    row = read_json(ROOT / "runs/B0" / f"seed{unit['seed']}" / "epochs" / f"epoch{unit['epoch']:02d}.json")
    return metric_view(row["validation"])


def main():
    pref, _ = training.preflight(ROOT)
    evaluator.STAGE2 = ROOT
    evaluator.PROTOCOL_ID = 'flame3_baseline_window_evaluation_20260921_v1'
    evaluator.expected_clean = expected_clean
    perturb, _ = evaluator.load_perturb()
    training.torch.backends.cudnn.benchmark = False
    training.torch.backends.cudnn.deterministic = True
    windows, recovery, rows = {}, [], []
    historical_paths = [ROOT / "HISTORICAL_STAGE3_SUMMARY.json", ROOT / "HISTORICAL_APPEND_SUMMARY.json"]
    for seed in SEEDS:
        historical_paths.append(ROOT.parent / "project_support/experiments/flame3_pidnet_s_fusion_manual_smoke_v11" / f"flame3_fusion_manual_smoke_v11_30e_seed{seed}/metrics.jsonl")
    context_before = {str(p): sha256(p) for p in historical_paths}
    for seed in SEEDS:
        folder = ROOT / "runs/B0" / f"seed{seed}"
        result = read_json(folder / "RESULT.json")
        if result["status"] != "COMPLETE_30_EPOCHS" or not result["input_hashes_unchanged"]:
            raise RuntimeError("Incomplete B0 training")
        old_rows = {r["epoch"]: r for r in read_jsonl(historical_paths[list(SEEDS).index(seed)+2])}
        values = []
        for epoch in EPOCHS:
            checkpoint = folder / f"epoch{epoch}.pth"
            if sha256(checkpoint) != result["checkpoint_sha256"][checkpoint.name]:
                raise RuntimeError("B0 checkpoint drift")
            unit = dict(kind="candidate", arm="baseline_v11", seed=seed, epoch=epoch, path=checkpoint)
            output = ROOT / "evaluation/B0" / f"seed{seed}" / f"epoch{epoch}.json"
            if output.exists():
                raise FileExistsError("No automatic overwrite or repeated evaluation")
            write_json(ROOT / "EVALUATION_STATUS.json", dict(status="RUNNING", seed=seed, epoch=epoch,
                       complete_units=len(rows), unit_total=15, test107_read=False))
            value = evaluator.evaluate_unit(unit, perturb, output)
            if value['clean_replay_contract_pass'] is not True:
                raise RuntimeError('B0 stage3 clean replay failed the frozen strict tolerance')
            values.append(value)
            check = recovery_check(value["conditions"]["clean"], metric_view(old_rows[epoch]["validation"]))
            recovery.append(dict(seed=seed, epoch=epoch, **check))
            rows.append(dict(seed=seed, epoch=epoch, checkpoint_sha256=value["checkpoint_sha256"],
                             clean_replay_max_abs=value["clean_replay_max_abs"], historical_recovery=check))
        windows[str(seed)] = {c["id"]: average_views([v["conditions"][c["id"]] for v in values]) for c in CONDITIONS}
    first = read_json(historical_paths[0])
    append = read_json(historical_paths[1])
    # These are complete same-window candidate evaluations, not epoch100 context.
    candidates = dict(first["robust_windows_epoch26_30"])
    candidates.update(append["append_robust_epoch26_30"])
    pairings = {arm: [dict(seed=seed, **paired_row(values[str(seed)], windows[str(seed)])) for seed in SEEDS]
                for arm, values in candidates.items()}
    attribution = []
    for condition in CONDITIONS[1:]:
        cond = condition["id"]
        for seed in SEEDS:
            for metric in windows[str(seed)][cond]:
                vals = {"B0": windows[str(seed)][cond][metric]}
                vals.update({arm: candidates[arm][str(seed)][cond][metric] for arm in ("R1", "R2", "R3")})
                attribution.append(dict(condition=cond, seed=seed, metric=metric,
                                        **{"Y_"+k: v for k,v in vals.items()},
                                        **interaction(vals["B0"], vals["R1"], vals["R2"], vals["R3"])))
    recovered = all(row["pass"] for row in recovery)
    summary = dict(status="COMPLETE_B0_REPLAY_AND_WINDOW_EVALUATION", baseline_window=windows,
                   historical_recovery_all_pass=recovered, historical_recovery=recovery,
                   historical_environment_identical=False,
                   formal_upgrading_allowed=False,
                   pairing_status="REQUIRES_PROVENANCE_REVIEW" if recovered else "DESCRIPTIVE_ONLY_REPLAY_NOT_RECOVERED",
                   paired_candidates=pairings, attribution=attribution,
                   units=rows, historical_reports_modified=False, test107_read=False,
                   predictions_saved=False, training_in_evaluation=False,
                   Fire_used_for_decision=False, S_used_for_decision=False,
                   no_extra_candidate_runs_started=True, finished_unix=time.time())
    after = {path: sha256(path) for path in pref["input_sha256"]}
    if after != pref["input_sha256"] or {str(p):sha256(p) for p in historical_paths} != context_before:
        raise RuntimeError("Input/context drift during evaluation")
    summary.update(input_hashes_unchanged=True, context_hashes_unchanged=True)
    write_json(ROOT / "BASELINE_WINDOW_REPORT.json", summary)
    write_json(ROOT / "EVALUATION_STATUS.json", dict(status="COMPLETE", unit_total=15,
               recovery_pass=recovered, formal_upgrading_allowed=False, test107_read=False))


if __name__ == "__main__":
    main()
