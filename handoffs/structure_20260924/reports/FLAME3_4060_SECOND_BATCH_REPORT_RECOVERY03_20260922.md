# FLAME3 4060Ti Second-Batch Report

Status: STAGE_A_ENDED_STOP_PENDING_NEW_AUTHORIZATION
Stop reason: Authorized Stage A ended. Later stages not started.

S2/S5/S7/S8 first formal runs. The historical nine arms were not retrained.
This independent 4060Ti group and old 4090 runs are not one matched leaderboard.
Whole-group 4060 baseline reuse: PASS
Same-checkpoint replay is not independent same-seed repeatability; the latter remains NOT_ESTABLISHED.
Formal started: 9; completed: 9; all-phase failure events: 1.
Engineering completed epochs: 6/8.

| Arm | State | Precision | Robust | Mean clean delta (pp) | Mean robustness gain (pp) |
|---|---|---|---|---:|---:|
| S7 | COMPLETE / PROMISING_NOT_PASS | False | False | 0.8402 | -3.0464 |
| S8 | COMPLETE / COMPLETE_NOT_PASS | False | False | 0.5452 | 1.0624 |
| S2 | ENGINEERING_BLOCKED / INCOMPLETE_NOT_JUDGED | None | None | NA | NA |
| S5 | COMPLETE / COMPLETE_NOT_PASS | False | False | -2.3952 | 2.2852 |

## Paired Seeds

| Arm | Seed | Clean IoU (%) | Clean delta (pp) | Noise .05 IoU (%) | Robust gain (pp) | Clean guard | FP abs guard | FP relative guard | RGB-zero delta (pp) | T-zero delta (pp) |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| S7 | 200 | 68.7738 | +0.2670 | 41.0751 | -12.1511 | True | True | True | -21.7489 | +4.1959 |
| S7 | 201 | 69.8869 | -0.8956 | 61.4439 | +5.4487 | True | True | True | +10.1945 | +1.1842 |
| S7 | 202 | 68.0136 | +3.1492 | 51.0461 | -2.4369 | True | True | True | -9.8339 | +2.6093 |
| S8 | 200 | 67.9407 | -0.5662 | 55.4922 | +3.0992 | True | True | True | +1.8125 | +4.0530 |
| S8 | 201 | 67.8339 | -2.9485 | 57.6286 | +3.6863 | False | True | True | -3.1867 | -2.5605 |
| S8 | 202 | 70.0148 | +5.1504 | 51.8861 | -3.5982 | True | True | True | -0.2539 | +2.8096 |
| S5 | 200 | 67.5423 | -0.9646 | 54.8602 | +2.8656 | True | True | True | -21.5037 | +5.4828 |
| S5 | 201 | 66.5295 | -4.2529 | 50.6308 | -2.0071 | False | True | True | +12.6910 | -3.1442 |
| S5 | 202 | 62.8964 | -1.9680 | 54.3629 | +5.9971 | False | True | False | -18.7114 | +0.3200 |

## Complete Evidence

WINDOW_METRICS.json/csv: all six conditions, all available seeds, all recorded metrics.
WINDOW_SUMMARY.json/csv: full-precision mean and sample SD. PAIRED_DELTAS.json/csv: fixed same-seed contrasts.
Fire/Heat, Background, mIoU and S are record-only. No checkpoint selection by curve or score.
Negative degradation values are preserved. Perturbed absolute IoU and FP are not replaced by robustness gains.
GMAC cap: 8.590500. 4060 latency is descriptive, no 10 ms hard gate. ORIGINAL_4090_LATENCY_STATUS=NOT_MEASURED.
S5 capacity reduction remains a mechanism-attribution limitation. Borrowed modules are not claimed as established originality.
Exploratory signals are not formal passes, significance claims, or authorization for further training.

Recommendation order: S7
No R1 parameter scan, N-series, stages B/C/D or 100-epoch confirmation was started.
See ENGINEERING_RESULTS.json and audit/failures for precise failure evidence; missing results are NA, not zero.

## Resource Interruption

Attempt 1 stopped before the first data batch: one worker failed while importing torch with MemoryError.
The three initialized worker audit files contained only their start record and no sample reads.
A residual FLAME3-related PowerShell process held about 53 GiB of Windows commit. User-authorized cleanup removed it.
UU GameViewer processes were preserved. No batch, workers, precision, seed, model mathematics or source snapshot changed.
The aborted S7 attempts remain untouched. Attempt 3 uses engineering/S7/attempt3_seed200, the same deterministic initialization and at most two engineering epochs.
Attempt2 was a pre-batch metadata-check failure (tuple versus JSON list); read-only diagnosis proved full JSON-semantic and tensor SHA identity. Recovery02 fixed only that comparison.
Recovery03 corrects the CPU diagnostic autocast context for Torch2.1; all folding thresholds/probes and training mathematics are unchanged. S7/S8 engineering results were reused without rerunning.
Original terminal queue, first-stop report and result tables are retained. These resumed reports are separate.
This resource incident is not a model-quality failure. No formal run was started before cleanup.
