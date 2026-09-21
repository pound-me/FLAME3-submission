# FLAME3 Network Structure Screen: Unified Stage 3 Report

Date: 2026-09-19

## Scope

The preregistered twelve research arms are S1-S8, R1-R3 and B1. E1 is an engineering control and is reported separately. S2, S5, S7 and S8 were not authorized for Stage 2 and remain not run; no values are imputed.

All evaluated runs use seeds 200/201/202 and the fixed epoch26-30 window. Fire/Heat values are record-only. Formal latency remains deferred.

## Twelve-arm table

| Arm | Status | Clean Smoke mean +/- SD | Mean delta vs v1.1 | Precision | Guards 3/3 | Drop@0.05 mean +/- SD | Robust | GMACs change | Fire full134 record-only | Overall |
|---|---|---:|---:|---|---|---:|---|---:|---:|---|
| S1 | COMPLETE_3_SEEDS_30_EPOCHS | 0.6806 +/- 0.0183 | -0.0195 | NO_DISTINGUISHABLE_EFFECT | False | 0.1569 +/- 0.0421 | NOT_DETERMINABLE_BASELINE_WINDOW_MISSING | 0.000% | 0.7488 | NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES |
| S2 | NOT_RUN_SECOND_BATCH_NOT_AUTHORIZED | N/A | N/A | NOT_EVALUATED | None | N/A | NOT_EVALUATED | -2.526% | N/A | NOT_RUN_NOT_A_FAILURE |
| S3 | COMPLETE_3_SEEDS_30_EPOCHS | 0.6764 +/- 0.0111 | -0.0237 | NO_DISTINGUISHABLE_EFFECT | False | 0.1923 +/- 0.0620 | NOT_DETERMINABLE_BASELINE_WINDOW_MISSING | -0.682% | 0.7094 | NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES |
| S4 | COMPLETE_3_SEEDS_30_EPOCHS | 0.6775 +/- 0.0111 | -0.0226 | NO_DISTINGUISHABLE_EFFECT | False | 0.1310 +/- 0.0199 | NOT_DETERMINABLE_BASELINE_WINDOW_MISSING | -4.504% | 0.7039 | NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES |
| S5 | NOT_RUN_SECOND_BATCH_NOT_AUTHORIZED | N/A | N/A | NOT_EVALUATED | None | N/A | NOT_EVALUATED | -4.884% | N/A | NOT_RUN_NOT_A_FAILURE |
| S6 | COMPLETE_3_SEEDS_30_EPOCHS | 0.6841 +/- 0.0142 | -0.0160 | NO_DISTINGUISHABLE_EFFECT | False | 0.1982 +/- 0.0840 | NOT_DETERMINABLE_BASELINE_WINDOW_MISSING | 10.106% | 0.7553 | NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES |
| S7 | NOT_RUN_SECOND_BATCH_NOT_AUTHORIZED | N/A | N/A | NOT_EVALUATED | None | N/A | NOT_EVALUATED | -0.143% | N/A | NOT_RUN_NOT_A_FAILURE |
| S8 | NOT_RUN_SECOND_BATCH_NOT_AUTHORIZED | N/A | N/A | NOT_EVALUATED | None | N/A | NOT_EVALUATED | 0.220% | N/A | NOT_RUN_NOT_A_FAILURE |
| R1 | COMPLETE_3_SEEDS_30_EPOCHS | 0.6582 +/- 0.0248 | -0.0419 | NO_DISTINGUISHABLE_EFFECT | False | 0.0433 +/- 0.0064 | NOT_DETERMINABLE_BASELINE_WINDOW_MISSING | 0.000% | 0.7219 | NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES |
| R2 | COMPLETE_3_SEEDS_30_EPOCHS | 0.6845 +/- 0.0061 | -0.0156 | NO_DISTINGUISHABLE_EFFECT | False | 0.2179 +/- 0.0878 | NOT_DETERMINABLE_BASELINE_WINDOW_MISSING | 0.246% | 0.7196 | NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES |
| R3 | COMPLETE_3_SEEDS_30_EPOCHS | 0.6783 +/- 0.0266 | -0.0218 | NO_DISTINGUISHABLE_EFFECT | False | 0.0418 +/- 0.0150 | NOT_DETERMINABLE_BASELINE_WINDOW_MISSING | 0.246% | 0.7148 | NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES |
| B1 | COMPLETE_3_SEEDS_30_EPOCHS | 0.6715 +/- 0.0263 | -0.0286 | NO_DISTINGUISHABLE_EFFECT | False | 0.1701 +/- 0.0653 | NOT_DETERMINABLE_BASELINE_WINDOW_MISSING | 0.000% | 0.7052 | NO_DISTINGUISHABLE_EFFECT_ON_DETERMINABLE_AXES |

## E1 engineering control

E1 clean Smoke IoU is 0.6990 +/- 0.0033. Static GMACs change is -5.066%. It passes only the engineering-efficiency control; it is not a research-arm precision win.

## R1/R2/R3/v1.1 2x2 attribution: Smoke IoU

| Condition | R1-B0 | R2-B0 | Interaction |
|---|---:|---:|---:|
| thermal_noise_002 | -0.0198 +/- 0.0228 | -0.0365 +/- 0.0151 | +0.0544 +/- 0.0116 |
| thermal_noise_005 | +0.1105 +/- 0.0435 | -0.0378 +/- 0.1043 | +0.0594 +/- 0.0908 |
| thermal_noise_010 | +0.2770 +/- 0.0922 | -0.0297 +/- 0.1899 | +0.0458 +/- 0.1948 |
| thermal_zero | -0.0654 +/- 0.0339 | -0.0346 +/- 0.0050 | +0.0571 +/- 0.0202 |
| rgb_zero | -0.2200 +/- 0.1512 | -0.0340 +/- 0.0793 | +0.0421 +/- 0.0328 |

These effects are descriptive, not formal, because the retained B0 perturbation checkpoint is v1.1 epoch100 while R1/R2/R3 use epoch26-30. Same-seed pairing is preserved, but the symmetric baseline window is unavailable.

## Record-only attribution: S

| Condition | R1-B0 | R2-B0 | Interaction |
|---|---:|---:|---:|
| thermal_noise_002 | -0.0725 +/- 0.0252 | -0.0804 +/- 0.0164 | +0.0858 +/- 0.0360 |
| thermal_noise_005 | +0.0117 +/- 0.0344 | -0.0573 +/- 0.0474 | +0.0643 +/- 0.0485 |
| thermal_noise_010 | +0.1875 +/- 0.0915 | +0.0223 +/- 0.0540 | -0.0100 +/- 0.0783 |
| thermal_zero | -0.0292 +/- 0.0151 | -0.0172 +/- 0.0025 | +0.0277 +/- 0.0103 |
| rgb_zero | -0.1527 +/- 0.0996 | -0.1145 +/- 0.1686 | +0.1169 +/- 0.1342 |

## Record-only attribution: Fire/Heat A001-A047

| Condition | R1-B0 | R2-B0 | Interaction |
|---|---:|---:|---:|
| thermal_noise_002 | -0.1010 +/- 0.0339 | -0.1020 +/- 0.0299 | +0.0975 +/- 0.0639 |
| thermal_noise_005 | -0.0630 +/- 0.0531 | -0.0555 +/- 0.0365 | +0.0479 +/- 0.0683 |
| thermal_noise_010 | +0.1191 +/- 0.1300 | +0.0823 +/- 0.1000 | -0.0764 +/- 0.0688 |
| thermal_zero | +0.0069 +/- 0.0073 | +0.0002 +/- 0.0001 | -0.0019 +/- 0.0033 |
| rgb_zero | -0.0486 +/- 0.0180 | -0.1581 +/- 0.2087 | +0.1443 +/- 0.1680 |

## Frozen decisions

- No evaluated research arm passes the clean Smoke precision rule.
- The formal robustness rule remains not determinable because v1.1 epoch26-30 perturbation checkpoints do not exist.
- E1 is retained only as an engineering-efficiency control.
- Fire/Heat, S and mIoU do not trigger any positive decision.
- The second batch was not started.

## Evidence

- First-batch Stage 3 summary: `docs\evidence\structure_20260919\STAGE3_SUMMARY.json`; SHA256 `3E7FF1682AFC0C8C2488B334E69AB0DF35BB8782BD9D8FF5748EB5F42D1E241F`
- S1/R2/R3 appended Stage 3 summary: `docs\evidence\structure_20260919\APPEND_STAGE3_SUMMARY.json`; SHA256 `3D69E39D3BAF602032F0B96EE883D36ADB211EB99A481C69B0368EB11493BDEC`
- R1/R2/R3/v1.1 attribution: `docs\evidence\structure_20260919\ATTRIBUTION_2X2.json`; SHA256 `4BA0773E56B8997B5DB852DEA6F512A89BF1B6BD047D8A849B8874C484CA656D`
- Stage 0 static cost table: `docs\evidence\structure_20260919\COST_SUMMARY_STAGE0.json`; SHA256 `F9F7F56F7C74919C245B3718A2C5A2ABB0BB5AF9B146D9EC4E9072AA4EF59478`
