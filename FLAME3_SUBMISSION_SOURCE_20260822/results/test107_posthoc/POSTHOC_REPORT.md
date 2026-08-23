# FLAME3 test107 post-hoc strata and perturbation analysis

> The 107-image test set had already been evaluated under frozen protocols. This artifact is post-hoc descriptive analysis and robustness checking of frozen checkpoints, not a new test-selection session.

## Clean equivalence

All six clean confusion matrices exactly reproduce the frozen August 20 evaluation. Inference uses `augment=False`; only unused auxiliary-head checkpoint keys are ignored after strict prefix checks and FP32 main-logit equivalence tests.

## Clean aggregate

| Arm | Background IoU | Smoke IoU | Fire/Heat IoU | mIoU | S | No-Fire joint FP |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.451451 | 0.414003 | 0.297684 | 0.387713 | 0.355844 | 0.000026 |
| v11 | 0.675997 | 0.642737 | 0.287545 | 0.535426 | 0.465141 | 0.000000 |

Paired v1.1-baseline clean deltas: Smoke IoU +0.228733, Fire/Heat IoU -0.010139, mIoU +0.147713, S +0.109297.

## Clean stratified metrics

| Arm | Stratum | Level | n | Background IoU | Smoke IoU | Fire/Heat IoU | mIoU | S | No-Fire joint FP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | brightness_bin | high | 36 | 0.084400 | 0.462049 | 0.219347 | 0.255265 | 0.340698 | 0.000000 |
| baseline | brightness_bin | low | 36 | 0.774234 | 0.422762 | 0.375515 | 0.524170 | 0.399139 | 0.000026 |
| baseline | brightness_bin | mid | 35 | 0.301313 | 0.361932 | 0.314551 | 0.325932 | 0.338241 | 0.000000 |
| baseline | fire_heat_scale_bin | high | 29 | 0.241199 | 0.404444 | 0.347066 | 0.330903 | 0.375755 | 0.000000 |
| baseline | fire_heat_scale_bin | low | 29 | 0.226722 | 0.439941 | 0.154239 | 0.273634 | 0.297090 | 0.000000 |
| baseline | fire_heat_scale_bin | mid | 29 | 0.250097 | 0.401591 | 0.316288 | 0.322659 | 0.358939 | 0.000000 |
| baseline | fire_heat_scale_bin | none | 20 | 0.955042 | 0.303210 | 0.000000 | 0.419417 | 0.151605 | 0.000026 |
| baseline | sample_class | Fire | 89 | 0.239808 | 0.414003 | 0.297729 | 0.317180 | 0.355866 | 0.000000 |
| baseline | sample_class | No Fire | 18 | 0.999974 | 0.000000 | 0.000000 | 0.333325 | 0.000000 | 0.000026 |
| baseline | smoke_coverage_bin | high | 30 | 0.036771 | 0.471858 | 0.276093 | 0.261574 | 0.373975 | 0.000000 |
| baseline | smoke_coverage_bin | low | 30 | 0.395192 | 0.341466 | 0.299271 | 0.345310 | 0.320369 | 0.000000 |
| baseline | smoke_coverage_bin | mid | 29 | 0.239070 | 0.430486 | 0.307101 | 0.325553 | 0.368794 | 0.000000 |
| baseline | smoke_coverage_bin | none | 18 | 0.999974 | 0.000000 | 0.000000 | 0.333325 | 0.000000 | 0.000026 |
| baseline | time_position_bin | early | 54 | 0.447190 | 0.434979 | 0.268029 | 0.383399 | 0.351504 | 0.000031 |
| baseline | time_position_bin | late | 53 | 0.456184 | 0.391408 | 0.315306 | 0.387633 | 0.353357 | 0.000020 |
| v11 | brightness_bin | high | 36 | 0.277780 | 0.618839 | 0.216432 | 0.371017 | 0.417635 | 0.000000 |
| v11 | brightness_bin | low | 36 | 0.906016 | 0.691146 | 0.363193 | 0.653452 | 0.527170 | 0.000000 |
| v11 | brightness_bin | mid | 35 | 0.636382 | 0.657591 | 0.302778 | 0.532250 | 0.480185 | 0.000000 |
| v11 | fire_heat_scale_bin | high | 29 | 0.517644 | 0.641224 | 0.346391 | 0.501753 | 0.493807 | 0.000000 |
| v11 | fire_heat_scale_bin | low | 29 | 0.530801 | 0.652769 | 0.139616 | 0.441062 | 0.396193 | 0.000000 |
| v11 | fire_heat_scale_bin | mid | 29 | 0.564356 | 0.629377 | 0.300245 | 0.497993 | 0.464811 | 0.000000 |
| v11 | fire_heat_scale_bin | none | 20 | 0.985059 | 0.703967 | 0.000000 | 0.563008 | 0.351983 | 0.000000 |
| v11 | sample_class | Fire | 89 | 0.542282 | 0.642737 | 0.287545 | 0.490854 | 0.465141 | 0.000000 |
| v11 | sample_class | No Fire | 18 | 1.000000 | 0.000000 | 0.000000 | 0.333333 | 0.000000 | 0.000000 |
| v11 | smoke_coverage_bin | high | 30 | 0.154222 | 0.621677 | 0.266170 | 0.347357 | 0.443924 | 0.000000 |
| v11 | smoke_coverage_bin | low | 30 | 0.788471 | 0.694967 | 0.284898 | 0.589445 | 0.489932 | 0.000000 |
| v11 | smoke_coverage_bin | mid | 29 | 0.498271 | 0.635848 | 0.300295 | 0.478138 | 0.468072 | 0.000000 |
| v11 | smoke_coverage_bin | none | 18 | 1.000000 | 0.000000 | 0.000000 | 0.333333 | 0.000000 | 0.000000 |
| v11 | time_position_bin | early | 54 | 0.672656 | 0.662928 | 0.251685 | 0.529090 | 0.457307 | 0.000000 |
| v11 | time_position_bin | late | 53 | 0.679191 | 0.620287 | 0.310192 | 0.536557 | 0.465239 | 0.000001 |

Stratified conclusion for v1.1: time-position S is stable (0.457307 early vs 0.465239 late); S decreases from 0.489932 to 0.443924 as Smoke coverage rises; small Fire/Heat regions are hardest (IoU 0.139616 vs 0.346391 for large regions); high-brightness scenes score below low-brightness scenes (0.417635 vs 0.527170); No-Fire joint FP remains 0.00000045.

## Largest v1.1 S degradation under perturbation

| Condition | Family | Delta S from clean | Delta Fire IoU | Delta Smoke IoU |
|---|---|---:|---:|---:|
| thermal_noise_sigma_0.10 | thermal_noise | -0.255128 | -0.021928 | -0.488327 |
| rgb_zero | modality_failure | -0.250921 | -0.051781 | -0.450061 |
| thermal_noise_sigma_0.05 | thermal_noise | -0.125225 | +0.013174 | -0.263625 |
| rgb_temperature_+0.2 | rgb_temperature | -0.103289 | -0.005365 | -0.201212 |
| thermal_zero | modality_failure | -0.096347 | -0.287545 | +0.094850 |
| rgb_temperature_-0.2 | rgb_temperature | -0.087115 | +0.011587 | -0.185817 |
| rgb_brightness_0.8 | rgb_brightness | -0.046703 | -0.000720 | -0.092686 |
| thermal_noise_sigma_0.02 | thermal_noise | -0.038214 | +0.004626 | -0.081054 |

## Robustness conclusion

Thermal additive noise is the earliest and strongest tested degradation pathway: sigma 0.02 already changes S by -0.038214, and sigma 0.10 produces the largest observed drop (-0.255128). Complete RGB failure is comparably severe (-0.250921).

Full per-image confusions, clean strata, condition aggregates, paired deltas, and degradation curves are provided as CSV files. No prediction images or tensors were saved.
