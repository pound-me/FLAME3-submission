# FLAME3 submission readiness audit

Status: `complete`. Failed checks: none.

## Frozen conclusions

- Conv1 initialization: `retain_random_four_channel_stem`. The RGB-copy/IR-mean arm improved manual three-class mIoU by +0.009761, but Fire/Heat IoU changed by -0.050595 and triggered the preregistered guard.
- RTX 4090 efficiency: 7.185 ms mean, 7.716 ms P95, 139.18 FPS; 7,623,939 parameters, 7.471 GMACs.
- Complete data manifest: 734 unique samples across train/val/test107; hashing performed without training or inference.
- FLAME2 is reported as a coverage-mismatched zero-shot lower-bound transfer estimate. test107 strata/perturbations remain explicitly post-hoc and are not method-selection evidence.

## Checks

| Check | Status |
|---|---|
| critical_artifacts_exist | passed |
| split_row_counts | passed |
| split_sha256 | passed |
| reproducibility_hash_index | passed |
| complete_per_file_manifest | passed |
| flame2_zero_shot_complete | passed |
| flame2_per_image_metrics | passed |
| test107_posthoc_complete | passed |
| conv1_ablation_frozen_decision | passed |
| conv1_six_run_metadata | passed |
| efficiency_protocol_complete | passed |
| no_sys_path_insert | passed |
| no_machine_paths_in_src_or_configs | passed |
| torchsummary_import_main_only | passed |
| no_checkpoint_binaries | passed |
| no_python_caches | passed |
