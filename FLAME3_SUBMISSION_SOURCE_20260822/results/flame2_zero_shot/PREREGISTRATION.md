# FLAME2 zero-shot transfer preregistration (2026-08-22)

- Source split: RoboFireFuseNet FLAME2 official `test_flm.txt`, 200 images, SHA256 `57e3a4b13be799809b5c013f028f76cecd64ae139caae5d15f9aabcf3fc8cb6a`.
- Models: six frozen SHA256-pinned FLAME3 `best_S.pth` checkpoints: baseline and v1.1, seeds 200/201/202.
- No training, fine-tuning, threshold adjustment, prediction saving, or FLAME3 test access.
- Mapping: FLAME2 black/gray/white = Background/Smoke/Fire-Heat. Independent Smoke exists. Visual review shows white Fire regions under dense RGB smoke aligned with IR hotspots, so Fire is not restricted to RGB-visible flame.
- Coverage caveat: FLAME2 Fire cannot be proven to include every residual-heat footprint covered by the broader FLAME3 Fire/Heat class. Results are a coverage-mismatched cross-dataset transfer estimate/lower bound and are not merged with FLAME3 main numbers.
- IR: exact official loader conversion `PIL.Image.convert("L")`, then divide by 255.
- Spatial preprocessing: deterministic 256x256 official FLAME2 resize/pad/crop, no augmentation. Model is built with `augment=False`.
- Endpoints: Background/Smoke/Fire-Heat IoU, precision, recall, 3-class mIoU, equal Fire-Smoke S, background-only joint false-positive ratio, per-image metrics, and paired v1.1-minus-baseline deltas.
- Ignore ratio: 0 for the official direct mapping; the coverage limitation is reported narratively rather than inventing post-hoc Ignore masks.
