# Data and third-party licensing statement

Accessed and recorded on 2026-08-22.

## FLAME3

The public FLAME3 Sycan Marsh dataset listing identifies the dataset license as MIT:

- https://www.kaggle.com/datasets/brycehopkins/flame-3-cv-dataset-sycan-marsh

This repository does not redistribute FLAME3 RGB images, thermal JPGs, Celsius TIFFs, or publisher metadata. Users must obtain the dataset from the original publisher and comply with the license and platform terms shown there. Because dataset-host metadata can change, authors should recheck the listing at submission/release time.

## FLAME2 and RoboFireFuseNet annotations

The official RoboFireFuseNet repository is MIT-licensed. Its FLAME2 annotation/data release states CC BY 4.0 for the processed RGB-thermal segmentation release:

- https://github.com/dimfot3/RoboFireFuseNet
- https://figshare.com/articles/dataset/FLAME2_RGB-Thermal_Dataset_for_Wildfire_Segmentation/31677823

This repository does not redistribute the FLAME2 raw images. The zero-shot experiment used a SHA256-pinned copy downloaded from the official RoboFireFuseNet release and reports the result only as a coverage-mismatched lower-bound transfer estimate.

## Project-created masks and manifests

The A001-A150 and test107 masks are project-created derivative annotations. No public redistribution license is assigned here without explicit author approval. Before publication, the authors should choose a compatible license for those annotations (for example CC BY 4.0), verify compatibility with the source-data terms, and add the final license file.

## Code

Vendored RoboFireFuseNet/PIDNet-derived code retains its upstream notices and license in `third_party/RoboFireFuseNet/`. New project code requires an explicit repository-level license before public release; no license is implied by source availability.
