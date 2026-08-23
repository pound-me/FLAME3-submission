# FLAME3 four-channel conv1 initialization ablation

Primary endpoint: epoch-30 `last.pth` on the frozen A001-A047 manual three-class development validation set. FLAME3 test107 was not read.

| Seed | Delta manual 3-class mIoU | Delta Fire/Heat IoU (val134) | Delta S | Delta No-Fire joint FP |
|---:|---:|---:|---:|---:|
| 200 | -0.003251 | -0.055756 | -0.014227 | -0.000834 |
| 201 | +0.022845 | -0.026157 | +0.013577 | -0.000506 |
| 202 | +0.009688 | -0.069872 | -0.007397 | -0.000969 |
| **Mean** | **+0.009761** | **-0.050595** | **-0.002682** | **-0.000770** |

Decision: `retain_random_four_channel_stem`; positive seeds: 2/3; guards clear: False.
