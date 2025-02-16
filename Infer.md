## Inference

### Feature Extraction

| Model       | Input Size | CPU_1 (ms) | CPU_2 (ms) | GPU (ms) |
|-------------|------------|------------|------------|----------|
| XFeat       | 640x360    | 18.70      | -          |          |
| XFeat(2048) | 640x360    | 13.14      | 13.40      |          |
| Superpoint  | 640x360    | 78.30      | 111.27     |          |

### Matching

| Model       | Keypoints  | Num Layers | CPU_1 (ms) | CPU_2 (ms) | GPU (ms) |
|-------------|------------|------------|------------|------------|----------|
| LighterGlue | 512        | 3          | 15.90      | -          |          |
| LighterGlue | 1024       | 3          | 27.41      | -          |          |
| LighterGlue | 2048       | 3          | 121.29     | -          |          |
| LighterGlue | 1024       | 6          | 52.58      | -          |          |
| LighterGlue | 2048       | 6          | 206.49     | -          |          |

* **XFeat**: XFeat model, with 1024 and 2048 feature size.
* **CPU_1**: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz
* **CPU_2**: Apple M4 @ 3.20GHz
