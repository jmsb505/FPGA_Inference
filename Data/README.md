# Local dataset directory

The private multimodal volumes, segmentation maps, generated caches, and calibration tensors are not part of this repository.

Place the dataset outside Git and pass its location with `--data-root` or the `REGISTRATION_DATA_ROOT` environment variable where supported. The final evaluation uses 18 held-out subjects arranged into nine deterministic cross-subject volume pairs.
