# AGENTS.md — MaeSy coding guide

## Project shape
MaeSy is a Python CLI framework for robot-soccer vision. The supported checkout uses the shared `_maesy_core` packages and the `maesy` application package. **Preserve the current main-branch architecture:** do not restore modules or paths from the historical `master-thesis` branch merely to satisfy stale code or tests.

Primary CLI entry point is `maesy.command_line:main` (registered in `setup.py`). `maesy/command_line.py` owns argparse and dispatches to:

- `_maesy_core/dataset/cli_dataset.py` for dataset operations
- `maesy/training/cli_train.py` for training dispatch
- `maesy/evaluation/cli_evaluate.py` for evaluation dispatch
- `_maesy_core/model/model_tools/cli_export.py` for export

Keep dispatchers thin. Training orchestration lives in `maesy/training/train_setups/`; shared training loops live in `maesy/training/base_trainer.py` and task-specific trainers. Dataset/model/inference implementations belong under `_maesy_core` unless an existing package boundary says otherwise. Trace behavior from argparse through the dispatched handler before changing a CLI option.

## Current model/config discovery
Architecture YAMLs are discovered recursively from `cfg/` using `_maesy_core/model/model_tools/model_factory.py::architecture_paths`. At the time this guide was updated, the checked-in IDs are `mae-multiscale` and `resnet50`. Resolve identifiers through the factory; do not hard-code removed IDs such as `rt-detr6` or assume a nested config tree that is not present. Supported model types and task-specific creation are defined by the factory and setup functions, not by old examples.

## Dataset and batch contracts
- Standard detection datasets use YOLO-style split directories: `<split>/images` and `<split>/labels`.
- `MaesyDataset` is in `_maesy_core/dataset/maesy_dataset.py`. Detection datasets are configured by a root `dataset.yaml` (or the YAML path directly) with `path`, split entries, `nc`, `names`, and `box_format` (`xyxy` or `cxcywh`). Label coordinates are normalized; detection targets expose normalized `boxes` in `xyxy` and integer `labels`.
- Optional line and ellipse classes are identified through YAML `lines`/`ellipses` names. Enabled geometry fields are normalized line endpoints (`line_points`, `x1 y1 x2 y2`) and ellipse parameters (`ellipses`, `center_x center_y log_a log_b cos(2*theta) sin(2*theta)`). Keep special target labels and geometry aligned.
- Classification `MaesyDataset` uses `annotation_type="classification"`; the label is the integer after the final underscore in each image filename, e.g. `frame_2.png`. `collate_classification_fn` turns those samples into `(images, LongTensor[B])`.
- Detection batch path is `MaesyDataset`/`ObjectDetectionDataset` → `maesy.training.collate_functions.collate_detection_fn` → `_maesy_core.dataset.utils.handle_raw_batch` → trainer/loss. `collate_detection_fn` must return `(images, List[target_dict])`; do not stack variable-length annotations.
- `handle_raw_batch` owns device transfer for trainer and inference batches. Add/retain tests in `tests/test_od_batch_contract.py` when altering these contracts.

## Checkpoints and transforms
- Checkpoints are MaeSy-specific and managed by `_maesy_core/model/model_tools/checkpoint_handler.py`; model config and backbone/head metadata must remain compatible across save/load changes.
- Inference and evaluation load models through `create_model_from_checkpoint`; preserve architecture-agnostic model behavior where the factory supports multiple model families.
- Train/validation/inference transforms should remain consistent for a task (the common detection setup uses 224×224 and ImageNet normalization). Geometry transformations must preserve the normalized target conventions.

## CLI behavior and safe data handling
- Confirm current CLI options with `python -m maesy.command_line <subcommand> -h` or the installed `maesy` command before editing help or tests. Current parser options are the contract; do not add tests for removed flags such as `--detector` unless deliberately restoring that feature through all layers.
- For a new mode/model, wire parser arguments → `cli_*` dispatcher → setup/model/trainer, and test forwarding at those boundaries.
- `dataset create --delete` is unsupported; it must fail clearly and never remove source folders. Dataset creation's destination is controlled by `--output-path` when supplied, otherwise `--path`.
- Use temporary directories, synthetic images/labels, and mocks for checkpoint/model loading, ROS APIs, and external services. Never run training or destructive dataset operations against the user's data as a smoke test.
- W&B is initialized in `BaseTrainer`; tests should disable it unless specifically testing integration.

## Tests and validation
- Tests are under `tests/`. Add behavior assertions that demonstrate outputs/errors/contracts, not merely importability or symbol existence. Keep tests offline and small; mock heavyweight model construction where model internals are not the subject.
- Discover available Python/dependency environments first. Do not assume a project `.venv` exists. Use a supported CI Python/PyTorch environment when available; otherwise report dependency-blocked tests truthfully.
- Suggested checks when installed: `pytest -q`, `ruff check .`, `python -m compileall -q maesy _maesy_core tests`, `python -m pip check`, and CLI help smoke checks for train/dataset/evaluate/export.
- After CLI/dataset changes, add parser-to-handler tests. After batch, model, trainer, or metrics changes, run the focused relevant tests and then the full suite where dependencies permit.
- Before finishing, run `git diff --check`, inspect the full diff and `git status`, and preserve unrelated user changes/untracked files. Do not commit unless asked.
