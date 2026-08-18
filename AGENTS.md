# AGENTS.md

## Scope
- MaeSy is a CLI-first Python framework for robot-soccer vision: dataset curation, MAE/classification pretraining, and object detection fine-tuning/inference.
- Main entrypoint is `maesy` (`setup.py` -> `maesy.command_line:main`).
- Existing AI-guidance scan found `AGENTS.md` and `README.md`.

## Source-of-Truth Architecture
- Dispatch boundary: `maesy/command_line.py` -> `_maesy_core/dataset/cli_dataset.py`, `maesy/training/cli_train.py`, `maesy/evaluation/cli_evaluate.py`, `_maesy_core/model/model_tools/cli_export.py`.
- Real training/inference orchestration lives in `maesy/training/train_setups/*.py`; keep `cli_*` thin.
- Training core is `BaseTrainer` (`maesy/training/base_trainer.py`); task behavior is in trainer overrides + setup files.
- Keep framework model-agnostic: do not assume DETR-only paths; support multiple architectures/losses wired through configs/setups.

## Data + Batch Contracts (Must Preserve)
- OD dataset format is YOLO-style: `<split>/images` + `<split>/labels` (`_maesy_core/dataset/object_detection_dataset.py`).
- Default training/inference uses `MaesyDataset` (`_maesy_core/dataset/maesy_dataset.py`) and requires a `dataset.yaml` in the dataset root (or pass the yaml path directly) with `path`, split keys (`train`/`val`/`test`), `box_format` (`xyxy` or `cxcywh`), `nc`, and `names`.
- Label rows are normalized; `MaesyDataset` accepts `cxcywh` or `xyxy` per `box_format` and always returns normalized `boxes` in `xyxy` plus `labels` long tensors.
- OD batch path is `MaesyDataset`/`ObjectDetectionDataset` -> `collate_detection_fn` -> `handle_raw_batch` -> detection loss matching.
- `collate_detection_fn` and `handle_raw_batch` are implemented in `maesy/training/collate_functions.py` (see `tests/test_od_batch_contract.py`).
- `collate_detection_fn` output must stay `(images, List[target_dict])` for OD trainers/losses.
- If line/ellipse detection is enabled, targets include normalized `target["line_points"]` (`x1 y1 x2 y2`) and `target["ellipses"]` (`center_x center_y log_a log_b cos(2*theta) sin(2*theta)`) when class names are mapped via `dataset.yaml` (`lines`, `ellipses`).

## Environment + Workflow
- Use the project venv: `MaeSy/.venv`; package is already editable-installed there.
- Prefer venv executables directly (for deterministic agent runs): `./.venv/bin/python`, `./.venv/bin/maesy`.
- Discover CLI surfaces before edits: `maesy train -h`, `maesy dataset -h`, `maesy evaluate -h`, `maesy export -h`.
- Common flows:
  - `maesy train mae <model_or_checkpoint> --dataset <dataset_root_or_yaml> [--resume] [--wandb]`
  - `maesy train od <model_or_checkpoint> --dataset <dataset_root_or_yaml> [--finetune] [--resume] [--backbone <ckpt>] [--enable-denoising ...] [--enable-line-detection] [--enable-ellipse-detection] [--wandb]`
  - `maesy evaluate infer <image_folder> <checkpoint> [-o out_dir] [--visualize]`
  - `maesy export <model_or_checkpoint> [-o out_dir] [--name NAME] [--num-classes N] [--enable-line-detection --line-class-id N] [--enable-ellipse-detection --ellipse-class-id N]`

## Testing + Validation Policy
- For code changes, add/update tests that cover new behavior, then run all tests in `.venv`.
- Repo has active pytest coverage under `tests/`; run `./.venv/bin/pytest` and report failures.
- After OD/training-path edits, run a startup smoke check (confirm training loop starts):
  - `./.venv/bin/maesy train od --dataset '/home/simon/Desktop/maesy-training/data/AllData (ObjectDetection)'`

## Integration Contracts
- W&B is initialized in `BaseTrainer`; run naming controls checkpoint subdirs.
- Checkpoints are custom (`backbone`, `head`, type/config metadata) in `_maesy_core/model/model_tools/checkpoint_handler.py`.
- If checkpoint schema/model contracts change, update save/load compatibility checks together.
- Inference/export load models from checkpoints via `create_model_from_checkpoint`; export can also take an architecture name from `cfg/*.yaml` when `--num-classes` (and special class ids) are provided.
- Dataset creation supports optional clustering (`resnet_kmeans`, `resnet_faiss`) via `DatasetManager.create_dataset`.

## Safe Change Strategy
- For new training/export modes/models: wire all three layers (`command_line.py` args -> `cli_*`/`cli_export.py` -> `training/train_setups/*`).
- Keep transforms consistent across train/val/infer (current default uses `224x224` + ImageNet normalization).
- Prefer CLI + `train_setups` behavior over stale examples/docs when they disagree.
