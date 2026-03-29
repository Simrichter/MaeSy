# AGENTS.md

## Scope
- MaeSy is a CLI-first Python framework for robot-soccer vision: dataset curation, MAE/classification pretraining, and object detection fine-tuning/inference.
- Main entrypoint is `maesy` (`setup.py` -> `maesy.command_line:main`).
- Existing AI-guidance scan found `AGENTS.md` and `README.md`.

## Source-of-Truth Architecture
- Dispatch boundary: `maesy/command_line.py` -> `maesy/dataset/cli_dataset.py`, `maesy/training/cli_train.py`, `maesy/evaluation/cli_evaluate.py`, `maesy/model_tools/cli_export.py`.
- Real training/inference orchestration lives in `maesy/training/train_setups/*.py`; keep `cli_*` thin.
- Training core is `BaseTrainer` (`maesy/training/base_trainer.py`); task behavior is in trainer overrides + setup files.
- Keep framework model-agnostic: do not assume DETR-only paths; support multiple architectures/losses wired through configs/setups.

## Data + Batch Contracts (Must Preserve)
- OD dataset format is YOLO-style: `<split>/images` + `<split>/labels` (`maesy/dataset/object_detection_dataset.py`).
- Label rows are normalized `class_id cx cy w h`; training expects normalized `cxcywh` boxes and `labels` long tensors.
- OD batch path is `ObjectDetectionDataset` -> `collate_detection_fn` -> `handle_raw_batch` -> detection loss matching.
- `collate_detection_fn` and `handle_raw_batch` are implemented in `maesy/training/utils/utils.py` (see `tests/test_od_batch_contract.py`).
- `collate_detection_fn` output must stay `(images, List[target_dict])` for OD trainers/losses.
- If line detection is enabled, `ObjectDetectionDataset` also emits normalized `target["line_points"]` (`x1 y1 x2 y2`) for `line_class_id` targets.

## Environment + Workflow
- Use the project venv: `MaeSy/.venv`; package is already editable-installed there.
- Prefer venv executables directly (for deterministic agent runs): `./.venv/bin/python`, `./.venv/bin/maesy`.
- Discover CLI surfaces before edits: `maesy train -h`, `maesy dataset -h`, `maesy evaluate -h`, `maesy export -h`.
- Common flows:
  - `maesy train mae --dataset <dataset_root> [--checkpoint ...] [--wandb]`
  - `maesy train od --dataset <dataset_root> [--checkpoint ...] [--resume] [--freeze] [--detector {detr,rt_detr}] [--enable-denoising ...] [--enable-line-detection --line-class-id N] [--wandb]`
  - `maesy evaluate infer <image_folder> <checkpoint> [-o out_dir] [--detector {auto,detr,rt_detr}] [--visualize]`
  - `maesy export <checkpoint> [-o out_dir] [--architecture {detr,rt_detr}]`

## Testing + Validation Policy
- For code changes, add/update tests that cover new behavior, then run all tests in `.venv`.
- Repo has active pytest coverage under `tests/`; run `./.venv/bin/pytest` and report failures.
- After OD/training-path edits, run a startup smoke check (confirm training loop starts):
  - `./.venv/bin/maesy train od --dataset '/home/simon/Desktop/maesy-training/data/AllData (ObjectDetection)'`

## Integration Contracts
- W&B is initialized in `BaseTrainer`; run naming controls checkpoint subdirs.
- Checkpoints are custom (`backbone`, `head`, type/config metadata) in `maesy/model_tools/checkpoint_handler.py`.
- If checkpoint schema/model contracts change, update save/load compatibility checks together.
- Inference/export can infer detector architecture from checkpoint `headtype`; explicit CLI overrides should remain supported.
- Dataset creation supports optional clustering (`resnet_kmeans`, `resnet_faiss`) via `DatasetManager.create_dataset`.

## Safe Change Strategy
- For new training/export modes/models: wire all three layers (`command_line.py` args -> `cli_*`/`cli_export.py` -> `training/train_setups/*`).
- Keep transforms consistent across train/val/infer (current default uses `224x224` + ImageNet normalization).
- Prefer CLI + `train_setups` behavior over stale examples/docs when they disagree.

