import argparse

import pytest

from maesy_ui.argparse_adapter import (
    ArgumentValidationError,
    build_argv,
    command_choices,
    fields_for_path,
    resolve_parser,
    validate_argv,
)


def make_nested_parser():
    parser = argparse.ArgumentParser(prog="tool")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--run-name", default="default")
    modes = train.add_subparsers(dest="mode", required=True)
    od = modes.add_parser("od")
    od.add_argument("model")
    od.add_argument("--dataset", nargs="+", required=True)
    od.add_argument("--epochs", type=int, default=2)
    od.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    od.add_argument("--resume", action="store_true")
    return parser


def test_adapter_discovers_nested_subcommands_and_fields():
    parser = make_nested_parser()

    assert command_choices(parser) == {"train": resolve_parser(parser, ["train"])}
    assert resolve_parser(parser, ["train", "od"]).prog.endswith("train od")
    assert [field.key for field in fields_for_path(parser, ["train", "od"])] == [
        ((), "verbose"),
        (("train",), "run_name"),
        (("train", "od"), "model"),
        (("train", "od"), "dataset"),
        (("train", "od"), "epochs"),
        (("train", "od"), "device"),
        (("train", "od"), "resume"),
    ]


def test_adapter_builds_and_validates_argv_for_nested_parser():
    parser = make_nested_parser()
    values = {
        ((), "verbose"): True,
        (("train",), "run_name"): "experiment",
        (("train", "od"), "model"): "rt-detr",
        (("train", "od"), "dataset"): ["/data one", "/data-two"],
        (("train", "od"), "epochs"): "5",
        (("train", "od"), "device"): "cuda",
        (("train", "od"), "resume"): True,
    }

    argv = build_argv(parser, ["train", "od"], values)

    assert argv == [
        "--verbose",
        "train",
        "--run-name",
        "experiment",
        "od",
        "rt-detr",
        "--dataset",
        "/data one",
        "/data-two",
        "--epochs",
        "5",
        "--device",
        "cuda",
        "--resume",
    ]
    parsed = validate_argv(parser, argv)
    assert parsed.model == "rt-detr"
    assert parsed.dataset == ["/data one", "/data-two"]
    assert parsed.epochs == 5
    assert parsed.resume is True


def test_adapter_omits_unchanged_optional_defaults():
    parser = make_nested_parser()
    values = {
        (("train", "od"), "model"): "rt-detr",
        (("train", "od"), "dataset"): "/data",
    }

    argv = build_argv(parser, ["train", "od"], values)

    assert argv == ["train", "od", "rt-detr", "--dataset", "/data"]
    parsed = validate_argv(parser, argv)
    assert parsed.epochs == 2
    assert parsed.device == "cpu"


def test_adapter_identifies_path_like_fields_from_destinations_and_help():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Architecture name or path to a checkpoint")
    parser.add_argument("--dataset", help="Training dataset")
    parser.add_argument("--name", help="Human-readable experiment name")

    fields = {field.action.dest: field for field in fields_for_path(parser, [])}

    assert fields["model"].is_path_like is True
    assert fields["dataset"].is_path_like is True
    assert fields["name"].is_path_like is False


def test_adapter_supports_fixed_length_sequence_fields():
    parser = argparse.ArgumentParser(prog="tool")
    parser.add_argument("--split", nargs=3, metavar=("TRAIN", "VAL", "TEST"))

    values = { ((), "split"): ["0.7", "0.2", "0.1"] }
    argv = build_argv(parser, [], values)

    assert argv == ["--split", "0.7", "0.2", "0.1"]
    parsed = validate_argv(parser, argv)
    assert list(parsed.split) == ["0.7", "0.2", "0.1"]


def test_adapter_reports_missing_required_values_without_exiting():
    parser = make_nested_parser()

    with pytest.raises(ArgumentValidationError, match="model"):
        build_argv(
            parser,
            ["train", "od"],
            {(("train", "od"), "dataset"): ["/data"]},
        )

    with pytest.raises(ArgumentValidationError, match="--dataset"):
        validate_argv(parser, ["train", "od", "rt-detr"])
