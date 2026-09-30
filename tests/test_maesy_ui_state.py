import argparse
import json
from types import SimpleNamespace

from maesy_ui import state
from maesy_ui.app import MaesyUiApp


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("--root", default=None, help="Root directory path")
    subparsers = parser.add_subparsers(dest="command")
    create = subparsers.add_parser("create")
    create.add_argument("source")
    create.add_argument("--workers", default=4)
    create_sub = create.add_subparsers(dest="mode")
    create_sub.add_parser("kmeans").add_argument("--clusters", default=8)
    create_sub.add_parser("faiss").add_argument("--nprobe", default=1)
    return parser


def test_state_file_uses_default_location(monkeypatch):
    monkeypatch.delenv("MAESY_UI_STATE_FILE", raising=False)
    assert state.state_file().name == ".maesy_ui_state.json"


def test_state_file_honours_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MAESY_UI_STATE_FILE", str(tmp_path / "custom.json"))
    assert state.state_file() == tmp_path / "custom.json"


def test_load_state_missing_file(tmp_path):
    assert state.load_state(tmp_path / "absent.json") == {}


def test_load_state_corrupt_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    assert state.load_state(path) == {}


def test_load_state_non_dict_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert state.load_state(path) == {}


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "nested" / "state.json"
    state.save_state({"target": "ClusterDevil", "flag": True}, path)
    assert state.load_state(path) == {"target": "ClusterDevil", "flag": True}


def test_save_state_stringifies_unserializable_values(tmp_path):
    path = tmp_path / "state.json"
    state.save_state({"values": {"a": object()}}, path)
    loaded = state.load_state(path)
    assert isinstance(loaded["values"]["a"], str)


def test_field_key_roundtrip():
    for path, dest in [((), "root"), (("train", "od"), "model"), (("a", "b", "c"), "x")]:
        encoded = state.encode_field_key(path, dest)
        assert state.decode_field_key(encoded) == (path, dest)


def test_decode_field_key_rejects_empty():
    assert state.decode_field_key("") is None
    assert state.decode_field_key("///") is None


def test_subcommand_key_roundtrip():
    for prefix in [(), ("train",), ("train", "od")]:
        encoded = state.encode_subcommand_key(prefix)
        assert state.decode_subcommand_key(encoded) == prefix


def test_target_state_roundtrip():
    parser = make_parser()
    subcommands = {(): "create", ("create",): "faiss"}
    form_values = {
        (("create",), "source"): "/data/imgs",
        (("create",), "nprobe"): 3,
        ((), "root"): "/data",
    }
    raw = state.target_state(subcommands, form_values)
    decoded_subcommands, decoded_values = state.decode_target_state(raw, parser)
    assert decoded_subcommands == subcommands
    assert decoded_values == form_values


def test_decode_target_state_rejects_garbage():
    parser = make_parser()
    assert state.decode_target_state(None, parser) == ({}, {})
    assert state.decode_target_state("nonsense", parser) == ({}, {})
    assert state.decode_target_state({"subcommands": 5, "values": [1]}, parser) == ({}, {})


def test_decode_target_state_falls_back_for_stale_subcommand():
    parser = make_parser()
    raw = {
        "subcommands": {"create": "removed-mode"},
        "values": {"create/source": "/data/imgs"},
    }
    subcommands, values = state.decode_target_state(raw, parser)
    assert subcommands == {(): "create", ("create",): "kmeans"}
    assert values == {(("create",), "source"): "/data/imgs"}


class _StringVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Root:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def make_app(tmp_path, monkeypatch, target_name="ClusterDevil"):
    """Build a MaesyUiApp without a display, wired to a temp state file."""
    monkeypatch.setenv("MAESY_UI_STATE_FILE", str(tmp_path / "state.json"))
    parser = make_parser()
    app = object.__new__(MaesyUiApp)
    app.root = _Root()
    app.target_var = _StringVar(target_name)
    app.subcommand_values = {}
    app.form_values = {}
    app.widget_fields = {}
    app._target_states = {}
    app._last_target_name = target_name
    app.target_by_name = {target_name: SimpleNamespace(display_name=target_name, parser=lambda: parser)}
    return app


def test_app_restore_state_remembers_inputs(tmp_path, monkeypatch):
    state.save_state(
        {
            "target": "ClusterDevil",
            "ClusterDevil": state.target_state(
                {(): "create", ("create",): "faiss"},
                {(("create",), "source"): "/data/imgs", ((), "root"): "/data"},
            ),
        },
        tmp_path / "state.json",
    )
    app = make_app(tmp_path, monkeypatch)
    app._restore_state()
    assert app.subcommand_values == {(): "create", ("create",): "faiss"}
    assert app.form_values == {(("create",), "source"): "/data/imgs", ((), "root"): "/data"}


def test_app_on_close_persists_current_inputs(tmp_path, monkeypatch):
    app = make_app(tmp_path, monkeypatch)
    app.subcommand_values = {(): "create", ("create",): "kmeans"}
    app.form_values = {(("create",), "source"): "/other/imgs", (("create",), "clusters"): 5}
    app._on_close()
    assert app.root.destroyed
    loaded = state.load_state(tmp_path / "state.json")
    assert loaded["target"] == "ClusterDevil"
    assert loaded["ClusterDevil"] == state.target_state(
        {(): "create", ("create",): "kmeans"},
        {(("create",), "source"): "/other/imgs", (("create",), "clusters"): 5},
    )


def test_app_target_switch_keeps_other_target_inputs(tmp_path, monkeypatch):
    state.save_state(
        {
            "target": "ClusterDevil",
            "MaeSy": state.target_state({}, {((), "root"): "/maesy/data"}),
            "ClusterDevil": state.target_state(
                {(): "create"}, {(("create",), "source"): "/cd/imgs"}
            ),
        },
        tmp_path / "state.json",
    )
    app = make_app(tmp_path, monkeypatch, target_name="ClusterDevil")
    app._render_form = lambda: None
    app.target_by_name["MaeSy"] = SimpleNamespace(display_name="MaeSy", parser=lambda: make_parser())
    app._restore_state()
    assert app.form_values == {(("create",), "source"): "/cd/imgs"}

    app.target_var.set("MaeSy")
    app._on_target_changed()
    assert app.form_values == {((), "root"): "/maesy/data"}
    assert app.subcommand_values == {(): "create", ("create",): "kmeans"}

    app.target_var.set("ClusterDevil")
    app._on_target_changed()
    assert app.form_values == {(("create",), "source"): "/cd/imgs"}


def test_full_persistence_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    saved = {
        "target": "ClusterDevil",
        "ClusterDevil": state.target_state({(): "create"}, {(("create",), "source"): "/x"}),
    }
    state.save_state(saved, path)
    loaded = state.load_state(path)
    assert loaded == json.loads(json.dumps(saved))
