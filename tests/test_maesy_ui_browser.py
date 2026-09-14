from pathlib import Path

from maesy_ui.app import browser_selection


def test_browser_selection_returns_highlighted_file():
    folder = Path("/tmp/data")
    selected = folder / "sample.yaml"

    assert browser_selection(folder, selected, highlighted_is_file=True) == selected


def test_browser_selection_returns_current_folder_without_file_selection():
    folder = Path("/tmp/data")

    assert browser_selection(folder, None, highlighted_is_file=False) == folder
    assert browser_selection(folder, folder / "subdir", highlighted_is_file=False) == folder
