import os
from typing import List


def _get_pyplot():
    """Import plotting only when a chart is actually generated."""
    import matplotlib.pyplot as plt

    return plt


def _recursive_search_results(
    folder: str,
    search_name: str = "evaluation_metrics.txt",
    mode: str = "strict",
) -> list[str]:
    """Find result files under a folder using strict or substring filename matching."""
    if mode not in {"strict", "contains"}:
        raise ValueError(f"Unsupported search mode: {mode}")

    results = []
    for current_root, _directories, filenames in os.walk(folder):
        for filename in filenames:
            matches = filename == search_name if mode == "strict" else search_name in filename
            if matches:
                results.append(os.path.abspath(os.path.join(current_root, filename)))
    return results


def _order_data_by_mtime(files: list[str]) -> list[str]:
    return sorted(files, key=lambda path: os.path.getmtime(path))


def _read_results(file_path: str) -> dict[str, str]:
    """Read key/value rows from an evaluator text report."""
    data = {}
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.partition(":")
            if separator:
                data[key.strip()] = value.strip()
    return data


def _merge_data(data_list: list[dict]) -> dict[str, list[str]]:
    """Merge report rows without shifting optional values between checkpoints."""
    keys = dict.fromkeys(key for data in data_list for key in data)
    merged = {
        key: [str(data.get(key, "N/A")) for data in data_list]
        for key in keys
    }
    if data_list:
        merged["Checkpoint"] = [
            str(data.get("Checkpoint") or f"Run {index + 1}")
            for index, data in enumerate(data_list)
        ]
    return merged


def _numeric_values(values: list[str]) -> list[float]:
    numeric = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            numeric.append(float("nan"))
    return numeric


def _has_metric(merged_results: dict, metric: str) -> bool:
    """Return whether any report has a value for a metric or composite metric."""
    return any(
        value not in (None, "", "N/A")
        for component in metric.split("/")
        for value in merged_results.get(component, [])
    )


def _metric_components(merged_results: dict, metric: str, index: int) -> list[str]:
    parts = []
    for component in metric.split("/"):
        component_values = merged_results.get(component, [])
        parts.append(
            str(component_values[index])
            if index < len(component_values)
            else "N/A"
        )
    return parts


def _metric_row_count(merged_results: dict, metrics: list[str]) -> int:
    components = {component for metric in metrics for component in metric.split("/")}
    return max((len(merged_results.get(key, [])) for key in components), default=0)


def _metric_values(merged_results: dict, metric: str, num_models: int) -> list[float]:
    """Average composite metric components per report; keep incomplete rows missing."""
    values = []
    for row in range(num_models):
        parts = _metric_components(merged_results, metric, row)
        try:
            numeric_parts = [float(part) for part in parts]
        except (TypeError, ValueError):
            values.append(float("nan"))
        else:
            values.append(sum(numeric_parts) / len(numeric_parts))
    return values


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _generate_barplot(merged_results: dict, output_dir: str, metrics: List[str]):
    metrics = [metric for metric in metrics if _has_metric(merged_results, metric)]
    if not metrics:
        raise ValueError("No requested comparison metrics are present in the result reports")

    plt = _get_pyplot()
    names = merged_results.get("Checkpoint", [])
    num_models = len(names)
    if num_models == 0:
        num_models = _metric_row_count(merged_results, metrics)
        names = [f"Run {index + 1}" for index in range(num_models)]
    if num_models == 0:
        raise ValueError("No checkpoint names or metric values available for comparison")

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    if not palette:
        palette = ["C0"]
    bar_width = 0.8 / len(metrics)
    x_positions = list(range(num_models))

    plt.figure(figsize=(12, 6))
    for idx, metric in enumerate(metrics):
        if "/" in metric:
            values = _metric_values(merged_results, metric, num_models)
        else:
            raw_values = merged_results.get(metric, [])
            values = _numeric_values(raw_values[:num_models])
            values.extend([float("nan")] * (num_models - len(values)))

        offset = (idx - (len(metrics) - 1) / 2) * bar_width
        positions = [x + offset for x in x_positions]
        plt.bar(
            positions,
            values,
            width=bar_width,
            label=metric,
            color=palette[idx % len(palette)],
        )

    plt.xticks(x_positions, names, rotation=45, ha="right")
    plt.xlabel("Checkpoints")
    plt.ylabel("Metric value (%)")
    plt.title("Comparison of metrics across checkpoints")
    plt.grid(axis="y")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_grouped_barplot.png"))
    plt.close()


def _generate_latex_table(merged_results: dict, output_dir: str, metrics: List[str]):
    names = merged_results.get("Checkpoint", [])
    metrics = [metric for metric in metrics if _has_metric(merged_results, metric)]
    headers = [_escape_latex(metric.replace("/", " / ")) for metric in metrics]
    rows = [f"Descriptor & {' & '.join(headers)}\\\\\n\\hline"]

    for index, name in enumerate(names):
        descriptor = str(name).replace("_", "-").removeprefix("rtdetr6(").removesuffix(")")
        values = []
        for metric in metrics:
            parts = _metric_components(merged_results, metric, index)
            values.append(_escape_latex(" / ".join(parts)))
        rows.append(f"{_escape_latex(descriptor)} & {' & '.join(values)}\\\\")

    with open(os.path.join(output_dir, "comparison_table.tex"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")


def compare(result_folders: list[str], output_dir: str):
    """Merge evaluation reports under the given folders and generate comparisons."""
    if not result_folders:
        raise ValueError("At least one result folder is required")

    if output_dir == "":
        output_dir = f"{result_folders[0].rstrip('/')}/comparison_results"

    result_txts = sorted(
        path
        for folder in result_folders
        for path in _recursive_search_results(folder)
    )
    if not result_txts:
        raise FileNotFoundError(
            f"No evaluation_metrics.txt files found in result folders: {result_folders}"
        )

    merged_results = _merge_data([_read_results(path) for path in result_txts])
    plot_metrics = [
        "total_mAP", "mAP50", "mAP50_95", "f1_50", "f0.25_50",
        "line_AP@0.10", "line_mAP",
    ]
    if not any(_has_metric(merged_results, metric) for metric in plot_metrics):
        raise ValueError(
            "Evaluation reports contain no comparable metrics "
            f"(supported metrics: {', '.join(plot_metrics)})"
        )

    os.makedirs(output_dir, exist_ok=True)
    table_metrics = [
        "total_mAP", "mAP50/mAP50_95", "f1_50/f0.25_50", "line_AP@0.10/line_mAP",
    ]

    _generate_barplot(merged_results, output_dir, plot_metrics)
    _generate_latex_table(merged_results, output_dir, table_metrics)
    print("Comparison results saved to:", output_dir)
