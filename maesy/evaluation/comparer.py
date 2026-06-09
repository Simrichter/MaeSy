import os
import matplotlib.pyplot as plt

def _recursive_search_results(folder: str, search_name: str = "evaluation_metrics.txt", mode: str="strict") -> list[str]:
    """
        Recursively descend into subfolders of all given folders to find evaluation result .txt-files

        Args:
            :param folder: Folder path to recursively search
            :param search_name: Name of evaluation result file. Default is 'evaluation_metrics.txt'
            :param mode: Search mode. Choice of ['strict', 'contains']. 'strict' means that file name must exactly match search_name, 'contains' means that search_name must be a substring of the file name. Default is 'strict'

        Returns list of absolute paths to found files
    """
    results_txts = []
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)):
            if (search_name == f and mode=="strict") or (search_name in f and mode=="contains"):
                results_txts.append(os.path.abspath(os.path.join(folder, f)))
        elif os.path.isdir(os.path.join(folder, f)):
            results_txts.extend(_recursive_search_results(os.path.join(folder, f), search_name, mode))
    return results_txts

def _order_data_by_mtime(files: list[str]) -> list[str]:
    return sorted(files, key=lambda x: os.path.getmtime(x))

def _read_results(file_path: str) -> dict:
    with open(file_path, "r") as f:
        lines = f.readlines()
        data_dict = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) != 2:
                continue
            k, v = parts[0].strip(), parts[1].strip()
            data_dict[k] = v
        return data_dict

def _merge_data(data_list: list[dict]) -> dict:
    merged_data = {}
    for data in data_list:
        for k, v in data.items():
            if k not in merged_data:
                merged_data[k] = []
            merged_data[k].append(v)
    return merged_data

def _generate_barplot(merged_results: dict, output_dir: str, metric: str):
    data = [float(x) for x in merged_results[metric]]
    names = merged_results["Checkpoint"]
    if len(data) != len(names):
        print(f"Warning: Inconsistent data lengths between {metric} and Checkpoint names")
        names = [f"Run {i+1}" for i in range(len(merged_results[metric]))]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(merged_results[metric])), data, color='blue')
    plt.xticks(range(len(merged_results[metric])), names, rotation=45, ha="right")
    plt.xlabel("Checkpoints")
    plt.ylabel(metric)
    plt.title("Comparison of mAP@50 across checkpoints")
    # plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_comparison_barplot.png"))
    plt.close()

expected_content = ["AP50_class_0",
                    "AP50_class_1",
                    "AP50_class_2",
                    "f0.25_50",
                    "f1_50",
                    "line_AP@0.02",
                    "line_AP@0.05",
                    "line_AP@0.10",
                    "line_endpoint_error@0.02",
                    "line_endpoint_error@0.05",
                    "line_endpoint_error@0.10",
                    "line_f1@0.02",
                    "line_f1@0.05",
                    "line_f1@0.10",
                    "line_mAP",
                    "line_precision@0.02",
                    "line_precision@0.05",
                    "line_precision@0.10",
                    "line_recall@0.02",
                    "line_recall@0.05",
                    "line_recall@0.10",
                    "mAP50",
                    "mAP50_95",
                    "num_gt_boxes",
                    "num_gt_lines",
                    "num_pred_boxes",
                    "num_pred_lines",
                    "precision50",
                    "recall50",
                    "total_mAP"
]


def _get_possible_metrics():
    pass
def compare(result_folders: list[str], output_dir: str):
    """
    Recursively detects 'evaluation_metrics.txt' files in the result_folders, merges the result data and generates comparison statistics.
    Main focus lies on plots.

    Parameters
        :param result_folders: List of string paths to result folders created by 'maesy evaluate test' command
        :param output_dir: Directory to save comparison results
    """
    print()
    print("="*60)
    print("Comparing result data")
    print("="*60)
    print()

    if output_dir == "":
        output_dir = f"{result_folders[0].rstrip('/')}/comparison_results"
    os.makedirs(output_dir, exist_ok=True)

    result_txts = []
    for f in result_folders:
        result_txts.extend(_recursive_search_results(f))
    # result_txts = _order_data_by_mtime(result_txts) # Bullshit, creation/moddified time is useless for sorting. Need modified timestamp in result.txt??

    print(f"Found {len(result_txts)} result files")
    merged_results = _merge_data([_read_results(txt) for txt in result_txts])

    print("Generating plots...")
    _generate_barplot(merged_results, output_dir, "mAP50")

    print("Generation finished. Comparison results saved to:", output_dir)
    print("="*60)
