import os
import warnings
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Use non-interactive backend for matplotlib
plt.switch_backend("agg")

# Set consistent style for plots
plt.rcParams.update({"text.usetex": True, "font.family": "Modern"})
plt.rcParams.update({"font.size": 12})
# Ignore warnings (all)
warnings.filterwarnings("ignore")


# Default file paths
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
CSV_DIR = os.path.join(REPORTS_DIR, "csv")

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Define parameter and metric columns
PARAM_COLUMNS: Dict[str, List[str]] = {
    "int": [
        "downsampling_factor",
        "bus_failure_id",
        "relay_failure_ids",
        "n_samples",
        "n_features",
        "n_timesteps",
        "n_windows",
    ],
    "float": [
        "window_length",
        "zeroing_duration_s",
        "period_of_interest",
        "step_length_seconds",
        "test_size",
        "time_diff",
    ],
    "bool": ["full_dataset", "current_loss", "voltage_loss"],
    "str": ["fault_target", "model"],
}

METRIC_COLUMNS: Dict[str, List[str]] = {"float": ["mean_f1_score", "precision", "recall"]}

DEFAULT_PARAMS: Dict[str, Any] = {
    "downsampling_factor": 1,
    "bus_failure_id": 0,
    "relay_failure_ids": 0,
    "zeroing_duration_s": 0.0,
    "current_loss": False,
    "voltage_loss": False,
    "phase_failure_id": "None",
}


def convert_datatypes(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the datatypes of columns in the DataFrame based on predefined types.

    Args:
        runs (pd.DataFrame): DataFrame containing MLflow run data.

    Returns:
        pd.DataFrame: The DataFrame with converted datatypes.
    """
    # Filter to params.n_samples=1000
    runs = runs[runs["params.n_samples"] == "4000"]

    runs.columns = (
        runs.columns.str.replace("params.", "")
        .str.replace("metrics.", "")
        .str.replace("tags.", "")
    )

    for col in PARAM_COLUMNS.get("int", []):
        if col in runs:
            runs[col] = pd.to_numeric(runs[col], errors="coerce").fillna(0).astype(int)

    for col in PARAM_COLUMNS.get("float", []):
        if col in runs:
            runs[col] = pd.to_numeric(runs[col], errors="coerce").astype(float)

    for col in PARAM_COLUMNS.get("bool", []):
        if col in runs:
            runs[col] = runs[col].astype(str).str.lower() == "true"

    for col in PARAM_COLUMNS.get("str", []):
        if col in runs:
            runs[col] = runs[col].astype(str)

    for col in METRIC_COLUMNS.get("float", []):
        if col in runs:
            runs[col] = pd.to_numeric(runs[col], errors="coerce").astype(float)

    output_path = os.path.join(CSV_DIR, "runs.csv")
    runs.to_csv(output_path, index=False, sep=";", decimal=",")
    print(f"Saved processed runs data to {output_path}")

    return runs


def create_heatmap_data(
    runs: pd.DataFrame, index_col: str, column_col: str, value_col: str, filt_col: str
) -> pd.DataFrame:
    """
    Create a pivot table for heatmap visualization.

    Args:
        runs (pd.DataFrame): DataFrame with MLflow run data.
        index_col (str): Column for the pivot table index.
        column_col (str): Column for pivot table columns.
        value_col (str): Column containing values to aggregate.
        filt_col (str): Filter value for 'fault_target'.

    Returns:
        pd.DataFrame: Pivot table for heatmap visualization.
    """
    runs_filtered = runs[runs["fault_target"] == filt_col]

    for key, value in DEFAULT_PARAMS.items():
        if key != column_col:
            runs_filtered = runs_filtered[runs_filtered[key] == value]

    return runs_filtered.pivot_table(
        index=index_col, columns=column_col, values=value_col, aggfunc="mean"
    )


def format_func(val: float) -> str:
    """Format heatmap values."""
    return f"{val:.4f}"


def plot_heatmap(data: pd.DataFrame, combo: Dict[str, str]) -> None:
    """
    Generate and save a heatmap.

    Args:
        data (pd.DataFrame): Pivot table for heatmap.
        combo (Dict[str, str]): Combination dictionary for plot parameters.

    Returns:
        None
    """
    filename = f"{combo['index_col']}_{combo['column_col']}_{combo['value_col']}_{combo['fault_filter']}.png"
    filepath = os.path.join(FIGURES_DIR, filename)

    # Modify the dataframe, so for each window the default value is taken as reference. For all other values, the difference to the default value is calculated.
    # This way, the heatmap will show the difference to the default value. Calculate as percentage difference.
    # data = data.apply(
    #     lambda x: (x - x[DEFAULT_PARAMS[combo["column_col"]]])
    #     / x[DEFAULT_PARAMS[combo["column_col"]]],
    #     axis=1,
    # )

    plt.figure(figsize=(12, 5.5))
    ax = sns.heatmap(
        data,
        cmap="viridis",
        annot=data.map(format_func),
        fmt="",
        vmin=0.2,
    )
    ax.set_ylabel("Window Length (ms)")
    ax.set_xlabel(combo["column_col"].replace("_", " ").title())
    plt.title(combo["title"])

    plt.savefig(filepath)
    # Save as .svg for better quality
    plt.savefig(filepath.replace(".png", ".svg"), format="svg")
    plt.close()
    print(f"Saved heatmap plot to {filepath}")


def generate_combinations(
    index_columns: List[str],
    column_columns: List[str],
    value_columns: List[str],
    filters: List[str],
) -> List[Dict[str, str]]:
    """
    Generate combinations of parameters for heatmap plotting.

    Args:
        index_columns (List[str]): List of index columns.
        column_columns (List[str]): List of column columns.
        value_columns (List[str]): List of value columns.
        filters (List[str]): List of filters.

    Returns:
        List[Dict[str, str]]: List of parameter combinations.
    """
    return [
        {
            "index_col": idx_col,
            "column_col": col_col,
            "value_col": val_col,
            "fault_filter": filt,
            "title": f"{val_col} Heatmap for {idx_col} vs {col_col} filtered by {filt}",
        }
        for idx_col in index_columns
        for col_col in column_columns
        for val_col in value_columns
        for filt in filters
    ]


def main() -> None:
    """Main execution function."""
    runs = mlflow.search_runs()
    runs = convert_datatypes(runs)

    data_sparsity_tasks = [
        "downsampling_factor",
        "zeroing_duration_s",
        "current_loss",
        "voltage_loss",
        "bus_failure_id",
        "relay_failure_ids",
        "phase_failure_id",
    ]

    combinations = generate_combinations(
        ["window_length"],
        data_sparsity_tasks,
        ["mean_f1_score"],
        ["fault_target", "fault"],
    )
    for combo in combinations:
        data = create_heatmap_data(
            runs,
            combo["index_col"],
            combo["column_col"],
            combo["value_col"],
            combo["fault_filter"],
        )
        plot_heatmap(data, combo)


if __name__ == "__main__":
    main()
