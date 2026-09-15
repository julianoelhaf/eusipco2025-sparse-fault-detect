import os
import warnings
from typing import Any, Dict, List, Optional

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


def calculate_parameter_impact(
    runs: pd.DataFrame, target_metric: str = "mean_f1_score", output_dir: str = "reports"
) -> Optional[pd.DataFrame]:
    """
    Calculate the correlation of selected parameters with the target metric
    and analyze the impact of each parameter's values.

    Args:
        runs (pd.DataFrame): DataFrame containing MLflow run data.
        target_metric (str): Metric to analyze correlation and impact (default: "mean_f1_score").
        output_dir (str): Directory to save results (default: "reports").

    Returns:
        Optional[pd.DataFrame]: DataFrame with correlation values, mean, and standard deviation
        of the target metric for each parameter, or None if processing fails.
    """

    # Define the parameters to analyze
    selected_params = [
        "downsampling_factor",
        "bus_failure_id",
        "relay_failure_ids",
        "zeroing_duration_s",
        "current_loss",
        "voltage_loss",
    ]

    # Ensure the target metric exists
    if target_metric not in runs.columns:
        print(f"Target metric '{target_metric}' not found in the DataFrame.")
        return None

    results = []

    for col in selected_params:
        if col in runs.columns:
            correlation = runs[col].corr(runs[target_metric])

            for value in sorted(runs[col].dropna().unique()):
                subset = runs[runs[col] == value][target_metric]
                mean_score = subset.mean()
                std_score = subset.std()

                results.append(
                    {
                        "parameter": col,
                        "value": value,
                        "correlation": correlation,
                        "mean_f1_score": mean_score,
                        "std_f1_score": std_score,
                    }
                )

    if not results:
        print("No valid results found for parameter impact analysis.")
        return None

    # Convert results to DataFrame
    impact_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    csv_path = os.path.join(output_dir, "parameter_impact.csv")
    md_path = os.path.join(output_dir, "parameter_impact.md")

    # Save results to CSV
    impact_df.to_csv(csv_path, index=False, sep=";", decimal=",")
    print(f"Saved parameter impact results to {csv_path}")

    # Save results as Markdown
    markdown_table = impact_df.to_markdown(index=False)
    with open(md_path, "w") as md_file:
        md_file.write("# Parameter Correlation Analysis\n\n")
        md_file.write(markdown_table)
    print(f"Saved parameter impact results to {md_path}")

    return impact_df


def calculate_parameter_correlation(
    runs: pd.DataFrame, target_metric: str = "mean_f1_score"
) -> pd.DataFrame:
    """
    Calculate correlation between parameters and the target metric.

    Args:
        runs (pd.DataFrame): DataFrame with MLflow run data.
        target_metric (str): Target metric for correlation.

    Returns:
        pd.DataFrame: DataFrame with correlation values.
    """
    selected_params = [
        "downsampling_factor",
        "bus_failure_id",
        "relay_failure_ids",
        "zeroing_duration_s",
        "current_loss",
        "voltage_loss",
    ]
    correlation_results = {
        col: runs[col].corr(runs[target_metric]) for col in selected_params if col in runs
    }

    correlation_df = pd.DataFrame(
        list(correlation_results.items()), columns=["parameter", "correlation"]
    )
    correlation_df.to_csv(
        os.path.join(CSV_DIR, "parameter_correlation.csv"), index=False, sep=";", decimal=","
    )

    return correlation_df


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


def create_performance_comparison_table(
    runs: pd.DataFrame,
    sparsity_column: str,
    metrics: List[str] = ["mean_f1_score", "precision", "recall"],
    output_dir: str = "reports",
) -> None:
    """
    Generate a performance comparison table for multiple metrics across different data sparsity scenarios.

    Args:
        runs (pd.DataFrame): DataFrame containing MLflow run data.
        sparsity_column (str): The column that defines the data sparsity type (e.g., downsampling_factor).
        metrics (List[str]): List of metric columns to include in the comparison table.
        output_dir (str): Directory to save the comparison table (default: "reports").

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Filter for relevant columns
    comparison_columns = [sparsity_column] + metrics
    runs_filtered = runs[comparison_columns].dropna(subset=[sparsity_column] + metrics)

    # Group by sparsity level and calculate the mean of each metric
    comparison_table = runs_filtered.groupby(sparsity_column).mean().reset_index()

    # Save the comparison table to CSV
    csv_filename = os.path.join(output_dir, f"performance_comparison_{sparsity_column}.csv")
    comparison_table.to_csv(csv_filename, index=False, sep=";", decimal=",")
    print(f"Saved performance comparison table to {csv_filename}")

    # Save the comparison table as Markdown
    md_filename = os.path.join(output_dir, f"performance_comparison_{sparsity_column}.md")
    markdown_table = comparison_table.to_markdown(index=False)
    with open(md_filename, "w") as md_file:
        md_file.write("# Performance Comparison Table\n\n")
        md_file.write(markdown_table)
    print(f"Saved performance comparison table to {md_filename}")

    # Save the data as a LaTeX table
    latex_filename = os.path.join(output_dir, f"performance_comparison_{sparsity_column}.tex")
    # Fix column names for LaTeX
    comparison_table.columns = [col.replace("_", " ").title() for col in comparison_table.columns]

    # Round everything to 3 decimal places
    comparison_table = comparison_table.round(3)

    # Remove trailing zeros
    comparison_table = comparison_table.applymap(lambda x: f"{x:.3f}".rstrip("0").rstrip("."))

    latex_table = comparison_table.to_latex(index=False, escape=False)
    with open(latex_filename, "w") as latex_file:
        latex_file.write(latex_table)
    print(f"Saved performance comparison table to {latex_filename}")


def plot_f1_vs_data_sparsity(
    runs: pd.DataFrame, sparsity_column: str, output_dir: str = "reports/figures"
) -> None:
    """
    Generate line plots for F1 score vs. different data sparsity levels.

    Args:
        runs (pd.DataFrame): DataFrame containing MLflow run data.
        sparsity_column (str): The column that defines the data sparsity type (e.g., downsampling_factor).
        output_dir (str): Directory to save the plot (default: "reports/figures").

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Filter the columns for plotting
    runs_filtered = runs[["mean_f1_score", sparsity_column]]
    runs_filtered = runs_filtered.dropna(subset=[sparsity_column, "mean_f1_score"])

    # Group by sparsity level and calculate mean F1 score
    f1_sparsity_grouped = (
        runs_filtered.groupby(sparsity_column)["mean_f1_score"].mean().reset_index()
    )

    # Plotting the line plot for F1 score vs. sparsity levels
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=f1_sparsity_grouped, x=sparsity_column, y="mean_f1_score", marker="o", color="b"
    )
    plt.title(f"F1 Score vs. {sparsity_column.replace('_', ' ').title()}")
    plt.xlabel(sparsity_column.replace("_", " ").title())
    plt.ylabel("Mean F1 Score")
    plt.grid(True)

    # Save the plot
    plot_filename = os.path.join(output_dir, f"f1_score_vs_{sparsity_column}.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved line plot for F1 Score vs. {sparsity_column} to {plot_filename}")


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

    # correlation_df = calculate_parameter_correlation(runs)
    # print("\n" + correlation_df.to_string(index=False))

    # impact_df = calculate_parameter_impact(runs)
    # if impact_df is not None:
    #     print("\n" + impact_df.to_string(index=False))

    # # # Assuming the runs DataFrame has been processed
    # plot_f1_vs_data_sparsity(runs, sparsity_column="downsampling_factor")

    # # Create multi-metric performance comparison table
    # for data_sparsity_task in data_sparsity_tasks:
    #     create_performance_comparison_table(runs, sparsity_column=data_sparsity_task)


if __name__ == "__main__":
    main()
