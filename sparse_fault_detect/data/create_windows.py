import logging
import math
import os
import re
from time import time
from typing import Generator, List, Tuple

import hydra
import natsort
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import MainConfig
from sparse_fault_detect import constants

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def generate_paths(config: MainConfig) -> Tuple[str, str]:
    output_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        config.dataset.data_directory,
        config.window_extraction.windows_directory,
    )
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    if config.full_dataset:
        naming_ext = ""
    else:
        naming_ext = f"_{config.n_samples}"
    X_path = os.path.join(
        output_dir,
        f"X_{config.window_extraction.window_length}_{config.window_extraction.step_length_seconds}{naming_ext}.npz",
    )
    y_path = os.path.join(
        output_dir,
        f"y_{config.window_extraction.window_length}_{config.window_extraction.step_length_seconds}{naming_ext}.npz",
    )

    return X_path, y_path


def create_sliding_windows(
    dataframe: pd.DataFrame,
    labels_series: pd.Series,
    config: MainConfig,
) -> List[pd.DataFrame]:
    """
    Creates sliding windows from a DataFrame based on the specified window length and overlap in seconds.

    Parameters:
        dataframe (pd.DataFrame): Input data.
        labels_series (pd.Series): Series containing event start and end times.
        config (MainConfig): Configuration object with necessary parameters.

    Returns:
        List[pd.DataFrame]: List of DataFrames, each representing a sliding window.
    """
    # Extract configuration parameters
    window_length_seconds = config.window_extraction.window_length
    step_length_seconds = config.window_extraction.step_length_seconds
    time_diff = config.window_extraction.time_diff
    period_of_interest = config.window_extraction.period_of_interest

    # Validate input parameters
    if time_diff <= 0:
        raise ValueError("time_diff must be greater than zero.")
    if window_length_seconds <= 0:
        raise ValueError("window_length_seconds must be greater than zero.")
    if step_length_seconds <= 0:
        raise ValueError("step_size must be greater than zero.")
    if period_of_interest <= 0:
        raise ValueError("period_of_interest must be greater than zero.")

    # Extract event start and end times from labels
    t_evnt_start = float(labels_series["t_evnt_start"])
    t_evnt_end = float(labels_series["t_evnt_end"])

    # Define the time interval of interest
    start_time = t_evnt_start - period_of_interest
    end_time = t_evnt_start + period_of_interest

    logger.debug(f"Event start: {t_evnt_start:.4f}, Event end: {t_evnt_end:.4f}")
    logger.debug(f"Interval start: {start_time:.4f} to {end_time:.4f}")

    # Filter the dataframe based on the event's time window
    dataframe["timestamp"] = dataframe["timestamp"].astype(np.float64)
    dataframe_filtered = dataframe[
        (dataframe["timestamp"] >= start_time) & (dataframe["timestamp"] <= end_time)
    ].reset_index(drop=True)

    logger.debug(f"Filtered dataframe size: {len(dataframe_filtered)} rows")

    # Convert window length and overlap to row indices based on time_diff
    window_size = math.ceil(window_length_seconds / time_diff)
    step_length = math.ceil(step_length_seconds / time_diff)
    if window_size <= step_length:
        step_length = window_size // 2

    logger.debug(f"Window size: {window_size}, Step length: {step_length}, Time diff: {time_diff}")

    # Initialize sliding window creation
    windows = []
    current_index = 0
    window_count = 0
    num_rows = len(dataframe_filtered)

    # Loop to create sliding windows
    while current_index + window_size <= num_rows:
        window = dataframe_filtered.iloc[current_index : current_index + window_size]

        if not window.empty:
            windows.append(window.reset_index(drop=True))
            window_count += 1
            logger.debug(
                f"Window {window_count}: Start index = {current_index}, End index = {current_index + window_size - 1}, Rows = {len(window)}"
            )
        else:
            logger.warning(
                f"Window {window_count + 1}: No data found between index {current_index} and {current_index + window_size - 1}"
            )

        current_index += step_length

    return windows


def assign_window_labels(
    window: pd.DataFrame,
    event_start: float,
    event_end: float,
    delta: float = constants.MIN_DELTA,
) -> int:
    """
    Assigns a label to the window based on the occurrence of an event start or end,

    Args:
        window (pd.DataFrame): A DataFrame representing a single window.
        event_start (float): The start time of the event.
        event_end (float): The end time of the event.
        delta (float): A small margin to adjust event boundaries.

    Returns:
        int: A label indicating if the event's start or end occurs within the window.
    """
    if window.empty:
        logger.debug("Window is empty. Assigning label 0.")
        return constants.LABEL_NO_FAULT_CLASS

    start_window_timestamp = window["timestamp"].iloc[0]
    end_window_timestamp = window["timestamp"].iloc[-1]

    if start_window_timestamp < event_start - delta and end_window_timestamp > event_start + delta:
        logger.debug(
            f"Fault occurs during window. Start window timestamp: {start_window_timestamp:.6f}, "
            f"End window timestamp: {end_window_timestamp:.6f}, Event start: {(event_start- delta):.6f}. "
            "Assigning label 1."
        )
        return constants.LABEL_FAULT_CLASS

    logger.debug(
        f"No fault occurs during window. Start window timestamp: {start_window_timestamp:.6f}, "
        f"End window timestamp: {end_window_timestamp:.6f}, Event start: {(event_start- delta):.6f}. "
        "Assigning label 0."
    )
    return constants.LABEL_NO_FAULT_CLASS


def adjust_windows_and_labels(
    windows: List[pd.DataFrame], labels: List[np.ndarray]
) -> Tuple[List[pd.DataFrame], List[np.ndarray]]:
    """
    Adjusts the windows and corresponding labels to ensure consistent shapes.

    Args:
        windows (List[pd.DataFrame]): List of window DataFrames.
        labels (List[np.ndarray]): List of labels for each window.

    Returns:
        Tuple[List[pd.DataFrame], List[np.ndarray]]: Adjusted windows and labels.
    """
    if not windows or not labels:
        raise ValueError("The list of windows and labels cannot be empty")

    window_shapes = [window.shape for window in windows]
    most_common_shape = max(set(window_shapes), key=window_shapes.count)

    valid_indices = [idx for idx, shape in enumerate(window_shapes) if shape == most_common_shape]

    removed_count = len(windows) - len(valid_indices)

    if removed_count > 0:
        logging.warning(f"Removed {removed_count} windows due to inconsistent shapes.")

    adjusted_windows = [windows[idx] for idx in valid_indices]
    adjusted_labels = [labels[idx] for idx in valid_indices]

    return adjusted_windows, adjusted_labels


def extract_windows_from_dataframe(
    dataframe: pd.DataFrame,
    labels_series: pd.Series,
    sample_id: int,
    config: MainConfig,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extracts sliding windows and assigns labels for a given DataFrame and label.

    Args:
        dataframe (pd.DataFrame): The DataFrame to extract windows from.
        labels_series (pd.Series): A Series containing event information.
        sample_id (int): The sample ID.
        window_length (float): The length of each window.
        overlap_time (float): The overlap between windows.
        period_of_interest (float): The period of interest around the event.
        time_diff (float): The time difference between samples

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: Arrays of features and corresponding labels.
    """

    # Extract event information from labels_df
    t_evnt_start = float(labels_series["t_evnt_start"])
    t_evnt_end = float(labels_series["t_evnt_end"])
    fault_target = str(labels_series["fault_target"])
    sc_location = float(labels_series["sc_location"])
    phase_select = int(labels_series["phase_select"])

    # Create sliding windows from the filtered DataFrame
    windows = create_sliding_windows(
        dataframe=dataframe,
        labels_series=labels_series,
        config=config,
    )

    # Prepare the feature array from the windows
    window_data = np.array([window.drop(columns=["timestamp"]).values for window in windows])

    # Assign labels to each window based on event times
    labels = [assign_window_labels(window, t_evnt_start, t_evnt_end) for window in windows]

    # Debugging information for unique labels
    logger.debug(f"Unique labels in windows: {np.unique(labels)}")

    window_start_times = [np.float64(window["timestamp"][0]) for window in windows]

    # Convert labels to numpy array
    fault_labels = np.array(labels, dtype=int)

    # Create a DataFrame for the labels
    window_label_dataframe = pd.DataFrame(
        {
            "fault": fault_labels,
            "fault_target": fault_target,
            "sc_location": sc_location,
            "phase_select": phase_select,
            "sample_id": sample_id,
            "start_time": window_start_times,
        }
    )

    # Set appropriate data types for the DataFrame columns
    window_label_dataframe = window_label_dataframe.astype(
        {"fault_target": "str", "sc_location": "float"}
    )

    # Reset index for the labels DataFrame
    window_label_dataframe.reset_index(drop=True, inplace=True)

    return window_data, window_label_dataframe


def extract_and_label_windows(
    list_of_dfs: List[pd.DataFrame],
    batch_sample_ids: List[int],
    labels_df: pd.DataFrame,
    config: MainConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the data by extracting windows and assigning labels.

    Args:
        list_of_dfs (List[pd.DataFrame]): List of DataFrames to process.
        labels_df (pd.DataFrame): DataFrame containing labels.
        window_length (float): The length of each window.
        overlap (float): The overlap between windows.
        period_of_interest (float): The period of interest around the event.
        time_diff (float): The time difference between samples.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Concatenated arrays of features and labels.
    """

    list_of_window_data, list_of_window_labels = [], []

    for sample_id, df in zip(batch_sample_ids, list_of_dfs):
        window_data, window_labels = extract_windows_from_dataframe(
            dataframe=df,
            labels_series=labels_df.loc[sample_id],
            sample_id=sample_id,
            config=config,
        )
        list_of_window_data.append(window_data)
        list_of_window_labels.append(window_labels)

    window_data = np.concatenate(list_of_window_data, dtype=np.float32)
    window_labels = np.concatenate(list_of_window_labels)

    logger.debug(f"Shape of batch X: {window_data.shape}, Shape of batch y: {window_labels.shape}")

    return window_data, window_labels


def save_windows_and_labels(
    window_data: np.ndarray, window_labels: np.ndarray, config: MainConfig, batch_id: int = 0
) -> None:
    """
    Save the preprocessed data to the specified directory.

    Args:
        X (np.ndarray): The feature array.
        y (np.ndarray): The label array.
        config (MainConfig): Configuration object with necessary parameters.
    """
    output_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        config.dataset.data_directory,
        config.window_extraction.windows_directory,
    )

    # Add relevant paramters to the file name
    window_data_filename = f"X_batch{batch_id}_{config.window_extraction.window_length}_{config.window_extraction.step_length_seconds}.npy"
    window_labels_filename = f"y_batch{batch_id}_{config.window_extraction.window_length}_{config.window_extraction.step_length_seconds}.npy"

    try:
        np.save(os.path.join(output_dir, window_data_filename), window_data)
        np.save(os.path.join(output_dir, window_labels_filename), window_labels)

        logger.debug("Batch saved successfully.")

    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise


def combine_windows_and_labels(config: MainConfig) -> None:
    """
    Combine the saved windows and labels into a single file.
    """

    output_dir = os.path.join(
        hydra.utils.get_original_cwd(),
        config.dataset.data_directory,
        config.window_extraction.windows_directory,
    )

    if not os.path.exists(output_dir):
        logger.error(f"Output directory does not exist: {output_dir}")
        return

    window_files = os.listdir(output_dir)

    # Extract values from config
    window_length = config.window_extraction.window_length
    step_length_seconds = config.window_extraction.step_length_seconds

    # Construct regex pattern dynamically based on config
    pattern = re.compile(rf"^X_batch\d+_{window_length}_{step_length_seconds}(?:\.\d+)?\.npy$")

    # Filter files based on the exact expected pattern
    selected_window_files = [file for file in window_files if pattern.match(file)]

    if not selected_window_files:
        logger.warning("No matching window files found. Skipping combination.")
        return

    # Limit log output for large datasets
    preview_files = selected_window_files[:5]
    logger.info(
        f"Selected {len(selected_window_files)} window files. Preview: {preview_files} ..."
    )

    X_list = []
    y_list = []

    for file in selected_window_files:
        X_filepath = os.path.join(output_dir, file)
        y_filepath = os.path.join(output_dir, file.replace("X", "y"))

        if not os.path.exists(y_filepath):
            logger.error(f"Missing label file for {file}. Skipping this batch.")
            continue  # Skip this batch if the corresponding y file is missing

        X_batch = np.load(X_filepath, allow_pickle=True)
        y_batch = np.load(y_filepath, allow_pickle=True)

        X_list.append(X_batch)
        y_list.append(y_batch)

        # Remove processed batch files to free memory
        os.remove(X_filepath)
        os.remove(y_filepath)

        logger.debug(f"Processed {file}: X shape {X_batch.shape}, y shape {y_batch.shape}")

    if not X_list:
        logger.error("No valid batches were processed. Exiting function.")
        return

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    X_path, y_path = generate_paths(config)

    np.savez_compressed(X_path, X=X)
    np.savez_compressed(y_path, y=y)
    logger.info(f"Shape of combined X: {X.shape}, Shape of combined y: {y.shape}")

    logger.info("Windows and labels combined and saved successfully.")


def load_windows_and_labels(config: MainConfig) -> Tuple[np.ndarray, pd.DataFrame, bool]:
    """
    Load the saved windows and labels.

    Args:
        config (MainConfig): Configuration object with necessary parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray, bool]: The feature array, label array, and a flag indicating if the full dataset was loaded.
    """
    load_time = time()
    logger.info("Loading windows and labels...")

    X_path, y_path = generate_paths(config)

    logger.debug(f"Loading X from {X_path} and y from {y_path}")

    X = np.load(X_path, allow_pickle=True)["X"]
    y = np.load(y_path, allow_pickle=True)["y"]

    X = np.array(X, dtype=np.float32)

    y = pd.DataFrame(
        y,
        columns=[
            "fault",
            "fault_target",
            "sc_location",
            "phase_select",
            "sample_id",
            "start_time",
        ],
    )
    # Set appropriate data types for the DataFrame columns
    y = y.astype({"fault_target": "str", "sc_location": "float"})
    y.reset_index(drop=True, inplace=True)

    logger.debug(f"Original X: {X.shape}, Original y: {y.shape}")

    if config.training.fault_target == "fault_target":
        # We only want to keep samples with faults, so we filter out the samples without faults
        y = y[y["fault"] == 1]
        X = X[y.index]

    if len(X) != len(y):
        raise ValueError(f"Number of samples in X ({len(X)}) and y ({len(y)}) do not match.")

    # Check if there are samples left
    if len(X) == 0:
        raise ValueError("No samples left after filtering.")

    logger.info(
        f"Loaded X: {X.shape}, y: {y.shape}. Time taken: {time() - load_time:.2f} seconds."
    )

    return X, y, config.full_dataset


def get_labels(config: MainConfig) -> pd.DataFrame:
    """
    Load the labels from the labels.csv file.

    Args:
        config (MainConfig): Configuration object with necessary parameters.

    Returns:
        pd.DataFrame: DataFrame containing the labels.
    """
    labels_path = os.path.join(
        hydra.utils.get_original_cwd(),
        config.dataset.data_directory,
        "ds_double_line_new_labels.csv",
    )
    labels = pd.read_csv(labels_path, sep=";", decimal=",").set_index("rep_id")

    logger.info(f"Loaded labels from {labels_path}")

    return labels


def get_preprocessed_data(
    config: MainConfig, batch_size: int = 500
) -> Generator[Tuple[List[pd.DataFrame], List[int]], None, None]:
    """
    Load the preprocessed data files in batches.

    Args:
        config (MainConfig): Configuration object with necessary parameters.
        batch_size (int): Number of files to load in one batch.

    Yields:
        List[pd.DataFrame]: A batch of DataFrames containing the preprocessed data.
    """
    preprocessed_data_directory = config.dataset.preprocessed_data_directory
    list_of_files = os.listdir(preprocessed_data_directory)
    # Sort files using natural sorting
    list_of_files = natsort.natsorted(list_of_files)
    list_of_sample_ids = [int(file.split("_")[0]) for file in list_of_files]

    # Process data in batches
    for i in range(0, len(list_of_files), batch_size):
        if not config.full_dataset and i >= config.n_samples:
            logger.info(f"Reached the specified number of samples ({config.n_samples}). Stopping.")
            break

        batch_files = list_of_files[i : i + batch_size]
        batch_sample_ids = list_of_sample_ids[i : i + batch_size]

        # If we are near the limit, truncate the batch
        if not config.full_dataset:
            remaining_samples = config.n_samples - i
            batch_files = batch_files[:remaining_samples]
            batch_sample_ids = batch_sample_ids[:remaining_samples]

        batch_dfs = [
            pd.read_pickle(os.path.join(preprocessed_data_directory, file))
            for file in batch_files
            if os.path.isfile(os.path.join(preprocessed_data_directory, file))
        ]

        logger.debug(f"Loaded batch of {len(batch_dfs)} files: {batch_sample_ids}")
        yield batch_dfs, batch_sample_ids


@hydra.main(version_base=None, config_path="../../config", config_name="main-config.yaml")
def create_windows(config: MainConfig) -> None:
    """
    Main function to create windows and save the data.

    Args:
        config (MainConfig): Configuration object with necessary parameters.
    """
    start_time = time()

    X_path, y_path = generate_paths(config)

    if os.path.exists(X_path) and os.path.exists(y_path) and not config.overwrite:
        logger.info("Windows and labels already exist. Skipping creation.")
        return

    # Get the labels and preprocessed data
    labels = get_labels(config)
    batch_size = config.batch_size
    total_files = len(labels)  # Total number of files

    processed_files = 0  # Track the number of files processed

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for batch_id, (batch_dataframes, batch_sample_ids) in enumerate(
            get_preprocessed_data(config, batch_size)
        ):
            # Extract windows and labels
            window_data, window_labels = extract_and_label_windows(
                batch_dataframes, batch_sample_ids, labels, config
            )

            # Save each batch separately
            save_windows_and_labels(window_data, window_labels, config=config, batch_id=batch_id)

            # Update progress bar by the number of files in this batch
            processed_files += len(batch_dataframes)
            pbar.update(len(batch_dataframes))

    # Combine the saved data
    combine_windows_and_labels(config)

    # Log total time taken
    end_time = time()
    logger.info(f"Process completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    create_windows()
