import logging
import os
import re
import sys
from typing import Dict, List

import hydra
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(".")

from config.config import MainConfig
from sparse_fault_detect import constants

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def validate_dataframe_structure(df: pd.DataFrame, config: MainConfig) -> bool:
    multiple_of = config.dataset.multiple_of
    additional = config.dataset.additional
    expected_modulo = (len(df.columns) - additional) % multiple_of
    if expected_modulo != 0:
        logger.debug(f"Column count mismatch: {len(df.columns)} columns found.")
        return False
    return True


def check_monotonic(series: pd.Series) -> bool:
    return series.is_monotonic_increasing


def validate_timestamp_column(df: pd.DataFrame, config: MainConfig) -> bool:
    timestamp_col_name = config.dataset.expected_name
    if timestamp_col_name not in df.columns:
        logger.debug(f"Missing timestamp column: {timestamp_col_name}")
        return False

    timestamp_col = df[timestamp_col_name][1:].astype(float, errors="ignore")

    if not check_monotonic(timestamp_col):
        logger.debug(f"Timestamp column is not monotonically increasing.")
        return False

    expected_duration = config.dataset.duration
    if not abs(timestamp_col.max() - expected_duration) < 1e-9:
        logger.debug(f"Timestamp max value mismatch: {timestamp_col.max()}.")
        return False

    expected_count = expected_duration / config.dataset.step
    if not abs(timestamp_col.gt(0).sum() - expected_count) < 1e-9:
        logger.debug(f"Positive timestamp count mismatch.")
        return False

    return True


def validate_measurement_columns(df: pd.DataFrame, config: MainConfig) -> bool:
    regex_pattern = re.compile(config.dataset.regex)
    timestamp_column = config.dataset.expected_name

    for column in df.columns:
        if column == timestamp_column:
            continue
        if not regex_pattern.fullmatch(column):
            logger.debug(f"Column {column} does not match regex.")
            return False

    return True


def validate_dataframe(df: pd.DataFrame, config: MainConfig) -> bool:
    return (
        validate_dataframe_structure(df, config)
        and validate_timestamp_column(df, config)
        and validate_measurement_columns(df, config)
    )


def transform_headers(df: pd.DataFrame, data_config: MainConfig) -> pd.DataFrame:
    timestamp_col_name = data_config.dataset.expected_name
    new_col_names = [data_config.dataset.new_name]

    for col in df.columns:
        if col == timestamp_col_name:
            continue

        try:
            col = col.split("\\")[1].replace("Bus", "Bus_")
            col = col[:5] + "_" + col[5:]

            if "." not in col:
                col += ".0"

            col = col.replace("ai_exp_ct_vt", "")
            old_suffix = "." + col.split(".")[1]

            new_suffix = constants.EMT_LABELS_TO_UNIT.get(old_suffix)

            col = col.replace(old_suffix, new_suffix)

        except IndexError:
            logger.error(f"Failed to transform column: {col}")
            raise ValueError(f"Column '{col}' cannot be split correctly.")

        new_col_names.append(col)

    df.columns = new_col_names
    return df


def transform_dataframe(df: pd.DataFrame, config: MainConfig) -> pd.DataFrame:
    df = transform_headers(df, config)
    df = df.iloc[1:].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    timestamp_col_name = config.dataset.new_name

    # Filter the DataFrame for positive timestamp values
    df = df[df[timestamp_col_name] > 0.0].reset_index(drop=True)

    keyword = "Line"
    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
    df_new = df.loc[:, df.columns.str.contains(keyword)].copy()
    df_new[timestamp_col_name] = df[timestamp_col_name]

    return df_new


def get_output_file_name(file: str, topology: str) -> str:
    base_name = file.split("_")[1].replace(".txt", "")
    return f"{base_name}_sample_{topology}.pkl"


def process_file(input_file_path: str, output_file_path: str, config: MainConfig) -> None:
    df = pd.read_csv(
        input_file_path,
        sep=";",
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )

    if validate_dataframe(df, config):
        transformed_df = transform_dataframe(df, config)
        transformed_df.to_pickle(output_file_path)
    else:
        logger.debug(f"Validation failed for file: {input_file_path}")


def process_files_in_parallel(
    config: MainConfig,
) -> None:
    # get hydra working directory
    working_dir = hydra.utils.get_original_cwd()
    raw_data_directory = os.path.join(
        working_dir, config.dataset.data_directory, config.dataset.raw_data_directory
    )
    preprocessed_data_directory = os.path.join(
        working_dir, config.dataset.data_directory, config.dataset.preprocessed_data_directory
    )
    topology = config.dataset.topology
    files = os.listdir(raw_data_directory)

    n_samples = config.n_samples

    if n_samples > 0:
        files = files[:n_samples]

    for file in tqdm(files):
        input_file_path = os.path.join(raw_data_directory, file)
        output_file_path = os.path.join(
            preprocessed_data_directory, get_output_file_name(file, topology)
        )

        if os.path.exists(output_file_path):
            logger.debug(f"Skipping already processed file: {output_file_path}")
            continue

        process_file(input_file_path, output_file_path, config)


@hydra.main(version_base=None, config_path="../../config", config_name="main-config.yaml")
def main(config: MainConfig) -> None:

    process_files_in_parallel(config)

    logger.info("Completed preprocessing")


if __name__ == "__main__":
    main()
