import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import hydra
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump
from mlflow.models.signature import infer_signature
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from config.config import MainConfig
from sparse_fault_detect.data.create_windows import load_windows_and_labels
from sparse_fault_detect.models import data_sparsity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

mlflow.set_tracking_uri("http://localhost:8080")


def create_model_from_name(config: MainConfig):
    model_name = config.model.model

    if model_name == "logistic_regression":
        return LogisticRegression(n_jobs=-1, random_state=42)
    elif model_name == "decision_tree_classifier":
        return DecisionTreeClassifier(random_state=42)
    elif model_name == "random_forest_classifier":
        return RandomForestClassifier(n_jobs=-1, random_state=42)
    elif model_name == "extra_tree_classifier":
        return ExtraTreeClassifier(random_state=42)

    else:
        raise ValueError(f"Model name {model_name} not recognized.")


def save_model(model, signature: dict, config: MainConfig) -> None:
    """
    Save the trained model to the specified directory with a timestamped filename.

    Args:
        model: Trained model instance.
        signature (dict): Model signature containing input and output schema.
        config (MainConfig): Configuration object containing the model directory and name.
    """
    model_name = config.model.model
    model_path = os.path.join(hydra.utils.get_original_cwd(), config.model.model_directory)
    os.makedirs(model_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_filepath = os.path.join(model_path, model_filename)

    dump(model, model_filepath)
    logger.info(f"Model saved to {model_filepath}")

    # Log model to MLflow
    mlflow.sklearn.log_model(model, model_name, signature=signature)
    logger.info(f"Model logged to MLflow as '{model_name}'")


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Evaluate the trained model on both the train and test sets and log relevant test metrics to MLflow.

    Args:
        model: Trained model instance.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels for the test set.
        config (MainConfig): Configuration object containing model and dataset settings.
    """

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return precision, recall, f1


def get_sample_ids_and_fault_targets(
    labels: pd.DataFrame, config: MainConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sample IDs and fault targets from the labels DataFrame.

    Args:
        labels (pd.DataFrame): DataFrame containing sample IDs and fault targets.
        config (MainConfig): Configuration object containing the fault target column.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Sample IDs and fault targets as NumPy arrays.

    Raises:
        ValueError: If the fault target column is missing or invalid.
    """
    # Ensure required column exists
    if "sample_id" not in labels.columns:
        raise ValueError("Missing required column: 'sample_id' in labels DataFrame.")

    sample_ids = labels["sample_id"].to_numpy(dtype=int)

    fault_target = config.training.fault_target

    # Ensure fault_target column exists
    if fault_target not in labels.columns:
        raise ValueError(f"Fault target column '{fault_target}' not found in labels DataFrame.")

    y = labels[fault_target]

    # Map fault_target to the correct dtype
    fault_target_types = {
        "fault": int,  # Binary or categorical fault detection
        "fault_target": str,  # Fault line identification (categorical)
        "sc_location": float,  # Fault localization (numerical)
    }

    if fault_target not in fault_target_types:
        raise ValueError(
            f"Invalid fault_target '{fault_target}'. "
            "Expected one of: 'fault', 'fault_target', 'sc_location'."
        )

    y = y.astype(fault_target_types[fault_target])

    if fault_target == "sc_location":
        # Map percentage [0, 100] directly to 10 classes by dividing by 10
        y = (y / 10).astype(int)
        y = np.clip(y, 0, 10)  # Ensure values stay in range

    logger.debug(f"Extracted {len(sample_ids)} samples with fault target '{fault_target}'.")
    y = np.array(y)

    return sample_ids, y


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Recursively flattens a nested dictionary, converting lists and unsupported types.

    Args:
        d (Dict[str, Any]): Input dictionary.
        parent_key (str, optional): Prefix for nested keys.
        sep (str, optional): Separator for nested keys.

    Returns:
        Dict[str, Any]: Flattened dictionary with MLflow-compatible values.
    """
    flattened = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Create hierarchical keys

        if isinstance(v, dict):
            flattened.update(flatten_dict(v, new_key, sep))  # Recursively flatten
        elif isinstance(v, list):
            flattened[new_key] = ",".join(map(str, v))  # Convert lists to comma-separated strings
        elif isinstance(v, (int, float, bool, str)) or v is None:
            flattened[new_key] = v  # Directly store supported types
        else:
            flattened[new_key] = str(v)  # Convert unsupported types to strings

    # remove entries that contain 'directory' in the key
    flattened = {k: v for k, v in flattened.items() if "directory" not in k}

    return flattened


def configure_mlflow_tags(config: MainConfig) -> None:
    """
    Configure MLflow tags based on the selected sub-configurations.

    Args:
        config (MainConfig): Configuration object containing dataset and model settings.
    """
    sub_configs = ["training", "window_extraction", "data_sparsity"]

    for sub_config in sub_configs:
        sub_config_data = getattr(config, sub_config, None)

        if sub_config_data is not None:
            sub_config_dict = OmegaConf.to_container(sub_config_data, resolve=True)
            flattened_config = flatten_dict(sub_config_dict)
            mlflow.log_params(flattened_config)

    logger.info("MLflow parameters logged successfully.")


def log_dataset_params(
    window_shape: Tuple[int, int, int], full_dataset: bool, config: MainConfig
) -> None:
    """
    Log dataset parameters to MLflow.

    Args:
        window_shape (Tuple[int, int, int]): Shape of the windowed dataset.
        full_dataset (bool): Whether the full dataset was used.
        config (MainConfig): Configuration object containing the dataset settings.
    """
    if full_dataset:
        mlflow.set_tag("full_dataset", "True")
    else:
        mlflow.set_tag("full_dataset", "False")
    n_windows, n_timesteps, n_features = window_shape
    mlflow.log_param("n_samples", config.n_samples)
    mlflow.log_param("n_windows", n_windows)
    mlflow.log_param("n_timesteps", n_timesteps)
    mlflow.log_param("n_features", n_timesteps * n_features)
    mlflow.log_param("model", config.model.model)


@hydra.main(version_base=None, config_path="../../config", config_name="main-config.yaml")
def main(config: MainConfig) -> None:
    """
    Main function to load data, preprocess, train a model, evaluate, and save it.
    """
    start_time = time.time()

    mlflow_port = config.cluster.port
    logger.info(f"Using MLflow tracking server on port {mlflow_port}")
    # Ensure MLflow connects to the right server
    mlflow.set_tracking_uri(f"http://127.0.0.1:{mlflow_port}")

    logger.info(f"Starting training with config: {config}")

    # Load preprocessed windows and labels
    windows, labels, full_dataset = load_windows_and_labels(config)

    # Extract sample IDs and labels
    sample_ids, y = get_sample_ids_and_fault_targets(labels, config)

    # Prepare data
    try:
        window_data, new_shape = data_sparsity.apply_sparsity_transform(windows, config)
    except ValueError as e:
        logger.error(f"Invalid data sparsity configuration: {e}")
        return

    with mlflow.start_run():
        # Initialize model
        configure_mlflow_tags(config)
        log_dataset_params(new_shape, full_dataset, config)

        model = create_model_from_name(config)

        grouped_k_fold = GroupKFold(n_splits=config.training.n_splits)
        fold_metrics = []
        fold_models = []
        for i, (train_idx, test_idx) in enumerate(
            grouped_k_fold.split(window_data, y, groups=sample_ids), start=0
        ):
            X_train, X_test = window_data[train_idx], window_data[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Train model
            model_time = time.time()
            model.fit(X_train, y_train)
            logger.info(f"Split {i+1} training time: {time.time()-model_time:.2f} seconds")

            # Evaluate model
            precision, recall, f1 = evaluate_model(model, X_test, y_test)
            fold_metrics.append({"precision": precision, "recall": recall, "f1_score": f1})
            fold_models.append(model)

        # Log all fold results at once
        logger.info("Cross-validation results:")
        for metric in ["precision", "recall", "f1_score"]:
            values = [m[metric] for m in fold_metrics]
            mlflow.log_metric(f"mean_{metric}", np.mean(values))
            mlflow.log_metric(f"std_{metric}", np.std(values))
            logger.info(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        # Get best model from all folds
        best_fold = np.argmax([m["f1_score"] for m in fold_metrics])
        model = fold_models[best_fold]

        signature = infer_signature(X_train, model.predict(X_train))
        # Save trained model
        save_model(model, signature, config=config)

    elapsed_time = time.time() - start_time

    logger.info(f"Execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
