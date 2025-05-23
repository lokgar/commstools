# commstools/utils/io.py
import numpy as np
import jax.numpy as jnp
import yaml
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def save_data_and_metadata(
    data_filepath: Union[str, Path],
    data: Any,
    metadata: Optional[Dict[str, Any]] = None,
    metadata_suffix: str = ".meta.yaml",
    overwrite: bool = False,
) -> None:
    """
    Saves data (e.g., NumPy/JAX array) to a file and optionally saves associated metadata to a YAML file.

    Args:
        data_filepath: Path to save the main data file (e.g., 'path/to/signal.npy').
        data: The data to save. If it's a JAX array, it will be converted to a NumPy array.
        metadata: Optional dictionary containing metadata to save.
        metadata_suffix: Suffix for the metadata file (e.g., '.meta.yaml', '.metadata.json').
                         The metadata filename will be data_filepath_base + metadata_suffix.
        overwrite: If True, overwrite existing files. Otherwise, raise FileExistsError.
    """
    data_filepath = Path(data_filepath)
    data_filepath.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and data_filepath.exists():
        raise FileExistsError(
            f"Data file already exists: {data_filepath}. Set overwrite=True to replace."
        )

    # Convert JAX array to NumPy array for saving, as np.save is more standard for .npy
    if hasattr(data, "__jax_array__"):  # Check if it's a JAX array
        data_to_save = np.asarray(data)
    else:
        data_to_save = data

    try:
        np.save(data_filepath, data_to_save)
        logger.info(f"Saved data to {data_filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {data_filepath}: {e}")
        raise

    if metadata:
        # Construct metadata filename by replacing/appending suffix to data_filepath
        # e.g., signal.npy -> signal.meta.yaml
        metadata_filename = data_filepath.stem + metadata_suffix
        metadata_filepath = data_filepath.with_name(metadata_filename)

        if not overwrite and metadata_filepath.exists():
            raise FileExistsError(
                f"Metadata file already exists: {metadata_filepath}. Set overwrite=True to replace."
            )

        try:
            with open(metadata_filepath, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved metadata to {metadata_filepath}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {metadata_filepath}: {e}")
            # Decide if this should also raise, or just warn
            # raise


def load_data_and_metadata(
    data_filepath: Union[str, Path],
    metadata_suffix: str = ".meta.yaml",
    load_as_jax: bool = False,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Loads data from a file and optionally loads associated metadata from a YAML file.

    Args:
        data_filepath: Path to the main data file.
        metadata_suffix: Suffix of the metadata file.
        load_as_jax: If True, convert loaded NumPy array to JAX array.

    Returns:
        A tuple (data, metadata). Metadata will be None if not found or not loadable.
    """
    data_filepath = Path(data_filepath)

    if not data_filepath.exists():
        raise FileNotFoundError(f"Data file not found: {data_filepath}")

    try:
        loaded_data = np.load(
            data_filepath, allow_pickle=False
        )  # allow_pickle=False for security
        if load_as_jax:
            loaded_data = jnp.asarray(loaded_data)
        logger.info(f"Loaded data from {data_filepath}")
    except Exception as e:
        logger.error(f"Failed to load data from {data_filepath}: {e}")
        raise

    metadata: Optional[Dict[str, Any]] = None
    metadata_filename = data_filepath.stem + metadata_suffix
    metadata_filepath = data_filepath.with_name(metadata_filename)

    if metadata_filepath.exists():
        try:
            with open(metadata_filepath, "r") as f:
                metadata = yaml.safe_load(f)
            logger.info(f"Loaded metadata from {metadata_filepath}")
        except Exception as e:
            logger.warning(
                f"Failed to load or parse metadata from {metadata_filepath}: {e}. Proceeding without metadata."
            )
            metadata = None  # Ensure metadata is None if loading failed

    return loaded_data, metadata
