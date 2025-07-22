from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy.linalg
import optax
from functools import partial

"""
The EI is a 2D array of shape (time, electrodes)
"""

def sodium_peaks(ei:jnp.ndarray) -> jnp.ndarray:
    return jnp.min(ei, axis=0)

def diffusion_peaks(ei: jnp.ndarray) -> jnp.ndarray:
    """
    Find the maximum value that occurs before the minimum value for each electrode.
    AKA capacitive peaks
    """
    # Get the minimum indices for each electrode
    min_indices = jnp.argmin(ei, axis=0)
    # Create a mask for values before the minimum
    time_indices = jnp.arange(ei.shape[0])[:, None]
    before_min_mask = time_indices < min_indices
    # Replace values after minimum with -inf using where
    masked_ei = jnp.where(before_min_mask, ei, -jnp.inf)
    # Get max of masked values
    max_before_min = jnp.max(masked_ei, axis=0)
    return max_before_min

def potassium_peaks(ei:jnp.ndarray) -> jnp.ndarray:
    """
    Find the maximum value that occurs after the minimum value for each electrode.
    """
    min_indices = jnp.argmin(ei, axis=0)
    # Create a mask for values after the minimum
    time_indices = jnp.arange(ei.shape[0])[:, None]
    after_min_mask = time_indices > min_indices
    # Replace values before minimum with -inf using where
    masked_ei = jnp.where(after_min_mask, ei, -jnp.inf)
    # Get max of masked values
    max_after_min = jnp.max(masked_ei, axis=0)
    return max_after_min

def ei_widths(ei, max_time_offset:int) -> jnp.ndarray:
    """
    Compute the width of the EI at each electrode.
    """
    min_indices = jnp.argmin(ei, axis=0)
    time_indices = jnp.arange(ei.shape[0])[:, None]
    too_late_mask = time_indices - max_time_offset > min_indices
    too_early_mask = time_indices + max_time_offset < min_indices

    _first_masked_ei = jnp.where(too_late_mask, 0.0, ei)
    masked_ei = jnp.where(too_early_mask, 0.0, _first_masked_ei)

    squared_time_offsets_from_min = ((time_indices - min_indices)**2).astype(jnp.float32)

    normalized_ei = jnp.square(masked_ei) / jnp.sum(jnp.square(masked_ei))

    return jnp.sqrt(jnp.sum(squared_time_offsets_from_min * normalized_ei, axis=0))



# def pairwise_sodium_peak_time_differences(ei:jnp.ndarray, normalize_beta:float=1.0) -> jnp.ndarray:
#     normalized_ei = jax.nn.softmax(-1 * normalize_beta * ei, axis=0)