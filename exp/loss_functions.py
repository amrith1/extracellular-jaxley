from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy.linalg
from functools import partial

"""
The EI is a 2D array of shape (time, electrodes)
"""

def upsample_ei(ei:jnp.ndarray, factor:int) -> jnp.ndarray:
    """
    Upsample the EI by a factor of 2
    """
    return jax.image.resize(ei, (ei.shape[0]*factor, ei.shape[1]), 'bilinear')

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

def get_masked_eis(ei:jnp.ndarray, mask_value:float=0.0) -> jnp.ndarray:
    """
    isolates the peaks of the EIs so softmax application returns a bell curve
    """

    #set 0 all data left of the diffusion peak and right of the potassium peak
    
    sodium_indices = jnp.argmin(ei, axis=0)

    #now calculate the diffusion indices
    time_indices = jnp.arange(ei.shape[0])[:, None]

    before_sodium_mask = time_indices < sodium_indices
    after_sodium_mask = time_indices > sodium_indices

    potassium_ei = jnp.where(before_sodium_mask, mask_value, ei)
    diffusion_ei = jnp.where(after_sodium_mask, mask_value, ei)

    potassium_indices = jnp.argmax(potassium_ei, axis=0)
    diffusion_indices = jnp.argmax(diffusion_ei, axis=0)

    before_diffusion_mask = time_indices < diffusion_indices
    after_potassium_mask = time_indices > potassium_indices

    _int_sodium_ei = jnp.where(before_diffusion_mask, mask_value, ei)
    sodium_ei = jnp.where(after_potassium_mask, mask_value, _int_sodium_ei)

    return sodium_ei, diffusion_ei, potassium_ei

def pairwise_time_differences(ei:jnp.ndarray, peak_value:float=10.0, component:str='sodium') -> jnp.ndarray:
    #normalize ei to a pdf with a spike at the sodium peak (sodium peak is negative)

    sodium_ei, diffusion_ei, potassium_ei = get_masked_eis(ei, mask_value=0.0)

    if component == 'sodium':
        betas = peak_value / jnp.min(sodium_ei, axis=0)
        normalized_ei = jax.nn.softmax(betas[None, :] * sodium_ei, axis=0)
    elif component == 'diffusion':
        betas = peak_value / jnp.max(diffusion_ei, axis=0)
        normalized_ei = jax.nn.softmax(betas[None, :] * diffusion_ei, axis=0)
    elif component == 'potassium':
        betas = peak_value / jnp.max(potassium_ei, axis=0)
        normalized_ei = jax.nn.softmax(betas[None, :] * potassium_ei, axis=0)

    num_electrodes = ei.shape[1]
    time_samples = normalized_ei.shape[0]
    pairwise_time_differences = jnp.zeros((num_electrodes, num_electrodes))
    # Time offset array centered at 0
    time_offsets = (jnp.arange(time_samples*2 -1) - time_samples + 1).astype(jnp.float32)
    
    for i in range(num_electrodes):
        for j in range(i+1, num_electrodes):
            # Compute cross correlation
            cross_corr = jnp.correlate(normalized_ei[:, i], normalized_ei[:, j], mode='full')
            expected_offset = jnp.sum(time_offsets * cross_corr)
            pairwise_time_differences = pairwise_time_differences.at[i,j].set(expected_offset)
            
    return pairwise_time_differences, normalized_ei

def ei_widths(ei:jnp.ndarray, peak_value:float=10.0, component_one:str='sodium', component_two:str='potassium') -> jnp.ndarray:
    sodium_ei, diffusion_ei, potassium_ei = get_masked_eis(ei, mask_value=0.0)

    sodium_betas = peak_value / jnp.min(sodium_ei, axis=0)
    potassium_betas = peak_value / jnp.max(potassium_ei, axis=0)
    diffusion_betas = peak_value / jnp.max(diffusion_ei, axis=0)

    normalized_sodium_ei = jax.nn.softmax(sodium_betas[None, :] * sodium_ei, axis=0)
    normalized_potassium_ei = jax.nn.softmax(potassium_betas[None, :] * potassium_ei, axis=0)
    normalized_diffusion_ei = jax.nn.softmax(diffusion_betas[None, :] * diffusion_ei, axis=0)

    if component_one == 'sodium':
        normalized_ei_one = normalized_sodium_ei
    elif component_one == 'diffusion':
        normalized_ei_one = normalized_diffusion_ei
    elif component_one == 'potassium':
        normalized_ei_one = normalized_potassium_ei

    if component_two == 'sodium':
        normalized_ei_two = normalized_sodium_ei
    elif component_two == 'diffusion':
        normalized_ei_two = normalized_diffusion_ei
    elif component_two == 'potassium':
        normalized_ei_two = normalized_potassium_ei

    num_electrodes = ei.shape[1]
    time_samples = normalized_ei_one.shape[0]
    time_differences = jnp.zeros((num_electrodes,))
    # Time offset array centered at 0
    time_offsets = (jnp.arange(time_samples*2 -1) - time_samples + 1).astype(jnp.float32)

    for i in range(num_electrodes):
        cross_corr = jnp.correlate(normalized_ei_one[:, i], normalized_ei_two[:, i], mode='full')
        expected_offset = jnp.sum(time_offsets * cross_corr)
        time_differences = time_differences.at[i].set(expected_offset)

    return time_differences
    
    




# def ei_widths(ei, max_time_offset:int) -> jnp.ndarray:
#     """
#     Compute the width of the EI at each electrode.
#     """
#     min_indices = jnp.argmin(ei, axis=0)
#     time_indices = jnp.arange(ei.shape[0])[:, None]
#     too_late_mask = time_indices - max_time_offset > min_indices
#     too_early_mask = time_indices + max_time_offset < min_indices

#     _first_masked_ei = jnp.where(too_late_mask, 0.0, ei)
#     masked_ei = jnp.where(too_early_mask, 0.0, _first_masked_ei)

#     squared_time_offsets_from_min = ((time_indices - min_indices)**2).astype(jnp.float32)

#     normalized_ei = jnp.abs(masked_ei) / jnp.sum(jnp.abs(masked_ei))

#     return jnp.sqrt(jnp.sum(squared_time_offsets_from_min * normalized_ei, axis=0))