from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy.linalg
import optax
from functools import partial


@partial(jit, static_argnums=(1,))
def compute_ei_width(ei, max_time_offset: int):
    """
    Compute the width of the EI signal around the minimum for each electrode.
    """
    min_ei, min_index = jnp.min(ei, axis=0), jnp.argmin(ei, axis=0)
    time_offset_squared = jnp.square(jnp.arange(-max_time_offset, max_time_offset+1, dtype=jnp.float32))
    
    def compute_single_width(elec_idx):
        start_idx = min_index[elec_idx] - max_time_offset
        end_idx = min_index[elec_idx] + max_time_offset + 1
        
        # Use dynamic_slice for safe indexing
        relevant_ei = jax.lax.dynamic_slice(ei[:, elec_idx], (start_idx,), (2*max_time_offset+1,))
        
        # Handle case where slice goes out of bounds
        relevant_ei = jnp.where(
            (start_idx >= 0) & (end_idx <= ei.shape[0]),
            relevant_ei,
            jnp.zeros_like(relevant_ei)
        )
        
        normalized_ei = jnp.square(relevant_ei) / jnp.sum(jnp.square(relevant_ei))
        return jnp.sqrt(jnp.sum(time_offset_squared * normalized_ei))
    
    width = jax.vmap(compute_single_width)(jnp.arange(ei.shape[1]))
    return width

@partial(jit, static_argnums=(2,))
def ei_width_loss(model_ei, real_ei, max_time_offset: int):
    """
    Functional version of EI width loss.
    
    Args:
        model_ei: Model EI tensor
        real_ei: Real EI tensor  
        max_time_offset: Maximum time offset for width computation
    
    Returns:
        Mean squared error between model and real EI widths
    """
    model_ei_width = compute_ei_width(model_ei, max_time_offset)
    real_ei_width = compute_ei_width(real_ei, max_time_offset)
    return jnp.mean(jnp.square(model_ei_width - real_ei_width))

@jit
def sodium_peak_loss(model_ei, real_ei):
    """
    Compute the L2 norm between the lowest values of model EI and real EI on each electrode.
    
    Args:
        model_ei: Model EI tensor
        real_ei: Real EI tensor
    
    Returns:
        Mean squared error between minimum values across electrodes
    """
    min_model_ei = jnp.min(model_ei, axis=0)
    min_real_ei = jnp.min(real_ei, axis=0)
    return jnp.mean(jnp.square(min_model_ei - min_real_ei))

@jit
def get_diffusion_peaks(ei):
    """
    Robust version that handles edge cases better using dynamic slicing.
    """
    min_ei, min_index = jnp.min(ei, axis=0), jnp.argmin(ei, axis=0)
    
    def compute_single_peak(elec_idx):
        # Use dynamic slice to get data before minimum
        slice_size = min_index[elec_idx]
        before_min = jax.lax.dynamic_slice(ei[:, elec_idx], (0,), (slice_size,))
        
        # Handle case where there's no data before minimum
        return jnp.where(
            slice_size > 0,
            jnp.max(before_min),
            jnp.float32(-jnp.inf)  # or ei[0, elec_idx] for first value
        )
    
    diffusion_peaks = jax.vmap(compute_single_peak)(jnp.arange(ei.shape[1]))
    
    return diffusion_peaks

@jit
def diffusion_peak_loss(model_ei, real_ei):
    """
    Compute the L2 loss between diffusion peaks of model and real EI.
    
    Args:
        model_ei: Model EI tensor
        real_ei: Real EI tensor
    
    Returns:
        Mean squared error between diffusion peaks
    """
    model_diffusion_peaks = get_diffusion_peaks(model_ei)
    real_diffusion_peaks = get_diffusion_peaks(real_ei)
    return jnp.mean(jnp.square(model_diffusion_peaks - real_diffusion_peaks))



@jit
def get_potassium_peak(ei):
    """
    Robust version that handles edge cases better using dynamic slicing.
    """
    min_ei, min_index = jnp.min(ei, axis=0), jnp.argmin(ei, axis=0)
    
    def compute_single_peak(elec_idx):
        # Calculate slice parameters
        start_idx = min_index[elec_idx]
        slice_size = ei.shape[0] - start_idx
        
        # Use dynamic slice to get data after minimum
        after_min = jax.lax.dynamic_slice(ei[:, elec_idx], (start_idx,), (slice_size,))
        
        # Handle case where there's no data after minimum (shouldn't happen with slice_size >= 1)
        return jnp.where(
            slice_size > 0,
            jnp.max(after_min),
            ei[start_idx, elec_idx]  # fallback to the minimum value
        )
    
    potassium_peaks = jax.vmap(compute_single_peak)(jnp.arange(ei.shape[1]))
    
    return potassium_peaks


@jit
def potassium_peak_loss(model_ei, real_ei):
    """
    Compute the L2 loss between diffusion peaks of model and real EI.
    
    Args:
        model_ei: Model EI tensor
        real_ei: Real EI tensor
    
    Returns:
        Mean squared error between diffusion peaks
    """
    model_potassium_peaks = get_potassium_peak(model_ei)
    real_potassium_peaks = get_potassium_peak(real_ei)
    return jnp.mean(jnp.square(model_diffusion_peaks - real_diffusion_peaks))

@partial(jit, static_argnums=(1, 2))
def pairwise_sodium_peak_time_differences(ei, time_step_ms: float, normalize_beta: float = 1.0):
    """
    Vectorized version using vmap for better performance.
    """
    normalized_ei = jax.nn.softmax(-1 * normalize_beta * ei, axis=0)
    offset_array = jnp.arange(-ei.shape[0] + 1, ei.shape[0], dtype=ei.dtype)
    
    def compute_single_cross_correlation(signal1, signal2):
        cross_correlation = jnp.correlate(signal1, signal2, mode='full')
        return jnp.dot(offset_array, cross_correlation)
    
    # Create all pairs of signals
    # This creates a (n_elec, n_elec, time) tensor
    signals_i = normalized_ei[:, :, None]  # (time, elec_i, 1)
    signals_j = normalized_ei[:, None, :]  # (time, 1, elec_j)
    
    # Vectorized cross-correlation computation
    vmap_correlate = jax.vmap(jax.vmap(compute_single_cross_correlation, in_axes=(2, None)), in_axes=(None, 2))
    all_correlations = vmap_correlate(signals_i, signals_j)
    
    # Create upper triangular mask
    n_elec = ei.shape[1]
    mask = jnp.triu(jnp.ones((n_elec, n_elec)), k=1)
    
    return all_correlations * mask * time_step_ms

@partial(jit, static_argnums=(1, 2))
def pairwise_diffusion_peak_time_differences(ei, time_step_ms: float, normalize_beta: float = 1.0):
    """
    Compute pairwise time differences between diffusion peaks using cross-correlation.
    
    Args:
        ei: EI tensor of shape (time, electrodes)
        time_step_ms: Time step in milliseconds
        normalize_beta: Normalization parameter for softmax
    
    Returns:
        Pairwise time differences matrix
    """
    # Softmax will be largest at the diffusion peak since its positive (no -1 multiplier)
    normalized_ei = jax.nn.softmax(normalize_beta * ei, axis=0)
    
    # Offset array for time differences
    offset_array = jnp.arange(-ei.shape[0] + 1, ei.shape[0], dtype=ei.dtype)
    
    def compute_single_cross_correlation(signal1, signal2):
        cross_correlation = jnp.correlate(signal1, signal2, mode='full')
        return jnp.dot(offset_array, cross_correlation)
    
    # Create all pairs of signals
    # This creates a (n_elec, n_elec, time) tensor
    signals_i = normalized_ei[:, :, None]  # (time, elec_i, 1)
    signals_j = normalized_ei[:, None, :]  # (time, 1, elec_j)
    
    # Vectorized cross-correlation computation
    vmap_correlate = jax.vmap(jax.vmap(compute_single_cross_correlation, in_axes=(2, None)), in_axes=(None, 2))
    all_correlations = vmap_correlate(signals_i, signals_j)
    
    # Create upper triangular mask (only compute for elec_two > elec_one)
    n_elec = ei.shape[1]
    mask = jnp.triu(jnp.ones((n_elec, n_elec)), k=1)
    
    return all_correlations * mask * time_step_ms


@partial(jit, static_argnums=(2, 3))
def ei_loss_cable_model(model_ei, real_ei, model_seg_distances, initial_seg_distances, 
                       continuity_beta: float, auto_gain: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Functional version of EI Loss Cable Model.
    
    Args:
        model_ei: Model EI tensor (time, channels)
        real_ei: Real EI tensor (time, channels)
        model_seg_distances: Model segment distances
        initial_seg_distances: Initial segment distances
        continuity_beta: Beta parameter for continuity loss weighting
        auto_gain: Whether to use automatic gain adjustment
        
    Returns:
        Tuple of (total_loss, ei_mse_loss, continuity_loss, align_index, aligned_gain_factor)
    """
    num_channels = model_ei.shape[1]
    num_offsets = model_ei.shape[0] + real_ei.shape[0] - 1
    total_samples = real_ei.shape[0] * real_ei.shape[1]

    model_ei_energy = jnp.sum(jnp.square(model_ei))
    real_ei_energy = jnp.sum(jnp.square(real_ei))

    # Pad the model_ei signal
    padding = ((real_ei.shape[0] - 1, real_ei.shape[0] - 1), (0, 0))
    padded_signal = jnp.pad(model_ei, padding, mode='constant', constant_values=0)

    # Vectorized correlation computation
    def compute_correlation_at_offset(offset):
        return jnp.sum(padded_signal[offset:offset+real_ei.shape[0], :] * real_ei)
    
    cross_correlation = jax.vmap(compute_correlation_at_offset)(jnp.arange(num_offsets))
    
    if auto_gain:
        gain_factors = cross_correlation / model_ei_energy 
    else:
        gain_factors = jnp.ones_like(cross_correlation)
    
    aligned_se_loss = (-2 * cross_correlation * gain_factors + 
                      (model_ei_energy * jnp.square(gain_factors)) + real_ei_energy)
    
    align_index = jnp.argmin(aligned_se_loss)
    aligned_gain_factor = gain_factors[align_index]
    
    # Recompute aligned MSE loss
    aligned_model = padded_signal[align_index:align_index+real_ei.shape[0], :]
    ei_mse_loss = jnp.sum(jnp.square(aligned_model - real_ei)) / total_samples

    continuity_loss = jnp.mean(jnp.square(model_seg_distances - initial_seg_distances))
    total_loss = ei_mse_loss + continuity_beta * continuity_loss
    
    return total_loss, ei_mse_loss, continuity_loss, align_index, aligned_gain_factor

