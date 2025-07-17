import jax
import jax.numpy as jnp

DISK_ELEC_RADIUS = 5.0 #um, not used here since point electrode simulation

def disk_elec_stim_potential_mV_per_uA(elec_radius_um: jnp.ndarray, extra_resistivity_ohm_cm: jnp.ndarray, 
                                       electrode_locations_um: jnp.ndarray, point_locations_um: jnp.ndarray) -> jnp.ndarray:
    """
    Compute disk electrode potential using analytical solution.
    """
    assert elec_radius_um.size == 1 and extra_resistivity_ohm_cm.size == 1,\
        "elec_radius_um and extra_resistivity_ohm_cm must be scalars"

    # Assert all electrode z-coordinates are 0
    assert jnp.allclose(electrode_locations_um[:, 2], 0.0), "All electrode z-coordinates must be 0"
    assert jnp.all(point_locations_um[:, 2] < 0.0), "All point locations must be below the z=0 plane"
    
    # z offset (distance below z=0 plane)
    z_off = -1 * point_locations_um[:, 2:3]  # shape: (n_points, 1)
    # radial distance in xy plane
    elec_xy = electrode_locations_um[:, :2]  # shape: (n_electrodes, 2)
    point_xy = point_locations_um[:, :2]     # shape: (n_points, 2)
    
    # Compute pairwise distances in xy plane
    r_off = jnp.sqrt(jnp.sum((elec_xy[:, None, :] - point_xy[None, :, :])**2, axis=2))  # shape: (n_electrodes, n_points)
    
    # Constant multiplier
    constant_multiplier = 1e1 * extra_resistivity_ohm_cm / (2 * jnp.pi * elec_radius_um)  # kohm
    
    # Distances from outer and inner edges of disk
    from_outer_dist = jnp.sqrt((r_off - elec_radius_um)**2 + z_off.T**2)  # shape: (n_electrodes, n_points)
    from_inner_dist = jnp.sqrt((r_off + elec_radius_um)**2 + z_off.T**2)  # shape: (n_electrodes, n_points)
    
    # Transfer impedance using analytical solution
    transfer_impedance = jnp.arcsin(
        2 * elec_radius_um / (from_outer_dist + from_inner_dist)
    )
    return constant_multiplier * transfer_impedance  # shape: (n_electrodes, n_points)