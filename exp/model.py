from typing import Dict, Optional, Tuple, List
import jax

jax.config.update('jax_platform_name', 'gpu')  # Force GPU usage
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import jaxley as jx
from jaxley.solver_gate import solve_gate_exponential, save_exp
from jaxley.channels import Channel
import jaxley.optimize.transforms as jt
import optax
import jax.lax as lax
from jax import jit, vmap, value_and_grad


SEED = 0
NUM_EPOCHS = 400
SIM_TIME_SAMPLES = 500
TIME_STEP = 2e-3 #ms
TOTAL_SIM_TIME = SIM_TIME_SAMPLES * TIME_STEP #ms

cell_params_list = [
    'radius', 'HH_gNa', 'HH_gK', 'axial_resistivity'
]
orientation_params_list = [
    'axon_origin_dist', 'axon_theta', 
    'axon_phi', 'axon_spin_angle',  
]

TOTAL_LENGTH = 1500.0
SEGMENT_LENGTH = 1.0
assert TOTAL_LENGTH % SEGMENT_LENGTH == 0, "TOTAL_LENGTH must be divisible by SEGMENT_LENGTH"
NUM_COMPARTMENTS = int(TOTAL_LENGTH/SEGMENT_LENGTH)

VOLTAGE_CLAMP_SEGMENT = 50
VOLTAGE_CLAMP_VALUE = 0.0

SAVE_NAME = f'seed_{SEED}'
np.random.seed(SEED)
ELEC_SPACING = 30.0
ELECTRODE_CONFIGURATION = 'triangle'
EXTRACELLULAR_CONDUCTIVITY = 1000 #ohm-cm
MEM_CAPACITANCE = 1.0 #uF/cm^2
ELEC_COORDS = None
DISK_ELEC_RADIUS = 5.0 #um, not used here since point electrode simulation

if ELECTRODE_CONFIGURATION == 'triangle':
    #place electrodes at equilateral traiangle in x=0 plane, center at (0,0,0), spaced ELEC_SPACING apart
    ELEC_COORDS = np.array([[0, 0, 0], [0, 1, 0], [0, 0.5, 0.866]])
    ELEC_COORDS = ELEC_COORDS - np.mean(ELEC_COORDS, axis=0)
    ELEC_COORDS = ELEC_SPACING*ELEC_COORDS
    ELEC_COORDS = jnp.array(ELEC_COORDS, dtype=jnp.float32)
    ELEC_COORDS = jax.device_put(ELEC_COORDS) # Move to default accelerator (GPU if available)

def _vtrap(x, y):
    return x / (save_exp(x / y) - 1.0)

class HH(Channel):
    """Hodgkin-Huxley channel."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True

        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 0.12,
            f"{prefix}_gK": 0.036,
            f"{prefix}_gLeak": 0.0003,
            f"{prefix}_eNa": 60.0,
            f"{prefix}_eK": -77.0,
            f"{prefix}_eLeak": -54.3,
        }
        self.channel_states = {
            f"{prefix}_m": 0.2,
            f"{prefix}_h": 0.2,
            f"{prefix}_n": 0.2,
        }
        self.current_name = f"i_HH"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Return updated HH channel state."""
        prefix = self._name
        m, h, n = states[f"{prefix}_m"], states[f"{prefix}_h"], states[f"{prefix}_n"]
        new_m = solve_gate_exponential(m, dt, *self.m_gate(v))
        new_h = solve_gate_exponential(h, dt, *self.h_gate(v))
        new_n = solve_gate_exponential(n, dt, *self.n_gate(v))
        return {f"{prefix}_m": new_m, f"{prefix}_h": new_h, f"{prefix}_n": new_n}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        """Return current through HH channels."""
        prefix = self._name
        m, h, n = states[f"{prefix}_m"], states[f"{prefix}_h"], states[f"{prefix}_n"]

        gNa = params[f"{prefix}_gNa"] * (m**3) * h  # S/cm^2
        gK = params[f"{prefix}_gK"] * n**4  # S/cm^2
        gLeak = params[f"{prefix}_gLeak"]  # S/cm^2

        return (
            gNa * (v - params[f"{prefix}_eNa"])
            + gK * (v - params[f"{prefix}_eK"])
            + gLeak * (v - params[f"{prefix}_eLeak"])
        )

    def init_state(self, states, v, params, delta_t):
        """Initialize the state such at fixed point of gate dynamics."""
        prefix = self._name
        alpha_m, beta_m = self.m_gate(v)
        alpha_h, beta_h = self.h_gate(v)
        alpha_n, beta_n = self.n_gate(v)
        return {
            f"{prefix}_m": alpha_m / (alpha_m + beta_m),
            f"{prefix}_h": alpha_h / (alpha_h + beta_h),
            f"{prefix}_n": alpha_n / (alpha_n + beta_n),
        }

    @staticmethod
    def m_gate(v):
        alpha = 2.725 * _vtrap(-(v + 35), 10)
        beta = 90.83 * save_exp(-(v + 60) / 20)
        return alpha, beta

    @staticmethod
    def h_gate(v):
        alpha = 1.817 * save_exp(-(v + 52) / 20)
        beta = 27.25 / (save_exp(-(v + 22) / 10) + 1)
        return alpha, beta

    @staticmethod
    def n_gate(v):
        alpha = 0.09575 * _vtrap(-(v + 37), 10)
        beta = 1.915 * save_exp(-(v + 47) / 80)
        return alpha, beta

def compute_membrane_current_density(cell,params: Dict[str, jnp.ndarray]):
    """Compute membrane currents using cell parameters."""
    current = jx.step_current(i_delay=1.0, i_dur=1.0, i_amp=0.0, delta_t=TIME_STEP, t_max=TOTAL_SIM_TIME)
    cell.branch(0).loc(0.0).stimulate(current)
    record_values = ["v", "i_HH"]
    for record_value in record_values:
        cell.record(record_value)

    cell_params = [{param_name: params[param_name]} for param_name in cell_params_list]
    outputs = jx.integrate(cell, delta_t=TIME_STEP, params=cell_params).reshape(
        len(record_values), NUM_COMPARTMENTS, -1)
    
    #compute transmembrane current at each compartment, capactive current as time derivative of membrane voltage and HH current
    membrane_voltage, hh_current = outputs[0, :, :], outputs[1, :, :]
    capacitive_current = MEM_CAPACITANCE * jnp.diff(membrane_voltage, axis=1) / (TIME_STEP * 1e3) #convert to mA/cm^2
    return capacitive_current + hh_current[:, 1:] #mA/cm^2, total current

def compute_eap(params: Dict[str, jnp.ndarray], cell):
    """Compute extracellular action potentials."""
    membrane_current = compute_membrane_current_density(cell, params) #mA/cm^2 (num_compartments, num_timepoints)

    compartment_surface_area = 2 * jnp.pi * params["radius"] * jnp.array(cell.nodes["length"]) #um^2
    compartment_surface_area = jax.device_put(compartment_surface_area)
    total_membrane_current = membrane_current * compartment_surface_area[:, None] * 1e-8 # convert to mA, still (num_compartments, num_timepoints)

    compartment_locations = compute_compartment_locations_from_orientations(params)

    #Calculate pairwise distances between electrodes and compartments
    elec_expanded = ELEC_COORDS[:, None, :] # shape: (n_electrodes, 1, 3)
    comp_expanded = compartment_locations[None, :, :] # shape: (1, n_compartments, 3)
    squared_diff = (elec_expanded - comp_expanded) ** 2 # shape: (n_electrodes, n_compartments, 3)
    elec_compartment_distances_cm = jnp.sqrt(jnp.sum(squared_diff, axis=2)) * 1e-4 # convert to cm # shape: (n_electrodes, n_compartments)

    # Compute 1/(4*pi*r) for each electrode-compartment pair (line source approximation for extracellular potential)
    current_to_eap_matrix = EXTRACELLULAR_CONDUCTIVITY / (4 * jnp.pi * elec_compartment_distances_cm)  # shape: (n_electrodes, n_compartments)
    return current_to_eap_matrix @ total_membrane_current # shape: (n_electrodes, n_timepoints)

def transform_point(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray, 
                    phi: jnp.ndarray, theta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Transform a point using phi and theta rotations.
    
    """
    final_z = z * jnp.cos(phi) - x * jnp.sin(phi)
    int_x = x * jnp.cos(phi) + z * jnp.sin(phi)
    final_x = int_x * jnp.cos(theta) - y * jnp.sin(theta)
    final_y = y * jnp.cos(theta) + int_x * jnp.sin(theta)
    return final_x, final_y, final_z

def compute_compartment_locations_from_orientations(
    params: Dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """
    Compute cell compartment locations from orientation parameters.
    """
    axon_spin_angle = params['axon_spin_angle']
    axon_origin_dist = params['axon_origin_dist']
    phi = params['axon_phi']
    theta = params['axon_theta']
    
    # Calculate original endpoints before phi and theta rotations using the spin angle
    original_first_x = -1 * TOTAL_LENGTH / 2 * jnp.cos(axon_spin_angle)
    original_first_y = -1 * TOTAL_LENGTH / 2 * jnp.sin(axon_spin_angle)
    original_first_z = axon_origin_dist
    
    # Transform the endpoints
    ff_x, ff_y, ff_z = transform_point(original_first_x, original_first_y, original_first_z, phi, theta)
    fl_x, fl_y, fl_z = transform_point(-1 * original_first_x, -1 * original_first_y, original_first_z, phi, theta)

    #return stacked interpolation between x,y,z endpoints in one tensor, make it compact(num_compartments, 3)
    return jnp.stack([
        jnp.linspace(ff_x, fl_x, NUM_COMPARTMENTS, endpoint=False),
        jnp.linspace(ff_y, fl_y, NUM_COMPARTMENTS, endpoint=False),
        jnp.linspace(ff_z, fl_z, NUM_COMPARTMENTS, endpoint=False)
    ], axis=1).reshape(-1, 3)

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

def point_source_stim_potential_mV_per_uA(extra_resistivity_ohm_cm: jnp.ndarray, 
                                       electrode_locations_um: jnp.ndarray, point_locations_um: jnp.ndarray) -> jnp.ndarray:
    """
    Compute point source potential using analytical solution.
    """
    assert extra_resistivity_ohm_cm.size == 1,\
        "elec_radius_um and extra_resistivity_ohm_cm must be scalars"
    
    elec_point_distances_cm = jnp.linalg.norm(electrode_locations_um[:, None, :] - point_locations_um[None, :, :], axis=2) * 1e-4 # convert to cm
    return extra_resistivity_ohm_cm / (4 * jnp.pi * elec_point_distances_cm) # shape: (n_electrodes, n_points)

def extracellular_triphasic_150us_stim_multielec(cell, adjacency_matrix: jnp.ndarray, params: Dict[str, jnp.ndarray], electrode_locations_um: jnp.ndarray, \
    electrode_intensities_uA: jnp.ndarray, compartment_locations_um: jnp.ndarray, extra_resistivity_ohm_cm: jnp.ndarray, time_step: float, t_pre: float,t_max: float ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute extracellular triphasic stimulus potential using analytical solution.
    """
    # Get number of electrodes and compartments
    n_electrodes = electrode_locations_um.shape[0]
    n_compartments = compartment_locations_um.shape[0]

    extracellular_potential_mV_per_elec = point_source_stim_potential_mV_per_uA(
        extra_resistivity_ohm_cm=extra_resistivity_ohm_cm, 
        electrode_locations_um=electrode_locations_um, 
        point_locations_um=compartment_locations_um
        ) #shape: (n_electrodes, n_compartments)
    
    extracellular_potential_mV = electrode_intensities_uA @ extracellular_potential_mV_per_elec
    assert extracellular_potential_mV.shape == (n_compartments,), "extracellular potential shape mismatch"

    axial_resistance_compartments = params['axial_resistivity'] * (jnp.array(cell.nodes["length"]) / (jnp.pi * params['radius']**2)) * 1e-2 # Mohm

    # Compute current flowing into each compartment per uA
    # For each connected pair, compute voltage difference and resulting current
    currents_compartments = jnp.zeros(n_compartments)
    connected_indices = jnp.nonzero(adjacency_matrix)
    # Assert adjacency matrix is symmetric
    assert jnp.allclose(adjacency_matrix, adjacency_matrix.T), "Adjacency matrix must be symmetric"
    
    # Assert diagonal is zero 
    assert jnp.allclose(jnp.diag(adjacency_matrix), 0), "Diagonal of adjacency matrix must be zero"

    for i, j in zip(*connected_indices):
        # Get voltage difference between compartments i and j
        v_diff = extracellular_potential_mV[j] - extracellular_potential_mV[i] #current flowing into i 
        # Compute axial resistance between compartments
        r_axial = (axial_resistance_compartments[i] + axial_resistance_compartments[j]) / 2
        # Add current contribution
        currents_compartments = currents_compartments.at[i].add(v_diff / r_axial)  # current (nA) flowing i

    assert 50e-3 % time_step == 0, "50us is not a multiple of time_step"
    assert t_pre % time_step == 0, "t_pre is not a multiple of time_step"
    assert t_max % time_step == 0, "t_max is not a multiple of time_step"

    # Loop through time points to set current time series values
    for t_idx in range(int(t_max / time_step)):
        if t_idx >= int(t_pre / time_step) and t_idx < int((t_pre + 50e-3) / time_step):
            current_time_series = current_time_series.at[:, t_idx].set(currents_compartments * (2/3))
        elif t_idx >= int((t_pre + 50e-3) / time_step) and t_idx < int((t_pre + 100e-3) / time_step):
            current_time_series = current_time_series.at[:, t_idx].set(currents_compartments * (-1))
        elif t_idx >= int((t_pre + 100e-3) / time_step) and t_idx < int((t_pre + 150e-3) / time_step):
            current_time_series = current_time_series.at[:, t_idx].set(currents_compartments * (1/3))

    cell.stimulate(current_time_series)
    record_values = ["v", "HH_m", "HH_h", "HH_n"]
    for record_value in record_values:
        cell.record(record_value)
    
    cell_params = [{param_name: params[param_name]} for param_name in cell_params_list]
    outputs = jx.integrate(cell, delta_t=TIME_STEP, params=cell_params_list).reshape(
        len(record_values), NUM_COMPARTMENTS, -1)
    
    v, m, h, n = outputs[0, :, :], outputs[1, :, :], outputs[2, :, :], outputs[3, :, :]
    return v, m, h, n
    



class SimpleCellEIAndLoss:
    def __init__(self):
        self.cell, self.adjacency_matrix = self.build_cell()
        self.cell.set("length", SEGMENT_LENGTH)
        self.cell.set("capacitance", MEM_CAPACITANCE)
        self.cell.set("axial_resistivity", 125.0)
        self.params = {
        'radius': jnp.array([3.0]),
        'HH_gNa': jnp.array([0.2]),
        'HH_gK': jnp.array([0.2]),
        'axial_resistivity': jnp.array([125.0]),
        'axon_origin_dist': jnp.array([20.0]),
        'axon_theta': jnp.array([0.0]),
        'axon_phi': jnp.array([jnp.pi/2]),
        'axon_spin_angle': jnp.array([0.0])
        }
        self.PARAM_BOUNDS = {
        'axon_origin_dist': (10.0, 30.0),
        'axon_theta': (-jnp.pi/2, jnp.pi/2),
        'axon_phi': (0.0, jnp.pi),
        'axon_spin_angle': (-jnp.pi/2, jnp.pi/2),
        'radius': (1.0, 5.0),
        'HH_gNa': (0.1, 0.3),
        'HH_gK': (0.1, 0.3),
        'axial_resistivity': (50.0, 200.0)
        }
        self.jitted_grad = jit(value_and_grad(self.loss, argnums=0))

    def build_cell(self):
        """Build a cell with the given parameters."""
        cell = jx.Cell([jx.Branch([jx.Compartment()]*NUM_COMPARTMENTS)], parents=[-1])
        cell.insert(HH())
        cell.set("HH_gLeak", 1e-4)    # Leak conductance in S/cm^2
        cell.set("HH_eNa", 60.60)     # Sodium reversal potential in mV
        cell.set("HH_eK", -101.34)     # Potassium reversal potential in mV
        cell.set("HH_eLeak", -64.58)  # Leak reversal potential in mV
        cell.set("HH_m", 0.0353)        # Initial value of m gate
        cell.set("HH_h", 0.9054)        # Initial value of h gate  
        cell.set("HH_n", 0.0677)        # Initial value of n gate
        cell.set("length", SEGMENT_LENGTH)
        cell.set("HH_gNa", 0.2)
        cell.set("HH_gK", 0.2)
        cell.set("axial_resistivity", 125.0)
        cell.set("radius", 3.0)
        cell.set("capacitance", MEM_CAPACITANCE)

        for param_name in cell_params_list:
            cell.make_trainable(param_name)

        adjacency_matrix = jnp.zeros((NUM_COMPARTMENTS, NUM_COMPARTMENTS))
        # Set adjacent compartments to 1 in adjacency matrix
        for i in range(NUM_COMPARTMENTS-1):
            adjacency_matrix = adjacency_matrix.at[i,i+1].set(1)
            adjacency_matrix = adjacency_matrix.at[i+1,i].set(1)

        return cell, adjacency_matrix


    def inverse_sigmoid(self, x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        normalized = (x - lower) / (upper - lower)
        return jnp.log(normalized / (1.0 - normalized))

    def sigmoid(self,x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
        normalized = 1.0 / (1.0 + jnp.exp(-x))
        return lower + (upper - lower) * normalized

    def inverse_sigmoid_transform_parameters(self,params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        return {param_name: self.inverse_sigmoid(param_value, self.PARAM_BOUNDS[param_name][0], self.PARAM_BOUNDS[param_name][1])
                for param_name, param_value in params.items()}

    def sigmoid_transform_parameters(self, params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        return {param_name: self.sigmoid(param_value, self.PARAM_BOUNDS[param_name][0], self.PARAM_BOUNDS[param_name][1])
                for param_name, param_value in params.items()}

    # def loss_function(opt_params):
    #     params = sigmoid_transform_parameters(opt_params)
    #     return jnp.sum(compute_eap(params)**2)

    # loss_grad_fn = jax.jit(jax.value_and_grad(loss_function))

    def predict(self, params):
        """
        Predict extracellular potentials using the current parameters.
        """
        outputs = compute_eap(params, self.cell)

        return outputs

    def loss(self, original_params, true_ei):
        """
        Computes a loss that is robust to temporal shifts by using a
        differentiable soft-alignment mechanism.
        """
        opt_params = self.sigmoid_transform_parameters(original_params)
        predicted_ei = self.predict(opt_params)

        # true_len = true_ei.shape[0]
        # pred_len = predicted_ei.shape[0]
        # n_electrodes = true_ei.shape[1]
        # max_offset = true_len - pred_len

        # def compute_loss_at_offset(offset):
        #     pred_window = lax.dynamic_slice(
        #         predicted_ei,
        #         start_indices=[offset, 0],
        #         slice_sizes=[true_len, n_electrodes]
        #     )

        #     epsilon = 1e-6
        #     pred_norm = (pred_window - jnp.mean(pred_window)) / (jnp.std(pred_window) + epsilon)
        #     true_norm = (true_ei - jnp.mean(true_ei)) / (jnp.std(true_ei) + epsilon)

        #     mse = jnp.mean((pred_norm - true_norm)**2)
        #     return mse

        # offsets = jnp.arange(max_offset + 1)
        # all_mse_losses = vmap(compute_loss_at_offset)(offsets)

        
        # temperature = 1.0
        # weights = jax.nn.softmax(-all_mse_losses * temperature)

        # final_loss = jnp.sum(weights * all_mse_losses)
        final_loss = jnp.sum(predicted_ei**2)

        return final_loss

    def train(self, data_point, num_epochs=NUM_EPOCHS, learning_rate=0.01):
        """
        Simplified training without Jaxley's ParamTransform.
        """
        self.data_point = data_point
        
        # Combine all parameters into one dictionary
        current_params = dict(self.params)
    
        # Transform to optimization space
        opt_params = self.inverse_sigmoid_transform_parameters(current_params)
        
        # Initialize optimizer
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(opt_params)
            
        epoch_losses = []
        
        for epoch in range(num_epochs):
            # Compute loss and gradients
            loss_val, gradients = self.jitted_grad(opt_params, data_point)
            print(gradients)
            
            # Check for NaN
            if jnp.isnan(loss_val):
                print(f"NaN loss detected at epoch {epoch}")
                break
            
            # Update parameters
            updates, opt_state = optimizer.update(gradients, opt_state)
            opt_params = optax.apply_updates(opt_params, updates)
            
            if epoch % 10 == 0:
                # Compute gradient norm for monitoring
                grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(gradients)))
                print(f"Epoch {epoch}, Loss: {loss_val:.6f}, Grad Norm: {grad_norm:.6f}")
            
            epoch_losses.append(float(loss_val))
        # Get final parameters
        final_params = self.sigmoid_transform_parameters(opt_params)
    
        # Update stored parameters
        self.params = final_params
        
        return final_params, epoch_losses






        