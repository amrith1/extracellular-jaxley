from typing import Dict, Tuple, List
import jax
import jax.numpy as jnp
import numpy as np
import jaxley as jx
import optax
import jax.lax as lax
from jax import jit, vmap, value_and_grad
from lib import HH, get_baseline_triphasic_stimulus
from loss_functions import sodium_peaks, diffusion_peaks, potassium_peaks, ei_widths, pairwise_time_differences

jax.config.update('jax_platform_name', 'cpu')  # Force GPU usage
jax.config.update('jax_enable_x64', True)

SEED = 0

NUM_EPOCHS = 400
SIM_TIME_SAMPLES = 1700
TIME_STEP = 1e-3 #ms
TOTAL_SIM_TIME = SIM_TIME_SAMPLES * TIME_STEP #ms

cell_params_list = [
    'radius', 'HH_gNa', 'HH_gK', 'axial_resistivity'
]
orientation_params_list = [
    'axon_origin_dist', 'axon_theta', 
    'axon_phi', 'axon_spin_angle',  
]

all_params_list = cell_params_list + orientation_params_list

PARAMETER_BOUNDS = {
        'axon_origin_dist': (10.0, 30.0),
        'axon_theta': (-jnp.pi/2, jnp.pi/2),
        'axon_phi': (0.0, jnp.pi),
        'axon_spin_angle': (-jnp.pi/2, jnp.pi/2),
        'radius': (1.0, 5.0),
        'HH_gNa': (0.1, 0.3),
        'HH_gK': (0.1, 0.3),
        'axial_resistivity': (50.0, 200.0)
        }

TOTAL_LENGTH = 2000.0
SEGMENT_LENGTH = 2.0
assert TOTAL_LENGTH % SEGMENT_LENGTH == 0, "TOTAL_LENGTH must be divisible by SEGMENT_LENGTH"
NUM_COMPARTMENTS = int(TOTAL_LENGTH/SEGMENT_LENGTH)

VOLTAGE_CLAMP_SEGMENT = 50
VOLTAGE_CLAMP_VALUE = 0.0

SAVE_NAME = f'seed_{SEED}'
np.random.seed(SEED)
ELEC_SPACING = 300.0
ELECTRODE_CONFIGURATION = 'triangle'
EXTRACELLULAR_RESISTIVITY = 1000 #ohm-cm
MEM_CAPACITANCE = 1.0 #uF/cm^2
ELEC_COORDS = None


if ELECTRODE_CONFIGURATION == 'triangle':
    #place electrodes at equilateral traiangle in x=0 plane, center at (0,0,0), spaced ELEC_SPACING apart
    ELEC_COORDS = np.array([[0, 0, 0], [0, 1, 0], [0, 0.5, 0.866]])
    ELEC_COORDS = ELEC_COORDS - np.mean(ELEC_COORDS, axis=0)
    ELEC_COORDS = ELEC_SPACING*ELEC_COORDS

    # Add (0,0,0) electrode to the configuration
    ELEC_COORDS = np.vstack([np.array([0, 0, 0]), ELEC_COORDS])

    ELEC_COORDS = jnp.array(ELEC_COORDS, dtype=jnp.float32)
    ELEC_COORDS = jax.device_put(ELEC_COORDS) # Move to default accelerator (GPU if available)


#checkpoint_levels = 1
#time_points = TOTAL_SIM_TIME // TIME_STEP + 2
#checkpoints = [int(np.ceil(time_points**(1/checkpoint_levels))) for _ in range(checkpoint_levels)]

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


def point_source_stim_potential_mV_per_uA(extra_resistivity_ohm_cm: jnp.ndarray, 
                                       source_locations_um: jnp.ndarray, record_locations_um: jnp.ndarray) -> jnp.ndarray:
    assert extra_resistivity_ohm_cm.size == 1, "extra_resistivity_ohm_cm must be scalars"
    record_source_distances_cm = jnp.linalg.norm(record_locations_um[:, None, :] - source_locations_um[None, :, :], axis=2) * 1e-4 # convert to cm
    return 1e-3 * extra_resistivity_ohm_cm / (4 * jnp.pi * record_source_distances_cm) # shape: (n_sources, n_records)

    
class StraightAxon:
    def __init__(self):
        self.PARAM_BOUNDS = PARAMETER_BOUNDS

        self.params = {}
        for param_name in all_params_list:
            bounds = PARAMETER_BOUNDS[param_name]
            self.params[param_name] = jnp.array([(bounds[0] + bounds[1])/2])

        self.ei_cell, self.adjacency_matrix = self.build_cell()

        assert jnp.allclose(self.adjacency_matrix, self.adjacency_matrix.T), "Adjacency matrix must be symmetric"
        assert jnp.allclose(jnp.diag(self.adjacency_matrix), 0), "Diagonal of adjacency matrix must be zero"

        ei_initial_stimulus_voltage = jnp.array(
            [0]*VOLTAGE_CLAMP_SEGMENT + [-70.0]*(NUM_COMPARTMENTS - VOLTAGE_CLAMP_SEGMENT)
            )
        self.ei_cell.set("v", ei_initial_stimulus_voltage)
        self.ei_stim_current = jnp.zeros((NUM_COMPARTMENTS, SIM_TIME_SAMPLES))
        self.ei_cell.stimulate(self.ei_stim_current)
        self.ei_cell_record_values = ["v", "i_HH"]
        for record_value in self.ei_cell_record_values:
            self.ei_cell.record(record_value)
        
        self.estim_cell, _ = self.build_cell()
        self.estim_cell_record_values = ["v", "HH_m", "HH_h", "HH_n"]
        for record_value in self.estim_cell_record_values:
            self.estim_cell.record(record_value)

        self.electrode_locations_um = ELEC_COORDS
        self.extra_resistivity_ohm_cm = jnp.array([EXTRACELLULAR_RESISTIVITY])

        self.triphasic_stimulus = get_baseline_triphasic_stimulus(
            t_pre=200e-3, t_max=2500e-3, time_step=TIME_STEP)

        self.jitted_loss = jit(self.loss)
        self.jitted_grad = jit(value_and_grad(self.loss, argnums=0))
        self.jitted_predict_ei = jit(self.predict_ei)
        self.jitted_xtra_estim = jit(self.extracellular_triphasic_150us_stim_multielec)


    def build_cell(self):
        """Build a cell with the given parameters."""
        cell = jx.Cell([jx.Branch([jx.Compartment()]*NUM_COMPARTMENTS)], parents=[-1])
        cell.insert(HH())
        cell.set("HH_gLeak", 1e-3)    # Leak conductance in S/cm^2
        cell.set("HH_eNa", 60.60)     # Sodium reversal potential in mV
        cell.set("HH_eK", -101.34)     # Potassium reversal potential in mV
        cell.set("HH_eLeak", -70.0)  # Leak reversal potential in mV
        cell.set("length", SEGMENT_LENGTH)
        cell.set("HH_gNa", 0.2)
        cell.set("HH_gK", 0.2)
        cell.set("axial_resistivity", 125.0)
        cell.set("radius", 3.0)
        cell.set("capacitance", MEM_CAPACITANCE)
        cell.set("length", SEGMENT_LENGTH)
        cell.set("capacitance", MEM_CAPACITANCE)
        cell.set("HH_m", 0.0353)
        cell.set("HH_h", 0.9054)
        cell.set("HH_n", 0.0677)
        cell.set("v", -70.0)

        for param_name in cell_params_list:
            cell.make_trainable(param_name)

        adjacency_matrix = jnp.zeros((NUM_COMPARTMENTS, NUM_COMPARTMENTS)).astype(jnp.float32)
        # Set adjacent compartments to 1 in adjacency matrix
        for i in range(NUM_COMPARTMENTS-1):
            adjacency_matrix = adjacency_matrix.at[i,i+1].set(1.0)
            adjacency_matrix = adjacency_matrix.at[i+1,i].set(1.0)

        return cell, adjacency_matrix


    def predict_ei(self, params):
        """
        Predict extracellular potentials using the current parameters.
        """
        cell_params = [{param_name: params[param_name]} for param_name in cell_params_list]

        compartment_locations = compute_compartment_locations_from_orientations(params)

        outputs = jx.integrate(self.ei_cell, delta_t=TIME_STEP, params=cell_params, voltage_solver='jaxley.stone').reshape(
            len(self.ei_cell_record_values), NUM_COMPARTMENTS, -1)
        
        _v, _i_HH = outputs[0, :, :], outputs[1, :, :]
        capacitive_current = MEM_CAPACITANCE * jnp.diff(_v, axis=1) / (TIME_STEP * 1e3) #convert to mA/cm^2
        mem_current_per_area = _i_HH[:, 0:-1] + capacitive_current #mA/cm^2
        mem_voltage = _v[:, 0:-1]
        
        compartment_surface_area = jax.device_put(2 * jnp.pi * params["radius"] * jnp.array(self.ei_cell.nodes["length"])) #um^2
        mem_current_uA = 1e-5 * mem_current_per_area * compartment_surface_area[:, None]# still (num_compartments, num_timepoints)
        compartment_locations = compute_compartment_locations_from_orientations(params)
        current_to_eap_matrix = point_source_stim_potential_mV_per_uA(
            extra_resistivity_ohm_cm=self.extra_resistivity_ohm_cm, 
            source_locations_um=compartment_locations, 
            record_locations_um=ELEC_COORDS
            ) #shape: (n_electrodes, n_compartments)

        eap_mV = current_to_eap_matrix @ mem_current_uA
        
        #time should be the first dimension
        return eap_mV.transpose(), mem_current_per_area.transpose(), mem_voltage.transpose()
    

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

    #def loss_fn(self, predicted_ei):
    #    return jnp.sum(sodium_peaks(predicted_ei)) + jnp.sum(diffusion_peaks(predicted_ei)) + jnp.sum(potassium_peaks(predicted_ei))\
    #    + jnp.sum(ei_widths(predicted_ei, component_one='sodium', component_two='potassium')) \
    #        + jnp.sum(pairwise_time_differences(predicted_ei, component='sodium'))

    def loss_fn(self, predicted_ei):
        #upsampled_ei = jax.image.resize(predicted_ei, (predicted_ei.shape[0]*2, predicted_ei.shape[1]), method='lanczos3')
        sodium_time_diffs, _ = pairwise_time_differences(predicted_ei, component='sodium')
        return sodium_time_diffs[1,2]

    def loss(self, opt_params):
        """
        Computes a loss that is robust to temporal shifts by using a
        differentiable soft-alignment mechanism.
        """
        transformed_params = self.sigmoid_transform_parameters(opt_params)
        predicted_ei, _, _ = self.jitted_predict_ei(transformed_params)
        final_loss = self.loss_fn(predicted_ei)

        return final_loss


    def extracellular_triphasic_150us_stim_multielec(self, params: Dict[str, jnp.ndarray], electrode_intensities_uA: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute extracellular triphasic stimulus potential using analytical solution.
        """

        compartment_locations_um = compute_compartment_locations_from_orientations(params)
        n_compartments = compartment_locations_um.shape[0]

        extracellular_potential_mV_per_elec = point_source_stim_potential_mV_per_uA(
            extra_resistivity_ohm_cm=self.extra_resistivity_ohm_cm, 
            source_locations_um=self.electrode_locations_um, 
            record_locations_um=compartment_locations_um
            ) #shape: (n_compartments, n_elecs)
        
        xtra_potential_comps_mV = extracellular_potential_mV_per_elec @ electrode_intensities_uA
        axial_resistance_compartments = params['axial_resistivity'] * (jnp.array(self.estim_cell.nodes["length"]) / (jnp.pi * params['radius']**2)) * 1e-2 # Mohm

        comp_potential_diffs = xtra_potential_comps_mV[None, :] - xtra_potential_comps_mV[:, None]
        inter_comp_axial_resistance = (axial_resistance_compartments[:, None] + axial_resistance_compartments[None, :]) / 2.0
        
        intra_comp_currents = (comp_potential_diffs / inter_comp_axial_resistance) * self.adjacency_matrix
        currents_compartments = jnp.sum(intra_comp_currents, axis=1)
        current_time_series = self.triphasic_stimulus[None, :] * currents_compartments[:, None]
        data_stimuli = self.estim_cell.data_stimulate(current_time_series)
        
        cell_params = [{param_name: params[param_name]} for param_name in cell_params_list]
        outputs = jx.integrate(self.estim_cell, delta_t=TIME_STEP, params=cell_params, voltage_solver='jaxley.stone', data_stimuli=data_stimuli).reshape(
            len(self.estim_cell_record_values), NUM_COMPARTMENTS, -1)
        
        v, m, h, n = outputs[0, :, :], outputs[1, :, :], outputs[2, :, :], outputs[3, :, :]
        return v, m, h, n

    def train(self, num_epochs=NUM_EPOCHS, learning_rate=0.01):
        """
        Simplified training without Jaxley's ParamTransform.
        """
        self.data_point = None
        
        # Combine all parameters into one dictionary
        current_params = dict(self.params)
    
        # Transform to optimization space
        opt_params = self.inverse_sigmoid_transform_parameters(current_params)
        
        # Initialize optimizer
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(opt_params)
            
        epoch_losses = []
        # Import time module for timing epochs
        import time
        epoch_times = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch} started")
            start_time = time.time()
            # Compute loss and gradients
            loss_val, gradients = self.jitted_grad(opt_params, self.data_point)
            print(gradients)
            
            # Check for NaN
            if jnp.isnan(loss_val):
                raise ValueError(f"NaN loss detected at epoch {epoch}")
            
            # Update parameters
            updates, opt_state = optimizer.update(gradients, opt_state)
            opt_params = optax.apply_updates(opt_params, updates)
            
            if epoch % 10 == 0:
                # Compute gradient norm for monitoring
                grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(gradients)))
                print(f"Epoch {epoch}, Loss: {loss_val:.6f}, Grad Norm: {grad_norm:.6f}")
            
            epoch_losses.append(float(loss_val))

            end_time = time.time()
            print(f"Epoch {epoch} took {end_time - start_time:.2f} seconds")
            epoch_times.append(end_time - start_time)
            print(f"Average epoch time: {np.mean(epoch_times):.2f} seconds")
            print(epoch_times)
            print('epoch 0 took', epoch_times[0], 'seconds')
        # Get final parameters
        final_params = self.sigmoid_transform_parameters(opt_params)
    
        # Update stored parameters
        self.params = final_params
        
        return final_params, epoch_losses

    def generate_random_gt_params_ei(self, seed=None):
        PERCENTILE_SAMPLE_AND_CLIP = 0.85
        if seed is not None:
            np.random.seed(seed)
        #generate a random model within the bounds to produce the physical ei
        ground_truth_model_params = {}
        for param in PARAMETER_BOUNDS.keys():
            mean = (PARAMETER_BOUNDS[param][0] + PARAMETER_BOUNDS[param][1]) / 2.0
            max_min_range = PARAMETER_BOUNDS[param][1] - PARAMETER_BOUNDS[param][0]
            CLIP_COEFF = np.abs(PERCENTILE_SAMPLE_AND_CLIP - 0.5)
            sampled_val = np.random.uniform(mean - CLIP_COEFF * max_min_range, mean + CLIP_COEFF * max_min_range)
            ground_truth_model_params[param] = jnp.array([sampled_val])

        ground_truth_model_params['total_length_um'] = jnp.array([TOTAL_LENGTH])
        ground_truth_model_params['segment_length_um'] = jnp.array([SEGMENT_LENGTH])
        ground_truth_model_params = {**PARAMETER_BOUNDS, **ground_truth_model_params}

        gt_ei, _, _ = self.jitted_predict_ei(ground_truth_model_params)
        
        return ground_truth_model_params, gt_ei

        