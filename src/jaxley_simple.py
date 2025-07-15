from typing import Dict, Optional, Tuple, List
import jax

jax.config.update('jax_platform_name', 'cpu')  # Force GPU usage
jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
import jaxley as jx
from jaxley.solver_gate import solve_gate_exponential, save_exp
from jaxley.channels import Channel
import jaxley.optimize.transforms as jt
import optax

SEED = 0
NUM_EPOCHS = 400
SIM_TIME_SAMPLES = 500
TIME_STEP = 2e-3 #ms
TOTAL_SIM_TIME = SIM_TIME_SAMPLES * TIME_STEP #ms


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
    location_params: Dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """
    Compute cell compartment locations from orientation parameters.
    """
    axon_spin_angle = location_params['axon_spin_angle']
    axon_origin_dist = location_params['axon_origin_dist']
    phi = location_params['axon_phi']
    theta = location_params['axon_theta']
    
    # Calculate original endpoints before phi and theta rotations using the spin angle
    original_first_x = -1 * TOTAL_LENGTH / 2 * jnp.cos(axon_spin_angle)
    original_first_y = -1 * TOTAL_LENGTH / 2 * jnp.sin(axon_spin_angle)
    original_first_z = axon_origin_dist
    
    # Transform the endpoints
    ff_x, ff_y, ff_z = transform_point(original_first_x, original_first_y, original_first_z, phi, theta)
    fl_x, fl_y, fl_z = transform_point(-1 * original_first_x, -1 * original_first_y, original_first_z, phi, theta)

    #return stacked interpolation between x,y,z endpoints in one tensor
    return jnp.stack([
        jnp.linspace(ff_x, fl_x, NUM_COMPARTMENTS, endpoint=False),
        jnp.linspace(ff_y, fl_y, NUM_COMPARTMENTS, endpoint=False),
        jnp.linspace(ff_z, fl_z, NUM_COMPARTMENTS, endpoint=False)
    ], axis=1)

def build_cell():
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
    cell.make_trainable("radius")
    cell.make_trainable("HH_gNa")
    cell.make_trainable("HH_gK")
    cell.make_trainable("axial_resistivity")
    return cell

cell = build_cell()
MEM_CAPACITANCE = cell.nodes["capacitance"][0] #uF/cm^2

def compute_membrane_current_density(cell_params: Dict[str, jnp.ndarray]):
    """Compute membrane currents using cell parameters."""
    current = jx.step_current(i_delay=1.0, i_dur=1.0, i_amp=0.0, delta_t=TIME_STEP, t_max=TOTAL_SIM_TIME)
    cell.branch(0).loc(0.0).stimulate(current)
    record_values = ["v", "i_HH"]
    for record_value in record_values:
        cell.record(record_value)

    outputs = jx.integrate(cell, delta_t=TIME_STEP, params=cell_params).reshape(
        len(record_values), NUM_COMPARTMENTS, -1)
    
    #compute transmembrane current at each compartment, capactive current as time derivative of membrane voltage and HH current
    membrane_voltage, hh_current = outputs[0, :, :], outputs[1, :, :]
    capacitive_current = MEM_CAPACITANCE * jnp.diff(membrane_voltage, axis=1) / (TIME_STEP * 1e3) #convert to mA/cm^2
    return capacitive_current + hh_current[:, 1:] #mA/cm^2, total current

def compute_eap(cell_params: Dict[str, jnp.ndarray], location_params: Dict[str, jnp.ndarray]):
    """Compute extracellular action potentials."""
    membrane_current = compute_membrane_current_density(cell_params) #mA/cm^2
    compartment_surface_area = 2 * jnp.pi * cell_params["radius"] * jnp.array(cell.nodes["length"]) #um^2
    compartment_surface_area = jax.device_put(compartment_surface_area)
    total_membrane_current = membrane_current * compartment_surface_area[:, None] * 1e-8 # convert to mA

    compartment_locations = compute_compartment_locations_from_orientations(location_params)

    # Calculate pairwise distances between electrodes and compartments
    elec_expanded = ELEC_COORDS[:, None, :] # shape: (n_electrodes, 1, 3)
    comp_expanded = compartment_locations[None, :, :] # shape: (1, n_compartments, 3)
    squared_diff = (elec_expanded - comp_expanded) ** 2 # shape: (n_electrodes, n_compartments, 3)
    elec_compartment_distances_cm = jnp.sqrt(jnp.sum(squared_diff, axis=2)) * 1e-4 # convert to cm # shape: (n_electrodes, n_compartments)
    
    # Compute 1/(4*pi*r) for each electrode-compartment pair (line source approximation for extracellular potential)
    current_to_eap_matrix = EXTRACELLULAR_CONDUCTIVITY / (4 * jnp.pi * elec_compartment_distances_cm)  # shape: (n_electrodes, n_compartments)
    return current_to_eap_matrix @ total_membrane_current # shape: (n_electrodes, n_timepoints)


cell_params_list = [
    'radius', 'HH_gNa', 'HH_gK', 'axial_resistivity'
]
orientation_params_list = [
    'axon_origin_dist', 'axon_theta', 
    'axon_phi', 'axon_spin_angle',  
]

PARAM_BOUNDS = {
    'axon_origin_dist': (10.0, 30.0),
    'axon_theta': (-jnp.pi/2, jnp.pi/2),
    'axon_phi': (0.0, jnp.pi),
    'axon_spin_angle': (-jnp.pi/2, jnp.pi/2),
    'radius': (1.0, 5.0),
    'HH_gNa': (0.1, 0.3),
    'HH_gK': (0.1, 0.3),
    'axial_resistivity': (50.0, 200.0)
}

cell_params = cell.get_parameters()



#jaxley specific parameter transforms
cell_params_transform = jx.ParamTransform(
    [{c_p: jt.SigmoidTransform(PARAM_BOUNDS[c_p][0], PARAM_BOUNDS[c_p][1])} for c_p in cell_params_list],
)
opt_cell_params = cell_params_transform.inverse(cell_params)

def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def inverse_sigmoid(x):
    return jnp.log(x / (1.0 - x))

def inverse_transform_param(x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    normalized = (x - lower) / (upper - lower)
    return inverse_sigmoid(normalized)

def forward_transform_param(x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    normalized = sigmoid(x)
    return lower + (upper - lower) * normalized

def transform_raw_orientation_param(raw_orientation_param: List[Dict[str, jnp.ndarray]]) -> List[Dict[str, jnp.ndarray]]:
    return [{param_entry.keys()[0]: forward_transform_param(raw_orientation_param[param_entry.keys()[0]], PARAM_BOUNDS[param_entry.keys()[0]][0], PARAM_BOUNDS[param_entry.keys()[0]][1])}
        for param_entry in raw_orientation_param]

def inverse_transform_orientation_param(orientation_params: List[Dict[str, jnp.ndarray]]) -> List[Dict[str, jnp.ndarray]]:
    return [{param_entry.keys()[0]: inverse_transform_param(orientation_params[param_entry.keys()[0]], PARAM_BOUNDS[param_entry.keys()[0]][0], PARAM_BOUNDS[param_entry.keys()[0]][1])}
        for param_entry in orientation_params]

orientation_params = {
    'axon_origin_dist': jnp.array([20.0]),
    'axon_theta': jnp.array([0.0]),
    'axon_phi': jnp.array([jnp.pi/2]),
    'axon_spin_angle': jnp.array([0.0]),
}
opt_orientation_params = inverse_transform_orientation_param(orientation_params)

def loss_function(opt_params):
    # Split optimization parameters into cell and orientation params
    opt_cell_params = opt_params[0:4]
    opt_orientation_params = opt_cell_params[4:]

    cell_params = cell_params_transform.forward(opt_cell_params)
    orientation_params = transform_raw_orientation_param(opt_orientation_params)

    return jnp.sum(compute_eap(cell_params, orientation_params)**2)

loss_function(opt_cell_params + opt_orientation_params)