from typing import List, Tuple
from collections import namedtuple
import torch
import torch.jit as jit
from torch import Tensor
from torch.nn import Parameter, Module
import adapt_neuron.constants as constants
from adapt_neuron.layers import precompute_forward_inputs_from_base_params, \
    forward_na_k_only, membrane_currents_to_electrical_image, get_connected_segment_distances
from adapt_neuron.estim import compute_estim_segment_bias_per_uA


LSTMState = namedtuple("LSTMState", ["v", "m", "h", "n"])

@jit.script
def convert_parameter_to_raw(param, min_val:float, max_val:float):
    assert torch.all(param >= min_val) and torch.all(param <= max_val), "Initial parameter values must be within bounds"
    
    ratio = (param - min_val) / (max_val - min_val)
    raw_val = torch.logit(ratio)

    return raw_val

@jit.script
def sigmoid_reparametrize(raw: Tensor, min_val: float, max_val: float) -> Tensor:
    ratio = torch.sigmoid(raw)


    return min_val + ratio * (max_val - min_val)

class HHCableRNNCellBase(jit.ScriptModule):
    def __init__(self):
        super().__init__()
        #print("Base Cell Initialized")
    
    @jit.script_method
    def get_connected_segment_distances(self):
        """
        Return connected segment distances
        """
        return get_connected_segment_distances(self.seg_locations, self.connected_segment_indices)
    
    @jit.script_method
    def forward(
        self, input_estim_bias: Tensor, state: Tuple[Tensor, Tensor, Tensor, Tensor],
        time_step:float
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        V_seg, m_seg, h_seg, n_seg = state
        trans_current_overtime, next_v, next_m, next_h, next_n = forward_na_k_only(
            V_seg=V_seg, m_seg=m_seg, h_seg=h_seg, n_seg=n_seg,
            seg_gna_ns_bar=self.segment_gna_nS, seg_gk_ns_bar=self.segment_gk_nS, seg_gl_ns=self.segment_gl_nS,
            seg_cap_pF=self.segment_cap_pF, 
            drift_matrix_sparse=self.drift_matrix_sparse, 
            ena=constants.E_NA, ek=constants.E_K, epas=constants.E_PAS,
            time_step=time_step,
            estim_bias_mV = input_estim_bias
            )
        return trans_current_overtime, (next_v, next_m, next_h, next_n)
    
    @jit.script_method
    def mem_currents_to_electrical_image(
        self, mem_currents: Tensor, electrode_locations_um: Tensor
    ) -> Tensor:
        return membrane_currents_to_electrical_image(
            seg_current_pA_overtime=mem_currents, segment_locations_um=self.seg_locations,
            electrode_locations_um=electrode_locations_um,
            extra_resistivity_ohm_cm=self.extra_resistivity,
            adc_count_to_uV=constants.LITKE_ADC_COUNT_TO_uV
            )
    
    @jit.script_method
    def compute_estim_segment_bias_per_uA(
        self, electrode_locations_um: Tensor, time_step:float, 
        elec_radius_um: float=8.0, extra_resistivity_ohm_cm :float = 1000.0 #ohm*cm
    ) -> Tensor:
        self.precompute_forward_inputs(time_step=time_step)
        return compute_estim_segment_bias_per_uA(
            electrode_location_um=electrode_locations_um, segment_locations_um= self.seg_locations,
            elec_radius_um=elec_radius_um, extra_resistivity_ohm_cm=extra_resistivity_ohm_cm,
            A_over_ms_matrix=self.A_over_ms_matrix, adjacency_matrix=self.seg_adjacency_matrix
            )

class HHCableRNNCellSimple(HHCableRNNCellBase):
    def __init__(
        self,
        axon_origin_dist,
        axon_theta,
        axon_phi,
        axon_spin_angle,
        fiber_radius_um,
        sodium_channel_density,
        potassium_channel_density,
        axial_resistivity,
        total_length_um,
        segment_length_um,
        extra_resistivity: float = 1000.0,
        gl_per_area: float = 1e-4,
        axon_origin_dist_bounds: Tuple[float, float] = (0.0, 100.0),
        axon_theta_bounds: Tuple[float, float] = (0.0, torch.pi),
        axon_phi_bounds: Tuple[float, float] = (0.0, 2 * torch.pi),
        axon_spin_angle_bounds: Tuple[float, float] = (0.0, 2 * torch.pi),
        fiber_radius_um_bounds: Tuple[float, float] = (0.1, 10.0),
        sodium_channel_density_bounds: Tuple[float, float] = (0.01, 1.0),
        potassium_channel_density_bounds: Tuple[float, float] = (0.01, 1.0),
        axial_resistivity_bounds: Tuple[float, float] = (50.0, 200.0)
    ):
        super().__init__()
        
        #print("Simple Cell Initialized")
        
        # Store bounds
        self.axon_origin_dist_bounds = axon_origin_dist_bounds
        self.axon_theta_bounds = axon_theta_bounds
        self.axon_phi_bounds = axon_phi_bounds
        self.axon_spin_angle_bounds = axon_spin_angle_bounds
        self.fiber_radius_um_bounds = fiber_radius_um_bounds
        self.sodium_channel_density_bounds = sodium_channel_density_bounds
        self.potassium_channel_density_bounds = potassium_channel_density_bounds
        self.axial_resistivity_bounds = axial_resistivity_bounds

        # Define raw parameters
        self.axon_origin_dist_raw = Parameter(convert_parameter_to_raw(torch.tensor([axon_origin_dist], dtype=torch.float32), *axon_origin_dist_bounds))
        self.axon_theta_raw = Parameter(convert_parameter_to_raw(torch.tensor([axon_theta], dtype=torch.float32), *axon_theta_bounds))
        self.axon_phi_raw = Parameter(convert_parameter_to_raw(torch.tensor([axon_phi], dtype=torch.float32), *axon_phi_bounds))
        self.axon_spin_angle_raw = Parameter(convert_parameter_to_raw(torch.tensor([axon_spin_angle], dtype=torch.float32), *axon_spin_angle_bounds))
        self.fiber_radius_um_raw = Parameter(convert_parameter_to_raw(torch.tensor([fiber_radius_um], dtype=torch.float32), *fiber_radius_um_bounds))
        self.sodium_channel_density_raw = Parameter(convert_parameter_to_raw(torch.tensor([sodium_channel_density], dtype=torch.float32), *sodium_channel_density_bounds))
        self.potassium_channel_density_raw = Parameter(convert_parameter_to_raw(torch.tensor([potassium_channel_density], dtype=torch.float32), *potassium_channel_density_bounds))
        self.axial_resistivity_raw = Parameter(convert_parameter_to_raw(torch.tensor([axial_resistivity], dtype=torch.float32), *axial_resistivity_bounds))
        
        self.total_length_um = float(total_length_um)
        self.num_segments = int(self.total_length_um / segment_length_um)

        self.seg_heights = torch.ones(self.num_segments).cuda() * segment_length_um
        self.seg_gl_per_area_bars = torch.ones(self.num_segments).cuda() * gl_per_area
        self.seg_adjacency_matrix = torch.zeros(self.num_segments, self.num_segments).to(torch.uint8).cuda()
        for i in range(self.num_segments-1):
            self.seg_adjacency_matrix[i, i+1] = 1
            self.seg_adjacency_matrix[i+1, i] = 1
        self.connected_segment_indices = torch.nonzero(self.seg_adjacency_matrix).cuda()
        self.extra_resistivity = torch.tensor(extra_resistivity).cuda()


        self.seg_heights.requires_grad = False
        self.seg_gl_per_area_bars.requires_grad = False
        self.seg_adjacency_matrix.requires_grad = False
        self.connected_segment_indices.requires_grad = False
        self.extra_resistivity.requires_grad = False

        self.axon_origin_dist = torch.empty(0)
        self.axon_theta = torch.empty(0)
        self.axon_phi = torch.empty(0)
        self.axon_spin_angle = torch.empty(0)
        self.fiber_radius_um = torch.empty(0)
        self.sodium_channel_density = torch.empty(0)
        self.potassium_channel_density = torch.empty(0)
        self.axial_resistivity = torch.empty(0)
        
        self.A_over_ms_matrix = torch.empty(0)
        self.segment_gna_nS = torch.empty(0)
        self.segment_gk_nS = torch.empty(0)
        self.segment_gl_nS = torch.empty(0)
        self.segment_cap_pF = torch.empty(0)
        self.drift_matrix_sparse = torch.empty(0)
        
        self.seg_radii = torch.empty(0)
        self.seg_locations = torch.empty(0)
        self.seg_gna_per_area_bars = torch.empty(0)
        self.seg_gk_per_area_bars = torch.empty(0)
        

    
    @jit.script_method
    def transform_point(self, x, y, z):
        
        phi = self.axon_phi[0]
        theta = self.axon_theta[0]
        
        #int_result = (z + 1j*x) * torch.exp(1j*self.axon_phi)
        #int_z = int_result.real
        #int_x = int_result.imag 
        #final_result = (int_x + 1j*y) * torch.exp(1j*self.axon_theta)
        
        final_z = z * torch.cos(phi) - x * torch.sin(phi)
        int_x = x * torch.cos(phi) + z * torch.sin(phi)
        
        final_x = int_x * torch.cos(theta) - y * torch.sin(theta)
        final_y = y * torch.cos(theta) + int_x * torch.sin(theta)
        
        return final_x, final_y, final_z
        
    
    @jit.script_method
    def reparametrize(self):
        # Use sigmoid to map raw parameters to their constrained form
        self.axon_origin_dist = sigmoid_reparametrize(self.axon_origin_dist_raw, *self.axon_origin_dist_bounds)
        self.axon_theta = sigmoid_reparametrize(self.axon_theta_raw, *self.axon_theta_bounds)
        self.axon_phi = sigmoid_reparametrize(self.axon_phi_raw, *self.axon_phi_bounds)
        self.axon_spin_angle = sigmoid_reparametrize(self.axon_spin_angle_raw, *self.axon_spin_angle_bounds)
        self.fiber_radius_um = sigmoid_reparametrize(self.fiber_radius_um_raw, *self.fiber_radius_um_bounds)
        self.sodium_channel_density = sigmoid_reparametrize(self.sodium_channel_density_raw, *self.sodium_channel_density_bounds)
        self.potassium_channel_density = sigmoid_reparametrize(self.potassium_channel_density_raw, *self.potassium_channel_density_bounds)
        self.axial_resistivity = sigmoid_reparametrize(self.axial_resistivity_raw, *self.axial_resistivity_bounds)
        
        self.seg_radii = torch.ones(self.num_segments).cuda() * self.fiber_radius_um
        self.seg_gna_per_area_bars = torch.ones(self.num_segments).cuda() * self.sodium_channel_density
        self.seg_gk_per_area_bars = torch.ones(self.num_segments).cuda() * self.potassium_channel_density
        
        
        #self.seg_locations = torch.ones(self.num_segments, 3).cuda() * self.axon_origin_dist[0] * self.axon_spin_angle[0] * self.axon_phi[0] * self.axon_theta[0]
        #lets see if this fixes the autograd bugs
        
        spin_angle = self.axon_spin_angle[0]
        
        original_first_x = -1 * self.total_length_um/2 * torch.cos(spin_angle)
        original_first_y = -1 * self.total_length_um/2 * torch.sin(spin_angle)
        original_first_z = self.axon_origin_dist[0]
        
        ff_x, ff_y, ff_z = self.transform_point(original_first_x, original_first_y, original_first_z)
        fl_x, fl_y, fl_z = self.transform_point(-1*original_first_x, -1*original_first_y, original_first_z)
        #assert distance between ff and fl is equal to total_length_um
        ff_fl_dist = torch.sqrt((ff_x - fl_x)**2 + (ff_y - fl_y)**2 + (ff_z - fl_z)**2)
        #assert torch.isclose(torch.tensor(ff_fl_dist), torch.tensor(self.total_length_um)), "Distance between ff and fl is not equal to total_length_um"
        #linspace derivative is not implemented, do something else
        x_space = (fl_x - ff_x) / (self.num_segments - 1)
        y_space = (fl_y - ff_y) / (self.num_segments - 1)
        z_space = (fl_z - ff_z) / (self.num_segments - 1)
        
        integer_list = torch.arange(self.num_segments, device=self.seg_gna_per_area_bars.device)
        
        self.seg_locations = torch.stack(
            [ff_x + integer_list * x_space, ff_y + integer_list * y_space, ff_z + integer_list * z_space], 
            dim=1)
    
    @jit.script_method
    def precompute_forward_inputs(self, time_step:float):
        self.reparametrize()
        self.segment_gna_nS, self.segment_gk_nS, self.segment_gl_nS, self.segment_cap_pF, self.drift_matrix_sparse, self.A_over_ms_matrix = \
            precompute_forward_inputs_from_base_params(
                seg_radii=self.seg_radii, seg_heights=self.seg_heights,
                seg_gna_per_area_bars=self.seg_gna_per_area_bars, seg_gk_per_area_bars=self.seg_gk_per_area_bars, seg_gl_per_area_bars=self.seg_gl_per_area_bars,
                membrane_cap_per_area=constants.MEM_CAP_PER_AREA , axial_resistivity=self.axial_resistivity,
                seg_adjacency_list=self.connected_segment_indices,
                time_step=time_step
                ) 





class CableIntegrator(torch.nn.Module):
#class CableIntegrator(Module):   
    def __init__(self, cell, **cell_kwargs):
        super().__init__()
        self.cell = cell(**cell_kwargs)

    #@jit.script_method
    def forward(
        self, input_estim: Tensor, state: Tuple[Tensor, Tensor, Tensor, Tensor], time_step:float, 
        compute_ei:bool = False, electrode_locations_um:Tensor = torch.empty(0), filter_ds_matrix: Tensor = torch.empty(0), num_mask_samples: int = 0
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        
        #the inputs are now the electrical stimulus input 
        inputs_estim_list = input_estim.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        seg_mem_voltages = torch.jit.annotate(List[Tensor], [])
                
        self.cell.precompute_forward_inputs(time_step=time_step)
        
        """RNN Model Implemented Here"""
        for i in range(len(inputs_estim_list)):
            curr_V, curr_m, curr_h, curr_n = state
            seg_mem_voltages += [curr_V]
            curr_mem_current, state = self.cell(
                input_estim_bias=inputs_estim_list[i], 
                state=state, 
                time_step=time_step
            )
            
            outputs += [curr_mem_current]
            
        mem_currents_over_time = torch.stack(outputs, dim=0)
        mem_voltage_over_time = torch.stack(seg_mem_voltages, dim=0) 
        assert torch.isfinite(mem_currents_over_time).all(), "Returned Membrane currents are not finite"        

        if not compute_ei:
            return torch.empty(0), mem_currents_over_time, mem_voltage_over_time, \
                self.cell.get_connected_segment_distances(), state

        filtered_ds_currents = filter_ds_matrix @ mem_currents_over_time[num_mask_samples:, :]
        electrical_image = self.cell.mem_currents_to_electrical_image(filtered_ds_currents, electrode_locations_um)
        
        return electrical_image, mem_currents_over_time, mem_voltage_over_time, \
            self.cell.get_connected_segment_distances(), state


class HHCableRNNCell(HHCableRNNCellBase):
#class HHCableRNNCell(Module):
    def __init__(
        self,
        seg_radii: Tensor,
        seg_heights: Tensor,
        seg_locations: Tensor,
        seg_gna_per_area_bars: Tensor,
        seg_gk_per_area_bars: Tensor,
        seg_gl_per_area_bars: Tensor,
        seg_adjacency_matrix: Tensor,
        seg_labels: List[str], #dendrite, soma, axon
        axial_resistivity: float,
        extra_resistivity: float
    ):
        super().__init__()
       
        #print("Full Cell Initialized")
        
        """Static Parameters First"""
        self.num_segments = seg_radii.size(0)
        self.seg_heights = seg_heights
        self.seg_adjacency_matrix = seg_adjacency_matrix
        self.seg_labels = seg_labels
        self.seg_gl_per_area_bars = seg_gl_per_area_bars
        #given the adjacency matrix and the locations, find initial distances between connected segments
        self.connected_segment_indices = torch.nonzero(seg_adjacency_matrix)
        self.extra_resistivity = torch.tensor(extra_resistivity) #now static parameter since using auto gain adjust
        self.axial_resistivity = torch.tensor(axial_resistivity) #now static parameter since using radius scaling 
        self.A_over_ms_matrix = torch.empty(0)
        self.segment_gna_nS = torch.empty(0)
        self.segment_gk_nS = torch.empty(0)
        self.segment_gl_nS = torch.empty(0)
        self.segment_cap_pF = torch.empty(0)
        self.drift_matrix_sparse = torch.empty(0)
        
        #self.axial_resistivity_raw = Parameter(
        #    convert_parameter_to_raw(torch.tensor(axial_resistivity), 
        #            constants.axial_resistivity_min, constants.axial_resistivity_max)
        #    )
        
        #self.extra_resistivity_raw = Parameter(
        #    convert_parameter_to_raw(torch.tensor(extra_resistivity), 
        #            constants.extra_resistivity_min, constants.extra_resistivity_max)
        #    )
        
        self.seg_radii_raw = Parameter(
            convert_parameter_to_raw(seg_radii, 
                    constants.seg_radii_min, constants.seg_radii_max)
            )

        self.radii_scaling_raw = Parameter(
            convert_parameter_to_raw(torch.tensor(1.0), 
                    constants.radii_scaling_min, constants.radii_scaling_max)
            )

        self.seg_gna_per_area_bars_raw = Parameter(
            convert_parameter_to_raw(seg_gna_per_area_bars, 
                    constants.seg_gna_per_area_min, constants.seg_gna_per_area_max)
            )
        self.seg_gk_per_area_bars_raw = Parameter(
            convert_parameter_to_raw(seg_gk_per_area_bars, 
                    constants.seg_gk_per_area_min, constants.seg_gk_per_area_max)
            )
        
        self.seg_z_locations_raw = Parameter(
            convert_parameter_to_raw(seg_locations[:,2], 
                    constants.z_loc_min, constants.z_loc_max)
            )
        
        self.seg_x_y_locations = Parameter(seg_locations[:,0:2])
        
        self.seg_locations = torch.empty(0)
        self.seg_radii = torch.empty(0)
        self.seg_gna_per_area_bars = torch.empty(0)
        self.seg_gk_per_area_bars = torch.empty(0)
        self.radii_scaling = torch.empty(0)

    @jit.script_method
    def sanity_check_vals(self):
        #check raw values are finite
        #axial_resistivity_check = torch.isfinite(self.axial_resistivity_raw).all()
        #extra_resistivity_check = torch.isfinite(self.extra_resistivity_raw).all()
        seg_radii_check = torch.isfinite(self.seg_radii_raw).all()
        seg_gna_check = torch.isfinite(self.seg_gna_per_area_bars_raw).all()
        seg_gk_check = torch.isfinite(self.seg_gk_per_area_bars_raw).all()
        seg_location_check = torch.isfinite(self.seg_locations).all()
        radii_scaling_check = torch.isfinite(self.radii_scaling_raw).all()
        
        all_finite = radii_scaling_check and \
            seg_radii_check and seg_gna_check and seg_gk_check and seg_location_check
                    
        if not all_finite:
            #raise a ValueError specifying which checks failed
            failed_check_string = "Failed parameters checks: "
            if not radii_scaling_check:
                failed_check_string += "radii_scaling, "
            #if not extra_resistivity_check:
            #    failed_check_string += "extra_resistivity, "
            if not seg_radii_check:
                failed_check_string += "seg_radii, "
            if not seg_gna_check:
                failed_check_string += "seg_gna_per_area_bars, "
            if not seg_gk_check:
                failed_check_string += "seg_gk_per_area_bars, "
            if not seg_location_check:
                failed_check_string += "seg_locations, "
            raise ValueError(failed_check_string)
                
    @jit.script_method
    def reparametrize(self):
        #self.axial_resistivity = sigmoid_reparametrize(self.axial_resistivity_raw,
        #    constants.axial_resistivity_min, constants.axial_resistivity_max)
        #self.extra_resistivity = sigmoid_reparametrize(self.extra_resistivity_raw,
        #    constants.extra_resistivity_min, constants.extra_resistivity_max)
        
        self.radii_scaling = sigmoid_reparametrize(self.radii_scaling_raw, 
            constants.radii_scaling_min, constants.radii_scaling_max)
        
        self.seg_radii = self.radii_scaling * sigmoid_reparametrize(self.seg_radii_raw, constants.seg_radii_min, constants.seg_radii_max) 
         
        self.seg_gna_per_area_bars = sigmoid_reparametrize(self.seg_gna_per_area_bars_raw, 
                                                           constants.seg_gna_per_area_min, constants.seg_gna_per_area_max)
        self.seg_gk_per_area_bars = sigmoid_reparametrize(self.seg_gk_per_area_bars_raw, 
                                                          constants.seg_gk_per_area_min, constants.seg_gk_per_area_max)
        self.seg_locations = torch.cat(
            [self.seg_x_y_locations, 
            sigmoid_reparametrize(self.seg_z_locations_raw, 
                                  constants.z_loc_min, constants.z_loc_max).unsqueeze(1)], 
            dim=1
            )
     
    @jit.script_method
    def precompute_forward_inputs(self, time_step:float):
        self.sanity_check_vals() 
        self.reparametrize() 
        self.segment_gna_nS, self.segment_gk_nS, self.segment_gl_nS, self.segment_cap_pF, self.drift_matrix_sparse, self.A_over_ms_matrix = \
            precompute_forward_inputs_from_base_params(
                seg_radii=self.seg_radii, seg_heights=self.seg_heights,
                seg_gna_per_area_bars=self.seg_gna_per_area_bars, seg_gk_per_area_bars=self.seg_gk_per_area_bars, seg_gl_per_area_bars=self.seg_gl_per_area_bars,
                membrane_cap_per_area=constants.MEM_CAP_PER_AREA , axial_resistivity=self.axial_resistivity,
                seg_adjacency_list=self.connected_segment_indices,
                time_step=time_step
                )
