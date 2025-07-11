from typing import Tuple
import torch
from torch import jit, Tensor
import torch.nn.functional as F
from torch.nn import Module

@jit.script
def compute_ei_width(ei, max_time_offset:int):
    min_ei, min_index = torch.min(ei, dim=0)
    time_offset_squared = torch.square(torch.arange(-max_time_offset, max_time_offset+1, dtype=torch.float32, device=ei.device))
    #make a torch list for the width at each electrode
    width = torch.zeros(ei.size(1), dtype=torch.float32, device=ei.device)
    
    for elec in range(ei.size(1)):
        relevant_ei = ei[min_index[elec]-max_time_offset:min_index[elec]+max_time_offset+1, elec]
        normalized_ei = torch.square(relevant_ei) / torch.sum(torch.square(relevant_ei))
        width[elec] = torch.sqrt(torch.sum(time_offset_squared * normalized_ei))
    #compute and return the width at each electrode, do not use inplace assignment
    return width

class EIWidthLoss(jit.ScriptModule):
    def __init__(self, max_time_offset:int):
        super(EIWidthLoss, self).__init__()
        self.max_time_offset = max_time_offset
    def forward(self, model_ei, real_ei):
        model_ei_width = compute_ei_width(model_ei, self.max_time_offset)
        real_ei_width = compute_ei_width(real_ei, self.max_time_offset)
        return torch.mean(torch.square(model_ei_width - real_ei_width))    

class SodiumPeakLoss(jit.ScriptModule):
    def __init__(self):
        super(SodiumPeakLoss, self).__init__()
        
    #@jit.script_method
    def forward(self, model_ei, real_ei):
        #compute the l2 norm between the lowest value of the model ei and the real ei on each electrode
        min_model_ei, model_index = torch.min(model_ei, dim=0)
        min_real_ei, real_index = torch.min(real_ei, dim=0)
        return torch.mean(torch.square(min_model_ei - min_real_ei))

@jit.script
def get_diffusion_peaks(ei):
    #compute the maximum peak of the ei before the global minimum
    min_ei, min_index = torch.min(ei, dim=0)
    diffusion_peaks = torch.zeros(ei.size(1), dtype=ei.dtype, device=ei.device)
    for elec in range(ei.size(1)):
        diffusion_peaks[elec] = torch.max(ei[:min_index[elec]]) 
    return diffusion_peaks

class DiffusionPeakLoss(jit.ScriptModule):
    def __init__(self):
        super(DiffusionPeakLoss, self).__init__()
        
    def forward(self, model_ei, real_ei):
        model_diffusion_peaks = get_diffusion_peaks(model_ei)
        real_diffusion_peaks = get_diffusion_peaks(real_ei)
        return torch.mean(torch.square(model_diffusion_peaks - real_diffusion_peaks))
    
@jit.script
def get_potassium_peak(ei):
    #compute the maximum peak of the ei after the global minimum
    min_ei, min_index = torch.min(ei, dim=0)
    potassium_peaks = torch.zeros(ei.size(1), dtype=ei.dtype, device=ei.device)
    for elec in range(ei.size(1)):
        potassium_peaks[elec] = torch.max(ei[min_index[elec]:])
    return potassium_peaks

class PotassiumPeakLoss(jit.ScriptModule):
    def __init__(self):
        super(PotassiumPeakLoss, self).__init__()
        
    def forward(self, model_ei, real_ei):
        model_potassium_peaks = get_potassium_peak(model_ei)
        real_potassium_peaks = get_potassium_peak(real_ei)
        return torch.mean(torch.square(model_potassium_peaks - real_potassium_peaks))

@jit.script
def pairwise_sodium_peak_time_differences(ei, time_step_ms:float, normalize_beta:float=1.0):

    #DECREASE BETA of normalization

    normalized_ei = torch.softmax(-1 * normalize_beta * ei, dim=0) #will be largest at the sodium peak since its negative
    pairwise_time_differences = torch.zeros(ei.size(1), ei.size(1), dtype=ei.dtype, device=ei.device)
    offset_array = torch.arange(-ei.size(0)+1, ei.size(0), dtype=ei.dtype, device=ei.device)
    
    for elec_one in range(ei.size(1)):
        for elec_two in range(elec_one+1, ei.size(1)):
            #compute the cross-correlation between the two signals
            #compute cross correlation using conv1d
            signal1 = normalized_ei[:, elec_one].unsqueeze(0).unsqueeze(0)
            signal2 = normalized_ei[:, elec_two].unsqueeze(0).unsqueeze(0)
            cross_correlation = torch.nn.functional.conv1d(signal1, signal2, padding=ei.size(0)-1).squeeze()
            
            #assert that the cross_crrelation sums to 1, this should always be true since normalized_ei sums to 1
            assert torch.isclose(torch.sum(cross_correlation), torch.tensor(1.0, dtype=cross_correlation.dtype, device=cross_correlation.device)), "cross_correlation does not sum to 1"

            pairwise_time_differences[elec_one, elec_two] = torch.dot(offset_array, cross_correlation)

    return pairwise_time_differences * time_step_ms

@jit.script
def pairwise_diffusion_peak_time_differences(ei, time_step_ms:float, normalize_beta:float=1.0):
    normalized_ei = torch.softmax(normalize_beta * ei, dim=0) #will be largest at the diffusion peak since its positive
    pairwise_time_differences = torch.zeros(ei.size(1), ei.size(1), dtype=ei.dtype, device=ei.device)
    offset_array = torch.arange(-ei.size(0)+1, ei.size(0), dtype=ei.dtype, device=ei.device)
    
    for elec_one in range(ei.size(1)):
        for elec_two in range(elec_one+1, ei.size(1)):
            #compute the cross-correlation between the two signals
            #compute cross correlation using conv1d
            signal1 = normalized_ei[:, elec_one].unsqueeze(0).unsqueeze(0)
            signal2 = normalized_ei[:, elec_two].unsqueeze(0).unsqueeze(0)
            cross_correlation = torch.nn.functional.conv1d(signal1, signal2, padding=ei.size(0)-1).squeeze()
            
            #assert that the cross_crrelation sums to 1, this should always be true since normalized_ei sums to 1
            assert torch.isclose(torch.sum(cross_correlation), torch.tensor(1.0, dtype=cross_correlation.dtype, device=cross_correlation.device)), "cross_correlation does not sum to 1"

            pairwise_time_differences[elec_one, elec_two] = torch.dot(offset_array, cross_correlation)

    return pairwise_time_differences * time_step_ms
    
    
    
    
    
"""
#This nearest segment calculation is actually useless, but stashing it here anyway
    #compute the location of the nearest segment to electrode one and two using a softmax so all segments are considered
    seg_distances_elec_one = torch.cdist(segment_locations_um, elec_locations_um[elec_one:elec_one+1])
    seg_distances_elec_two = torch.cdist(segment_locations_um, elec_locations_um[elec_two:elec_two+1])

    elec_one_nearest_seg_location = torch.softmax(1/ seg_distances_elec_one, dim=0) @ segment_locations_um
    elec_two_nearest_seg_location = torch.softmax(1/ seg_distances_elec_two, dim=0) @ segment_locations_um
    
    #assert that this doesnt deviate by more than 3um ferom the argmin
    elec_one_nearest_seg_index = torch.argmin(seg_distances_elec_one)
    elec_two_nearest_seg_index = torch.argmin(seg_distances_elec_two)
    assert torch.norm(elec_one_nearest_seg_location - segment_locations_um[elec_one_nearest_seg_index]) < 3e0, "elec_one_nearest_seg_location deviates by more than 3um from the argmin, velocity calculation may be incorrect"
    assert torch.norm(elec_two_nearest_seg_location - segment_locations_um[elec_two_nearest_seg_index]) < 3e0, "elec_two_nearest_seg_location deviates by more than 3um from the argmin, velocity calculation may be incorrect"
    
    electrode_distance_along_axon = torch.norm(elec_one_nearest_seg_location - elec_two_nearest_seg_location)
""" 
    
    
    
        
class EILossCableModel(jit.ScriptModule):
#class EILossCableModel(Module):
    def __init__(self, continuity_beta:float, auto_gain=True):
        super(EILossCableModel, self).__init__()
        self.continuity_beta = torch.tensor(continuity_beta)
        self.auto_gain = auto_gain
        
    @jit.script_method
    def forward(self, model_ei, real_ei,
                model_seg_distances, initial_seg_distances) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Ensure that both inputs are 2D tensors and have the same number of channels
        assert model_ei.size(1) == real_ei.size(1), "model_ei and real_ei must have the same number of channels"
        assert model_ei.ndim == 2 and model_ei.ndim == 2, "model_ei and real_ei must be 2D tensors"

        num_channels = model_ei.size(1)
        num_offsets = model_ei.size(0) + real_ei.size(0) - 1
        total_samples = real_ei.size(0) * real_ei.size(1)

        # Initialize cross_correlation with the appropriate size
        cross_correlation = torch.zeros(num_offsets, dtype=model_ei.dtype, device=model_ei.device)
        model_ei_energy = model_ei.pow(2).sum()
        real_ei_energy = real_ei.pow(2).sum()

        # Pad the model_ei signal along the time dimension but not the channel dim
        padded_signal = F.pad(model_ei, (0, 0, real_ei.size(0) - 1, real_ei.size(0) - 1))

        # Compute correlation at each offset
        for offset in range(num_offsets):
            cross_correlation[offset] = torch.sum(padded_signal[offset:offset+real_ei.size(0), :] * real_ei)
            
        if self.auto_gain:
            gain_factors = cross_correlation / model_ei_energy 
        else:
            gain_factors = torch.ones_like(cross_correlation)
        
        aligned_se_loss = (-2 * cross_correlation * gain_factors + (model_ei_energy * gain_factors.pow(2)) + real_ei_energy)
        min_loss, align_index = torch.min(aligned_se_loss, dim=0)
        aligned_gain_factor = gain_factors[align_index]
        
        #ei_mse_loss = min_loss / total_samples #division so that its mse

        #recompute the ei_mse_loss just add the alignment and only non-zero physical_ei values
        ei_mse_loss = torch.sum(torch.square(padded_signal[align_index:align_index+real_ei.size(0), :] - real_ei)) / total_samples

        continuity_loss = F.mse_loss(model_seg_distances, initial_seg_distances)
        total_loss = ei_mse_loss + self.continuity_beta * continuity_loss
        
        return total_loss, ei_mse_loss, continuity_loss, align_index, aligned_gain_factor

@jit.script
def get_connected_segment_distances(segment_locations, connected_segments):
    """
    Returns the distances between connected segments
    """
    num_connections = connected_segments.size(0)
    assert connected_segments.size(1) == 2, "connected_segments must have 2 columns"
    left_locations = segment_locations[connected_segments[:,0]]
    right_locations = segment_locations[connected_segments[:,1]]
    return torch.norm(left_locations - right_locations, dim=1)

@jit.script
def get_membrane_to_electrode_matrix(
    segment_locations_um, electrode_locations_um, 
    extra_resistivity_ohm_cm, adc_count_to_uV:float
    ):
    seg_elec_distances = torch.cdist(segment_locations_um, electrode_locations_um)
    #ohm * cm /um = 10*4ohm, so we need to multiply by 1e-2 to get Mohm
    return (1e-2 * extra_resistivity_ohm_cm/4/torch.pi/adc_count_to_uV) / seg_elec_distances

@jit.script
def membrane_currents_to_electrical_image(
    seg_current_pA_overtime, #(time, segment)
    segment_locations_um, electrode_locations_um,
    extra_resistivity_ohm_cm, #now expecting a tensor of shape (1,) instead of a float so it can have gradients
    adc_count_to_uV:float
    ):
    membrane_pA_to_elec_ADC_count = \
        get_membrane_to_electrode_matrix(
            segment_locations_um=segment_locations_um,
            electrode_locations_um=electrode_locations_um,
            extra_resistivity_ohm_cm=extra_resistivity_ohm_cm,
            adc_count_to_uV=adc_count_to_uV
        )
    #compute electrode voltage in ADC counts
    return seg_current_pA_overtime @ membrane_pA_to_elec_ADC_count

@jit.script
def safe_exp(x, max:float=20.0):
    return torch.exp(torch.clamp(x, max=max))

def get_alpha_m(V_seg:Tensor) -> Tensor:
    return torch.where(
        torch.abs(V_seg + 35) < 2e0,
        27.25 + 1.3625 * (V_seg + 35),
        (-2.725 * (V_seg + 35)) / (safe_exp(-0.1 * (V_seg + 35)) - 1)
    )

def get_alpha_n(V_seg:Tensor) -> Tensor:
    return torch.where(
        torch.abs(V_seg + 37) < 2e0,
        0.9575 + 0.0479 * (V_seg + 37),
        (-0.09575 * (V_seg + 37)) / (safe_exp(-0.1 * (V_seg + 37)) - 1)
    )


@jit.script
def forward_na_k_only(V_seg,m_seg,h_seg,n_seg,#internal state variables
                      
                      seg_gna_ns_bar, seg_gk_ns_bar, seg_gl_ns, #max transmembrane conductances of segment ion channels
                      seg_cap_pF, #capacitance of segment
                      drift_matrix_sparse, #sparse matrix describing intracellular charge drift across membrane
                      ena:float, ek:float, epas:float,#nernst potentials (mV)
                      time_step:float, #time step (ms),
                      estim_bias_mV #what membrane potential tends to during estim
                      ):
    seg_current_g_na_ns = seg_gna_ns_bar * m_seg**3 * h_seg
    seg_current_g_k_ns = seg_gk_ns_bar * n_seg**4
    ionic_current = seg_current_g_na_ns * (V_seg - ena) + seg_current_g_k_ns * (V_seg - ek) + seg_gl_ns * (V_seg - epas) #pA
    
    #just do forward euler
    intermed_op = (V_seg - ionic_current * time_step / seg_cap_pF)
    
    #outputs are computed, now compute the next internal state
    #BS implementation of forward euler
    #dV/dt = a*V + b
    #V(t) + b/a = (V0 + b/a) * exp(a*t)
    #a_term = -1* (seg_gl_ns + seg_current_g_na_ns + seg_current_g_k_ns) / seg_cap_pF
    #b_term = (epas * seg_gl_ns + ena * seg_current_g_na_ns + ek * seg_current_g_k_ns) / seg_cap_pF
    #bias_term = -1 * b_term / a_term
    #intermed_op = bias_term + (V_seg - bias_term) * torch.exp(a_term * time_step)

    assert torch.isfinite(intermed_op).all(), "intermed_op has non-finite values"
    total_bias_mV = estim_bias_mV + epas
    next_V_seg = total_bias_mV + torch.sparse.mm(drift_matrix_sparse, (intermed_op - total_bias_mV).unsqueeze(1)).squeeze(1)
    assert torch.isfinite(next_V_seg).all(), "next_V_seg has non-finite values"
    
    assert torch.isfinite(V_seg).all(), "V_seg has non-finite values prior to gating variable computation"
    #alpha_m = (-2.725 * (V_seg + 35)) / (torch.exp(-0.1 * (V_seg + 35)) - 1)
    alpha_h = 1.817 * torch.exp(-1 * (V_seg + 52)/20)
    #alpha_n = (-0.09575 * (V_seg + 37)) / (torch.exp(-0.1 * (V_seg + 37)) - 1)
    beta_m = 90.83 * torch.exp(-1 * (V_seg + 60) / 20)
    beta_h = 27.25/(torch.exp(-0.1 * (V_seg + 22)) + 1)
    beta_n = 1.915 * torch.exp(-1 * (V_seg + 47) / 80)

    #rewrite kinetics using torch.where to avoid division by zero
    #use hard coded values for alpha_m and alph_n and beta_h when within 1e-1 of -35, -37, and -22 respectively
    #the hard coded values are taken using lhospitals rule 
    TOL = 2e0
    ALPHA_M_LIM_DIS = 27.25
    ALPHA_M_DERIV = 1.3625
    ALPHA_N_LIM_DIS = 0.9575
    ALPHA_N_DERIV = 0.0479
    EPSILON = 1e-6
    #alpha_m = torch.where(torch.abs(V_seg + 35) < TOL, ALPHA_M_LIM_DIS, (-2.725 * (V_seg + 35)) / (torch.exp(-0.1 * (V_seg + 35)) - 1))
    #alpha_n = torch.where(torch.abs(V_seg + 37) < TOL, ALPHA_N_LIM_DIS, (-0.09575 * (V_seg + 37)) / (torch.exp(-0.1 * (V_seg + 37)) - 1))

    #compute alpha m and alpha n unstable part using first order taylor expansion
    
    #alpha_m = torch.where(
    #    torch.abs(V_seg + 35) < TOL,
    #    ALPHA_M_LIM_DIS + (V_seg + 35) * ALPHA_M_DERIV,
    #    (-2.725 * (V_seg + 35)) / (safe_exp(-0.1 * (V_seg + 35)) - 1 + EPSILON) 
    #)
    
    #alpha_n = torch.where(
    #    torch.abs(V_seg + 37) < TOL,
    #    ALPHA_N_LIM_DIS + (V_seg + 37) * ALPHA_N_DERIV,
    #    (-0.09575 * (V_seg + 37)) / (safe_exp(-0.1 * (V_seg + 37)) - 1 + EPSILON)
    #)

    alpha_m = get_alpha_m(V_seg)
    alpha_n = get_alpha_n(V_seg)

    #alpha_c = (-1.362 * (V_seg + 13)) / (torch.exp(-0.1 * (V_seg + 13)) - 1)
    #beta_c = 45.41 * torch.exp(-1 * (V_seg + 38) / 18)
    
    if not torch.isfinite(alpha_m).all():
        #get the indices and input values that are causing the issue
        problem_indices = torch.nonzero(~torch.isfinite(alpha_m))
        problem_values = V_seg[problem_indices]
        raise ValueError(f"alpha_m has non-finite values when input {problem_values}")
    if not torch.isfinite(beta_h).all():
        problem_indices = torch.nonzero(~torch.isfinite(beta_h))
        problem_values = V_seg[problem_indices]
        raise ValueError(f"beta_h has non-finite values when input {problem_values}")
    if not torch.isfinite(alpha_n).all():
        problem_indices = torch.nonzero(~torch.isfinite(alpha_n))
        problem_values = V_seg[problem_indices]
        raise ValueError(f"alpha_n has non-finite values when input {problem_values}")
    if not torch.isfinite(beta_m).all():
        problem_indices = torch.nonzero(~torch.isfinite(beta_m))
        problem_values = V_seg[problem_indices]
        raise ValueError(f"beta_m has non-finite values when input {problem_values}")
    
    """
    compute using 1st order decay approximation
     dm/dt = alpha_m - (alpha_m + beta_m) * m
    bias_m = alpha_m / (alpha_m + beta_m)
    next_m_seg = bias_m + \
        (m_seg - bias_m) * torch.exp(-1 * (alpha_m + beta_m) * time_step)
    bias_h = alpha_h / (alpha_h + beta_h)
    next_h_seg = bias_h + \
        (h_seg - bias_h) * torch.exp(-1 * (alpha_h + beta_h) * time_step)
    bias_n = alpha_n / (alpha_n + beta_n)
    next_n_seg = bias_n + \
        (n_seg - bias_n) * torch.exp(-1 * (alpha_n + beta_n) * time_step)
    """
    #just do forward euler
    next_m_seg = torch.clamp(m_seg + (alpha_m * (1 - m_seg) - beta_m * m_seg) * time_step, 0, 1)
    next_h_seg = torch.clamp(h_seg + (alpha_h * (1 - h_seg) - beta_h * h_seg) * time_step, 0, 1)
    next_n_seg = torch.clamp(n_seg + (alpha_n * (1 - n_seg) - beta_n * n_seg) * time_step, 0, 1)
    
    if not torch.isfinite(next_m_seg).all():
        problem_indices = torch.nonzero(~torch.isfinite(next_m_seg))
        problem_values = m_seg[problem_indices]
        problem_alpha_m = alpha_m[problem_indices]
        problem_beta_m = beta_m[problem_indices]
        raise ValueError(f"next_m_seg has non-finite values when alpha_m {problem_alpha_m} beta_m {problem_beta_m} last_m {problem_values}")
    
    #assert torch.isfinite(next_m_seg).all(), "next_m_seg has non-finite values"
    assert torch.isfinite(next_h_seg).all(), "next_h_seg has non-finite values"
    assert torch.isfinite(next_n_seg).all(), "next_n_seg has non-finite values"
    

    capacitive_current = (next_V_seg - V_seg) * seg_cap_pF / time_step #pA
    transmembrane_current_pA = ionic_current + capacitive_current

    return transmembrane_current_pA, next_V_seg, next_m_seg, next_h_seg, next_n_seg

@jit.script
def precompute_forward_inputs_from_base_params(seg_radii, seg_heights, #um
                                               seg_gna_per_area_bars, seg_gk_per_area_bars, seg_gl_per_area_bars, #S/cm^2
                                               membrane_cap_per_area:float, axial_resistivity, #uF/cm^2, ohm*cm,
                                               seg_adjacency_list, #(connections, 2) tensor of segment indices
                                               time_step:float,
                                               drift_mat_tolerance:float=1e-5):
    num_segments = seg_radii.size(0)
    lateral_area = 2 * torch.pi * seg_radii * seg_heights
    seg_cap_pF = membrane_cap_per_area * lateral_area * 1e-2 #pF
    seg_gna_ns = seg_gna_per_area_bars * lateral_area * 1e1 #nS
    seg_gk_ns = seg_gk_per_area_bars * lateral_area * 1e1 #nS
    seg_gl_ns = seg_gl_per_area_bars * lateral_area * 1e1 #nS

    cross_area = torch.pi * seg_radii**2  
    seg_axial_conductance = 1e5 * (cross_area/seg_heights)/axial_resistivity #nS

    conductance_matrix = torch.zeros(num_segments, num_segments, dtype=torch.float32, device=seg_radii.device)
    conductance_matrix[seg_adjacency_list[:,0], seg_adjacency_list[:,1]] = \
        (seg_axial_conductance[seg_adjacency_list[:,0]] + seg_axial_conductance[seg_adjacency_list[:,1]]) / 2
    neg_row_sum = -1 * conductance_matrix.sum(dim=1)    
    diag_conductance_matrix = conductance_matrix + torch.diag(neg_row_sum)
    A_over_ms_matrix = diag_conductance_matrix / seg_cap_pF.unsqueeze(1)
    #now divide each row by the segment capacitance and take the exponential
    drift_matrix_raw = torch.matrix_exp(time_step * A_over_ms_matrix)
    assert torch.isfinite(drift_matrix_raw).all(), "Drift matrix has non-finite values"
    assert (torch.abs(drift_matrix_raw) > 1).sum() == 0, "Drift matrix has values greater than 1"  

    #make the drift matrix sparse and normalized now
    drift_matrix_zeroed = torch.relu(drift_matrix_raw - drift_mat_tolerance)
    drift_matrix_normalized = drift_matrix_zeroed / drift_matrix_zeroed.sum(dim=1).unsqueeze(1)
    

    drift_matrix_sparse = drift_matrix_normalized.to_sparse()

    return seg_gna_ns, seg_gk_ns, seg_gl_ns, seg_cap_pF, drift_matrix_sparse, A_over_ms_matrix
