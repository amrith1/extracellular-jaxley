import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, required=False)
parser.add_argument('--gpu', type=int, default=0, required=False)
args = parser.parse_args()

SEED = args.seed if args.seed is not None else 0
GPU = args.gpu if args.gpu is not None else 0

import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)


import datetime
import argparse
from typing import List
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
import visionloader as vl
from adapt_neuron.model import CableIntegrator, HHCableRNNCellSimple, LSTMState
import adapt_neuron.constants as constants
from adapt_neuron.utils import with_ttl, time_clip_zeros_ei, get_filter_downsample_matrix
from adapt_neuron.layers import EILossCableModel, get_connected_segment_distances, SodiumPeakLoss, DiffusionPeakLoss, PotassiumPeakLoss, EIWidthLoss, pairwise_sodium_peak_time_differences, pairwise_diffusion_peak_time_differences
from adapt_neuron.ei_analysis import load_cell_data
import numpy as np
import matplotlib.pyplot as plt
#torch.autograd.set_detect_anomaly(True)
from initial_location_guess import generate_location_guess






NUM_EPOCHS = 400
SIM_TIME_SAMPLES = 700
TIME_STEP = 2e-3 #ms
ELEC_SPACING = 30.0
DUMP_PERIOD = 20
TOTAL_LENGTH = 1500.0
SEGMENT_LENGTH = 1.0
VOLTAGE_CLAMP_SEGMENT = 50
VOLTAGE_CLAMP_VALUE = 0.0

STOP_CRITERION = 0.05

PERCENTILE_SAMPLE_AND_CLIP = 0.85

#SAVE_NAME = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
SAVE_NAME = f'seed_{SEED}'
np.random.seed(SEED)


ELECTRODE_CONFIGURATION = 'hexagon'
elec_coords = None

if ELECTRODE_CONFIGURATION == 'triangle':
    #place electrodes at equilateral traiangle in x=0 plane, center at (0,0,0), spaced ELEC_SPACING apart
    elec_coords = np.array([[0, 0, 0], [0, 1, 0], [0, 0.5, 0.866]])
    elec_coords = elec_coords - np.mean(elec_coords, axis=0)
    elec_coords = ELEC_SPACING*elec_coords
    elec_coords = torch.tensor(elec_coords).cuda().to(torch.float32)
    
elif ELECTRODE_CONFIGURATION == 'hexagon':
    #place electrodes at square in x=0 plane, center at (0,0,0), spaced ELEC_SPACING apart, 6 electrodes 30um apart and one at origin
    elec_coords = []
    for elec in range(6):
        angle = 2*np.pi*elec/6.0
        elec_coords.append([0, ELEC_SPACING*np.cos(angle), ELEC_SPACING*np.sin(angle)])
    elec_coords.append([0, 0, 0])
    elec_coords = torch.tensor(elec_coords).cuda().to(torch.float32)

print("elec_coords", elec_coords)

_zeros = torch.zeros(int(TOTAL_LENGTH/SEGMENT_LENGTH)).cuda()
V_seg, m_seg, h_seg, n_seg = _zeros + constants.V_INIT, _zeros + constants.m_INIT, _zeros + constants.h_INIT, _zeros + constants.n_INIT
V_seg[0:VOLTAGE_CLAMP_SEGMENT] = VOLTAGE_CLAMP_VALUE

initial_state = LSTMState(V_seg.clone(), m_seg.clone(), h_seg.clone(), n_seg.clone())
input_estim_over_time = torch.zeros(SIM_TIME_SAMPLES, V_seg.shape[0]).cuda()
filter_ds_matrix = torch.eye(SIM_TIME_SAMPLES).cuda()


MODEL_BOUNDS = {
    'axon_origin_dist_bounds': (10.0, 30.0),
    'axon_theta_bounds': (-torch.pi/2, torch.pi/2),
    'axon_phi_bounds': (0.0, torch.pi),
    'axon_spin_angle_bounds': (-torch.pi/2, torch.pi/2),
    'fiber_radius_um_bounds': (1.0, 5.0),
    'sodium_channel_density_bounds': (0.1, 0.3),
    'potassium_channel_density_bounds': (0.1, 0.3),
    'axial_resistivity_bounds': (50.0, 200.0)
}

params_to_optimize = [
    'axon_origin_dist',
    'axon_theta',
    'axon_phi',
    'axon_spin_angle',
    'fiber_radius_um',
    'sodium_channel_density',
    'potassium_channel_density',
    #'axial_resistivity'
]

print(f'Optimizing {params_to_optimize}')

#generate a random model within the bounds to produce the physical ei
ground_truth_model_kwargs = {}
for param in MODEL_BOUNDS.keys():
    param_name = param.replace('_bounds', '')
    
    if param_name in params_to_optimize:
        #change so as to only draw from 15th to 85th percentile of the bounds
        mean = (MODEL_BOUNDS[param_name + '_bounds'][0] + MODEL_BOUNDS[param_name + '_bounds'][1]) / 2.0
        max_min_range = MODEL_BOUNDS[param_name + '_bounds'][1] - MODEL_BOUNDS[param_name + '_bounds'][0]
        CLIP_COEFF = np.abs(PERCENTILE_SAMPLE_AND_CLIP - 0.5)
        ground_truth_model_kwargs[param_name] = np.random.uniform(mean - CLIP_COEFF * max_min_range, mean + CLIP_COEFF * max_min_range)
    else:
        ground_truth_model_kwargs[param_name] = (MODEL_BOUNDS[param_name + '_bounds'][0] + MODEL_BOUNDS[param_name + '_bounds'][1]) / 2.0


ground_truth_model_kwargs['total_length_um'] = TOTAL_LENGTH
ground_truth_model_kwargs['segment_length_um'] = SEGMENT_LENGTH

ground_truth_model_kwargs = {**MODEL_BOUNDS, **ground_truth_model_kwargs}

gt_model = CableIntegrator(cell=HHCableRNNCellSimple, **ground_truth_model_kwargs)
gt_model.eval()
gt_model.cuda()
gt_model_output = gt_model(input_estim=input_estim_over_time, state=initial_state, time_step=TIME_STEP,
        compute_ei=True, filter_ds_matrix=filter_ds_matrix, num_mask_samples=0,
        electrode_locations_um=elec_coords)
physical_ei = gt_model_output[0].detach()
model_voltages = gt_model_output[2].detach()
initial_seg_distances = gt_model.cell.get_connected_segment_distances().detach()

#plot the physical ei
plt.figure()
plot_ei = time_clip_zeros_ei(physical_ei)
#for elec in range(elec_coords.shape[0]):
#for elec in [5]:
#    plt.plot(plot_ei.cpu().numpy()[:,elec], label=f'Electrode {elec}')
#plt.legend()
#plot electrode 1 and then also plot electrode 4 very faintly
plt.plot(plot_ei.cpu().numpy()[:,1], label='Electrode 1')
plt.plot(plot_ei.cpu().numpy()[:,4], label='Electrode 4', alpha=0.5)
plt.legend()

#plot horizontal line at y=0
plt.axhline(y=0, color='k', linewidth=0.5)
plt.xlabel('Time Samples(500kHz)')
plt.ylabel('EI (uV)')
plt.title('Electrical Image Loss Function Features')
plt.savefig(f'physical_ei_seed_{SEED}.png')
exit()

gt_pairwise_sodium_peak_time_differences = pairwise_sodium_peak_time_differences(physical_ei, TIME_STEP).detach()
gt_pairwise_diffusion_peak_time_differences = pairwise_diffusion_peak_time_differences(physical_ei, TIME_STEP).detach()

print('MICROSECONDS gt_pairwise_sodium_peak_time_differences', gt_pairwise_sodium_peak_time_differences*1000.0)

print(ground_truth_model_kwargs)

#generate the initial guess for the model location
max_ei_values, max_ei_indices = torch.max(physical_ei, dim=0)
elec_distances_guess = 1/max_ei_values
distance_g, phi_g, theta_g, spin_angle_g = \
    generate_location_guess(gt_pairwise_sodium_peak_time_differences.cpu().numpy(), 
                            elec_distances_guess.cpu().numpy(), 
                            elec_coords.cpu().numpy(), 
                            MODEL_BOUNDS['axon_origin_dist_bounds'], 
                            MODEL_BOUNDS['axon_phi_bounds'], 
                            MODEL_BOUNDS['axon_theta_bounds'], 
                            MODEL_BOUNDS['axon_spin_angle_bounds']
                            )
location_guess_dict = {
    'axon_origin_dist': distance_g,
    'axon_phi': phi_g,
    'axon_theta': theta_g,
    'axon_spin_angle': spin_angle_g
}

print('initial guess', distance_g, phi_g, theta_g, spin_angle_g)
print('ground truth', ground_truth_model_kwargs['axon_origin_dist'], ground_truth_model_kwargs['axon_phi'], ground_truth_model_kwargs['axon_theta'], ground_truth_model_kwargs['axon_spin_angle'])

#now that the physical ei is generated, we can generate the model to try to match it
#pick the midway point if we are optimizing the parameter, else pick the ground truth value
initial_guess_model_kwargs = {}
for param in ground_truth_model_kwargs.keys():
    if param in params_to_optimize:
        initial_guess_model_kwargs[param] = (MODEL_BOUNDS[param + '_bounds'][0] + MODEL_BOUNDS[param + '_bounds'][1]) / 2.0
    else:
        initial_guess_model_kwargs[param] = ground_truth_model_kwargs[param]
        
    if param in location_guess_dict.keys() and param in params_to_optimize:
        initial_guess_model_kwargs[param] = location_guess_dict[param]
        
model = CableIntegrator(cell=HHCableRNNCellSimple, **initial_guess_model_kwargs)
model.eval()
model.cuda()

#run the model once
model_output = model(input_estim=input_estim_over_time, state=initial_state, time_step=TIME_STEP,
        compute_ei=True, filter_ds_matrix=filter_ds_matrix, num_mask_samples=0,
        electrode_locations_um=elec_coords)
model_ei = model_output[0].detach()
model_seg_currents = model_output[1].detach()
model_seg_voltages = model_output[2].detach()
del model_ei, model_seg_currents, model_seg_voltages

param_name_to_model_raw = {
    'axon_origin_dist': model.cell.axon_origin_dist_raw,
    'axon_theta': model.cell.axon_theta_raw,
    'axon_phi': model.cell.axon_phi_raw,
    'axon_spin_angle': model.cell.axon_spin_angle_raw,
    'fiber_radius_um': model.cell.fiber_radius_um_raw,
    'sodium_channel_density': model.cell.sodium_channel_density_raw,
    'potassium_channel_density': model.cell.potassium_channel_density_raw,
    'axial_resistivity': model.cell.axial_resistivity_raw
}


#print the model vs the ground truth parameters and also the bounds
for param_bound in MODEL_BOUNDS.keys():
    param_name = param_bound.replace('_bounds', '')
    values = f'{param_name}: {initial_guess_model_kwargs[param_name]:.5} vs {ground_truth_model_kwargs[param_name]:.5} bounds: {MODEL_BOUNDS[param_bound]}'
    print(values)


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Update the optimizer to use AdamW with lr=1e0

adam_lr = 1e-1
adam_betas = (0.9, 0.999)

optimizers = {
    'axon_origin_dist': torch.optim.Adam([model.cell.axon_origin_dist_raw], lr=adam_lr, betas=adam_betas),
    'axon_theta': torch.optim.Adam([model.cell.axon_theta_raw], lr=adam_lr, betas=adam_betas),
    'axon_phi': torch.optim.Adam([model.cell.axon_phi_raw], lr=adam_lr, betas=adam_betas),   
    'axon_spin_angle': torch.optim.Adam([model.cell.axon_spin_angle_raw], lr=adam_lr, betas=adam_betas),
    'fiber_radius_um': torch.optim.Adam([model.cell.fiber_radius_um_raw], lr=adam_lr, betas=adam_betas),
    'sodium_channel_density': torch.optim.Adam([model.cell.sodium_channel_density_raw], lr=adam_lr, betas=adam_betas),
    'potassium_channel_density': torch.optim.Adam([model.cell.potassium_channel_density_raw], lr=adam_lr, betas=adam_betas),
    'axial_resistivity': torch.optim.Adam([model.cell.axial_resistivity_raw], lr=adam_lr, betas=adam_betas)
}
# Use CosineAnnealingWarmRestarts for learning rate scheduling
# schedulers = {
#     'axon_origin_dist': CosineAnnealingWarmRestarts(optimizers['axon_origin_dist'], T_0=5, T_mult=1),
#     'axon_theta': CosineAnnealingWarmRestarts(optimizers['axon_theta'], T_0=5, T_mult=1),
#     'axon_phi': CosineAnnealingWarmRestarts(optimizers['axon_phi'], T_0=5, T_mult=1), 
#     'axon_spin_angle': CosineAnnealingWarmRestarts(optimizers['axon_spin_angle'], T_0=5, T_mult=1),
#     'fiber_radius_um': CosineAnnealingWarmRestarts(optimizers['fiber_radius_um'], T_0=5, T_mult=1),
#     'sodium_channel_density': CosineAnnealingWarmRestarts(optimizers['sodium_channel_density'], T_0=5, T_mult=1),
#     'potassium_channel_density': CosineAnnealingWarmRestarts(optimizers['potassium_channel_density'], T_0=5, T_mult=1),
#     'axial_resistivity': CosineAnnealingWarmRestarts(optimizers['axial_resistivity'], T_0=5, T_mult=1)
# }

#set all schedulers to None
schedulers = {
    'axon_origin_dist': None,
    'axon_theta': None,
    'axon_phi': None,
    'axon_spin_angle': None, 
    'fiber_radius_um': None,
    'sodium_channel_density': None,
    'potassium_channel_density': None,
    'axial_resistivity': None
}

width_loss_fn = EIWidthLoss(max_time_offset=125)
cable_loss_fn = EILossCableModel(continuity_beta=1.0, auto_gain=False)
sodium_peak_loss_fn = SodiumPeakLoss()
diffusion_peak_loss_fn = DiffusionPeakLoss()
potassium_peak_loss_fn = PotassiumPeakLoss()

def train_one_epoch(params_to_optimize: List[str], input_estim: torch.Tensor, initial_state:LSTMState,
                    time_step:float, filter_ds_matrix:torch.Tensor, num_mask_samples:int):
    # Define a function to add noise to the input
    def add_noise_to_input(input_tensor, noise_std=0.01):
        noise = torch.normal(mean=0.0, std=noise_std, size=input_tensor.shape).to(input_tensor.device)
        return input_tensor + noise

    # Flag to determine if we need to redo the forward pass with noisy input
    need_retry = True
    
    while need_retry:
        # Zero the gradients for the selected optimizers
        for param_name in params_to_optimize:
            optimizers[param_name].zero_grad()

        # Forward pass of the model
        model_ei, model_seg_currents, model_seg_voltages, connected_seg_distances, final_state = \
            model(input_estim=input_estim, state=initial_state, time_step=time_step,
                  compute_ei=True, electrode_locations_um=elec_coords,
                  filter_ds_matrix=filter_ds_matrix, num_mask_samples=num_mask_samples
                  )

        # Check that the cell actually spiked
        INDICATOR_SEG = int(model.cell.num_segments * 0.7)
        relevant_voltages = model_seg_voltages[num_mask_samples:, INDICATOR_SEG]
        # Assert at least one voltage is above 0mV
        assert torch.any(relevant_voltages > 0.0), 'Cell did not spike to input'

        # Also assert that voltage is between -100 and 100 at all segments
        relevant_voltages = model_seg_voltages[num_mask_samples:, :]
        assert torch.all(relevant_voltages > constants.E_K) and torch.all(relevant_voltages < constants.E_NA), 'Voltage out of bounds'

        # Compute loss
        # total_loss, ei_loss, continuity_loss, align_index, aligned_gain_factor = \
        #     model_loss_fn(model_ei=model_ei, real_ei=physical_ei,
        #                   model_seg_distances=connected_seg_distances,
        #                   initial_seg_distances=initial_seg_distances)


        width_loss = width_loss_fn(model_ei=model_ei, real_ei=physical_ei)
        cable_loss = cable_loss_fn(model_ei=model_ei, real_ei=physical_ei,
                                   model_seg_distances=connected_seg_distances,
                                   initial_seg_distances=initial_seg_distances)
        
        
        sodium_peak_loss = sodium_peak_loss_fn(model_ei=model_ei, real_ei=physical_ei)
        diffusion_peak_loss = diffusion_peak_loss_fn(model_ei=model_ei, real_ei=physical_ei)
        potassium_peak_loss = potassium_peak_loss_fn(model_ei=model_ei, real_ei=physical_ei)
        
        #compute loss between sodium and diffusion time differences
        model_sodium_peak_time_differences = pairwise_sodium_peak_time_differences(model_ei, TIME_STEP)
        model_diffusion_peak_time_differences = pairwise_diffusion_peak_time_differences(model_ei, TIME_STEP)
        
        sodium_velocity_loss = torch.mean(torch.square(model_sodium_peak_time_differences - gt_pairwise_sodium_peak_time_differences)) * 1e6 #loss in us^2
        diffusion_velocity_loss = torch.mean(torch.square(model_diffusion_peak_time_differences - gt_pairwise_diffusion_peak_time_differences)) * 1e6 #loss in us^2

        #compute loss between model and gt time differences

        #switch back and forth randomly between the two loss functions

        total_loss = (sodium_peak_loss + diffusion_peak_loss + potassium_peak_loss * 4.0 + width_loss * 500.0 + sodium_velocity_loss * 50.0 + diffusion_velocity_loss * 50.0) * 0.3
        # Backward pass
        
        try:
            total_loss.backward()
        except Exception as e:
            print(f"Error during backward pass: {e}")
            input_estim = add_noise_to_input(input_estim)
            continue

        # Check if any of the gradients are `inf` or `NaN`
        gradient_is_finite = True
        for param_name in params_to_optimize:
            for param in optimizers[param_name].param_groups[0]['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        gradient_is_finite = False
                        break
            if not gradient_is_finite:
                break

        if gradient_is_finite:
            # If all gradients are finite, proceed with optimization and exit loop
            need_retry = False
        else:
            # If gradients are `inf` or `NaN`, retry the forward pass with noisy input
            print("\n\nNon-finite gradient detected, retrying with noisy input.\n\n", flush=True)
            input_estim = add_noise_to_input(input_estim)

    # Perform the optimization step for the selected parameters
    for param_name in params_to_optimize:
        optimizers[param_name].step()
        raw_param = param_name_to_model_raw[param_name]
        with torch.no_grad():
            #if the value of the param is out of bounds, reset the optimizer
            if torch.lt(raw_param, torch.logit(torch.tensor(1.0-PERCENTILE_SAMPLE_AND_CLIP).cuda())).any() or torch.gt(raw_param, torch.logit(torch.tensor(PERCENTILE_SAMPLE_AND_CLIP).cuda())).any():
                #reset the optimizer entirely
                optimizers[param_name] = torch.optim.Adam([raw_param], lr=adam_lr, betas=adam_betas)
            #set the clips to be the logit of the sample_and_clip_bounds
            assert PERCENTILE_SAMPLE_AND_CLIP > 0.5 and PERCENTILE_SAMPLE_AND_CLIP < 1.0, 'PERCENTILE_SAMPLE_AND_CLIP must be between 0.5 and 1'
            raw_param.clamp_(torch.logit(torch.tensor(1.0-PERCENTILE_SAMPLE_AND_CLIP).cuda()), torch.logit(torch.tensor(PERCENTILE_SAMPLE_AND_CLIP).cuda()))
            
        
    #step the reduce on plateau schedulers
    for param_name in params_to_optimize:
        if schedulers[param_name] is not None:
            schedulers[param_name].step(epoch_index)

    #reset the input_estim to zeros
    input_estim = torch.zeros_like(input_estim)

    # Get current learning rates for each parameter
    learning_rates = {}
    for param_name in params_to_optimize:
        learning_rates[param_name] = optimizers[param_name].param_groups[0]['lr']

    # Return any values you need to monitor
    return model_ei, model_seg_currents, total_loss, sodium_peak_loss, diffusion_peak_loss, potassium_peak_loss, width_loss, sodium_velocity_loss, diffusion_velocity_loss, learning_rates


if __name__ == "__main__":

    os.system('mkdir -p training_checkpoints/loss_curves')
    os.system('mkdir -p training_checkpoints/epoch_data')

    model.train(True)
    filter_ds_matrix.requires_grad = False

    epoch_outputs = []
    og_ei_loss = None
    pbar = tqdm(range(NUM_EPOCHS))
    for epoch_index in pbar:
        
        initial_state = LSTMState(V_seg.clone(), m_seg.clone(), h_seg.clone(), n_seg.clone())
        input_estim_overtime = torch.zeros(SIM_TIME_SAMPLES, V_seg.shape[0]).cuda() #zeros ie no estim

        model_ei, model_seg_currents, total_loss, sodium_peak_loss, diffusion_peak_loss, potassium_peak_loss, width_loss, sodium_velocity_loss, diffusion_velocity_loss, learning_rates=\
            train_one_epoch(params_to_optimize=params_to_optimize, 
                            input_estim=input_estim_overtime,
                            initial_state=initial_state,
                            time_step=TIME_STEP,
                            filter_ds_matrix=filter_ds_matrix,
                            num_mask_samples= 0
                            )
        
        #copy to cpu and store, we dont want to consume GPU memory
        epoch_params = {
            'axon_origin_dist': model.cell.axon_origin_dist.detach().item(),
            'axon_theta': model.cell.axon_theta.detach().item(),
            'axon_phi': model.cell.axon_phi.detach().item(),
            'axon_spin_angle': model.cell.axon_spin_angle.detach().item(),
            'fiber_radius_um': model.cell.fiber_radius_um.detach().item(),
            'sodium_channel_density': model.cell.sodium_channel_density.detach().item(),
            'potassium_channel_density': model.cell.potassium_channel_density.detach().item(),
            'axial_resistivity': model.cell.axial_resistivity.detach().item(),
            'total_length_um': TOTAL_LENGTH,
            'segment_length_um': SEGMENT_LENGTH
        }
        
        epoch_learning_rates = {}
        for param_name in params_to_optimize:
            epoch_learning_rates[param_name] = learning_rates[param_name]
        
        epoch_save_dict = {'params':epoch_params, 'learning_rates':epoch_learning_rates, 'model_bounds':MODEL_BOUNDS, 'ground_truth_model_kwargs':ground_truth_model_kwargs, 'params_to_optimize':params_to_optimize, 'seed':SEED}
    
        epoch_outputs.append(epoch_save_dict)
        pbar.set_description(
            f'Sodium Peak Loss: {torch.sqrt(sodium_peak_loss):.5} Diffusion Peak Loss: {torch.sqrt(diffusion_peak_loss):.5} Potassium Peak Loss: {torch.sqrt(potassium_peak_loss):.5} Width Loss: {torch.sqrt(width_loss):.5} Sodium Velocity Loss: {torch.sqrt(sodium_velocity_loss):.5} Diffusion Velocity Loss: {torch.sqrt(diffusion_velocity_loss):.5}'
            ) 
        
        #print out the gradient of loss with respect to axon_origin_dist and the current value of axon_origin_dist
        #print(f'Gradient of loss with respect to axon_origin_dist: {model.cell.axon_origin_dist_raw.grad.item():.5}')
        #print(f'Current value of axon_origin_dist: {model.cell.axon_origin_dist.item():.5}')
        #and ground truth value
        #print(f'Ground truth value of axon_origin_dist: {ground_truth_model_kwargs["axon_origin_dist"]:.5}')
        
        #determine if stop criterion is met
        stop_criterion_met = True
        param_errors = {}
        for param_name in params_to_optimize:
            bound_range = MODEL_BOUNDS[param_name + '_bounds'][1] - MODEL_BOUNDS[param_name + '_bounds'][0]
            gt_value = ground_truth_model_kwargs[param_name]
            current_value = epoch_outputs[-1]['params'][param_name]
            param_errors[param_name] = np.abs(current_value - gt_value)/np.abs(bound_range)
            if np.abs(current_value - gt_value) > STOP_CRITERION * np.abs(bound_range):
                stop_criterion_met = False
                continue
        
        if epoch_index % DUMP_PERIOD == 0 or stop_criterion_met:
            #print('Generating plot')
            #print(f'Epoch {epoch_index} learning rates: {learning_rates}')
            
            #make a 2 column 5 row plot of the converging parameters
            #reserve the bottom 2 plots for the model and real ei
            fig, axs = plt.subplots(5, 2, figsize=(10, 25))
            flat_axs = axs.flatten()
            #at each of the param plots, plot a horizontal black line for the ground truth value
            #make the ylimit the bounds of the param
            #plot the value of the param over epochs
            for i, param in enumerate(MODEL_BOUNDS.keys()):
                param_name = param.replace('_bounds', '')
                #generate param over time series
                param_over_time = [epoch_outputs[j]['params'][param_name] for j in range(len(epoch_outputs))]
                
                flat_axs[i].axhline(y=ground_truth_model_kwargs[param_name], color='k', linestyle='--')
                flat_axs[i].plot(param_over_time, linewidth=1)
                flat_axs[i].set_ylim(MODEL_BOUNDS[param_name + '_bounds'])
                flat_axs[i].set_title(param_name)
                flat_axs[i].set_xlabel('Epoch')
                flat_axs[i].set_ylabel(param_name)
            
            #now plot the physical ei
            flat_axs[-2].plot(time_clip_zeros_ei(physical_ei).cpu().detach().numpy(), label='Physical EI')
            flat_axs[-2].set_title('EIs')
            flat_axs[-2].set_xlabel('Time')
            flat_axs[-2].set_ylabel('EI')
            
            flat_axs[-2].plot(time_clip_zeros_ei(model_ei).cpu().detach().numpy(), label='Model EI')
            flat_axs[-2].set_title('Model EI')
            flat_axs[-2].set_xlabel('Time')
            flat_axs[-2].set_ylabel('EI')
            flat_axs[-2].legend()
            #now plot the model ei at each epoch
            # for i, epoch_output in enumerate(epoch_outputs):
            #     #clipped_model_ei = time_clip_zeros_ei(epoch_output['model_ei'], energy_percent_each_side=0.001)
            #     #flat_axs[-1].plot(clipped_model_ei)
            #     flat_axs[-1].plot(epoch_output['model_ei'])
            # flat_axs[-1].set_title('Model EI')
            # flat_axs[-1].set_xlabel('Time')
            # flat_axs[-1].set_ylabel('EI')
            
            #plot the learning rates for each param on flat_axs[-1]
            for i, param in enumerate(params_to_optimize):
                flat_axs[-1].plot([epoch_outputs[j]['learning_rates'][param] for j in range(len(epoch_outputs))], label=param)
                flat_axs[-1].set_title('Learning Rates')
                flat_axs[-1].set_xlabel('Epoch')
                flat_axs[-1].set_ylabel('Learning Rate')
            #make y axis log scale
            flat_axs[-1].set_yscale('log')
            flat_axs[-1].legend()
            
            #print the param errors in the suptitle
            suptitle = ''
            for param_name in params_to_optimize:
                suptitle += f'{param_name} Error: {param_errors[param_name]:.5} \n'
            plt.suptitle(suptitle)
            
            plt.savefig(f'training_checkpoints/loss_curves/{SAVE_NAME}.png')
            plt.close()
            
            #save the epoch_outputs to a pickle file
            with open(f'training_checkpoints/epoch_data/{SAVE_NAME}.pkl', 'wb') as f:
                pickle.dump(epoch_outputs, f)
            
        #    with open(f'training_checkpoints/{SAVE_NAME}.pkl', 'wb') as f:
        #        pickle.dump((static_params, epoch_outputs), f)
        #    pbar.set_description('Model dumped')
        
            #make a list of dicts of param values over time

        del model_ei, model_seg_currents
        
        if stop_criterion_met:
            print('Stop criterion met')
            break
        
    