#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import multielec_src.fitting as fitting
import multielec_src.multielec_utils as mutils
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
import visionloader as vl
import h5py


# In[2]:


mapping = loadmat('/Volumes/Lab/Users/AlexG/pystim/files/adj-elecs-tools/newLVChan2Elec519.mat')['channel2Elec519'].flatten()

def is_valid_set(numbers, is_30um=True):
    """
    Checks if the input set of numbers (1-512) is valid.
    A valid set contains numbers only from either all odd or all even index groups.

    Parameters:
        numbers (array-like): List or array of unique numbers in the range 1-512.

    Returns:
        bool: True if valid, False otherwise.
    """
    numbers = np.asarray(numbers)
    if is_30um:
        transformed_numbers = []
        for number in numbers:
            ind = np.where(mapping == number)[0][0] + 1
            transformed_numbers.append(ind)
    
        numbers = np.asarray(transformed_numbers)

    # Compute the chunk indices (0-7) for each number
    chunk_indices = (numbers - 1) // 64

    # Check if all indices are even or all are odd
    return np.all(chunk_indices % 2 == 0) or np.all(chunk_indices % 2 == 1)    


# Constants
ESTIM_ANALYSIS_BASE = '/Volumes/Lab/Users/praful/outputs/pp_out'
WNOISE_ANALYSIS_BASE = '/Volumes/Analysis'

# Dummy channels to remove (0-indexed)
DUMMY_CHANNELS_519 = [0, 1, 130, 259, 260, 389, 390, 519]

# Dataset configurations
datasets = ['2020-09-29-2', '2020-09-29-2', '2020-09-29-2', '2020-10-06-7', '2020-10-18-0', '2020-10-18-5', '2021-02-13-6', '2021-03-12-0', '2021-03-12-3']
dataruns = ['data007', 'data008', 'data009', 'data003', 'data003', 'data006', 'data003', 'data002', 'data003']
wnoises = ['kilosort_data006/data006', 'kilosort_data006/data006', 'kilosort_data006/data006', 'kilosort_data000/data000', 
           'kilosort_data000/data000', 'kilosort_data002/data002', 'kilosort_data001/data001', 'kilosort_data000/data000',
           'kilosort_data000/data000']

# Output file for structured data
output_file = '/Volumes/Lab/Users/seijiy/extracellular-jaxley/data-packaging/triplet_outputs/dense_triplets.h5'

# Dictionary to store data organized by cells
cells_data = {}

def convert_to_dense_vector(stim_elecs, amps_gsort):
    """
    Convert sparse stimulation data to dense vector with dummy channels removed.
    
    Args:
        stim_elecs: Array of 3 electrode indices (1-indexed)
        amps_gsort: Array of 3 current values corresponding to the electrodes
    
    Returns:
        dense_vector: (NUM_ELECTRODES - 8)-dimensional array with zeros everywhere except at stim_elecs positions
    """
    dense_vector = np.zeros(NUM_ELECTRODES)
    
    # Convert to 0-indexed for array indexing
    stim_elecs_0indexed = np.array(stim_elecs) - 1
    
    # Place the current values at the correct electrode positions
    for i, (elec_idx, current) in enumerate(zip(stim_elecs_0indexed, amps_gsort)):
        dense_vector[elec_idx] = current
    
    # Remove dummy channels
    valid_indices = [i for i in range(NUM_ELECTRODES) if i not in DUMMY_CHANNELS_519]
    dense_vector_clean = dense_vector[valid_indices]
    
    return dense_vector_clean

print("Processing datasets to create dense stimulation vectors...")

# Get electrode coordinates from the first dataset to use for all datasets
first_dataset = datasets[0]
first_datarun = dataruns[0]
first_wnoise = wnoises[0]
vstim_datapath = os.path.join(WNOISE_ANALYSIS_BASE, first_dataset, first_wnoise)
vstim_datarun = os.path.basename(os.path.normpath(vstim_datapath))
vcd = vl.load_vision_data(vstim_datapath, vstim_datarun,
                        include_neurons=True,
                        include_ei=True,
                        include_params=True,
                        include_noise=True)
coords = vcd.electrode_map

if coords is None:
    raise ValueError("electrode_map is None - cannot proceed")
NUM_ELECTRODES = coords.shape[0]
print(f"Number of electrodes: {NUM_ELECTRODES}")

for dataset, datarun, wnoise in zip(datasets, dataruns, wnoises):
    print(f'Processing {dataset} {datarun} {wnoise}')
    basename = f'/Volumes/Analysis/{dataset}/gsort'

    # Load electrical data and g-sort data
    outpath = os.path.join(basename, datarun, wnoise)
    parameters = loadmat(os.path.join(outpath, 'parameters.mat'))

    cells = parameters['cells'].flatten()
    patterns = parameters['patterns'].flatten()
    num_cells = len(cells)
    num_patterns = max(patterns)
    num_movies = parameters['movies'].flatten()[0]

    all_trials = np.array(np.memmap(os.path.join(outpath, 'trial.dat'),mode='r',shape=(num_patterns, num_movies), dtype='int16'), dtype=int)
    if dataset != '2020-10-06-7':
        amps_gsort = mutils.get_stim_amps_newlv(os.path.join(ESTIM_ANALYSIS_BASE, dataset, datarun), 
                                                len(all_trials))
    else:
        amps_gsort = mutils.get_stim_amps_newlv(os.path.join(ESTIM_ANALYSIS_BASE, dataset, 'data005'), 
                                                len(all_trials))

    path = os.path.join(basename, datarun, wnoise)
    file_list = os.listdir(path)

    vstim_datapath = os.path.join(WNOISE_ANALYSIS_BASE, dataset, wnoise)
    vstim_datarun = os.path.basename(os.path.normpath(vstim_datapath))
    vcd = vl.load_vision_data(vstim_datapath, vstim_datarun,
                            include_neurons=True,
                            include_ei=True,
                            include_params=True,
                            include_noise=True)
    coords = vcd.electrode_map
    
    # Get number of electrodes from the electrode map
    if coords is None:
        print(f"Warning: electrode_map is None for dataset {dataset}")
        continue
    NUM_ELECTRODES = coords.shape[0]
    print(f"Number of electrodes: {NUM_ELECTRODES}")

    patterns = {}
    for file in file_list:
        if file.endswith('.mat') and file.startswith('fit'):
            pattern_cell = file.split('.mat')[0].split('_')[-1]
            p, c = pattern_cell.replace('p', '').split('c')
            p = int(p)
            c = int(c)

            if dataset != '2020-10-06-7':
                stim_elecs = mutils.get_stim_elecs_newlv(os.path.join(ESTIM_ANALYSIS_BASE, dataset, datarun), 
                                                        p)
            else:
                stim_elecs = mutils.get_stim_elecs_newlv(os.path.join(ESTIM_ANALYSIS_BASE, dataset, 'data005'), 
                                                        p)
            if not is_valid_set(stim_elecs):
                continue

            # Load fitted data for response probabilities
            fit_data = loadmat(os.path.join(path, file))
            params = fit_data['params_true']
            X = fit_data['amps_fit']
            probs_fit = fit_data['probs_fit'].flatten()
            
            # Check if GPR data is available
            gpr_available = 'gpr_predictions' in fit_data
            if gpr_available:
                gpr_predictions = fit_data['gpr_predictions'].flatten()
                print(f"    GPR data found for cell {c}, pattern {p}")
            else:
                print(f"    No GPR data for cell {c}, pattern {p}")
            
            # Check if multi-site data is available
            multisite_available = 'probs_multisite' in fit_data
            if multisite_available:
                probs_multisite = fit_data['probs_multisite'].flatten()
                params_multisite = fit_data['params_multisite']
                print(f"    Multi-site data found for cell {c}, pattern {p}")
            else:
                print(f"    No multi-site data for cell {c}, pattern {p}")
            
            # Find indices of rows in amps_gsort corresponding to rows in X, preserving order
            indices = [np.where(np.all(amps_gsort == row, axis=1))[0][0] for row in X]
            remaining_inds = np.setdiff1d(np.arange(len(amps_gsort)), indices)
            amps_remaining = deepcopy(amps_gsort[remaining_inds])

            # Create full probability array for all trials
            probs_flipped = np.zeros(len(amps_gsort))
            probs_flipped[indices] = probs_fit

            # Calculate probabilities for remaining trials using fitted parameters
            probs_remaining = fitting.sigmoidND_nonlinear(sm.add_constant(amps_remaining, has_constant='add'),
                                                        params)
            probs_flipped[remaining_inds] = probs_remaining
            
            # Handle GPR predictions if available
            if gpr_available:
                # Create full GPR prediction array for all trials
                gpr_flipped = np.zeros(len(amps_gsort))
                gpr_flipped[indices] = gpr_predictions
                
                # For remaining trials, we could use the same sigmoid function or set to NaN
                # For now, let's use the same approach as probabilities
                gpr_remaining = fitting.sigmoidND_nonlinear(sm.add_constant(amps_remaining, has_constant='add'),
                                                          params)
                gpr_flipped[remaining_inds] = gpr_remaining
                
                # GPR data is available (no need to track in metadata)
                pass
            else:
                # If no GPR data, create array of NaN values
                gpr_flipped = np.full(len(amps_gsort), np.nan)
            
            # Handle multi-site predictions if available
            if multisite_available:
                # Create full multi-site prediction array for all trials
                multisite_flipped = np.zeros(len(amps_gsort))
                multisite_flipped[indices] = probs_multisite
                
                # For remaining trials, use multi-site parameters
                multisite_remaining = fitting.sigmoidND_nonlinear(sm.add_constant(amps_remaining, has_constant='add'),
                                                                params_multisite)
                multisite_flipped[remaining_inds] = multisite_remaining
                
                # Multi-site data is available (no need to track in metadata)
                pass
            else:
                # If no multi-site data, create array of NaN values
                multisite_flipped = np.full(len(amps_gsort), np.nan)

            # Get voltage trace from EI data (electrical image) - always get this
            # print ei shape 
            ei_data = vcd.get_ei_for_cell(c).ei
            print(f"Cell {c}: ei_data shape {ei_data.shape}")
            
            # Initialize cell data if not exists
            # Use actual cell ID for the key, but we'll renumber them 1-37 when saving
            cell_key = f"cell_{c:03d}"
            if cell_key not in cells_data:
                # Remove dummy channels from EI data
                dummy_ei_channels = [channel - 1 for channel in DUMMY_CHANNELS_519[1:]]
                valid_indices = [i for i in range(ei_data.shape[0]) if i not in dummy_ei_channels]
                ei_clean = ei_data[valid_indices]
                
                cells_data[cell_key] = {
                    'stim_inputs': [],
                    'stim_outputs': [],
                    'gpr_outputs': [],  # Add GPR predictions
                    'multisite_outputs': [],  # Add multi-site predictions
                    'voltage_trace': ei_clean,  # Add voltage trace data
                    'cell_id': c,
                    'dataset': dataset,
                    'datarun': datarun,
                    'patterns': [],
                    'cell_type': vcd.get_cell_type_for_cell(c) or 'unknown'
                }
            
            # Convert each stimulation trial to dense vector
            print(f"    Processing {len(amps_gsort)} trials for pattern {p}, cell {c}")
            for trial_idx, amps_trial in enumerate(amps_gsort):
                dense_vector = convert_to_dense_vector(stim_elecs, amps_trial)
                cells_data[cell_key]['stim_inputs'].append(dense_vector)
                
                # Use actual response probability from fitted data
                stim_output = probs_flipped[trial_idx]
                cells_data[cell_key]['stim_outputs'].append(stim_output)
                
                # Add GPR prediction
                gpr_output = gpr_flipped[trial_idx]
                cells_data[cell_key]['gpr_outputs'].append(gpr_output)
                
                # Add multi-site prediction
                multisite_output = multisite_flipped[trial_idx]
                cells_data[cell_key]['multisite_outputs'].append(multisite_output)
                
                # Add pattern info
                if p not in cells_data[cell_key]['patterns']:
                    cells_data[cell_key]['patterns'].append(p)

# Convert lists to numpy arrays for each cell
print("Converting data to numpy arrays...")
for cell_key, cell_data in cells_data.items():
    cell_data['stim_inputs'] = np.array(cell_data['stim_inputs'], dtype=np.float32)
    cell_data['stim_outputs'] = np.array(cell_data['stim_outputs'], dtype=np.float32)
    cell_data['gpr_outputs'] = np.array(cell_data['gpr_outputs'], dtype=np.float32)
    cell_data['multisite_outputs'] = np.array(cell_data['multisite_outputs'], dtype=np.float32)

print(f"Created data for {len(cells_data)} cells")

# Save to HDF5 file with structured format
print(f"Saving to {output_file}")
import json

with h5py.File(output_file, 'w') as f:
    # Group cells by dataset (date)
    dataset_cells = {}
    dataset_wnoises = {}  # Store wnoise for each dataset
    for cell_key, cell_data in cells_data.items():
        dataset = cell_data['dataset']
        if dataset not in dataset_cells:
            dataset_cells[dataset] = []
            # Get wnoise for this dataset from the datasets/dataruns/wnoises lists
            dataset_idx = datasets.index(dataset)
            dataset_wnoises[dataset] = wnoises[dataset_idx]
        dataset_cells[dataset].append((cell_key, cell_data))
    
    # Sort datasets by date
    sorted_datasets = sorted(dataset_cells.keys())
    print(f"Datasets found: {sorted_datasets}")
    
    # Save each dataset's cells
    global_cell_counter = 1  # Global counter across all datasets
    
    for dataset in sorted_datasets:
        print(f"Processing dataset: {dataset}")
        dataset_group = f.create_group(dataset)
        
        # Store wnoise at dataset level
        dataset_group.create_dataset('wnoise', data=dataset_wnoises[dataset].encode('utf-8'))
        
        # Get cells for this dataset and sort them by cell_id
        dataset_cell_list = dataset_cells[dataset]
        dataset_cell_list.sort(key=lambda x: x[1]['cell_id'])  # Sort by actual cell_id
        
        for i, (cell_key, cell_data) in enumerate(dataset_cell_list):
            if i % 10 == 0:  # Progress update every 10 cells
                print(f"  Saving cell {global_cell_counter}/{sum(len(dataset_cells[d]) for d in sorted_datasets)}: {cell_key}")
            
            # Create enumerated cell key with global counter
            enumerated_cell_key = f"cell_{global_cell_counter:03d}"
            cell_group = dataset_group.create_group(enumerated_cell_key)
            
            # Increment global counter
            global_cell_counter += 1
            
            # Save stim_inputs
            cell_group.create_dataset('stim_inputs', data=cell_data['stim_inputs'], 
                                     dtype=np.float32, compression='gzip')
            
            # Save stim_outputs
            cell_group.create_dataset('stim_outputs', data=cell_data['stim_outputs'], 
                                     dtype=np.float32, compression='gzip')
            
            # Save gpr_outputs
            cell_group.create_dataset('gpr_outputs', data=cell_data['gpr_outputs'], 
                                     dtype=np.float32, compression='gzip')
            
            # Save multisite_outputs
            cell_group.create_dataset('multisite_outputs', data=cell_data['multisite_outputs'], 
                                     dtype=np.float32, compression='gzip')
            
            # Save voltage_trace
            cell_group.create_dataset('voltage_trace', data=cell_data['voltage_trace'], 
                                     dtype=np.float32, compression='gzip')
            
            # Save metadata directly under cell group
            cell_group.create_dataset('cell_id', data=cell_data['cell_id'])
            cell_group.create_dataset('dataset', data=cell_data['dataset'].encode('utf-8'))
            cell_group.create_dataset('datarun', data=cell_data['datarun'].encode('utf-8'))
            cell_group.create_dataset('cell_type', data=cell_data['cell_type'].encode('utf-8'))
            cell_group.create_dataset('patterns', data=cell_data['patterns'])
    
    # Create globals group
    globals_group = f.create_group('globals')
    
    # Save electrode positions (removing dummy channels)
    # Note: This assumes all datasets use the same electrode array
    valid_indices = [i for i in range(coords.shape[0]) if i not in DUMMY_CHANNELS_519]
    coords_clean = coords[valid_indices]
    globals_group.create_dataset('electrode_positions', data=coords_clean, 
                                dtype=np.float32, compression='gzip')
    
    # Save acquisition info
    acquisition_info = {
        'num_datasets': len(datasets),
        'datasets': datasets,
        'dataruns': dataruns,
        'wnoises': wnoises,
        'num_electrodes': NUM_ELECTRODES,
        'num_electrodes_clean': NUM_ELECTRODES - len(DUMMY_CHANNELS_519),
        'dummy_channels_removed': DUMMY_CHANNELS_519,
        'creation_date': str(np.datetime64('now'))
    }
    globals_group.attrs['acquisition_info'] = json.dumps(acquisition_info)

print("Done, dataset saved to HDF5 file.")
print(f"Total cells: {len(cells_data)}")
print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")




