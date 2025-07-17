#!/usr/bin/env python3
# coding: utf-8

import h5py
import numpy as np

def print_h5_structure(h5_file_path):
    """Print the complete structure of an HDF5 file without accessing data."""
    print(f"Analyzing HDF5 file: {h5_file_path}")
    print("=" * 60)
    try:
        with h5py.File(h5_file_path, 'r') as f:
            print("File structure:")
            print("dense_triplets.h5")
            
            # Get all top-level keys
            top_keys = list(f.keys())
            print(f"Top-level keys: {top_keys}")
            
            for key in top_keys:
                item = f[key]
                print(f"\n├── /{key}/")
                
                if isinstance(item, h5py.Group):
                    # It's a group (dataset or globals)
                    sub_keys = list(item.keys())
                    print(f"│   Group with {len(sub_keys)} items: {sub_keys}")
                    
                    for sub_key in sub_keys:
                        sub_item = item[sub_key]
                        print(f"│   ├── {sub_key}")
                        
                        if isinstance(sub_item, h5py.Group):
                            # It's a cell group
                            cell_keys = list(sub_item.keys())
                            for cell_key in cell_keys:
                                cell_item = sub_item[cell_key]
                                if isinstance(cell_item, h5py.Dataset):
                                    # Print value for metadata fields
                                    if cell_key in ['cell_id', 'cell_type', 'datarun', 'patterns']:
                                        value = cell_item[()]
                                        # Decode bytes to string if needed
                                        if isinstance(value, bytes):
                                            value = value.decode('utf-8')
                                        print(f"│   │   │   ├── {cell_key}: {value}")
                                    else:
                                        print(f"│   │   │   ├── {cell_key}: {cell_item.dtype}, shape {list(cell_item.shape)}")
                                else:
                                    print(f"│   │   │   ├── {cell_key}: {type(cell_item).__name__}")
                        elif isinstance(sub_item, h5py.Dataset):
                            # Its a dataset (like wnoise)
                            print(f"│   │   ├── {sub_key}: {sub_item.dtype}, shape {list(sub_item.shape)}")
                        else:
                            print(f"│   │   ├── {sub_key}: {type(sub_item).__name__}")
                elif isinstance(item, h5py.Dataset):
                    # It's a dataset
                    print(f"│   Dataset: {item.dtype}, shape {list(item.shape)}")
                else:
                    print(f"│   Other: {type(item).__name__}")
            
            # Print summary statistics
            print(f"\n" + "=" * 60)
            print("SUMMARY:")
            
            # Count datasets and cells
            total_datasets = 0
            total_cells = 0
            total_stimuli = 0
            
            for key in top_keys:
                if key == 'globals':
                    continue
                    
                item = f[key]
                if isinstance(item, h5py.Group):
                    total_datasets += 1
                    
                    for sub_key in item.keys():
                        if sub_key.startswith('cell_'):
                            total_cells += 1
                            cell_group = item[sub_key]
                            if 'stim_inputs' in cell_group:
                                total_stimuli += cell_group['stim_inputs'].shape[0]
            
            print(f"Total datasets (dates): {total_datasets}")
            print(f"Total cells: {total_cells}")
            print(f"Total stimuli across all cells: {total_stimuli}")
            
            # Show electrode info
            if 'globals' in f and 'electrode_positions' in f['globals']:
                electrode_positions = f['globals']['electrode_positions']
                print(f"Electrode positions: {electrode_positions.dtype}, shape {list(electrode_positions.shape)}")
            
            print(f"\n✓ HDF5 file structure analysis complete!")
            
    except FileNotFoundError:
        print(f"Error: File {h5_file_path} not found.")
    except Exception as e:
        print(f"Error analyzing structure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    h5file = '/Volumes/Lab/Users/seijiy/extracellular-jaxley/data-packaging/triplet_outputs/dense_triplets.h5'
    print_h5_structure(h5file) 