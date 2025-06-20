import os
import streamlit as st
import glob
import pandas as pd
from plotting_component import render_plotting_component

def get_available_files():
    """Get all pickle files from both pickle and subset_pickle directories."""
    all_files = []
    # Get full sets from ./pickle
    if os.path.exists('pickle'):
        for file_path in sorted(glob.glob('pickle/*.pkl')):
            all_files.append({
                'name': os.path.basename(file_path),
                'path': file_path,
                'type': 'full'
            })
    # Get subsets from ./subset_pickle
    if os.path.exists('subset_pickle'):
        os.makedirs('subset_pickle', exist_ok=True)
        for file_path in sorted(glob.glob('subset_pickle/*.pkl')):
            all_files.append({
                'name': os.path.basename(file_path),
                'path': file_path,
                'type': 'subset'
            })
    return all_files

def show_plotting_page():
    """Display the plotting page with integrated dataset selection and management."""
    # Add a reset button at the top right of the page
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        if st.button("↺", key="reset_plotting_page", help="Reset all plotting controls and selections"):
            # Collect all keys related to plotting to delete them
            keys_to_delete = []
            for key in st.session_state.keys():
                if key.startswith(('plot_', 'select_')) or key in [
                    "plots_generated", "primary_dim", "secondary_dim", "y_axis",
                    "show_titles", "comparison_mode", "plot_params", "show_outliers",
                    "fig_size_mode", "fig_width_cm", "fig_height_cm", "output_format",
                    "plot_titles", "axes_label_mode", "x_axis_label", "y_axis_label",
                    "selected_subsets"
                ]:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Get all available data files
    available_files = get_available_files()
    
    if not available_files:
        st.info("No data files found. Create subsets in the 'Data Filtering' page or add files to the 'pickle' directory.")
        return
    
    # --- Dataset Selection ---
    st.markdown("##### Select Datasets")
    
    if 'selected_subsets' not in st.session_state:
        st.session_state.selected_subsets = []

    selected_paths = []
    
    for file_info in available_files:
        file_path = file_info['path']
        file_name = file_info['name']
        file_type = file_info['type']
        
        label = f"{file_name} (full set)" if file_type == 'full' else file_name
        
        # Layout for checkbox and delete button
        col1, col2 = st.columns([0.9, 0.1])
        
        with col1:
            # Use the unique file path for the checkbox key
            is_selected = st.checkbox(
                label, 
                key=f"select_{file_path}", 
                value=(file_path in st.session_state.selected_subsets)
            )
            if is_selected:
                selected_paths.append(file_path)

        with col2:
            # Show delete button only for subsets
            if file_type == 'subset':
                if st.button("⨯", key=f"delete_{file_path}"):
                    try:
                        os.remove(file_path)
                        if file_path in st.session_state.selected_subsets:
                            st.session_state.selected_subsets.remove(file_path)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {str(e)}")

    # If the selection has changed, update the session state and rerun
    if selected_paths != st.session_state.selected_subsets:
        st.session_state.selected_subsets = selected_paths
        st.rerun()
    
    # --- Plotting Section ---
    if st.session_state.selected_subsets:
        st.markdown("---")
        
        # Load the selected dataframes
        loaded_dataframes = []
        loaded_filenames = []
        try:
            for path in st.session_state.selected_subsets:
                df = pd.read_pickle(path)
                loaded_dataframes.append(df)
                loaded_filenames.append(os.path.basename(path))
            
            # Render the main plotting component
            render_plotting_component(loaded_dataframes, loaded_filenames)
        except Exception as e:
            st.error(f"Error loading dataframes: {str(e)}")
    else:
        st.info("Select at least one dataset to start plotting.")