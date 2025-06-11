import os
import streamlit as st # type: ignore
import glob
import pandas as pd
from plotting_component import render_plotting_component

def get_subset_files():
    """Get all pickle files from the subset_pickle directory"""
    if not os.path.exists('subset_pickle'):
        os.makedirs('subset_pickle')
        return []
    return [os.path.basename(file) for file in sorted(glob.glob('subset_pickle/*.pkl'))]

def show_plotting_page():
    """Display the plotting page"""
    # Get subset files
    subset_files = get_subset_files()
    
    if not subset_files:
        st.info("No subset files found. Create some subsets in the Data Filtering page first.")
        return
    
    # Subset manager (collapsible)
    with st.expander("Subset Manager", expanded=False):
        # Delete subsets
        if subset_files:
            st.markdown("##### Delete Subset")
            file_to_delete = st.selectbox("Select file to delete", subset_files, key="delete_file_select")
            if st.button("Delete", key="delete_file_button"):
                try:
                    os.remove(f"subset_pickle/{file_to_delete}")
                    st.success(f"Deleted {file_to_delete}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting file: {str(e)}")
        
        # Rename subsets
        if subset_files:
            st.markdown("##### Rename Subset")
            file_to_rename = st.selectbox("Select file to rename", subset_files, key="rename_file_select")
            new_name = st.text_input("New name (without .pkl extension)")
            
            if st.button("Rename", key="rename_file_button"):
                if not new_name:
                    st.error("Please enter a new name")
                else:
                    try:
                        os.rename(f"subset_pickle/{file_to_rename}", f"subset_pickle/{new_name}.pkl")
                        st.success(f"Renamed {file_to_rename} to {new_name}.pkl")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error renaming file: {str(e)}")
    
    # Select subsets for future use
    st.markdown("##### Select Subsets")
    if 'selected_subsets' not in st.session_state:
        st.session_state.selected_subsets = []
    
    selected_files = []
    for pkl_file in subset_files:
        if st.checkbox(pkl_file, key=f"select_{pkl_file}"):
            selected_files.append(pkl_file)
    
    if selected_files != st.session_state.selected_subsets:
        st.session_state.selected_subsets = selected_files
    
    # Plotting section
    if st.session_state.selected_subsets:
        st.markdown("---")
        st.markdown("### Data Visualization")
        
        # Load the selected dataframes
        loaded_dataframes = []
        try:
            for file in st.session_state.selected_subsets:
                df = pd.read_pickle(f"subset_pickle/{file}")
                loaded_dataframes.append(df)
            
            # Render the plotting component
            render_plotting_component(loaded_dataframes, st.session_state.selected_subsets)
        except Exception as e:
            st.error(f"Error loading dataframes: {str(e)}")
    else:
        st.info("Select at least one subset to start plotting.")