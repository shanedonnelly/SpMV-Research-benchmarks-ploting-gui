import os
import pandas as pd
import streamlit as st # type: ignore
import configparser
import glob
import re
from filter_component import filter_dataframe
from plotting_page import show_plotting_page

# Set page config
st.set_page_config(page_title="Benchmark Plotting App", page_icon="./logo.jpeg")


# Add CSS to limit width of components
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# Create directories if they don't exist
def ensure_directories():
    directories = ['pickle', 'subset_pickle']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            st.warning(f"Created missing directory: {directory}")

# Load configuration
def load_config():
    config = configparser.ConfigParser()
    try:
        config.read('config.ini')
        return {
            'max_unique': config.getint('General', 'max_unique', fallback=20),
            'show_df_by_default': config.getboolean('General', 'show_df_by_default', fallback=True)
        }
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {'max_unique': 20, 'show_df_by_default': True}

# Save configuration
def save_config(max_unique, show_df_by_default):
    config = configparser.ConfigParser()
    config['General'] = {
        'max_unique': str(max_unique),
        'show_df_by_default': str(show_df_by_default)
    }
    
    try:
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        st.success("Configuration saved successfully!")
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")

# Get available pickle files
def get_pickle_files():
    if not os.path.exists('pickle'):
        return []
    return [os.path.basename(file) for file in sorted(glob.glob('pickle/*.pkl'))]

# Find next index for subset file
def find_next_subset_index(base_name):
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.pkl$")
    existing_files = glob.glob(f"subset_pickle/{base_name}_*.pkl")
    
    # Extract existing indices
    indices = []
    for file in existing_files:
        match = pattern.search(file)
        if match:
            indices.append(int(match.group(1)))
    
    # Return next index
    if indices:
        return max(indices) + 1
    else:
        return 1

# Get default subset name without extension
def get_default_subset_name(original_name):
    base_name = original_name.split('.')[0]
    next_index = find_next_subset_index(base_name)
    return f"{base_name}_{next_index}"

# Save dataframe as subset
def save_subset(df, original_name, custom_name=None):
    # Ensure subset_pickle directory exists
    if not os.path.exists('subset_pickle'):
        os.makedirs('subset_pickle')
    
    if custom_name:
        # Use custom name but ensure it has .pkl extension
        if not custom_name.endswith('.pkl'):
            new_file_name = f"{custom_name}.pkl"
        else:
            new_file_name = custom_name
    else:
        # Use default naming convention
        base_name = original_name.split('.')[0]
        next_index = find_next_subset_index(base_name)
        new_file_name = f"{base_name}_{next_index}.pkl"
    
    try:
        df.to_pickle(f"subset_pickle/{new_file_name}")
        return True, new_file_name
    except Exception as e:
        return False, str(e)

# Settings UI
def show_settings():
    st.header("Settings")
    
    config_data = load_config()
    
    with st.form("config_form"):
        max_unique = st.number_input(
            "Maximum number of unique values for categorical filters",
            min_value=1,
            max_value=1000,
            value=config_data['max_unique']
        )
        
        show_df_default = st.checkbox(
            "Show dataframe by default",
            value=config_data['show_df_by_default']
        )
        
        if st.form_submit_button("Save Settings"):
            save_config(max_unique, show_df_default)
            st.session_state.show_settings = False
            st.rerun()
    
    if st.button("Close"):
        st.session_state.show_settings = False
        st.rerun()

# Data Filtering page
def show_data_filtering():
    config_data = load_config()
    
    # Get pickle files
    pickle_files = get_pickle_files()
    
    if not pickle_files:
        st.warning("No pickle files found. Please add CSV files to the csv/ folder and run setup.py.")
        return
    
    # Initialize session state for dataframe selection
    if 'selected_df_file' not in st.session_state:
        st.session_state.selected_df_file = pickle_files[0]
    if 'current_df' not in st.session_state:
        st.session_state.current_df = pd.read_pickle(f"pickle/{st.session_state.selected_df_file}")
    
    # Dataframe selector with properly aligned OK button
    col1, col2 = st.columns([3, 1])
    with col1:
        new_selection = st.selectbox(
            "Select DataFrame",
            options=pickle_files,
            index=pickle_files.index(st.session_state.selected_df_file) if st.session_state.selected_df_file in pickle_files else 0
        )

    # Add vertical space to align button with dropdown, not label
    with col2:
        st.write("")  # Add some vertical space
        if st.button("OK", key="load_df_button"):
            if new_selection != st.session_state.selected_df_file:
                st.session_state.selected_df_file = new_selection
                st.session_state.current_df = pd.read_pickle(f"pickle/{new_selection}")
                st.rerun()
    
    # Filter component - wrap in a container to constrain width
    with st.container():
        filtered_df = filter_dataframe(
            st.session_state.current_df, 
            max_unique=config_data['max_unique'],
            show_df_by_default=config_data['show_df_by_default']
        )
    
    # Save subset section
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    
    # Create a popover for save subset functionality
    with st.popover("Save Subset", use_container_width=False):
        st.subheader("Save Filtered DataFrame")
        
        # Get default name for the subset
        default_name = get_default_subset_name(st.session_state.selected_df_file)
        
        # Text input with default name (auto-selected)
        subset_name = st.text_input(
            "Enter a name for this subset:",
            value=default_name,
            key="subset_name_input"
        )
        
        # Save button
        if st.button("Save", key="save_confirm_button"):
            success, result = save_subset(filtered_df, st.session_state.selected_df_file, subset_name)
            if success:
                st.success(f"Saved as {result}")
            else:
                st.error(f"Error saving subset: {result}")
    
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    # Ensure directories exist
    ensure_directories()
    
    # Initialize session state
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Data Filtering"
    
    # Settings icon at top right
    col1, col2 = st.columns([20, 1])
    with col2:
        if st.button("⚙️"):
            st.session_state.show_settings = not st.session_state.show_settings
            st.rerun()
    
    # Show settings dialog if needed
    if st.session_state.show_settings:
        show_settings()
    else:
        # Simple tab navigation
        tabs = st.tabs(["Data Filtering", "Plotting"])

        # Show content based on active tab
        with tabs[0]:
            show_data_filtering()
            
        with tabs[1]:
            show_plotting_page()

if __name__ == "__main__":
    main()