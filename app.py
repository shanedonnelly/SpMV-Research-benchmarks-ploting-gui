\
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import os
import configparser
from filter_component import filter_dataframe # type: ignore

# --- Configuration --- 
def load_config():
    config = configparser.ConfigParser()
    if os.path.exists("config.ini"):
        config.read("config.ini")
    else: # Default config if file not found
        config['General'] = {'max_unique': '20', 'show_df_by_default': 'True'}
        with open("config.ini", "w") as f:
            config.write(f)
    return config

def save_config(config):
    with open("config.ini", "w") as configfile:
        config.write(configfile)

config = load_config()
max_unique_val = config.getint("General", "max_unique", fallback=20)
show_df_default = config.getboolean("General", "show_df_by_default", fallback=True)

# --- Session State Initialization ---
if 'show_df' not in st.session_state:
    st.session_state.show_df = show_df_default
if 'selected_pickle_file' not in st.session_state:
    st.session_state.selected_pickle_file = None
if 'df_to_filter' not in st.session_state:
    st.session_state.df_to_filter = None
if 'filtered_df_result' not in st.session_state:
    st.session_state.filtered_df_result = None
if 'show_config_editor' not in st.session_state:
    st.session_state.show_config_editor = False
if 'selected_subsets_for_plotting' not in st.session_state:
    st.session_state.selected_subsets_for_plotting = []

# --- Helper Functions ---
def get_pickle_files():
    pickle_dir = "pickle/"
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
        return []
    return sorted([f for f in os.listdir(pickle_dir) if f.endswith(".pkl")])

def load_dataframe(file_path):
    try:
        return pd.read_pickle(file_path)
    except Exception as e:
        st.error(f"Error loading dataframe: {e}")
        return None

def get_next_subset_filename(original_filename):
    pickle_dir = "pickle/"
    base, ext = os.path.splitext(original_filename)
    # Remove any existing _number suffix from base to correctly find next index
    parts = base.split('_')
    if parts[-1].isdigit():
        base = "_".join(parts[:-1])

    i = 1
    while True:
        new_filename = f"{base}_{i}{ext}"
        if not os.path.exists(os.path.join(pickle_dir, new_filename)):
            return new_filename
        i += 1

# --- Settings Icon and Editor ---
with st.sidebar:
    st.title("Settings")
    if st.button("⚙️ Edit Configuration", key="edit_config_button"):
        st.session_state.show_config_editor = not st.session_state.show_config_editor

if st.session_state.show_config_editor:
    with st.expander("Configuration Editor", expanded=True):
        st.subheader("Application Configuration")
        new_max_unique = st.number_input(
            "Max Unique Values for Categorical Filter", 
            min_value=1, 
            value=max_unique_val, 
            key="config_max_unique"
        )
        new_show_df_default = st.checkbox(
            "Show DataFrame by Default on Load", 
            value=show_df_default,
            key="config_show_df_default"
        )

        if st.button("Save Configuration", key="save_config_button"):
            config["General"]["max_unique"] = str(new_max_unique)
            config["General"]["show_df_by_default"] = str(new_show_df_default)
            save_config(config)
            st.success("Configuration saved! Please rerun the app for changes to take full effect if they alter initial state.")
            # Update live values for current session where possible
            max_unique_val = new_max_unique
            show_df_default = new_show_df_default 
            # Note: show_df_default change might need a full rerun or careful state management if it affects 'show_df' initial value logic
            st.session_state.show_config_editor = False # Optionally close editor
            st.rerun()

# --- Page Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📄 Data Filtering", "📊 Plotting"], key="page_nav")

# --- Main Application Logic ---
if page == "📄 Data Filtering":
    st.header("📄 Data Filtering")
    pickle_files = get_pickle_files()

    if not pickle_files:
        st.warning("No pickle files found in the 'pickle/' directory. Please run `setup.py` or add files.")
    else:
        # Dataframe Selector
        col1, col2 = st.columns([3,1])
        with col1:
            selected_file_from_dropdown = st.selectbox(
                "Select a DataFrame to load:", 
                pickle_files, 
                index=pickle_files.index(st.session_state.selected_pickle_file) if st.session_state.selected_pickle_file in pickle_files else 0,
                key="df_selector"
            )
        with col2:
            if st.button("Load DataFrame", key="load_df_button", use_container_width=True):
                if selected_file_from_dropdown:
                    st.session_state.selected_pickle_file = selected_file_from_dropdown
                    st.session_state.df_to_filter = load_dataframe(os.path.join("pickle/", st.session_state.selected_pickle_file))
                    st.session_state.filtered_df_result = None # Reset filtered result on new load
                    st.session_state.show_df = show_df_default # Reset to default visibility
                     # Reset filter component's internal state by clearing its session variables
                    if 'filters_applied' in st.session_state: del st.session_state['filters_applied']
                    if 'previous_columns' in st.session_state: del st.session_state['previous_columns']
                    # Add any other specific keys used by filter_component that need reset
                    # This is a bit of a hack; ideally, the component would have a reset function.
                    # For now, we list known keys. A more robust way would be for the component to manage this.
                    # Example: for key in list(st.session_state.keys()): if key.startswith("filter_multiselect_cols") or key.endswith(("_reset_flag", "_pending_changes", "_expander_open", "_selected_values", "_search_text", "_filter_min", "_filter_max", "_unique_value", "_numeric_expanded", "_date_range", "_text_filter")) : del st.session_state[key]
                    st.rerun()

        if st.session_state.df_to_filter is not None:
            st.subheader(f"Filtering: {st.session_state.selected_pickle_file}")
            st.session_state.filtered_df_result = filter_dataframe(st.session_state.df_to_filter, max_unique=max_unique_val)

            # Show DataFrame checkbox and display
            if st.session_state.filtered_df_result is not None:
                new_show_df_state = st.checkbox("Show DataFrame", value=st.session_state.show_df, key="show_df_checkbox_main")
                if new_show_df_state != st.session_state.show_df:
                    st.session_state.show_df = new_show_df_state
                    st.rerun()
                
                st.caption(f"Filtered rows: {len(st.session_state.filtered_df_result)} / Original rows: {len(st.session_state.df_to_filter)}")
                if st.session_state.show_df:
                    st.dataframe(st.session_state.filtered_df_result)

                # Save Subset button
                st.markdown("---_---_---") # Visual separator
                cols_save = st.columns([1,2,1]) # Centering the button
                with cols_save[1]:
                    if st.button("💾 Save Filtered Subset", key="save_subset_button", use_container_width=True):
                        if st.session_state.filtered_df_result is not None and st.session_state.selected_pickle_file:
                            subset_filename = get_next_subset_filename(st.session_state.selected_pickle_file)
                            save_path = os.path.join("pickle/", subset_filename)
                            try:
                                st.session_state.filtered_df_result.to_pickle(save_path)
                                st.success(f"Subset saved as `{subset_filename}`")
                            except Exception as e:
                                st.error(f"Error saving subset: {e}")
                        else:
                            st.warning("No filtered data to save or original file not selected.")
        else:
            st.info("Select a DataFrame and click 'Load DataFrame' to begin filtering.")

elif page == "📊 Plotting":
    st.header("📊 Plotting")
    st.write("Manage and select data subsets for plotting.")

    pickle_files = get_pickle_files()
    subset_files = [f for f in pickle_files if '_' in os.path.splitext(f)[0] and os.path.splitext(f)[0].split('_')[-1].isdigit()]

    with st.expander("Subset Manager", expanded=True):
        if not subset_files:
            st.info("No subsets found. Save a filtered version from the 'Data Filtering' page first.")
        else:
            for i, subset_file in enumerate(subset_files):
                col1, col2, col3 = st.columns([3,1,1])
                with col1:
                    new_name_base = os.path.splitext(subset_file)[0]
                    # Attempt to remove numeric suffix for cleaner display in rename input
                    parts = new_name_base.split('_')
                    display_name_for_rename = "_".join(parts[:-1]) if len(parts) > 1 and parts[-1].isdigit() else new_name_base
                    
                    new_subset_name_input = st.text_input(f"Rename '{subset_file}' to (excluding _index.pkl):", value=display_name_for_rename, key=f"rename_{subset_file}_{i}")
                
                with col2:
                    if st.button("Rename", key=f"rename_btn_{subset_file}_{i}", use_container_width=True):
                        original_path = os.path.join("pickle/", subset_file)
                        # Ensure the new name is valid and construct the full new filename
                        # It should retain its original numeric suffix or get a new one if base name changes significantly
                        # For simplicity, let's assume renaming changes the base part, and we might need a new index if it clashes
                        # This logic can be complex. A simpler rename might just change the descriptive part.
                        # For now, let's try to keep the suffix or re-evaluate.
                        
                        # Simplified: User provides new base name. We find next available index for THAT base.
                        # This means original_df_1.pkl renamed to new_name.pkl becomes new_name_1.pkl
                        # This is not ideal. A better rename would be original_df_1.pkl to my_subset_1.pkl
                        # Let's stick to renaming the base part and keeping the index for now.
                        
                        original_base, original_suffix_ext = os.path.splitext(subset_file)
                        original_parts = original_base.split('_')
                        original_index_str = ""
                        if original_parts[-1].isdigit():
                            original_index_str = original_parts[-1]
                            # new_name_base_for_rename = "_".join(original_parts[:-1]) # This was for display
                        
                        # The user input new_subset_name_input is the new base (e.g. "my_new_name")
                        # We need to append the original index to it: "my_new_name_1.pkl"
                        if original_index_str: # if it had an index
                            new_full_filename = f"{new_subset_name_input}_{original_index_str}{original_suffix_ext}"
                        else: # if it somehow didn't have an index (shouldn't happen for subsets by convention)
                             new_full_filename = f"{new_subset_name_input}{original_suffix_ext}" 

                        new_path = os.path.join("pickle/", new_full_filename)

                        if new_subset_name_input and new_path != original_path:
                            if not os.path.exists(new_path):
                                try:
                                    os.rename(original_path, new_path)
                                    st.success(f"Renamed '{subset_file}' to '{new_full_filename}'")
                                    # Update selection if the renamed file was selected
                                    if subset_file in st.session_state.selected_subsets_for_plotting:
                                        st.session_state.selected_subsets_for_plotting.remove(subset_file)
                                        st.session_state.selected_subsets_for_plotting.append(new_full_filename)
                                    st.rerun()
                                except OSError as e:
                                    st.error(f"Error renaming file: {e}")
                            else:
                                st.error(f"File '{new_full_filename}' already exists.")
                        elif new_path == original_path:
                            st.warning("New name is the same as the old name.")
                        else:
                            st.warning("New name cannot be empty.")

                with col3:
                    if st.button("Delete", key=f"delete_{subset_file}_{i}", type="secondary", use_container_width=True):
                        try:
                            os.remove(os.path.join("pickle/", subset_file))
                            st.success(f"Deleted '{subset_file}'")
                            if subset_file in st.session_state.selected_subsets_for_plotting:
                                st.session_state.selected_subsets_for_plotting.remove(subset_file)
                            st.rerun()
                        except OSError as e:
                            st.error(f"Error deleting file: {e}")
                st.markdown("---_---")

    st.subheader("Select Subsets for Plotting")
    if not subset_files:
        st.info("No subsets available to select.")
    else:
        current_selection = st.session_state.selected_subsets_for_plotting[:]
        for subset_file in subset_files:
            is_selected = subset_file in current_selection
            if st.checkbox(subset_file, value=is_selected, key=f"select_plot_{subset_file}"):
                if not is_selected:
                    current_selection.append(subset_file)
            elif is_selected:
                current_selection.remove(subset_file)
        
        if current_selection != st.session_state.selected_subsets_for_plotting:
            st.session_state.selected_subsets_for_plotting = current_selection
            # st.rerun() # Rerun if selection changes to update UI or trigger downstream logic

    st.write("Selected subsets for plotting:", st.session_state.selected_subsets_for_plotting)
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True) # Empty space for future plots
    st.info("Plotting area - to be implemented.")


# --- Entry Point for Streamlit --- 
# (No explicit main function needed for streamlit, it runs top-down)
# Ensure `streamlit_app.py` is the file you run with `streamlit run streamlit_app.py`
