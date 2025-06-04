import pandas as pd
import streamlit as st # type: ignore
import configparser
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_integer_dtype,
    is_float_dtype
)

st.title("Auto Filter Dataframes in Streamlit")

# Initialize session state variables
if 'removed_filters' not in st.session_state:
    st.session_state.removed_filters = set()
if 'show_df' not in st.session_state:
    st.session_state.show_df = False
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = {}

def filter_dataframe(df: pd.DataFrame, max_unique: int) -> pd.DataFrame:
    """Adds a UI to filter dataframe columns"""
    df_copy = df.copy()

    # Convert date-like object columns to datetime
    for col in df_copy.columns:
        if is_object_dtype(df_copy[col]):
            try:
                sample = df_copy[col].dropna().iloc[:10] if not df_copy[col].empty else []
                likely_dates = all(isinstance(s, str) and (len(s) >= 8 and ('/' in s or '-' in s or ':' in s)) for s in sample)
                if likely_dates:
                    temp_series = pd.to_datetime(df_copy[col], errors='coerce', format="%d/%m/%Y")
                    if temp_series.notna().sum() > 0.5 * df_copy[col].count():
                        df_copy[col] = temp_series
            except Exception:
                pass
        if is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.tz_localize(None)

    with st.container():
        to_filter_columns = st.multiselect("Filter dataframe on", df_copy.columns)

        # Manage filter state for removed/re-added columns
        active_filter_columns = set(to_filter_columns)
        if 'previous_columns' not in st.session_state:
            st.session_state.previous_columns = set()

        removed_cols_from_multiselect = st.session_state.previous_columns - active_filter_columns
        for col_to_remove_filter_state in removed_cols_from_multiselect:
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith(f"{col_to_remove_filter_state}_")]
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            if col_to_remove_filter_state in st.session_state.filters_applied:
                del st.session_state.filters_applied[col_to_remove_filter_state]
        st.session_state.previous_columns = active_filter_columns
        
        # This will be the DataFrame that is progressively filtered by "OK" button clicks
        # Start with a fresh copy each time filter_dataframe is called
        # Actual filtering happens conditionally later
        intermediate_filtered_df = df_copy.copy()

        for column in to_filter_columns:
            # Adjust column ratio to give slightly more space for column names (was 3:17)
            left, right = st.columns((5, 15))
            # Use markdown for slightly larger font for the column name
            left.markdown(f"##### ↳ {column}") 

            is_categorical = isinstance(df_copy[column].dtype, pd.CategoricalDtype)
            is_low_cardinality = (
                not df_copy[column].empty
                and df_copy[column].nunique() < max_unique
                and not is_float_dtype(df_copy[column])
                and not is_integer_dtype(df_copy[column])
            )

            # Initialize common session state keys for each filter type
            reset_key_generic = f"{column}_reset_flag"
            if reset_key_generic not in st.session_state:
                st.session_state[reset_key_generic] = False
            
            if f"{column}_expander_open" not in st.session_state: # For categorical
                st.session_state[f"{column}_expander_open"] = False

            # Add state to track numeric filter expandability
            if f"{column}_numeric_expanded" not in st.session_state:
                st.session_state[f"{column}_numeric_expanded"] = True

            if is_categorical or is_low_cardinality:
                unique_values_for_col = df_copy[column].unique()
                filter_container = right.container()
                # Adjust ratios: make OK and Reset button columns (2nd and 3rd) equally small
                # Old: filter_container.columns([9, 4, 1])
                col1, col2_ok, col3_reset = filter_container.columns([6, 1, 1])

                # Initialize states for categorical filter
                if f"{column}_selected_values" not in st.session_state:
                    st.session_state[f"{column}_selected_values"] = list(unique_values_for_col)
                if f"{column}_search_text" not in st.session_state:
                    st.session_state[f"{column}_search_text"] = ""

                if st.session_state[reset_key_generic]:
                    st.session_state[f"{column}_selected_values"] = list(unique_values_for_col)
                    st.session_state[f"{column}_search_text"] = ""
                    if column in st.session_state.filters_applied:
                        del st.session_state.filters_applied[column]
                    st.session_state[reset_key_generic] = False # Reset the flag

                selected_count = len(st.session_state[f"{column}_selected_values"])
                total_count = len(unique_values_for_col)
                
                expander_label = f"Values ({selected_count}/{total_count} selected)"
                with col1.expander(expander_label, expanded=st.session_state[f"{column}_expander_open"]):
                    st.session_state[f"{column}_expander_open"] = True # Keep it open during interaction

                    search_text_current = st.session_state[f"{column}_search_text"]
                    new_search_text = st.text_input("Search values", value=search_text_current, key=f"{column}_search_widget")

                    if new_search_text != search_text_current:
                        st.session_state[f"{column}_search_text"] = new_search_text
                        if new_search_text: # If search text is not empty, auto-select matching
                            st.session_state[f"{column}_selected_values"] = [
                                val for val in unique_values_for_col if new_search_text.lower() in str(val).lower()
                            ]
                        # If search text is empty, selection remains as is, user can see all options
                        st.session_state[f"{column}_expander_open"] = True
                        st.rerun() # Rerun to reflect search and selection changes

                    # Determine values to display based on current search
                    display_values = sorted([
                        val for val in unique_values_for_col if st.session_state[f"{column}_search_text"].lower() in str(val).lower()
                    ]) if st.session_state[f"{column}_search_text"] else sorted(unique_values_for_col)

                    st.write('<div style="display:flex;gap:10px">', unsafe_allow_html=True)
                    if st.button("Select All Visible", key=f"{column}_select_all_visible"):
                        current_sel = set(st.session_state[f"{column}_selected_values"])
                        for val in display_values: current_sel.add(val)
                        st.session_state[f"{column}_selected_values"] = list(current_sel)
                        st.session_state[f"{column}_expander_open"] = True
                        st.rerun()
                    if st.button("Deselect All Visible", key=f"{column}_deselect_all_visible"):
                        current_sel = set(st.session_state[f"{column}_selected_values"])
                        st.session_state[f"{column}_selected_values"] = list(current_sel - set(display_values))
                        st.session_state[f"{column}_expander_open"] = True
                        st.rerun()
                    st.write('</div>', unsafe_allow_html=True)

                    # Checkboxes for displayed values
                    temp_selection = st.session_state[f"{column}_selected_values"][:] # Operate on a copy
                    changed_in_checkboxes = False
                    for i, value in enumerate(display_values):
                        is_checked = value in temp_selection
                        if st.checkbox(str(value), value=is_checked, key=f"{column}_val_cb_{i}"):
                            if not is_checked:
                                temp_selection.append(value)
                                changed_in_checkboxes = True
                        elif is_checked:
                            temp_selection.remove(value)
                            changed_in_checkboxes = True
                    
                    if changed_in_checkboxes:
                        st.session_state[f"{column}_selected_values"] = temp_selection
                        st.session_state[f"{column}_expander_open"] = True
                        st.rerun()

                if col2_ok.button("OK", key=f"{column}_ok_cat"):
                    st.session_state.filters_applied[column] = True
                    st.session_state[f"{column}_expander_open"] = False # Close expander on OK
                    st.rerun()

                if col3_reset.button("↺", key=f"{column}_reset_cat_btn"):
                    st.session_state[reset_key_generic] = True
                    st.session_state[f"{column}_expander_open"] = False
                    # Preserve show_df state
                    current_show_df = st.session_state.show_df
                    st.session_state.show_df = current_show_df
                    st.rerun()
                
            elif is_numeric_dtype(df_copy[column]):
                if df_copy[column].empty:
                    right.text(f"{column}: No data to filter.")
                    continue

                col_min_original = float(df_copy[column].min())
                col_max_original = float(df_copy[column].max())
                is_int = is_integer_dtype(df_copy[column]) or df_copy[column].equals(df_copy[column].round(0))
                
                # Ensure step is float to avoid MixedNumericTypesError
                step, num_format = (1.0, None) if is_int else (0.1, "%.5f")

                # Create a container for the numeric filter
                numeric_container = right.container()
                
                # Initialize states for numeric filter
                min_key, max_key, unique_key_val = f"{column}_filter_min", f"{column}_filter_max", f"{column}_unique_value"
                if min_key not in st.session_state: st.session_state[min_key] = col_min_original
                if max_key not in st.session_state: st.session_state[max_key] = col_max_original
                if unique_key_val not in st.session_state: st.session_state[unique_key_val] = None

                # Reset handling
                if st.session_state[reset_key_generic]:
                    st.session_state[min_key] = col_min_original
                    st.session_state[max_key] = col_max_original
                    st.session_state[unique_key_val] = None
                    if column in st.session_state.filters_applied:
                        del st.session_state.filters_applied[column]
                    st.session_state[reset_key_generic] = False
                    # Don't auto-expand on reset - let user control

                # Create a collapsible section for numeric details
                with numeric_container.expander("Filter details", expanded=st.session_state.get(f"{column}_numeric_expanded", True)):
                    # Fix - directly use the expander instead of creating nested columns
                    # This line created the nesting issue:
                    # main_area, button_area = st.columns([9, 1])
                    
                    # Min, Max, Unique inputs side by side
                    cols1, cols2, cols3 = st.columns(3)
                    
                    # Min input
                    cols1.text("Min")
                    new_min_val = cols1.number_input(
                        "Min", 
                        col_min_original, 
                        col_max_original, 
                        st.session_state[min_key], 
                        step, 
                        format=num_format, 
                        label_visibility="collapsed", 
                        key=f"{column}_min_input_widget"
                    )
                    if new_min_val != st.session_state[min_key]:
                        st.session_state[min_key] = new_min_val
                        st.rerun()

                    # Max input
                    cols2.text("Max")
                    new_max_val = cols2.number_input(
                        "Max", 
                        col_min_original, 
                        col_max_original, 
                        st.session_state[max_key], 
                        step, 
                        format=num_format, 
                        label_visibility="collapsed", 
                        key=f"{column}_max_input_widget"
                    )
                    if new_max_val != st.session_state[max_key]:
                        st.session_state[max_key] = new_max_val
                        st.rerun()
                    
                    # Unique input
                    cols3.text("Unique")
                    current_unique_val_state = st.session_state[unique_key_val]
                    new_unique_val_input = cols3.number_input(
                        "Unique", 
                        col_min_original, 
                        col_max_original, 
                        current_unique_val_state, 
                        step, 
                        format=num_format, 
                        label_visibility="collapsed", 
                        key=f"{column}_unique_input_widget"
                    )
                    
                    if new_unique_val_input != current_unique_val_state:
                        st.session_state[unique_key_val] = new_unique_val_input
                        if new_unique_val_input is not None:
                            st.session_state[min_key] = new_unique_val_input
                            st.session_state[max_key] = new_unique_val_input
                        st.rerun()
                    
                    # Slider 
                    current_slider_min = st.session_state[min_key]
                    current_slider_max = st.session_state[max_key]
                    
                    # Ensure slider values are valid
                    if current_slider_min > current_slider_max: current_slider_min = current_slider_max
                    if current_slider_max < current_slider_min: current_slider_max = current_slider_min
                    
                    slider_value = (current_slider_min, current_slider_max)
                    
                    new_slider_vals = st.slider(
                        f"Range for {column}", 
                        col_min_original, 
                        col_max_original, 
                        value=slider_value, 
                        step=step, 
                        key=f"{column}_slider_widget"
                    )
                    if new_slider_vals != slider_value:
                        st.session_state[min_key], st.session_state[max_key] = new_slider_vals
                        st.session_state[unique_key_val] = None # Clear unique if slider is moved
                        st.rerun()
                    
                    # Fix OK button width with CSS to prevent text wrapping
                    st.markdown("""
                    <style>
                    /* Make button text stay on one line */
                    div[data-testid="stButton"] button {
                        min-width: 48px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons in a row
                    button_cols = st.columns([1, 1, 8])
                    
                    # OK button inside expander
                    if button_cols[0].button("OK", key=f"{column}_ok_num"):
                        st.session_state.filters_applied[column] = True
                        st.rerun()
                    
                    # Reset button inside expander
                    if button_cols[1].button("↺", key=f"{column}_reset_num_btn"):
                        st.session_state[reset_key_generic] = True
                        current_show_df = st.session_state.show_df
                        st.session_state.show_df = current_show_df
                        st.rerun()

            elif is_datetime64_any_dtype(df_copy[column]):
                if df_copy[column].empty:
                    right.text(f"{column}: No data to filter.")
                    continue
                
                min_date_orig = df_copy[column].min().date()
                max_date_orig = df_copy[column].max().date()
                date_range_key = f"{column}_date_range"

                if date_range_key not in st.session_state:
                    st.session_state[date_range_key] = (min_date_orig, max_date_orig)

                if st.session_state[reset_key_generic]:
                    st.session_state[date_range_key] = (min_date_orig, max_date_orig)
                    if column in st.session_state.filters_applied:
                        del st.session_state.filters_applied[column]
                    st.session_state[reset_key_generic] = False

                date_cols = right.columns([9, 1, 1])
                current_date_range = st.session_state[date_range_key]
                new_date_input = date_cols[0].date_input(f"Values for {column}", value=current_date_range, min_value=min_date_orig, max_value=max_date_orig, key=f"{column}_date_input_widget")
                
                if len(new_date_input) == 2 and new_date_input != current_date_range :
                    st.session_state[date_range_key] = new_date_input
                    st.rerun()

                if date_cols[1].button("OK", key=f"{column}_ok_date"):
                    st.session_state.filters_applied[column] = True
                    st.rerun()
                
                if date_cols[2].button("↺", key=f"{column}_reset_date_btn"):
                    st.session_state[reset_key_generic] = True
                    current_show_df = st.session_state.show_df
                    st.session_state.show_df = current_show_df
                    st.rerun()

            else:  # Text input
                text_filter_key = f"{column}_text_filter"
                if text_filter_key not in st.session_state:
                    st.session_state[text_filter_key] = ""

                if st.session_state[reset_key_generic]:
                    st.session_state[text_filter_key] = ""
                    if column in st.session_state.filters_applied:
                        del st.session_state.filters_applied[column]
                    st.session_state[reset_key_generic] = False

                text_cols = right.columns([9, 1, 1])
                current_text_filter = st.session_state[text_filter_key]
                new_text_input = text_cols[0].text_input(f"Substring or regex in {column}", value=current_text_filter, key=f"{column}_text_input_widget")

                if new_text_input != current_text_filter:
                    st.session_state[text_filter_key] = new_text_input
                    st.rerun()

                if text_cols[1].button("OK", key=f"{column}_ok_text"):
                    st.session_state.filters_applied[column] = True
                    st.rerun()
                
                if text_cols[2].button("↺", key=f"{column}_reset_text_btn"):
                    st.session_state[reset_key_generic] = True
                    current_show_df = st.session_state.show_df
                    st.session_state.show_df = current_show_df
                    st.rerun()
            
            # Apply filter to intermediate_filtered_df if "OK" was pressed for this column
            if column in st.session_state.filters_applied and st.session_state.filters_applied[column]:
                if is_categorical or is_low_cardinality:
                    selected_vals = st.session_state.get(f"{column}_selected_values", [])
                    intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].isin(selected_vals)]
                elif is_numeric_dtype(df_copy[column]):
                    min_val_to_apply = st.session_state.get(f"{column}_filter_min", col_min_original)
                    max_val_to_apply = st.session_state.get(f"{column}_filter_max", col_max_original)
                    intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].between(min_val_to_apply, max_val_to_apply)]
                elif is_datetime64_any_dtype(df_copy[column]):
                    date_range_to_apply = st.session_state.get(f"{column}_date_range")
                    if date_range_to_apply and len(date_range_to_apply) == 2:
                        start_date, end_date = pd.to_datetime(date_range_to_apply[0]), pd.to_datetime(date_range_to_apply[1])
                        intermediate_filtered_df = intermediate_filtered_df.loc[intermediate_filtered_df[column].between(start_date, end_date)]
                else: # Text
                    text_to_apply = st.session_state.get(f"{column}_text_filter", "")
                    if text_to_apply:
                        intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].astype(str).str.contains(text_to_apply, case=False, na=False)]
        
        return intermediate_filtered_df


# Load data
try:
    df_original = pd.read_parquet("synthetic_benchmarks_all-devices_all.parquet")
except FileNotFoundError:
    st.error("Required file not found. Please ensure 'synthetic_benchmarks_all-devices_all.parquet' exists.")
    st.stop()

# Load configuration
config = configparser.ConfigParser()
try:
    with open("config.ini", "r") as f:
        config.read_file(f)
    max_unique_val = config.getint("General", "max_unique", fallback=50)
except FileNotFoundError:
    max_unique_val = 50

# Filter and display dataframe
filtered_df_result = filter_dataframe(df_original, max_unique=max_unique_val)

with st.container():
    # Use st.session_state.show_df for the checkbox's value
    new_show_df_state = st.checkbox("Show DataFrame", value=st.session_state.show_df, key="show_df_checkbox")
    if new_show_df_state != st.session_state.show_df:
        st.session_state.show_df = new_show_df_state
        st.rerun() # Rerun if checkbox state changes to immediately show/hide

    st.caption(f"Filtered rows: {len(filtered_df_result)}")

if st.session_state.show_df:
    st.dataframe(filtered_df_result)