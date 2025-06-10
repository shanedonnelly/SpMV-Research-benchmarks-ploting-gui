\
import pandas as pd
import streamlit as st # type: ignore
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_integer_dtype,
)

# Helper function for categorical filters
def _handle_categorical_filter(df_col: pd.Series, col_name: str, container: st.container):
    reset_key = f"{col_name}_reset_flag"
    pending_key = f"{col_name}_pending_changes"
    expander_open_key = f"{col_name}_expander_open"
    selected_values_key = f"{col_name}_selected_values"
    search_text_key = f"{col_name}_search_text"

    if expander_open_key not in st.session_state:
        st.session_state[expander_open_key] = False
    
    unique_values = list(df_col.unique())
    if selected_values_key not in st.session_state:
        st.session_state[selected_values_key] = unique_values
    if search_text_key not in st.session_state:
        st.session_state[search_text_key] = ""

    if st.session_state.get(reset_key, False): # Use .get for safety
        st.session_state[selected_values_key] = unique_values
        st.session_state[search_text_key] = ""
        if col_name in st.session_state.filters_applied:
            del st.session_state.filters_applied[col_name]
        st.session_state[pending_key] = False
        st.session_state[reset_key] = False
        # Expander remains as is, user controls it

    selected_count = len(st.session_state[selected_values_key])
    total_count = len(unique_values)
    expander_label = f"Values ({selected_count}/{total_count} selected)"

    with container.expander(expander_label, expanded=st.session_state[expander_open_key]):
        st.session_state[expander_open_key] = True # Keep open during interaction

        search_text_current = st.session_state[search_text_key]
        new_search_text = st.text_input("Search values", value=search_text_current, key=f"{col_name}_search_widget")
        if new_search_text != search_text_current:
            st.session_state[search_text_key] = new_search_text
            st.session_state[pending_key] = True
            if new_search_text:
                st.session_state[selected_values_key] = [
                    val for val in unique_values if new_search_text.lower() in str(val).lower()
                ]
            else: # if search is cleared, show all unique values again
                st.session_state[selected_values_key] = unique_values
            st.rerun()

        display_values = sorted([
            val for val in unique_values if st.session_state[search_text_key].lower() in str(val).lower()
        ]) if st.session_state[search_text_key] else sorted(unique_values)

        button_cols_select = st.columns(2)
        if button_cols_select[0].button("Select All Visible", key=f"{col_name}_select_all_visible"):
            current_sel = set(st.session_state[selected_values_key])
            for val in display_values: current_sel.add(val)
            st.session_state[selected_values_key] = list(current_sel)
            st.session_state[pending_key] = True
            st.rerun()
        if button_cols_select[1].button("Deselect All Visible", key=f"{col_name}_deselect_all_visible"):
            current_sel = set(st.session_state[selected_values_key])
            st.session_state[selected_values_key] = list(current_sel - set(display_values))
            st.session_state[pending_key] = True
            st.rerun()

        temp_selection = st.session_state[selected_values_key][:]
        changed_in_checkboxes = False
        for i, value in enumerate(display_values):
            is_checked = value in temp_selection
            # Use a more unique key for the checkbox itself to avoid conflicts if values are similar
            checkbox_key = f"{col_name}_val_cb_{str(value)}_{i}"
            if st.checkbox(str(value), value=is_checked, key=checkbox_key):
                if not is_checked: temp_selection.append(value); changed_in_checkboxes = True
            elif is_checked: temp_selection.remove(value); changed_in_checkboxes = True
        
        if changed_in_checkboxes:
            st.session_state[selected_values_key] = temp_selection
            st.session_state[pending_key] = True
            st.rerun()

        action_button_cols = st.columns([1,1,8]) # OK, Reset, spacer
        button_type = "primary" if st.session_state.get(pending_key, False) else "secondary"
        if action_button_cols[0].button("OK", key=f"{col_name}_ok_cat", type=button_type):
            st.session_state.filters_applied[col_name] = True # Mark as applied
            st.session_state[pending_key] = False
            st.rerun()
        if action_button_cols[1].button("↺", key=f"{col_name}_reset_cat_btn"):
            st.session_state[reset_key] = True
            st.rerun()

# Helper function for numeric filters
def _handle_numeric_filter(df_col: pd.Series, col_name: str, container: st.container):
    if df_col.empty:
        container.text(f"{col_name}: No data to filter.")
        return

    reset_key = f"{col_name}_reset_flag"
    pending_key = f"{col_name}_pending_changes"
    min_val_key, max_val_key = f"{col_name}_filter_min", f"{col_name}_filter_max"
    unique_val_key = f"{col_name}_unique_value"
    expanded_key = f"{col_name}_numeric_expanded"

    col_min_orig, col_max_orig = float(df_col.min()), float(df_col.max())
    # Check if all values are integers or can be represented as integers
    is_int_equivalent = (df_col.dropna() == df_col.dropna().round(0)).all()
    is_int = is_integer_dtype(df_col) or is_int_equivalent
    
    step, num_format = (1.0, "%.0f") if is_int else (0.01, "%.5f") 
    if col_min_orig == col_max_orig and is_int: # Handle single integer value case
        num_format = "%.0f"
    elif col_min_orig == col_max_orig: # Single float value
        num_format = "%.5f"


    if min_val_key not in st.session_state: st.session_state[min_val_key] = col_min_orig
    if max_val_key not in st.session_state: st.session_state[max_val_key] = col_max_orig
    if unique_val_key not in st.session_state: st.session_state[unique_val_key] = None # Explicitly None
    if expanded_key not in st.session_state: st.session_state[expanded_key] = True


    if st.session_state.get(reset_key, False): # Use .get for safety
        st.session_state[min_val_key], st.session_state[max_val_key] = col_min_orig, col_max_orig
        st.session_state[unique_val_key] = None
        if col_name in st.session_state.filters_applied: del st.session_state.filters_applied[col_name]
        st.session_state[pending_key] = False
        st.session_state[reset_key] = False

    with container.expander("Filter details", expanded=st.session_state[expanded_key]):
        num_cols = st.columns(3)
        
        # Min Input
        current_min_val_for_input = st.session_state[min_val_key]
        new_min = num_cols[0].number_input("Min", min_value=col_min_orig, max_value=col_max_orig, value=current_min_val_for_input, step=step, format=num_format, key=f"{col_name}_min_in")
        if new_min != current_min_val_for_input: # Compare with value used in widget
            st.session_state[min_val_key] = new_min
            st.session_state[pending_key] = True; st.rerun()
        
        # Max Input
        current_max_val_for_input = st.session_state[max_val_key]
        new_max = num_cols[1].number_input("Max", min_value=col_min_orig, max_value=col_max_orig, value=current_max_val_for_input, step=step, format=num_format, key=f"{col_name}_max_in")
        if new_max != current_max_val_for_input: # Compare with value used in widget
            st.session_state[max_val_key] = new_max
            st.session_state[pending_key] = True; st.rerun()

        # Unique Input
        current_unique_val_for_input = st.session_state[unique_val_key]
        new_unique = num_cols[2].number_input("Unique", min_value=col_min_orig, max_value=col_max_orig, value=current_unique_val_for_input if current_unique_val_for_input is not None else col_min_orig, step=step, format=num_format, key=f"{col_name}_unique_in", help="Set Min/Max to this value")
        # The number_input for "Unique" might return a float if empty, handle this.
        # For now, assume if it's not None, it's a valid number.
        # The main issue is if the user clears it, what value it gets. Streamlit's number_input behavior can be tricky.
        # Let's assume if it's different from session_state, it's a change.
        if new_unique != st.session_state[unique_val_key]: # This comparison might need care if new_unique can be None from clearing input
            st.session_state[unique_val_key] = new_unique
            if new_unique is not None: # If a unique value is set
                st.session_state[min_val_key], st.session_state[max_val_key] = new_unique, new_unique
            st.session_state[pending_key] = True; st.rerun()

        # Slider
        # Ensure slider values are valid and within bounds before passing to st.slider
        current_slider_min = float(st.session_state[min_val_key])
        current_slider_max = float(st.session_state[max_val_key])

        # Clamp values to be within the original column min/max and ensure min <= max
        current_slider_min = max(col_min_orig, min(col_max_orig, current_slider_min))
        current_slider_max = max(col_min_orig, min(col_max_orig, current_slider_max))
        if current_slider_min > current_slider_max: current_slider_min = current_slider_max
        
        slider_vals = (current_slider_min, current_slider_max)

        # Disable slider if min and max of column are the same
        disable_slider = col_min_orig == col_max_orig

        new_slider_vals = st.slider(f"Range for {col_name}", col_min_orig, col_max_orig, slider_vals, step, key=f"{col_name}_slider", disabled=disable_slider)
        if not disable_slider and new_slider_vals != slider_vals:
            st.session_state[min_val_key], st.session_state[max_val_key] = new_slider_vals
            st.session_state[unique_val_key] = None # Clear unique if slider moved
            st.session_state[pending_key] = True; st.rerun()

        action_button_cols = st.columns([1,1,8])
        button_type = "primary" if st.session_state.get(pending_key, False) else "secondary"
        if action_button_cols[0].button("OK", key=f"{col_name}_ok_num", type=button_type):
            st.session_state.filters_applied[col_name] = True # Mark as applied
            st.session_state[pending_key] = False; st.rerun()
        if action_button_cols[1].button("↺", key=f"{col_name}_reset_num_btn"):
            st.session_state[reset_key] = True; st.rerun()

# Helper function for datetime filters
def _handle_datetime_filter(df_col: pd.Series, col_name: str, container: st.container):
    if df_col.empty:
        container.text(f"{col_name}: No data to filter.")
        return

    reset_key = f"{col_name}_reset_flag"
    pending_key = f"{col_name}_pending_changes"
    range_key = f"{col_name}_date_range"

    # Ensure conversion to datetime if not already, then extract date part
    df_col_datetime = pd.to_datetime(df_col).dt.date
    min_date_orig, max_date_orig = df_col_datetime.min(), df_col_datetime.max()

    if range_key not in st.session_state:
        st.session_state[range_key] = (min_date_orig, max_date_orig)

    if st.session_state.get(reset_key, False): # Use .get for safety
        st.session_state[range_key] = (min_date_orig, max_date_orig)
        if col_name in st.session_state.filters_applied: del st.session_state.filters_applied[col_name]
        st.session_state[pending_key] = False
        st.session_state[reset_key] = False

    date_cols = container.columns([9, 1, 1]) # Adjusted for potentially wider date input
    current_range = st.session_state[range_key]
    
    # Ensure current_range values are valid dates before passing to date_input
    # This can happen if session_state was somehow corrupted or not initialized correctly
    try:
        val_for_widget = tuple(pd.to_datetime(d).date() for d in current_range)
    except: # Fallback if conversion fails
        val_for_widget = (min_date_orig, max_date_orig)


    new_range = date_cols[0].date_input(f"Values for {col_name}", value=val_for_widget, min_value=min_date_orig, max_value=max_date_orig, key=f"{col_name}_date_in")
    
    # date_input returns a tuple of datetime.date objects
    if len(new_range) == 2 and new_range != current_range : # Compare with session state value
        st.session_state[range_key] = new_range
        st.session_state[pending_key] = True; st.rerun()

    button_type = "primary" if st.session_state.get(pending_key, False) else "secondary"
    if date_cols[1].button("OK", key=f"{col_name}_ok_date", type=button_type):
        st.session_state.filters_applied[col_name] = True # Mark as applied
        st.session_state[pending_key] = False; st.rerun()
    if date_cols[2].button("↺", key=f"{col_name}_reset_date_btn"):
        st.session_state[reset_key] = True; st.rerun()

# Helper function for text filters
def _handle_text_filter(df_col: pd.Series, col_name: str, container: st.container):
    reset_key = f"{col_name}_reset_flag"
    pending_key = f"{col_name}_pending_changes"
    text_filter_key = f"{col_name}_text_filter"

    if text_filter_key not in st.session_state: st.session_state[text_filter_key] = ""

    if st.session_state.get(reset_key, False): # Use .get for safety
        st.session_state[text_filter_key] = ""
        if col_name in st.session_state.filters_applied: del st.session_state.filters_applied[col_name]
        st.session_state[pending_key] = False
        st.session_state[reset_key] = False

    text_cols = container.columns([9, 1, 1]) # Adjusted for potentially wider text input
    current_text = st.session_state[text_filter_key]
    new_text = text_cols[0].text_input(f"Substring or regex in {col_name}", current_text, key=f"{col_name}_text_in")
    if new_text != current_text:
        st.session_state[text_filter_key] = new_text
        st.session_state[pending_key] = True; st.rerun()

    button_type = "primary" if st.session_state.get(pending_key, False) else "secondary"
    if text_cols[1].button("OK", key=f"{col_name}_ok_text", type=button_type):
        st.session_state.filters_applied[col_name] = True # Mark as applied
        st.session_state[pending_key] = False; st.rerun()
    if text_cols[2].button("↺", key=f"{col_name}_reset_text_btn"):
        st.session_state[reset_key] = True; st.rerun()


def filter_dataframe(df: pd.DataFrame, max_unique: int = 20) -> pd.DataFrame:
    """Adds a UI to filter dataframe columns.
    Manages its own session state for filters applied and previous columns.
    """
    df_copy = df.copy() # Work on a copy

    # Apply CSS for button width globally for this UI section
    # This is fine here as it's specific to this component's buttons
    st.markdown("""<style> div[data-testid="stButton"] > button { min-width: 48px; } </style>""", unsafe_allow_html=True)

    # Initialize component-specific session state if not already present by the main app
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = {}
    if 'previous_columns' not in st.session_state:
        st.session_state.previous_columns = set()

    with st.container(): # Main container for the filter UI
        # Use list(st.session_state.previous_columns) for default, as multiselect expects a list
        to_filter_columns = st.multiselect(
            "Filter dataframe on", 
            df_copy.columns, 
            default=list(st.session_state.previous_columns),
            key="filter_multiselect_cols" # Add a key for stability
        )
        
        active_filter_columns = set(to_filter_columns)
        # Columns that were previously selected but now are not
        removed_cols_from_multiselect = st.session_state.previous_columns - active_filter_columns
        
        # Cleanup state for columns that were removed from selection
        for col_to_remove_filter_state in removed_cols_from_multiselect:
            # More robust key deletion: iterate over a copy of keys
            keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(f"{col_to_remove_filter_state}_")]
            for key in keys_to_delete:
                if key in st.session_state: del st.session_state[key]
            if col_to_remove_filter_state in st.session_state.filters_applied:
                del st.session_state.filters_applied[col_to_remove_filter_state]
        
        st.session_state.previous_columns = active_filter_columns.copy() # Update previous columns
        
        # This will be the dataframe that gets progressively filtered
        intermediate_filtered_df = df_copy.copy()

        for column in to_filter_columns: # Iterate over currently selected columns for filtering
            # Column header and filter UI container
            # Using st.container for each filter's UI elements for better layout control if needed
            filter_ui_container = st.container() 
            left, right = filter_ui_container.columns((5, 15)) # Layout for label and filter widget
            
            left.markdown(f"##### ↳ {column}") 

            # Initialize common session state keys for each filter if not already present
            reset_key_generic = f"{column}_reset_flag"
            pending_changes_key = f"{column}_pending_changes" # Tracks if changes are made but not "OK'd"
            
            if reset_key_generic not in st.session_state: st.session_state[reset_key_generic] = False
            if pending_changes_key not in st.session_state: st.session_state[pending_changes_key] = False
            
            df_column_series = df_copy[column] # Get the original data for this column for filter setup

            # Determine filter type based on column properties
            is_categorical_type_col = isinstance(df_column_series.dtype, pd.CategoricalDtype)
            is_low_cardinality_obj_col = (
                not df_column_series.empty and
                df_column_series.nunique() < max_unique and
                is_object_dtype(df_column_series) and
                not is_numeric_dtype(df_column_series) # Explicitly non-numeric objects
            )
            
            # Refined conditions for choosing the filter type
            if is_categorical_type_col or is_low_cardinality_obj_col or \
               (not df_column_series.empty and df_column_series.nunique() < max_unique and not is_numeric_dtype(df_column_series) and not is_datetime64_any_dtype(df_column_series)):
                _handle_categorical_filter(df_column_series, column, right)
            elif is_numeric_dtype(df_column_series):
                _handle_numeric_filter(df_column_series, column, right)
            elif is_datetime64_any_dtype(df_column_series):
                _handle_datetime_filter(df_column_series, column, right)
            else:  # Default to text input for other types (e.g., objects with high cardinality)
                _handle_text_filter(df_column_series, column, right)
            
            # Apply the current column's filter to intermediate_filtered_df IF it has been "OK'd"
            if column in st.session_state.filters_applied and st.session_state.filters_applied[column]:
                # Re-check type for applying filter, using original df_column_series for type checks
                if is_categorical_type_col or is_low_cardinality_obj_col or \
                   (not df_column_series.empty and df_column_series.nunique() < max_unique and not is_numeric_dtype(df_column_series) and not is_datetime64_any_dtype(df_column_series)):
                    selected_vals = st.session_state.get(f"{column}_selected_values", [])
                    if selected_vals: # Only filter if there are selected values
                         intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].isin(selected_vals)]
                elif is_numeric_dtype(df_column_series):
                    min_val = st.session_state.get(f"{column}_filter_min", df_column_series.min())
                    max_val = st.session_state.get(f"{column}_filter_max", df_column_series.max())
                    intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].between(min_val, max_val)]
                elif is_datetime64_any_dtype(df_column_series):
                    date_range = st.session_state.get(f"{column}_date_range")
                    if date_range and len(date_range) == 2:
                        # Convert date objects from date_input to datetime for comparison if column is datetime64
                        start_date = pd.to_datetime(date_range[0])
                        end_date = pd.to_datetime(date_range[1])
                        # Ensure comparison is between compatible types (datetime64 Series vs datetime objects)
                        intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].between(start_date, end_date)]
                else: # Text filter
                    text_to_apply = st.session_state.get(f"{column}_text_filter", "")
                    if text_to_apply: # Only apply if text is not empty
                        intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].astype(str).str.contains(text_to_apply, case=False, na=False)]
        
        return intermediate_filtered_df
