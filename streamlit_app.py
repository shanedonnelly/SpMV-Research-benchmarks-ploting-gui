import pandas as pd
import streamlit as st # type: ignore
import configparser
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_integer_dtype,
)

st.title("Auto Filter Dataframes in Streamlit")

# Initialize session state variables
if 'removed_filters' not in st.session_state:
    st.session_state.removed_filters = set()
if 'show_df' not in st.session_state:
    st.session_state.show_df = False
if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = {}
if 'previous_columns' not in st.session_state:
    st.session_state.previous_columns = set()

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

    if st.session_state[reset_key]:
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
            if st.checkbox(str(value), value=is_checked, key=f"{col_name}_val_cb_{i}"):
                if not is_checked: temp_selection.append(value); changed_in_checkboxes = True
            elif is_checked: temp_selection.remove(value); changed_in_checkboxes = True
        
        if changed_in_checkboxes:
            st.session_state[selected_values_key] = temp_selection
            st.session_state[pending_key] = True
            st.rerun()

        action_button_cols = st.columns([1,1,8]) # OK, Reset, spacer
        button_type = "primary" if st.session_state.get(pending_key, False) else "secondary"
        if action_button_cols[0].button("OK", key=f"{col_name}_ok_cat", type=button_type):
            st.session_state.filters_applied[col_name] = True
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
    is_int = is_integer_dtype(df_col) or df_col.equals(df_col.round(0))
    step, num_format = (1.0, None) if is_int else (0.01, "%.5f") # Adjusted step for float

    if min_val_key not in st.session_state: st.session_state[min_val_key] = col_min_orig
    if max_val_key not in st.session_state: st.session_state[max_val_key] = col_max_orig
    if unique_val_key not in st.session_state: st.session_state[unique_val_key] = None
    if expanded_key not in st.session_state: st.session_state[expanded_key] = True


    if st.session_state[reset_key]:
        st.session_state[min_val_key], st.session_state[max_val_key] = col_min_orig, col_max_orig
        st.session_state[unique_val_key] = None
        if col_name in st.session_state.filters_applied: del st.session_state.filters_applied[col_name]
        st.session_state[pending_key] = False
        st.session_state[reset_key] = False

    with container.expander("Filter details", expanded=st.session_state[expanded_key]):
        # CSS for button width is now global in filter_dataframe
        num_cols = st.columns(3)
        new_min = num_cols[0].number_input("Min", col_min_orig, col_max_orig, st.session_state[min_val_key], step, format=num_format, key=f"{col_name}_min_in")
        if new_min != st.session_state[min_val_key]:
            st.session_state[min_val_key] = new_min
            st.session_state[pending_key] = True; st.rerun()
        
        new_max = num_cols[1].number_input("Max", col_min_orig, col_max_orig, st.session_state[max_val_key], step, format=num_format, key=f"{col_name}_max_in")
        if new_max != st.session_state[max_val_key]:
            st.session_state[max_val_key] = new_max
            st.session_state[pending_key] = True; st.rerun()

        new_unique = num_cols[2].number_input("Unique", col_min_orig, col_max_orig, st.session_state[unique_val_key], step, format=num_format, key=f"{col_name}_unique_in")
        if new_unique != st.session_state[unique_val_key]:
            st.session_state[unique_val_key] = new_unique
            if new_unique is not None:
                st.session_state[min_val_key], st.session_state[max_val_key] = new_unique, new_unique
            st.session_state[pending_key] = True; st.rerun()

        current_slider_min, current_slider_max = st.session_state[min_val_key], st.session_state[max_val_key]
        if current_slider_min > current_slider_max: current_slider_min = current_slider_max
        if current_slider_max < current_slider_min: current_slider_max = current_slider_min
        
        slider_vals = (current_slider_min, current_slider_max)
        new_slider_vals = st.slider(f"Range for {col_name}", col_min_orig, col_max_orig, slider_vals, step, key=f"{col_name}_slider")
        if new_slider_vals != slider_vals:
            st.session_state[min_val_key], st.session_state[max_val_key] = new_slider_vals
            st.session_state[unique_val_key] = None # Clear unique if slider moved
            st.session_state[pending_key] = True; st.rerun()

        action_button_cols = st.columns([1,1,8])
        button_type = "primary" if st.session_state.get(pending_key, False) else "secondary"
        if action_button_cols[0].button("OK", key=f"{col_name}_ok_num", type=button_type):
            st.session_state.filters_applied[col_name] = True
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

    min_date_orig, max_date_orig = df_col.min().date(), df_col.max().date()
    if range_key not in st.session_state:
        st.session_state[range_key] = (min_date_orig, max_date_orig)

    if st.session_state[reset_key]:
        st.session_state[range_key] = (min_date_orig, max_date_orig)
        if col_name in st.session_state.filters_applied: del st.session_state.filters_applied[col_name]
        st.session_state[pending_key] = False
        st.session_state[reset_key] = False

    date_cols = container.columns([9, 1, 1])
    current_range = st.session_state[range_key]
    new_range = date_cols[0].date_input(f"Values for {col_name}", current_range, min_date_orig, max_date_orig, key=f"{col_name}_date_in")
    if len(new_range) == 2 and new_range != current_range:
        st.session_state[range_key] = new_range
        st.session_state[pending_key] = True; st.rerun()

    ok_label = "OK" # Text remains OK
    button_type = "primary" if st.session_state.get(pending_key, False) else "secondary"
    if date_cols[1].button(ok_label, key=f"{col_name}_ok_date", type=button_type):
        st.session_state.filters_applied[col_name] = True
        st.session_state[pending_key] = False; st.rerun()
    if date_cols[2].button("↺", key=f"{col_name}_reset_date_btn"):
        st.session_state[reset_key] = True; st.rerun()

# Helper function for text filters
def _handle_text_filter(df_col: pd.Series, col_name: str, container: st.container):
    reset_key = f"{col_name}_reset_flag"
    pending_key = f"{col_name}_pending_changes"
    text_filter_key = f"{col_name}_text_filter"

    if text_filter_key not in st.session_state: st.session_state[text_filter_key] = ""

    if st.session_state[reset_key]:
        st.session_state[text_filter_key] = ""
        if col_name in st.session_state.filters_applied: del st.session_state.filters_applied[col_name]
        st.session_state[pending_key] = False
        st.session_state[reset_key] = False

    text_cols = container.columns([9, 1, 1])
    current_text = st.session_state[text_filter_key]
    new_text = text_cols[0].text_input(f"Substring or regex in {col_name}", current_text, key=f"{col_name}_text_in")
    if new_text != current_text:
        st.session_state[text_filter_key] = new_text
        st.session_state[pending_key] = True; st.rerun()

    ok_label = "OK" # Text remains OK
    button_type = "primary" if st.session_state.get(pending_key, False) else "secondary"
    if text_cols[1].button(ok_label, key=f"{col_name}_ok_text", type=button_type):
        st.session_state.filters_applied[col_name] = True
        st.session_state[pending_key] = False; st.rerun()
    if text_cols[2].button("↺", key=f"{col_name}_reset_text_btn"):
        st.session_state[reset_key] = True; st.rerun()


def filter_dataframe(df: pd.DataFrame, max_unique: int = 20) -> pd.DataFrame:
    """Adds a UI to filter dataframe columns"""
    df_copy = df.copy()
    
    # Apply CSS for button width globally for this UI section
    st.markdown("""<style> div[data-testid="stButton"] > button { min-width: 48px; } </style>""", unsafe_allow_html=True)

    with st.container():
        to_filter_columns = st.multiselect("Filter dataframe on", df_copy.columns, default=list(st.session_state.previous_columns))
        
        active_filter_columns = set(to_filter_columns)
        removed_cols_from_multiselect = st.session_state.previous_columns - active_filter_columns
        for col_to_remove_filter_state in removed_cols_from_multiselect:
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith(f"{col_to_remove_filter_state}_")]
            for key in keys_to_delete:
                if key in st.session_state: del st.session_state[key]
            if col_to_remove_filter_state in st.session_state.filters_applied:
                del st.session_state.filters_applied[col_to_remove_filter_state]
        st.session_state.previous_columns = active_filter_columns
        
        intermediate_filtered_df = df_copy.copy()

        for column in to_filter_columns:
            left, right = st.columns((5, 15))
            left.markdown(f"##### ↳ {column}") 

            # Initialize common session state keys for each filter
            reset_key_generic = f"{column}_reset_flag"
            pending_changes_key = f"{column}_pending_changes"
            if reset_key_generic not in st.session_state: st.session_state[reset_key_generic] = False
            if pending_changes_key not in st.session_state: st.session_state[pending_changes_key] = False
            
            df_column_series = df_copy[column]

            is_categorical_type = isinstance(df_column_series.dtype, pd.CategoricalDtype)
            is_low_cardinality_obj = (
                not df_column_series.empty and
                df_column_series.nunique() < max_unique and
                is_object_dtype(df_column_series) and # Ensure it's object type for low cardinality non-numeric
                not is_numeric_dtype(df_column_series) # Redundant if already object, but safe
            )
            
            # Refined conditions for filter types
            if is_categorical_type or is_low_cardinality_obj or \
               (df_column_series.nunique() < max_unique and not is_numeric_dtype(df_column_series) and not is_datetime64_any_dtype(df_column_series)):
                _handle_categorical_filter(df_column_series, column, right)
            elif is_numeric_dtype(df_column_series):
                _handle_numeric_filter(df_column_series, column, right)
            elif is_datetime64_any_dtype(df_column_series):
                _handle_datetime_filter(df_column_series, column, right)
            else:  # Default to text input for other types (likely objects with high cardinality)
                _handle_text_filter(df_column_series, column, right)
            
            # Apply filter to intermediate_filtered_df if "OK"/"Apply" was pressed for this column
            if column in st.session_state.filters_applied and st.session_state.filters_applied[column]:
                if is_categorical_type or is_low_cardinality_obj or \
                   (df_column_series.nunique() < max_unique and not is_numeric_dtype(df_column_series) and not is_datetime64_any_dtype(df_column_series)):
                    selected_vals = st.session_state.get(f"{column}_selected_values", [])
                    intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].isin(selected_vals)]
                elif is_numeric_dtype(df_column_series):
                    min_val = st.session_state.get(f"{column}_filter_min", df_column_series.min())
                    max_val = st.session_state.get(f"{column}_filter_max", df_column_series.max())
                    intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].between(min_val, max_val)]
                elif is_datetime64_any_dtype(df_column_series):
                    date_range = st.session_state.get(f"{column}_date_range")
                    if date_range and len(date_range) == 2:
                        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                        intermediate_filtered_df = intermediate_filtered_df[intermediate_filtered_df[column].between(start_date, end_date)]
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