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

# Track removed filters to reset when re-added
if 'removed_filters' not in st.session_state:
    st.session_state.removed_filters = set()

def filter_dataframe(df: pd.DataFrame, max_unique: int) -> pd.DataFrame:
    """Adds a UI to filter dataframe columns"""
    df_copy = df.copy()

    # Convert only date-like object columns to datetime
    for col in df_copy.columns:
        if is_object_dtype(df_copy[col]):
            try:
                # Check if column might contain dates (quick heuristic)
                sample = df_copy[col].dropna().iloc[:10] if not df_copy[col].empty else []
                likely_dates = all(isinstance(s, str) and (len(s) >= 8 and ('/' in s or '-' in s or ':' in s)) for s in sample)
                
                if likely_dates:
                    temp_series = pd.to_datetime(df_copy[col], errors='coerce', format="%d/%m/%Y")
                    # Only convert if most values successfully parsed
                    if temp_series.notna().sum() > 0.5 * df_copy[col].count():
                        df_copy[col] = temp_series
            except Exception:
                pass
                
        if is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.tz_localize(None)

    with st.container():
        to_filter_columns = st.multiselect("Filter dataframe on", df_copy.columns)
        
        # Reset filters for columns that were removed and re-added
        for column in to_filter_columns:
            if column in st.session_state.removed_filters:
                keys_to_delete = [k for k in st.session_state.keys() 
                                 if k.startswith(f"{column}_")]
                for key in keys_to_delete:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.removed_filters.remove(column)
                
        # Track columns that have been removed
        previous_columns = getattr(st.session_state, 'previous_columns', set())
        removed_columns = previous_columns - set(to_filter_columns)
        st.session_state.removed_filters.update(removed_columns)
        st.session_state.previous_columns = set(to_filter_columns)
        
        # Create a copy for filtering
        filtered_df = df_copy.copy()
        
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            
            # Categorical or low-cardinality columns
            is_categorical = isinstance(filtered_df[column].dtype, pd.CategoricalDtype)
            is_low_cardinality = not filtered_df[column].empty and filtered_df[column].nunique() < max_unique
            
            if is_categorical or is_low_cardinality:
                unique_values = filtered_df[column].unique()
                filter_container = right.container()
                col1, col2 = filter_container.columns([10, 1])
                
                # Initialize values on first run
                reset_key = f"{column}_reset_cat"
                if reset_key not in st.session_state:
                    st.session_state[reset_key] = False
                    
                if f"{column}_selected_values" not in st.session_state:
                    st.session_state[f"{column}_selected_values"] = list(unique_values)
                
                # Handle reset request
                if st.session_state[reset_key]:
                    st.session_state[f"{column}_selected_values"] = list(unique_values)
                    st.session_state[reset_key] = False
                
                user_cat_input = col1.multiselect(
                    f"Values for {column}",
                    unique_values,
                    default=st.session_state[f"{column}_selected_values"],
                )
                
                # Store selected values
                st.session_state[f"{column}_selected_values"] = user_cat_input
                
                # Reset button 
                if col2.button("↺", key=f"{column}_cat_reset_btn"):
                    st.session_state[reset_key] = True
                    st.rerun()  # Fixed: use st.rerun() instead of experimental_rerun
                    
                filtered_df = filtered_df[filtered_df[column].isin(user_cat_input)]
                
            elif is_numeric_dtype(filtered_df[column]):
                if filtered_df[column].empty:
                    right.text(f"{column}: No data to filter.")
                    continue

                col_min_original = float(df_copy[column].min())
                col_max_original = float(df_copy[column].max())
                is_int = is_integer_dtype(filtered_df[column]) or filtered_df[column].equals(filtered_df[column].round(0))

                # Ensure integers for int columns
                if is_int:
                    col_min_original = int(col_min_original)
                    col_max_original = int(col_max_original)
                    step = 1
                    num_format = None  # Let Streamlit handle int format
                else:
                    step = 0.1
                    num_format = "%.5f"  # 5 decimal precision for floats

                filter_col1, filter_col2, filter_col3, filter_col4 = right.columns([3, 3, 3, 1])
                
                # Initialize or get filter values
                reset_key = f"{column}_reset_num"
                if reset_key not in st.session_state:
                    st.session_state[reset_key] = False
                
                if f"{column}_filter_min" not in st.session_state:
                    st.session_state[f"{column}_filter_min"] = col_min_original
                if f"{column}_filter_max" not in st.session_state:
                    st.session_state[f"{column}_filter_max"] = col_max_original
                
                # Handle reset request
                if st.session_state[reset_key]:
                    st.session_state[f"{column}_filter_min"] = col_min_original
                    st.session_state[f"{column}_filter_max"] = col_max_original
                    st.session_state[f"{column}_unique_value"] = None
                    st.session_state[reset_key] = False
                
                with filter_col1:
                    filter_col1.text("Min")  # Label above input
                    min_val = filter_col1.number_input(
                        "Minimum",
                        min_value=col_min_original,
                        max_value=col_max_original, 
                        value=st.session_state[f"{column}_filter_min"],
                        step=step,
                        format=num_format,
                        label_visibility="collapsed",
                        key=f"{column}_min_input"
                    )
                    st.session_state[f"{column}_filter_min"] = min_val
                    
                with filter_col2:
                    filter_col2.text("Max")  # Label above input
                    max_val = filter_col2.number_input(
                        "Maximum",
                        min_value=col_min_original, 
                        max_value=col_max_original,
                        value=st.session_state[f"{column}_filter_max"],
                        step=step,
                        format=num_format,
                        label_visibility="collapsed",
                        key=f"{column}_max_input"
                    )
                    st.session_state[f"{column}_filter_max"] = max_val
                    
                with filter_col3:
                    filter_col3.text("Unique")  # Label above input
                    unique_key = f"{column}_unique_value"
                    
                    if unique_key not in st.session_state:
                        st.session_state[unique_key] = None
                        
                    # Use an empty string as placeholder
                    unique_val = filter_col3.number_input(
                        "Unique Value",
                        min_value=col_min_original,
                        max_value=col_max_original,
                        value=st.session_state.get(unique_key),
                        step=step,
                        format=num_format,
                        label_visibility="collapsed",
                        key=f"{column}_unique_input"
                    )
                    
                    # Apply unique value button
                    if filter_col3.button("Apply", key=f"{column}_unique_apply"):
                        if unique_val is not None:
                            st.session_state[unique_key] = unique_val
                            # Update min and max to make them equal to unique value
                            st.session_state[f"{column}_filter_min"] = unique_val
                            st.session_state[f"{column}_filter_max"] = unique_val
                            st.rerun()  # Fixed: use st.rerun() instead of experimental_rerun
                
                # Reset button
                with filter_col4:
                    if filter_col4.button("↺", key=f"{column}_reset_button"):
                        st.session_state[reset_key] = True
                        st.session_state[unique_key] = None
                        st.rerun()  # Fixed: use st.rerun() instead of experimental_rerun
                
                # Calculate correct slider values
                slider_min = min_val if min_val is not None else col_min_original
                slider_max = max_val if max_val is not None else col_max_original
                
                # Update slider based on min/max inputs
                slider_vals = right.slider(
                    f"Range for {column}",
                    min_value=col_min_original,
                    max_value=col_max_original,
                    value=(slider_min, slider_max),
                    step=step,
                    key=f"{column}_slider"
                )
                
                # Update min/max from slider if they changed
                if slider_vals[0] != st.session_state[f"{column}_filter_min"] or slider_vals[1] != st.session_state[f"{column}_filter_max"]:
                    st.session_state[f"{column}_filter_min"] = slider_vals[0]
                    st.session_state[f"{column}_filter_max"] = slider_vals[1]
                    
                # Apply the filter (strictly enforced)
                filtered_df = filtered_df[filtered_df[column].between(slider_vals[0], slider_vals[1])]

            elif is_datetime64_any_dtype(filtered_df[column]):
                if filtered_df[column].empty:
                    right.text(f"{column}: No data to filter.")
                    continue
                    
                min_date = filtered_df[column].min().date()
                max_date = filtered_df[column].max().date()
                
                date_container = right.container()
                date_col1, date_col2 = date_container.columns([10, 1])
                
                # Reset management
                reset_key = f"{column}_reset_date"
                if reset_key not in st.session_state:
                    st.session_state[reset_key] = False
                
                # Initialize date range
                if f"{column}_date_range" not in st.session_state:
                    st.session_state[f"{column}_date_range"] = (min_date, max_date)
                
                # Handle reset
                if st.session_state[reset_key]:
                    st.session_state[f"{column}_date_range"] = (min_date, max_date)
                    st.session_state[reset_key] = False
                
                user_date_input = date_col1.date_input(
                    f"Values for {column}",
                    value=st.session_state[f"{column}_date_range"],
                    min_value=min_date,
                    max_value=max_date,
                )
                
                # Store date range
                if len(user_date_input) == 2:
                    st.session_state[f"{column}_date_range"] = user_date_input
                
                # Reset button
                if date_col2.button("↺", key=f"{column}_date_reset"):
                    st.session_state[reset_key] = True
                    st.rerun()  # Fixed: use st.rerun() instead of experimental_rerun
                
                if user_date_input and len(user_date_input) == 2:
                    start_date, end_date = pd.to_datetime(user_date_input[0]), pd.to_datetime(user_date_input[1])
                    filtered_df = filtered_df.loc[filtered_df[column].between(start_date, end_date)]
                    
            else:  # Text input
                text_container = right.container()
                text_col1, text_col2 = text_container.columns([10, 1])
                
                # Reset management
                reset_key = f"{column}_reset_text"
                if reset_key not in st.session_state:
                    st.session_state[reset_key] = False
                
                # Initialize text filter
                if f"{column}_text_filter" not in st.session_state:
                    st.session_state[f"{column}_text_filter"] = ""
                
                # Handle reset
                if st.session_state[reset_key]:
                    st.session_state[f"{column}_text_filter"] = ""
                    st.session_state[reset_key] = False
                
                user_text_input = text_col1.text_input(
                    f"Substring or regex in {column}",
                    value=st.session_state[f"{column}_text_filter"],
                )
                
                # Store text filter
                st.session_state[f"{column}_text_filter"] = user_text_input
                
                # Reset button
                if text_col2.button("↺", key=f"{column}_text_reset"):
                    st.session_state[reset_key] = True
                    st.rerun()  # Fixed: use st.rerun() instead of experimental_rerun
                
                if user_text_input:
                    filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(user_text_input, case=False, na=False)]
    
        return filtered_df

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
filtered_df = filter_dataframe(df_original, max_unique=max_unique_val)

with st.container():
    show_df = st.checkbox("Show DataFrame", value=True)
    st.caption(f"Filtered rows: {len(filtered_df)}")

if show_df:
    st.dataframe(filtered_df)