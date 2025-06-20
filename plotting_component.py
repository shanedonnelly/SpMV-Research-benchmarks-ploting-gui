import streamlit as st
import pandas as pd
import os
import io
import zipfile
import numpy as np
from matplotlib_interface import (
    create_side_by_side_plot, 
    create_stacked_plots, 
    save_plot_to_buffer
)

def check_columns_match(dataframes):
    """
    Check if all dataframes have exactly the same column names
    """
    if not dataframes or len(dataframes) < 2:
        return True
    
    reference_columns = set(dataframes[0].columns)
    
    for df in dataframes[1:]:
        if set(df.columns) != reference_columns:
            return False
    
    return True

def is_numerical_column(df, column):
    """
    Check if a column contains numerical data
    """
    return pd.api.types.is_numeric_dtype(df[column])

def bin_numerical_data(df, column, n_bins=4):
    """
    Split numerical data into bins for grouping
    Not fully implemented - basic functionality only
    """
    min_val = df[column].min()
    max_val = df[column].max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    # Create bin labels
    bin_labels = []
    for i in range(n_bins):
        bin_labels.append(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}")
    
    # Assign data to bins
    binned_column = pd.cut(df[column], bins=bin_edges, labels=bin_labels, include_lowest=True)
    
    return binned_column

def render_plotting_component(dataframes, filenames):
    """
    Render the plotting component with controls and visualization
    """
    if not dataframes:
        st.warning("Please select at least one subset to plot")
        return
    
    # Check if column names match across all dataframes
    if not check_columns_match(dataframes):
        st.warning("Please select dataframes with identical column structures")
        return
    
    # Initialize session state for plots
    if 'plots_generated' not in st.session_state:
        st.session_state.plots_generated = False
    
    # Reset button beside the title
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        st.markdown("#### Plotting Controls")
    with col2:
        if st.button("↺", key="reset_plots"):
            # Full reset of all state variables
            for key in list(st.session_state.keys()):
                if key.startswith("plot_") or key in [
                    "plots_generated", "primary_dim", "secondary_dim", 
                    "y_axis", "show_titles", "comparison_mode", "plot_params",
                    "show_outliers", "fig_size_mode", "fig_width_cm", 
                    "fig_height_cm", "output_format", "plot_titles",
                    "axes_label_mode", "x_axis_label", "y_axis_label"
                ]:
                    del st.session_state[key]
            st.rerun()
    
    # Get column names from first dataframe
    columns = list(dataframes[0].columns)
    
    # Plot type (disabled for now as we only support boxplots)
    st.selectbox("Plot Type", ["Boxplot"], key="plot_mode", disabled=True)
    
    # Replace multiselect with separate primary and secondary dimension selectors

    col1, col2 = st.columns(2)
    with col1:
        primary_dim = st.selectbox("Primary Dimension", columns, key="primary_dim", 
                                 help="First dimension for grouping (required)")
    with col2:
        # None option for secondary dimension
        secondary_options = ["None"] + columns
        secondary_dim = st.selectbox("Secondary Dimension", secondary_options, key="secondary_dim",
                                   help="Second dimension for coloring (optional)")
    
    # Y-axis selection in its own row
    default_y = "gflops" if "gflops" in columns else columns[0]
    y_axis = st.selectbox("Y-axis", columns, 
                         index=columns.index(default_y) if default_y in columns else 0, 
                         key="y_axis")
    
    # Add checkbox for outliers
    show_outliers = st.checkbox("Show outliers", value=False, key="show_outliers")
    
    # Add axes label control
    axes_label_mode = st.segmented_control(
        "Axes Labels",
        ["Auto", "Manual"],
        key="axes_label_mode",
        default="Auto"
    )
    
    # If Manual is selected, add text inputs for X and Y labels
    x_label, y_label = None, None
    if axes_label_mode == "Manual":
        col1, col2 = st.columns(2)
        with col1:
            x_label = st.text_input("X-axis Label", value=primary_dim, key="x_axis_label")
        with col2:
            y_label = st.text_input("Y-axis Label", value=y_axis, key="y_axis_label")
    
    # Comparison mode as segmented control - switch order so Separate is on the left
    comparison_mode = st.segmented_control(
        "Comparison Mode", 
        ["Separate", "Side by Side"],  # Changed order here
        key="comparison_mode",
        default="Separate",
        help="Side by Side: traditional layout. Separate: overlapping boxplots"
    )
    
    # Figure size controls
    col1, col2 = st.columns([1, 2])
    with col1:
        fig_size_mode = st.segmented_control(
            "Figure Size",
            ["Auto", "Manual"],
            key="fig_size_mode",
            default="Auto"
        )
        
    with col2:
        if fig_size_mode == "Manual":
            col_w, col_h = st.columns(2)
            with col_w:
                fig_width_cm = st.slider("Width (cm)", 4, 100, 40, 1, key="fig_width_cm")
            with col_h:
                fig_height_cm = st.slider("Height (cm)", 4, 100, 40, 1, key="fig_height_cm")
        else:
            fig_width_cm = None
            fig_height_cm = None
    
    # Output format selection
    output_format = st.segmented_control(
        "Output Format",
        ["PNG", "PDF"],
        key="output_format",
        default="PNG"
    )
    
    # Title controls
    show_titles = st.checkbox("Add titles to plots", value=False, key="show_titles")
    
    # If titles are enabled, show input fields for each subset
    plot_titles = {}
    if show_titles:
        st.write("Enter titles for each subset:")
        cols = st.columns(min(3, len(filenames)))
        for i, filename in enumerate(filenames):
            col_idx = i % len(cols)
            with cols[col_idx]:
                default_title = os.path.splitext(filename)[0]
                plot_titles[filename] = st.text_input(f"Title for {default_title}", 
                                                    value=default_title, 
                                                    key=f"title_{i}")
    
    # Generate plot button with styling
    st.markdown(
        """
        <style>
        div.stButton > button[kind="primary"] {
            background-color: #ff4b4b;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("Generate Plot", key="generate_plot", type="primary"):
        # Convert secondary_dim to None if "None" is selected
        secondary_dimension = None if secondary_dim == "None" else secondary_dim
        
        # Build list of group by dimensions
        group_by_cols = [primary_dim]
        if secondary_dimension:
            group_by_cols.append(secondary_dimension)
        
        st.session_state.plots_generated = True
        st.session_state.plot_params = {
            'dataframes': dataframes,
            'filenames': filenames,
            'primary_dim': primary_dim,
            'secondary_dim': secondary_dimension,
            'y_axis': y_axis,
            'show_titles': show_titles,
            'comparison_mode': comparison_mode,
            'show_outliers': show_outliers,
            'fig_size_mode': fig_size_mode,
            'fig_width_cm': fig_width_cm,
            'fig_height_cm': fig_height_cm,
            'output_format': output_format,
            'plot_titles': plot_titles,
            'axes_label_mode': axes_label_mode,
            'x_label': x_label,
            'y_label': y_label
        }
    
    # Show plots if they've been generated
    if st.session_state.plots_generated:
        generate_individual_boxplots(**st.session_state.plot_params)

def generate_individual_boxplots(dataframes, filenames, primary_dim, secondary_dim, y_axis, 
                               show_titles=False, comparison_mode="Separate", 
                               show_outliers=False, fig_size_mode="Auto", fig_width_cm=None, 
                               fig_height_cm=None, output_format="PNG", plot_titles=None,
                               axes_label_mode="Auto", x_label=None, y_label=None):
    """
    Generate separate boxplots for each dataframe
    """
    plot_buffers = []
    need_zip = len(dataframes) > 1
    
    for file_idx, (df, filename) in enumerate(zip(dataframes, filenames)):
        st.markdown(f"### {os.path.splitext(filename)[0]}")
        
        # Preprocess data
        modified_df = preprocess_data(df, primary_dim, secondary_dim)
        
        try:
            # Get plot title if provided
            title = plot_titles.get(filename, os.path.splitext(filename)[0]) if show_titles else None
            
            # Create appropriate plot based on mode - fixed order logic
            if comparison_mode == "Side by Side":
                fig = create_side_by_side_plot(modified_df, primary_dim, secondary_dim, y_axis, 
                                           show_titles, title, show_outliers, 
                                           fig_size_mode, fig_width_cm, fig_height_cm,
                                           axes_label_mode, x_label, y_label)
            else:  # Separate mode
                fig = create_stacked_plots(modified_df, primary_dim, secondary_dim, y_axis, 
                                       show_titles, title, show_outliers,
                                       fig_size_mode, fig_width_cm, fig_height_cm,
                                       axes_label_mode, x_label, y_label)
            
            # Display the plot
            st.pyplot(fig)
            
            # Create download buffer
            buffer = save_plot_to_buffer(fig, format=output_format.lower())
            extension = "pdf" if output_format == "PDF" else "png"
            plot_buffers.append((buffer, f"{os.path.splitext(filename)[0]}.{extension}"))
            
            # Individual download button
            st.download_button(
                label=f"download as {output_format.lower()} : {os.path.splitext(filename)[0]}",
                data=buffer,
                file_name=f"{os.path.splitext(filename)[0]}_plot.{extension}",
                mime=f"image/{extension}",
                key=f"download_plot_{file_idx}"
            )
        
        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")
            import traceback
            st.error(traceback.format_exc())  # Add traceback for better debugging
    
    # Add zip download option if multiple plots
    if need_zip and plot_buffers:
        create_zip_download(plot_buffers, output_format.lower())

def preprocess_data(df, primary_dim, secondary_dim):
    """Process data for visualization, handling numerical columns"""
    modified_df = df.copy()
    numerical_warning_shown = False
    
    # Process primary dimension if numerical with many unique values
    if is_numerical_column(df, primary_dim) and df[primary_dim].nunique() > 10:
        if not numerical_warning_shown:
            st.warning("Numerical column with many unique values detected. Binning into groups (not fully implemented).")
            numerical_warning_shown = True
        modified_df[primary_dim] = bin_numerical_data(df, primary_dim)
    
    # Process secondary dimension if it exists and is numerical with many unique values
    if secondary_dim and is_numerical_column(df, secondary_dim) and df[secondary_dim].nunique() > 10:
        if not numerical_warning_shown:
            st.warning("Numerical column with many unique values detected. Binning into groups (not fully implemented).")
        modified_df[secondary_dim] = bin_numerical_data(df, secondary_dim)
    
    return modified_df

def create_zip_download(plot_buffers, format='png'):
    """Create a ZIP file containing all plots"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
        for buffer, name in plot_buffers:
            zip_file.writestr(name, buffer.getvalue())
    
    zip_buffer.seek(0)
    st.download_button(
        label=f"download all as {format}",
        data=zip_buffer,
        file_name=f"all_plots.zip",
        mime="application/zip",
        key="download_all_plots"
    )