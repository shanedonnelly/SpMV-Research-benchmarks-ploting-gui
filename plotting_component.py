import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import os
import io
import zipfile
import numpy as np
import matplotlib.colors as mcolors # type: ignore
from matplotlib.patches import Patch # type: ignore

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

def get_pastel_colors(n):
    """
    Generate a list of pastel colors for the plots
    """
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    # Make colors more pastel by mixing with white
    pastel_colors = []
    
    for i in range(n):
        color_idx = i % len(base_colors)
        base_color = np.array(mcolors.to_rgb(base_colors[color_idx]))
        # Mix with white (0.6 base color, 0.4 white)
        pastel_color = 0.65 * base_color + 0.35 * np.array([1, 1, 1])
        pastel_colors.append(tuple(pastel_color))
    
    return pastel_colors

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
            st.session_state.plots_generated = False
            # Clear other selections
            if "primary_dim" in st.session_state:
                del st.session_state.primary_dim
            if "secondary_dim" in st.session_state:
                del st.session_state.secondary_dim
            if "y_axis" in st.session_state:
                del st.session_state.y_axis
            if "show_titles" in st.session_state:
                del st.session_state.show_titles
            if "comparison_mode" in st.session_state:
                del st.session_state.comparison_mode
            if "plot_params" in st.session_state:
                del st.session_state.plot_params
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
    
    # Comparison mode selection
    comparison_mode = st.selectbox("Comparison Mode",["Separate","Side by Side"], 
                                 key="comparison_mode",
                                 help="Side by Side: traditional layout. Separate: overlapping boxplots")
    
    # Add title checkbox
    show_titles = st.checkbox("Add subset name as title", value=False, key="show_titles")
    
    # Generate plot button
    if st.button("Generate Plot", key="generate_plot"):
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
            'comparison_mode': comparison_mode
        }
    
    # Show plots if they've been generated
    if st.session_state.plots_generated:
        generate_individual_boxplots(**st.session_state.plot_params)

def generate_individual_boxplots(dataframes, filenames, primary_dim, secondary_dim, y_axis, show_titles=False, comparison_mode="Side by Side"):
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
            # Create appropriate plot based on mode
            if comparison_mode == "Side by Side":
                fig = create_side_by_side_plot(modified_df, primary_dim, secondary_dim, y_axis, show_titles, filename)
            else:  # Separate mode
                fig = create_stacked_plots(modified_df, primary_dim, secondary_dim, y_axis, show_titles, filename)
            
            # Display the plot
            st.pyplot(fig)
            
            # Create download buffer
            buffer = save_plot_to_buffer(fig)
            plot_buffers.append((buffer, os.path.splitext(filename)[0] + '.png'))
            
            # Individual download button
            st.download_button(
                label=f"download : {os.path.splitext(filename)[0]}",
                data=buffer,
                file_name=f"{os.path.splitext(filename)[0]}_plot.png",
                mime="image/png",
                key=f"download_plot_{file_idx}"
            )
        
        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")
    
    # Add zip download option if multiple plots
    if need_zip and plot_buffers:
        create_zip_download(plot_buffers)

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

def try_numeric_sort(values):
    """
    Try to sort values in a natural order, handling numeric ranges in strings
    For strings like '[2-4]', '[4-8]', will sort by the first number
    """
    def extract_sort_key(val):
        # Convert to string to handle all types
        val_str = str(val)
        
        # Try to extract first number from string (e.g., from "[2-4]" extract 2)
        import re
        numbers = re.findall(r'[-+]?\d+\.?\d*', val_str)
        if numbers:
            return float(numbers[0])
        
        # If no numbers, return string for lexicographical sort
        return val_str
    
    try:
        return sorted(values, key=extract_sort_key)
    except Exception:
        # Fallback to normal sorting
        return sorted(values)

def create_side_by_side_plot(df, primary_dim, secondary_dim, y_axis, show_titles, filename):
    """Create side-by-side boxplot visualization"""
    
    primary_values_unique = df[primary_dim].unique()
    num_primary_values = len(primary_values_unique)
    
    base_width = 5.0  # Base width for the plot
    width_per_primary_category = 1.0
    
    if secondary_dim:
        secondary_values_unique = df[secondary_dim].unique()
        num_secondary_values = len(secondary_values_unique)
        # Adjust width based on number of secondary categories to accommodate grouped bars
        width_per_primary_category = 0.5 + 0.25 * num_secondary_values 
    
    dynamic_width = base_width + num_primary_values * width_per_primary_category
    # Clamp width to a reasonable range
    dynamic_width = max(8.0, min(20.0, dynamic_width))
    
    fig, ax = plt.subplots(figsize=(dynamic_width, 8), dpi=300)
    
    if secondary_dim:
        # Create grouped boxplots colored by secondary dimension
        primary_values = try_numeric_sort(primary_values_unique)
        secondary_values = try_numeric_sort(secondary_values_unique)
        colors = get_pastel_colors(len(secondary_values))
        
        # Calculate positions and collect data
        positions = []
        data = []
        box_indices = {}  # Map to track box indices
        
        for i, prim_val in enumerate(primary_values):
            for j, sec_val in enumerate(secondary_values):
                subset = df[(df[primary_dim] == prim_val) & (df[secondary_dim] == sec_val)]
                if not subset.empty:
                    pos = i + j * 0.25
                    positions.append(pos)
                    data.append(subset[y_axis].values)
                    box_indices[(i, j)] = len(positions) - 1
        
        # Create boxplot
        bp = ax.boxplot(data, positions=positions, patch_artist=True, 
                       labels=[""] * len(positions), widths=0.15)
        
        # Set primary dimension tick positions and labels
        ax.set_xticks([i + 0.125 * (len(secondary_values) - 1) for i in range(len(primary_values))])
        ax.set_xticklabels(primary_values)
        
        # Color boxes by secondary dimension
        for j, sec_val in enumerate(secondary_values):
            for i, prim_val in enumerate(primary_values):
                if (i, j) in box_indices:
                    idx = box_indices[(i, j)]
                    bp['boxes'][idx].set_facecolor(colors[j])
        
        # Add legend - standard format with consistent placement in upper right
        legend_elements = [
            Patch(facecolor=colors[j], edgecolor='black', label=str(sec_val))
            for j, sec_val in enumerate(secondary_values)
        ]
        ax.legend(handles=legend_elements, title=secondary_dim, loc='upper right')
        
    else:
        # Simple boxplot with primary dimension only
        # Sort primary values
        primary_values = try_numeric_sort(df[primary_dim].unique())
        grouped_data = []
        labels = []
        
        for prim_val in primary_values:
            group = df[df[primary_dim] == prim_val]
            grouped_data.append(group[y_axis].values)
            labels.append(str(prim_val))
        
        # Create boxplot
        bp = ax.boxplot(grouped_data, labels=labels, patch_artist=True)
        
        # Set consistent colors
        colors = get_pastel_colors(len(labels))
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
    
    # Customize plot
    if show_titles:
        ax.set_title(os.path.splitext(filename)[0], fontsize=14)
    ax.set_ylabel(y_axis, fontsize=12)
    ax.set_xlabel(primary_dim, fontsize=12)
    
    # Rotate x-axis labels if needed
    if len(ax.get_xticklabels()) > 3 or any(len(str(label.get_text())) > 10 for label in ax.get_xticklabels()):
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right') # Reduced rotation to 30
    
    plt.tight_layout()
    return fig

def create_stacked_plots(df, primary_dim, secondary_dim, y_axis, show_titles, filename):
    """Create stacked subplot visualization with each secondary value in its own subplot"""
    if not secondary_dim:
        # No secondary dimension, just create a regular boxplot
        return create_side_by_side_plot(df, primary_dim, None, y_axis, show_titles, filename)
    
    # Get unique values for each dimension and sort them
    primary_values_unique = df[primary_dim].unique()
    secondary_values_unique = df[secondary_dim].unique()
    
    primary_values = try_numeric_sort(primary_values_unique)
    secondary_values = try_numeric_sort(secondary_values_unique)
    n_subplots = len(secondary_values)

    num_primary_values = len(primary_values)
    base_width = 5.0
    width_per_primary_category = 1.0
    dynamic_width = base_width + num_primary_values * width_per_primary_category
    dynamic_width = max(8.0, min(20.0, dynamic_width)) # Clamp width

    # Adjust height based on number of subplots, with a minimum
    fig_height_per_subplot = 2.5
    min_fig_height = 5.0
    dynamic_height = max(min_fig_height, n_subplots * fig_height_per_subplot)
    
    fig, axes = plt.subplots(n_subplots, 1, figsize=(dynamic_width, dynamic_height), dpi=300, 
                             sharex=True, squeeze=False) # squeeze=False to always get an array
    
    axes = axes.flatten() # Ensure axes is always a flat array

    # Colors for each subplot
    colors = get_pastel_colors(n_subplots)
    
    # Create boxplots for each secondary dimension value
    for idx, (sec_val, color) in enumerate(zip(secondary_values, colors)):
        ax = axes[idx]
        
        # Filter data for this secondary value
        sec_df = df[df[secondary_dim] == sec_val]
        
        # Only proceed if we have data for this combination
        if not sec_df.empty:
            # Group by primary dimension
            grouped_data = []
            labels = []
            
            # Ensure we have entries for all primary values (for consistent x-axis)
            for prim_val in primary_values:
                group = sec_df[sec_df[primary_dim] == prim_val]
                if not group.empty:
                    grouped_data.append(group[y_axis].values)
                    labels.append(str(prim_val))
                else:
                    # Add empty placeholder for missing combinations
                    grouped_data.append([])
                    labels.append(str(prim_val))
            
            # Create boxplot
            bp = ax.boxplot(grouped_data, labels=labels, patch_artist=True)
            
            # Color the boxes
            for box in bp['boxes']:
                box.set_facecolor(color)
            
            # Set y-axis limits based on this subplot's data only
            if not sec_df.empty:
                y_min = sec_df[y_axis].min()
                y_max = sec_df[y_axis].max()
                padding = (y_max - y_min) * 0.05 if y_max > y_min else y_max * 0.05
                ax.set_ylim(y_min - padding, y_max + padding)
        
        # Set y-axis label
        ax.set_ylabel(y_axis, fontsize=10)
        
        # Only add x-axis label to bottom subplot
        if idx == n_subplots - 1:
            ax.set_xlabel(primary_dim, fontsize=12)
        
        # Format x-axis labels
        if len(primary_values) > 3 or any(len(str(val)) > 10 for val in primary_values):
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right') # Reduced rotation to 30
    
    # --- Improved Title and Legend Layout ---
    rect_top = 0.95 # Default top of plotting area
    legend_height_fraction = 0.0
    
    if secondary_dim:
        legend_elements = [
            Patch(facecolor=colors[j], edgecolor='black', label=str(sec_val))
            for j, sec_val in enumerate(secondary_values)
        ]
        # Make legend wider if many items, up to 4 columns
        legend_ncol = min(max(1, len(secondary_values) // (2 if len(secondary_values) > 4 else 1)), 4)
        
        # Estimate legend height based on rows
        num_legend_rows = (len(legend_elements) + legend_ncol - 1) // legend_ncol
        legend_height_fraction = num_legend_rows * 0.04 # Approximate fraction per row
        
        # Place legend
        # Adjust bbox_to_anchor y based on whether title is present
        legend_y_anchor = 0.98 if not show_titles else 0.92 
        fig.legend(handles=legend_elements, title=secondary_dim, 
                  loc='upper center', bbox_to_anchor=(0.5, legend_y_anchor), 
                  ncol=legend_ncol, frameon=False)
        rect_top -= legend_height_fraction # Make space for legend

    if show_titles:
        fig.suptitle(os.path.splitext(filename)[0], fontsize=16, y=0.98)
        rect_top -= 0.05 # Make space for suptitle
    
    # Ensure rect_top doesn't become too small
    rect_top = max(0.7, rect_top)

    # Adjust layout to make room for title and legend
    # Add padding: left, bottom, right, top
    try:
        plt.tight_layout(rect=[0.03, 0.05, 0.97, rect_top])
    except ValueError:
        # Fallback if rect values are problematic (e.g., top < bottom)
        plt.tight_layout()


    # Remove old subplots_adjust logic
    # plt.subplots_adjust(top=top_margin - legend_height) # This line is removed
    
    return fig

def save_plot_to_buffer(fig):
    """Save plot to a buffer for downloading"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    return buffer

def create_zip_download(plot_buffers):
    """Create a ZIP file containing all plots"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
        for buffer, name in plot_buffers:
            zip_file.writestr(name, buffer.getvalue())
    
    zip_buffer.seek(0)
    st.download_button(
        label="download all",
        data=zip_buffer,
        file_name="all_plots.zip",
        mime="application/zip",
        key="download_all_plots"
    )