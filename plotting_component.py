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

def create_side_by_side_plot(df, primary_dim, secondary_dim, y_axis, show_titles, title, 
                           show_outliers=False, fig_size_mode="Auto", fig_width_cm=None, fig_height_cm=None,
                           axes_label_mode="Auto", x_label=None, y_label=None):
    """Create side-by-side boxplot visualization"""
    
    primary_values_unique = df[primary_dim].unique()
    num_primary_values = len(primary_values_unique)
    
    # Determine figure size
    if fig_size_mode == "Manual" and fig_width_cm and fig_height_cm:
        # Convert cm to inches (1 cm ≈ 0.3937 inches)
        fig_width = fig_width_cm * 0.3937
        fig_height = fig_height_cm * 0.3937
    else:
        # Auto sizing logic - made consistent with stacked plots
        base_width = 5.0
        width_per_primary_category = 1.0
        
        if secondary_dim:
            secondary_values_unique = df[secondary_dim].unique()
            num_secondary_values = len(secondary_values_unique)
            width_per_primary_category = 0.5 + 0.25 * num_secondary_values 
        
        fig_width = base_width + num_primary_values * width_per_primary_category
        fig_width = max(8.0, min(20.0, fig_width))
        fig_height = 8.0 * 1.1  # Slight height increase for consistency with stacked plots
    
    # Create figure with higher DPI for better resolution
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=1200)
    
    # Use thinner lines for better appearance
    plt.rcParams['lines.linewidth'] = 0.8
    plt.rcParams['boxplot.flierprops.linewidth'] = 0.8
    plt.rcParams['boxplot.boxprops.linewidth'] = 0.8
    plt.rcParams['boxplot.whiskerprops.linewidth'] = 0.8
    plt.rcParams['boxplot.capprops.linewidth'] = 0.8
    plt.rcParams['boxplot.medianprops.linewidth'] = 0.8
    
    if secondary_dim:
        # Create grouped boxplots colored by secondary dimension
        primary_values = try_numeric_sort(primary_values_unique)
        secondary_values = try_numeric_sort(df[secondary_dim].unique())
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
        
        # Create boxplot with outlier option
        data = [d for d in data if len(d) > 0]  # Ensure data is not empty
        if not data:
            st.warning(f"No data to plot for the selected combination")
            # Return an empty figure if no data
            return fig
        
        bp = ax.boxplot(data, positions=positions, patch_artist=True, 
                       labels=[""] * len(positions), widths=0.15,
                       showfliers=show_outliers)
        
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
            if not group.empty:  # Only add non-empty groups
                grouped_data.append(group[y_axis].values)
                labels.append(str(prim_val))
        
        if not grouped_data:
            st.warning(f"No data to plot for the selected combination")
            # Return an empty figure if no data
            return fig
            
        # Create boxplot with outlier option
        bp = ax.boxplot(grouped_data, labels=labels, patch_artist=True, showfliers=show_outliers)
        
        # Set consistent colors
        colors = get_pastel_colors(len(labels))
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
    
    # Customize plot
    if show_titles and title:
        # Sanitize title to remove special characters that might cause font issues
        title = ''.join(c for c in title if c.isprintable())
        ax.set_title(title, fontsize=16)
    
    # Set axis labels based on mode - sanitize labels
    x_axis_label = x_label if axes_label_mode == "Manual" and x_label else primary_dim
    y_axis_label = y_label if axes_label_mode == "Manual" and y_label else y_axis
    
    # Sanitize labels to prevent font issues
    x_axis_label = ''.join(c for c in str(x_axis_label) if c.isprintable())
    y_axis_label = ''.join(c for c in str(y_axis_label) if c.isprintable())
    
    ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel(y_axis_label, fontsize=16)
    
    # Sanitize tick labels to prevent font issues
    sanitized_labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        sanitized = ''.join(c for c in text if c.isprintable())
        label.set_text(sanitized)
        sanitized_labels.append(sanitized)
    
    # Rotate x-axis labels if needed
    if len(sanitized_labels) > 3 or any(len(label) > 10 for label in sanitized_labels):
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Tighter layout with less padding
    plt.tight_layout(pad=1.0)
    return fig

def create_stacked_plots(df, primary_dim, secondary_dim, y_axis, show_titles, title, 
                       show_outliers=False, fig_size_mode="Auto", fig_width_cm=None, fig_height_cm=None,
                       axes_label_mode="Auto", x_label=None, y_label=None):
    """Create stacked subplot visualization with each secondary value in its own subplot"""
    if not secondary_dim:
        # No secondary dimension, just create a regular boxplot
        return create_side_by_side_plot(df, primary_dim, None, y_axis, show_titles, title, 
                                      show_outliers, fig_size_mode, fig_width_cm, fig_height_cm,
                                      axes_label_mode, x_label, y_label)
    
    # Get unique values for each dimension and sort them
    primary_values_unique = df[primary_dim].unique()
    secondary_values_unique = df[secondary_dim].unique()
    
    primary_values = try_numeric_sort(primary_values_unique)
    secondary_values = try_numeric_sort(secondary_values_unique)
    n_subplots = len(secondary_values)

    # Determine figure size
    if fig_size_mode == "Manual" and fig_width_cm and fig_height_cm:
        # Convert cm to inches (1 cm ≈ 0.3937 inches)
        fig_width = fig_width_cm * 0.3937
        fig_height = fig_height_cm * 0.3937
    else:
        # Auto sizing logic
        num_primary_values = len(primary_values)
        base_width = 5.0
        width_per_primary_category = 1.0
        fig_width = base_width + num_primary_values * width_per_primary_category
        fig_width = max(8.0, min(20.0, fig_width)) # Clamp width

        # Improved auto height calculation for better subplot visibility
        fig_height_per_subplot = 4.0
        min_fig_height = 5.0
        # Add extra height for legend + apply a scaling factor for better proportion
        fig_height = max(min_fig_height, n_subplots * fig_height_per_subplot) * 1.1
    
    # Create figure with higher DPI for better resolution
    try:
        fig, axes = plt.subplots(n_subplots, 1, figsize=(fig_width, fig_height), dpi=1200, 
                                sharex=True, squeeze=False)
        axes = axes.flatten()  # Ensure axes is always a flat array
    except ValueError as e:
        st.error(f"Error creating figure: {e}")
        # Create a fallback figure
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=1200)
        return fig
    
    # Use thinner lines for better appearance
    plt.rcParams['lines.linewidth'] = 0.8
    plt.rcParams['boxplot.flierprops.linewidth'] = 0.8
    plt.rcParams['boxplot.boxprops.linewidth'] = 0.8
    plt.rcParams['boxplot.whiskerprops.linewidth'] = 0.8
    plt.rcParams['boxplot.capprops.linewidth'] = 0.8
    plt.rcParams['boxplot.medianprops.linewidth'] = 0.8
    #median line in black

    
    # Colors for each subplot
    colors = get_pastel_colors(n_subplots)
    
    # Create boxplots for each secondary dimension value
    for idx, (sec_val, color) in enumerate(zip(secondary_values, colors)):
        if idx >= len(axes):  # Ensure we don't try to access non-existent axes
            break
            
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
                    # Sanitize label to prevent font issues
                    label = ''.join(c for c in str(prim_val) if c.isprintable())
                    labels.append(label)
                else:
                    # Add empty placeholder for missing combinations
                    grouped_data.append([])
                    # Sanitize label to prevent font issues
                    label = ''.join(c for c in str(prim_val) if c.isprintable())
                    labels.append(label)
            
            # Skip if no data
            if all(len(data) == 0 for data in grouped_data):
                continue
                
            # Create boxplot with outlier option
            try:
                bp = ax.boxplot(grouped_data, labels=labels, patch_artist=True, showfliers=show_outliers)
                
                # Color the boxes
                for box in bp['boxes']:
                    box.set_facecolor(color)
            except (ValueError, TypeError) as e:
                st.warning(f"Could not create boxplot for {sec_val}: {e}")
                continue
            
            # Set y-axis limits based on this subplot's data only
            if not sec_df.empty:
                y_min = sec_df[y_axis].min()
                y_max = sec_df[y_axis].max()
                padding = (y_max - y_min) * 0.05 if y_max > y_min else y_max * 0.05
                ax.set_ylim(y_min - padding, y_max + padding)
        
        # Set axis labels based on mode with increased font size
        x_axis_label = x_label if axes_label_mode == "Manual" and x_label else primary_dim
        y_axis_label = y_label if axes_label_mode == "Manual" and y_label else y_axis
        
        # Sanitize labels to prevent font issues
        x_axis_label = ''.join(c for c in str(x_axis_label) if c.isprintable())
        y_axis_label = ''.join(c for c in str(y_axis_label) if c.isprintable())
        
        # Increased font size for axis labels
        ax.set_ylabel(y_axis_label, fontsize=16)  # Increased from 10
        
        # Only add x-axis label to bottom subplot
        if idx == n_subplots - 1:
            ax.set_xlabel(x_axis_label, fontsize=16)  # Increased from 12
    
    # --- Improved Title and Legend Layout ---
    # Fix vertical spacing issues
    rect_top = 0.95 # Default top of plotting area
    
    if secondary_dim:
        legend_elements = [
            Patch(facecolor=colors[j], edgecolor='black', label=str(sec_val))
            for j, sec_val in enumerate(secondary_values)
        ]
        # Make legend wider if many items, up to 4 columns
        legend_ncol = min(max(1, len(secondary_values) // (2 if len(secondary_values) > 4 else 1)), 4)
        
        # Zero spacing between legend and plot
        legend_y_anchor = 1.02 if not show_titles else 1.0
        
        fig.legend(handles=legend_elements, title=secondary_dim, 
                  loc='upper center', bbox_to_anchor=(0.5, legend_y_anchor), 
                  ncol=legend_ncol, frameon=False,
                  fontsize=12, title_fontsize=14)
        
        # Calculate space needed for legend
        num_legend_rows = (len(legend_elements) + legend_ncol - 1) // legend_ncol
        legend_height_fraction = num_legend_rows * 0.02
        rect_top -= legend_height_fraction

    if show_titles and title:
        # Sanitize title to remove special characters
        title = ''.join(c for c in title if c.isprintable())
        # Position title right at the top
        fig.suptitle(title, fontsize=18, y=0.995) 
        rect_top -= 0.02
    
    # Ensure rect_top doesn't become too small
    rect_top = max(0.85, rect_top)
    
    # Use tighter layout with minimal padding
    try:
        plt.tight_layout(rect=[0.03, 0.05, 0.97, rect_top], pad=0.3, h_pad=0.3) 
        # Removed subplot adjustment that was changing the default spacing
    except ValueError:
        # Fallback if rect values are problematic
        plt.tight_layout(pad=0.3, h_pad=0.3)
    
    return fig

def save_plot_to_buffer(fig, format='png'):
    """Save plot to a buffer for downloading"""
    buffer = io.BytesIO()
    # Optimize saving with tight bbox to reduce margins
    fig.savefig(buffer, format=format, dpi=600, bbox_inches='tight')
    buffer.seek(0)
    return buffer

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