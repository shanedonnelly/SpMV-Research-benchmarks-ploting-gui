import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def _sanitize_text(text):
    """Remove non-printable characters from a string."""
    if text is None:
        return ""
    return ''.join(c for c in str(text) if c.isprintable())

def _get_unique_values(series):
    """Get unique, non-null values from a series, preserving category order."""
    if pd.api.types.is_categorical_dtype(series):
        # For categoricals, use the defined category order
        uniques = series.cat.categories
    else:
        uniques = series.unique()
    
    # Filter out any null/NaN values
    return [v for v in uniques if pd.notna(v)]

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

def try_numeric_sort(values):
    """
    Try to sort values in a natural order, handling numeric ranges in strings
    For strings like '[2-4]', '[4-8]', will sort by the first number
    """
    def extract_sort_key(val):
        # Handle NaN or None values
        if pd.isna(val):
            return (2, val) # Group NaNs last

        # Convert to string to handle all types
        val_str = str(val)
        
        # Try to extract first number from string (e.g., from "[2-4]" extract 2)
        numbers = re.findall(r'[-+]?\d+\.?\d*', val_str)
        if numbers:
            return (0, float(numbers[0])) # Group numeric-like strings first
        
        # If no numbers, return string for lexicographical sort
        return (1, val_str) # Group other strings second
    
    try:
        # The key returns tuples, which are sorted element by element.
        # This prevents TypeError from comparing different types.
        return sorted(values, key=extract_sort_key)
    except TypeError:
        # Fallback to normal sorting, converting all to string to be safe
        return sorted(values, key=str)

def create_side_by_side_plot(df, primary_dim, secondary_dim, y_axis, show_titles, title, 
                           show_outliers=False, fig_size_mode="Auto", fig_width_cm=None, fig_height_cm=None,
                           axes_label_mode="Auto", x_label=None, y_label=None):
    """Create side-by-side boxplot visualization"""
    
    # --- Parameter initialization with comments ---
    DPI = 300  # Resolution of the output figure (lower for reasonable file size)
    FONT_SIZE_AXIS = 14  # Font size for axis labels
    FONT_SIZE_TITLE = 16  # Font size for plot title
    FONT_SIZE_LEGEND = 13  # Font size for legend text
    FONT_SIZE_LEGEND_TITLE = 14  # Font size for legend title
    LINE_WIDTH = 0.8  # Width of lines for boxplots
    
    # Get unique values, filter NaNs, and sort
    primary_values_unique = _get_unique_values(df[primary_dim])
    primary_values = try_numeric_sort(primary_values_unique)
    num_primary_values = len(primary_values)
    
    # Determine figure size
    if fig_size_mode == "Manual" and fig_width_cm and fig_height_cm:
        # Convert cm to inches (1 cm ≈ 0.3937 inches)
        fig_width = fig_width_cm * 0.3937
        fig_height = fig_height_cm * 0.3937
    else:
        # Auto sizing logic - made consistent with stacked plots
        base_width = 5.0  # Base width of the plot
        width_per_primary_category = 1.0  # Extra width for each primary category
        
        if secondary_dim:
            secondary_values_unique = _get_unique_values(df[secondary_dim])
            secondary_values = try_numeric_sort(secondary_values_unique)
            width_per_primary_category = 0.5 + 0.25 * len(secondary_values)  # Adjust width for secondary categories
        
        fig_width = base_width + num_primary_values * width_per_primary_category
        fig_width = max(8.0, min(20.0, fig_width))  # Constrain width between 8 and 20 inches
        fig_height = 8.0  # Standard height for side-by-side plots
    
    # Create figure with reasonable DPI for better file size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=DPI)
    
    # Use thinner lines for better appearance
    plt.rcParams['lines.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.flierprops.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.boxprops.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.whiskerprops.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.capprops.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.medianprops.linewidth'] = LINE_WIDTH
    
    if secondary_dim:
        # Create grouped boxplots colored by secondary dimension
        secondary_values_unique = _get_unique_values(df[secondary_dim])
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
        
        # Create boxplot with outlier option and black median line
        data = [d for d in data if len(d) > 0]  # Ensure data is not empty
        if not data:
            st.warning(f"No data to plot for the selected combination")
            # Return an empty figure if no data
            return fig
        
        bp = ax.boxplot(data, positions=positions, patch_artist=True, 
                       labels=[""] * len(positions), widths=0.15,
                       showfliers=show_outliers,
                       medianprops={'color': 'black', 'linewidth': LINE_WIDTH})
        
        # Set primary dimension tick positions and labels
        ax.set_xticks([i + 0.125 * (len(secondary_values) - 1) for i in range(len(primary_values))])
        ax.set_xticklabels(primary_values)
        
        # Color boxes by secondary dimension
        for j, sec_val in enumerate(secondary_values):
            for i, prim_val in enumerate(primary_values):
                if (i, j) in box_indices:
                    idx = box_indices[(i, j)]
                    bp['boxes'][idx].set_facecolor(colors[j])
        
        # Add legend - enhanced size with larger font
        legend_elements = [
            Patch(facecolor=colors[j], edgecolor='black', label=str(sec_val))
            for j, sec_val in enumerate(secondary_values)
        ]
        ax.legend(handles=legend_elements, title=secondary_dim, loc='upper right',
                 fontsize=FONT_SIZE_LEGEND, title_fontsize=FONT_SIZE_LEGEND_TITLE)
        
    else:
        # Simple boxplot with primary dimension only
        # primary_values is already sorted from above
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
            
        # Create boxplot with outlier option and black median line
        bp = ax.boxplot(grouped_data, labels=labels, patch_artist=True, showfliers=show_outliers,
                       medianprops={'color': 'black', 'linewidth': LINE_WIDTH})
        
        # Set consistent colors
        colors = get_pastel_colors(len(labels))
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % len(colors)])
    
    # Customize plot
    if show_titles and title:
        ax.set_title(_sanitize_text(title), fontsize=FONT_SIZE_TITLE)
    
    # Set axis labels based on mode
    x_axis_label = x_label if axes_label_mode == "Manual" and x_label else primary_dim
    y_axis_label = y_label if axes_label_mode == "Manual" and y_label else y_axis
    
    ax.set_xlabel(_sanitize_text(x_axis_label), fontsize=FONT_SIZE_AXIS)
    ax.set_ylabel(_sanitize_text(y_axis_label), fontsize=FONT_SIZE_AXIS)
    
    # Sanitize tick labels
    sanitized_labels = []
    for label in ax.get_xticklabels():
        sanitized_text = _sanitize_text(label.get_text())
        label.set_text(sanitized_text)
        sanitized_labels.append(sanitized_text)
    
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
    
    # --- Parameter initialization with comments ---
    DPI = 300  # Resolution of the output figure (lower for reasonable file size)
    FONT_SIZE_AXIS = 14  # Font size for axis labels
    FONT_SIZE_TITLE = 16  # Font size for plot title
    FONT_SIZE_LEGEND = 12  # Font size for legend text
    FONT_SIZE_LEGEND_TITLE = 14  # Font size for legend title
    LINE_WIDTH = 0.8  # Width of lines for boxplots
    SUBPLOT_HEIGHT = 3.2  # Height per subplot (adjusted to be less tall but still adequate)
    HSPACE = 0 # Vertical space between subplots
    
    # Get unique values for each dimension, filter NaNs, and sort them
    primary_values_unique = _get_unique_values(df[primary_dim])
    secondary_values_unique = _get_unique_values(df[secondary_dim])
    
    primary_values = try_numeric_sort(primary_values_unique)
    secondary_values = try_numeric_sort(secondary_values_unique)
    n_subplots = len(secondary_values)

    if n_subplots == 0:
        # Handle case where secondary dimension has no valid data
        return create_side_by_side_plot(df, primary_dim, None, y_axis, show_titles, title, 
                                      show_outliers, fig_size_mode, fig_width_cm, fig_height_cm,
                                      axes_label_mode, x_label, y_label)

    # Determine figure size
    if fig_size_mode == "Manual" and fig_width_cm and fig_height_cm:
        # Convert cm to inches (1 cm ≈ 0.3937 inches)
        fig_width = fig_width_cm * 0.3937
        fig_height = fig_height_cm * 0.3937
    else:
        # Auto sizing logic
        num_primary_values = len(primary_values)
        base_width = 5.0  # Base width of the plot
        width_per_primary_category = 1.0  # Extra width for each primary category
        fig_width = base_width + num_primary_values * width_per_primary_category
        fig_width = max(8.0, min(20.0, fig_width))  # Constrain width between 8 and 20 inches

        # Calculate total figure height based on number of subplots
        min_fig_height = 5.0  # Minimum figure height
        title_space = 0.5 if show_titles else 0  # Space for title if needed
        
        # Calculate total height
        fig_height = max(min_fig_height, n_subplots * SUBPLOT_HEIGHT + title_space)
    
    # Create figure with reasonable DPI for better file size
    fig, axes = plt.subplots(n_subplots, 1, figsize=(fig_width, fig_height), dpi=DPI, 
                           sharex=True, squeeze=False)
    # Flatten and reverse axes so smallest secondary value is at the bottom
    axes = axes.flatten()[::-1]
    
    # Use thinner lines for better appearance
    plt.rcParams['lines.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.flierprops.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.boxprops.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.whiskerprops.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.capprops.linewidth'] = LINE_WIDTH
    plt.rcParams['boxplot.medianprops.linewidth'] = LINE_WIDTH
    
    # Colors for each subplot
    colors = get_pastel_colors(n_subplots)
    
    # Create boxplots for each secondary dimension value
    for idx, ax in enumerate(axes):
        # sec_val corresponds to the axis index (since both are sorted ascending)
        sec_val = secondary_values[idx]
        color = colors[idx]
            
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
                    labels.append(_sanitize_text(prim_val))
                else:
                    # Add empty placeholder for missing combinations
                    grouped_data.append([])
                    labels.append(_sanitize_text(prim_val))
            
            # Skip if no data
            if all(len(data) == 0 for data in grouped_data):
                continue
                
            # Create boxplot with outlier option and black median line
            try:
                bp = ax.boxplot(grouped_data, labels=labels, patch_artist=True, showfliers=show_outliers,
                              medianprops={'color': 'black', 'linewidth': LINE_WIDTH})
                
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
        
        # Set axis labels based on mode
        x_axis_label = x_label if axes_label_mode == "Manual" and x_label else primary_dim
        y_axis_label = y_label if axes_label_mode == "Manual" and y_label else y_axis
        
        # Add axis labels with specified font size
        ax.set_ylabel(_sanitize_text(y_axis_label), fontsize=FONT_SIZE_AXIS)
        
        # Only add x-axis label to bottom subplot (which is now axes[0])
        if idx == 0:
            ax.set_xlabel(_sanitize_text(x_axis_label), fontsize=FONT_SIZE_AXIS)
    
    # Add title if requested
    if show_titles and title:
        fig.suptitle(_sanitize_text(title), fontsize=FONT_SIZE_TITLE, y=1)
    
    # Create legend on the right side of the plot
    if secondary_dim:
        legend_elements = [
            Patch(facecolor=colors[j], edgecolor='black', label=str(sec_val))
            for j, sec_val in enumerate(secondary_values)
        ]
        
        # Place legend to the right of the plot
        fig.legend(handles=legend_elements, title=secondary_dim, 
                  bbox_to_anchor=(0.90, 0.5), loc='center left',
                  fontsize=FONT_SIZE_LEGEND, title_fontsize=FONT_SIZE_LEGEND_TITLE)
    
    # Rotate x-axis labels on the bottom plot if needed
    bottom_ax = axes[0]
    sanitized_labels = [_sanitize_text(label.get_text()) for label in bottom_ax.get_xticklabels()]
    if len(sanitized_labels) > 3 or any(len(label) > 10 for label in sanitized_labels):
        plt.setp(bottom_ax.get_xticklabels(), rotation=30, ha='right')

    # Adjust the layout to make room for the legend
    plt.tight_layout()
    
    # Additional adjustment for the legend on the right
    if secondary_dim:
        plt.subplots_adjust(right=0.85, hspace=HSPACE)
    
    return fig

def save_plot_to_buffer(fig, format='png'):
    """Save plot to a buffer for downloading"""
    buffer = io.BytesIO()
    # Optimize saving with tight bbox to reduce margins, reduced DPI for better file size
    fig.savefig(buffer, format=format, dpi=300, bbox_inches='tight')
    buffer.seek(0)
    return buffer
