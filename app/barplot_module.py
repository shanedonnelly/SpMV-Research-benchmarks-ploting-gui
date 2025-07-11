import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import io
import zipfile

from coloring_module import render_color_picker_component
from matplotlib_interface import save_plot_to_buffer, get_colors_from_session, _sanitize_text

def _create_barplot(df, primary_dim, secondary_dim, y_axis, title, fig_size_params, axes_label_params, secondary_dim_label_mode):
    """
    Generates a bar plot figure based on the provided parameters.
    """
    fig, ax = plt.subplots(figsize=fig_size_params['figsize'])

    # Group data for plotting
    if secondary_dim and secondary_dim != "None":
        grouped = df.groupby(secondary_dim)
        group_labels = list(grouped.groups.keys())
        num_groups = len(group_labels)
        
        primary_values = df[primary_dim].unique()
        num_primary = len(primary_values)
        
        bar_width = 0.8 / num_primary
        group_positions = np.arange(num_groups)
        
        colors = get_colors_from_session(num_primary)
        color_map = {val: colors[i % len(colors)] for i, val in enumerate(primary_values)}

        for i, primary_val in enumerate(primary_values):
            heights = []
            for group_name in group_labels:
                group_df = grouped.get_group(group_name)
                row = group_df[group_df[primary_dim] == primary_val]
                heights.append(row[y_axis].iloc[0] if not row.empty else 0)
            
            position_offset = (i - num_primary / 2 + 0.5) * bar_width
            ax.bar(group_positions + position_offset, heights, width=bar_width, label=primary_val, color=color_map[primary_val])

        ax.set_xticks(group_positions)
        if secondary_dim_label_mode == "Labels":
            sanitized_labels = [_sanitize_text(label) for label in group_labels]
            ax.set_xticklabels(sanitized_labels)
            if len(sanitized_labels) > 3 or any(len(str(label)) > 10 for label in sanitized_labels):
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        else: # Default to "ID"
            ax.set_xticklabels(range(1, num_groups + 1)) # Labels are 1 to N
        ax.set_xlabel(axes_label_params.get('x_label') or secondary_dim)

    else: # No secondary dimension
        primary_values = df[primary_dim].unique()
        num_primary = len(primary_values)
        colors = get_colors_from_session(num_primary)
        color_map = {val: colors[i % len(colors)] for i, val in enumerate(primary_values)}
        
        bar_colors = [color_map[val] for val in df[primary_dim]]
        ax.bar(df[primary_dim], df[y_axis], color=bar_colors)
        ax.set_xlabel(axes_label_params.get('x_label') or primary_dim)

    # Common plot settings
    ax.set_ylabel(axes_label_params.get('y_label') or y_axis)
    if title:
        ax.set_title(title)
    
    if secondary_dim and secondary_dim != "None":
        legend_elements = [Patch(facecolor=color_map[val], label=val) for val in primary_values]
        # Place legend to the right of the plot
        fig.legend(handles=legend_elements, title=primary_dim, 
                   bbox_to_anchor=(0.90, 0.5), loc='center left')

    plt.tight_layout()
    # Adjust layout to make room for the legend
    if secondary_dim and secondary_dim != "None":
        plt.subplots_adjust(right=0.85)
        
    return fig

def render_barplot_component(dataframes, filenames):
    """
    Renders the UI for configuring and displaying a bar plot.
    Handles single or multiple dataframes.
    """
    if not dataframes:
        st.warning("Please select at least one subset to plot.")
        return

    # --- UI Controls ---
    st.markdown("#### Bar Plot Controls")

    # Use the first dataframe for UI default values
    df_for_ui = dataframes[0]
    columns = list(df_for_ui.columns)
    numerical_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df_for_ui[col])]

    if not numerical_cols:
        st.warning("The selected dataframe has no numerical columns to plot for the Y-axis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        secondary_options = ["None"] + columns
        default_secondary = 'matrix_name' if 'matrix_name' in secondary_options else 'None'
        secondary_dim = st.selectbox("Secondary Dimension (X-axis groups)", secondary_options, 
                                     index=secondary_options.index(default_secondary), key="bp_secondary_dim")
    with col2:
        default_primary = 'format_name' if 'format_name' in columns else columns[0]
        primary_dim = st.selectbox("Primary Dimension (X-axis clusters)", columns, 
                                   index=columns.index(default_primary), key="bp_primary_dim")

    secondary_dim_label_mode = "ID"
    if secondary_dim and secondary_dim != "None":
        secondary_dim_label_mode = st.segmented_control(
            "Secondary Dimension Labels", ["ID", "Labels"], key="bp_secondary_dim_label_mode", default="ID",
            help="Use numeric IDs or actual labels for the secondary dimension on the X-axis."
        )

    default_y = "gflops" if "gflops" in numerical_cols else numerical_cols[0]
    y_axis = st.selectbox("Y-axis", numerical_cols, 
                         index=numerical_cols.index(default_y) if default_y in numerical_cols else 0, 
                         key="bp_y_axis")

    axes_label_mode = st.segmented_control("Axes Labels", ["Auto", "Manual"], key="bp_axes_label_mode", default="Auto")
    x_label, y_label = None, None
    if axes_label_mode == "Manual":
        col1, col2 = st.columns(2)
        with col1:
            x_label = st.text_input("X-axis Label", value=(secondary_dim if secondary_dim != 'None' else primary_dim), key="bp_x_axis_label")
        with col2:
            y_label = st.text_input("Y-axis Label", value=y_axis, key="bp_y_axis_label")

    col1, col2 = st.columns([1, 2])
    with col1:
        fig_size_mode = st.segmented_control("Figure Size", ["Auto", "Manual"], key="bp_fig_size_mode", default="Auto")
    with col2:
        fig_width_cm, fig_height_cm = None, None
        if fig_size_mode == "Manual":
            col_w, col_h = st.columns(2)
            with col_w:
                fig_width_cm = st.slider("Width (cm)", 4, 100, 40, 1, key="bp_fig_width_cm")
            with col_h:
                fig_height_cm = st.slider("Height (cm)", 4, 100, 40, 1, key="bp_fig_height_cm")

    output_format = st.segmented_control("Output Format", ["PNG", "PDF"], key="bp_output_format", default="PNG")

    show_titles = st.checkbox("Add titles to plots", value=False, key="bp_show_titles")
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
                                                    key=f"bp_title_{i}")

    combine_plots = False
    if len(dataframes) > 1:
        combine_plots = st.checkbox(
            "Combine plots into a single image", 
            value=False, 
            key="bp_combine_plots"
        )

    render_color_picker_component()

    if st.button("Generate Plot", key="bp_generate_plot", type="primary"):
        st.session_state.bp_last_params = {
            'primary_dim': primary_dim,
            'secondary_dim': secondary_dim,
            'y_axis': y_axis,
            'show_titles': show_titles,
            'plot_titles': plot_titles,
            'fig_size_mode': fig_size_mode,
            'fig_width_cm': fig_width_cm,
            'fig_height_cm': fig_height_cm,
            'axes_label_mode': axes_label_mode,
            'x_label': x_label,
            'y_label': y_label,
            'output_format': output_format,
            'combine_plots': combine_plots,
            'secondary_dim_label_mode': secondary_dim_label_mode
        }

    if 'bp_last_params' in st.session_state:
        params = st.session_state.bp_last_params
        
        figsize = (params['fig_width_cm'] / 2.54, params['fig_height_cm'] / 2.54) if params['fig_size_mode'] == 'Manual' else (12, 8)
        
        figures = []
        for i, df in enumerate(dataframes):
            filename = filenames[i]
            title = params['plot_titles'].get(filename, os.path.splitext(filename)[0]) if params['show_titles'] else None

            # --- Data Validation and Preparation for each df ---
            required_cols = ['mem_footprint', 'matrix_name', 'format_name']
            if 'mem_footprint' in df.columns:
                df = df.copy()
                df.sort_values(by='mem_footprint', ascending=True, inplace=True, kind='stable')

            fig = _create_barplot(
                df=df,
                primary_dim=params['primary_dim'],
                secondary_dim=params['secondary_dim'],
                y_axis=params['y_axis'],
                title=title,
                fig_size_params={'figsize': figsize},
                axes_label_params={'x_label': params['x_label'], 'y_label': params['y_label']},
                secondary_dim_label_mode=params['secondary_dim_label_mode']
            )
            figures.append((fig, filename))
        
        # --- Display Logic ---
        if params['combine_plots'] and len(figures) > 1:
            # This part can be enhanced with grid layout logic like in plotting_component
            st.warning("Combined plot view for bar plots is not yet implemented. Displaying individually.")

        # Display individual plots
        cols = st.columns(min(3, len(figures)))
        zip_buffer = io.BytesIO()
        
        for i, (fig, filename) in enumerate(figures):
            with cols[i % len(cols)]:
                st.pyplot(fig)
                
                buffer = save_plot_to_buffer(fig, format=params['output_format'].lower())
                
                # Add to zip file
                with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zf:
                    file_extension = "pdf" if params['output_format'].lower() == "pdf" else "png"
                    zf.writestr(f"{os.path.splitext(filename)[0]}_bar.{file_extension}", buffer.getvalue())

                st.download_button(
                    label=f"Download {os.path.splitext(filename)[0]}",
                    data=buffer,
                    file_name=f"{os.path.splitext(filename)[0]}_bar.{params['output_format'].lower()}",
                    mime=f"image/{params['output_format'].lower()}",
                    key=f"bp_download_button_{i}"
                )
        
        if len(figures) > 1:
            st.download_button(
                label="Download All as ZIP",
                data=zip_buffer.getvalue(),
                file_name="bar_plots.zip",
                mime="application/zip",
                key="bp_download_all"
            )