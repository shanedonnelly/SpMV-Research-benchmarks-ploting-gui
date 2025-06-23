import streamlit as st
import pandas as pd
import os
import io
import zipfile
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
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

def render_binning_controls(df, column_name, key_prefix):
    """Renders performant sliders and inputs for numerical binning."""
    st.markdown(f"##### Binning for `{column_name}`")
    min_val = float(df[column_name].min())
    max_val = float(df[column_name].max())

    n_bins_key = f"plot_{key_prefix}_n_bins"
    boundaries_key = f"plot_{key_prefix}_boundaries"

    # Use a callback on the first slider to reset boundaries when n_bins changes
    def n_bins_changed():
        n_bins = st.session_state[n_bins_key]
        if n_bins > 1:
            # Calculate new, equally spaced boundaries
            new_boundaries = np.linspace(min_val, max_val, n_bins + 1)[1:-1]
            st.session_state[boundaries_key] = [round(b, 1) for b in new_boundaries]
        elif boundaries_key in st.session_state:
            # Clear boundaries if n_bins is 1
            del st.session_state[boundaries_key]

    n_bins = st.slider(
        "Number of ranges",
        min_value=1, max_value=10, step=1,
        key=n_bins_key,
        on_change=n_bins_changed,
        # Set default value without overwriting existing state
        value=st.session_state.get(n_bins_key, 1)
    )

    if n_bins <= 1:
        return None

    # Ensure boundaries are initialized if they don't exist for the current n_bins
    if boundaries_key not in st.session_state or len(st.session_state.get(boundaries_key, [])) != n_bins - 1:
        n_bins_changed()

    boundaries = st.session_state.get(boundaries_key, [])
    
    st.caption("Fine-tune range delimiters")
    
    # Determine step based on column dtype
    is_int = pd.api.types.is_integer_dtype(df[column_name])
    step = 1.0 if is_int else 0.1
    
    num_inputs = n_bins - 1
    inputs_per_row = 5
    new_boundaries = []
    
    # Create a list of all the number inputs to be created
    all_input_indices = list(range(num_inputs))

    for i in range(0, num_inputs, inputs_per_row):
        row_indices = all_input_indices[i:i + inputs_per_row]
        cols = st.columns(len(row_indices))
        
        for j, input_idx in enumerate(row_indices):
            with cols[j]:
                # Set min/max for each input to prevent overlap
                min_b = boundaries[input_idx-1] if input_idx > 0 else min_val
                max_b = boundaries[input_idx+1] if input_idx < len(boundaries) - 1 else max_val
                
                val = st.number_input(
                    label=f"boundary_{input_idx}",
                    label_visibility="collapsed",
                    value=float(boundaries[input_idx]),
                    min_value=float(min_b),
                    max_value=float(max_b),
                    step=step,
                    format="%.4f",
                    key=f"{boundaries_key}_input_{input_idx}"
                )
                new_boundaries.append(val)

    # Update session state only if there's a change
    if new_boundaries != boundaries:
        st.session_state[boundaries_key] = sorted(new_boundaries)
        # Rerun to update the min/max constraints of the number inputs
        st.rerun()

    final_boundaries = st.session_state.get(boundaries_key, [])
    return [min_val] + final_boundaries + [max_val] if final_boundaries else None

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
                    "axes_label_mode", "x_axis_label", "y_axis_label",
                    "combine_plots", "combined_plot_title",
                    "grid_layout_mode", "grid_rows", "grid_cols"
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
    
    # --- Binning controls for numerical dimensions ---
    primary_bin_edges = None
    secondary_bin_edges = None

    if primary_dim and is_numerical_column(dataframes[0], primary_dim) and dataframes[0][primary_dim].nunique() > 5:
        primary_bin_edges = render_binning_controls(dataframes[0], primary_dim, "primary")
        st.divider()

    if secondary_dim not in [None, "None"] and is_numerical_column(dataframes[0], secondary_dim) and dataframes[0][secondary_dim].nunique() > 5:
        secondary_bin_edges = render_binning_controls(dataframes[0], secondary_dim, "secondary")
        st.divider()
    
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
    
    # New feature: Combine plots
    combine_plots = False
    combined_plot_title = ""
    if len(dataframes) > 1:
        combine_plots = st.checkbox(
            "Combine plots into a single image", 
            value=False, 
            key="combine_plots", 
            # help="Combine all plots into a single grid image. Only available for multiple subsets."
        )
        if combine_plots:
            combined_plot_title = st.text_input(
                "Overall plot title", 
                value="", 
                key="combined_plot_title", 
                placeholder="Optional title for the combined plot"
            )
            st.segmented_control(
                "Grid Layout",
                ["Auto", "Manual"],
                key="grid_layout_mode",
                default="Auto"
            )
            if st.session_state.get("grid_layout_mode") == "Manual":
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input("Rows", min_value=1, value=1, key="grid_rows")
                with col2:
                    st.number_input("Columns", min_value=1, value=1, key="grid_cols")

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
        
        st.session_state.last_plot_params = {
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
            'y_label': y_label,
            'primary_bin_edges': primary_bin_edges,
            'secondary_bin_edges': secondary_bin_edges,
            'combine_plots': combine_plots,
            'combined_plot_title': combined_plot_title,
            'grid_layout_mode': st.session_state.get('grid_layout_mode', 'Auto'),
            'grid_rows': st.session_state.get('grid_rows', 1),
            'grid_cols': st.session_state.get('grid_cols', 1)
        }
        # Clear any previous cache when generating a new plot
        if 'combined_plot_cache' in st.session_state:
            del st.session_state.combined_plot_cache
        if 'combined_plot_cache_key' in st.session_state:
            del st.session_state.combined_plot_cache_key


    # Show plots if they've been generated and parameters haven't changed
    if 'last_plot_params' in st.session_state:
        secondary_dimension = None if secondary_dim == "None" else secondary_dim
        current_params_check = {
            'dataframes': dataframes, 'filenames': filenames, 'primary_dim': primary_dim,
            'secondary_dim': secondary_dimension, 'y_axis': y_axis, 'show_titles': show_titles,
            'comparison_mode': comparison_mode, 'show_outliers': show_outliers,
            'fig_size_mode': fig_size_mode, 'fig_width_cm': fig_width_cm, 'fig_height_cm': fig_height_cm,
            'output_format': output_format, 'plot_titles': plot_titles, 'axes_label_mode': axes_label_mode,
            'x_label': x_label, 'y_label': y_label, 'primary_bin_edges': primary_bin_edges,
            'secondary_bin_edges': secondary_bin_edges, 'combine_plots': combine_plots,
            'combined_plot_title': combined_plot_title,
            'grid_layout_mode': st.session_state.get('grid_layout_mode', 'Auto'),
            'grid_rows': st.session_state.get('grid_rows', 1),
            'grid_cols': st.session_state.get('grid_cols', 1)
        }
        
        # To avoid the "truth value of a DataFrame is ambiguous" error,
        # we compare the parameters without the dataframes themselves.
        # The list of filenames serves as a proxy for the data.
        params_for_comparison_current = current_params_check.copy()
        params_for_comparison_last = st.session_state.last_plot_params.copy()
        del params_for_comparison_current['dataframes']
        del params_for_comparison_last['dataframes']

        # # If params have changed, inform the user and don't show the plot
        # if params_for_comparison_current != params_for_comparison_last:
        #     st.info("Plot settings have changed. Click 'Generate Plot' to update.")
        # else:
        if not params_for_comparison_current != params_for_comparison_last:
            generate_individual_boxplots(**st.session_state.last_plot_params)

def generate_individual_boxplots(dataframes, filenames, primary_dim, secondary_dim, y_axis, 
                               show_titles=False, comparison_mode="Separate", 
                               show_outliers=False, fig_size_mode="Auto", fig_width_cm=None, 
                               fig_height_cm=None, output_format="PNG", plot_titles=None,
                               axes_label_mode="Auto", x_label=None, y_label=None,
                               primary_bin_edges=None, secondary_bin_edges=None,
                               combine_plots=False, combined_plot_title="",
                               grid_layout_mode="Auto", grid_rows=3, grid_cols=3):
    """
    Generate separate boxplots for each dataframe, or a single combined plot.
    """
    # If combine plots is selected, call the dedicated function and exit
    if combine_plots and len(dataframes) > 1:
        generate_combined_plot(
            dataframes=dataframes, filenames=filenames, primary_dim=primary_dim, 
            secondary_dim=secondary_dim, y_axis=y_axis, show_titles=show_titles, 
            comparison_mode=comparison_mode, show_outliers=show_outliers, 
            fig_size_mode=fig_size_mode, fig_width_cm=fig_width_cm, 
            fig_height_cm=fig_height_cm, output_format=output_format, 
            plot_titles=plot_titles, axes_label_mode=axes_label_mode, 
            x_label=x_label, y_label=y_label, primary_bin_edges=primary_bin_edges, 
            secondary_bin_edges=secondary_bin_edges, combined_plot_title=combined_plot_title,
            grid_layout_mode=grid_layout_mode, grid_rows=grid_rows, grid_cols=grid_cols
        )
        return

    plot_buffers = []
    need_zip = len(dataframes) > 1
    
    for file_idx, (df, filename) in enumerate(zip(dataframes, filenames)):
        st.markdown(f"### {os.path.splitext(filename)[0]}")
        
        # Preprocess data
        modified_df = preprocess_data(df, primary_dim, secondary_dim, primary_bin_edges, secondary_bin_edges)
        
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

def generate_combined_plot(dataframes, filenames, primary_dim, secondary_dim, y_axis, 
                           show_titles=False, comparison_mode="Separate", 
                           show_outliers=False, fig_size_mode="Auto", fig_width_cm=None, 
                           fig_height_cm=None, output_format="PNG", plot_titles=None,
                           axes_label_mode="Auto", x_label=None, y_label=None,
                           primary_bin_edges=None, secondary_bin_edges=None,
                           combined_plot_title="",
                           grid_layout_mode="Auto", grid_rows=3, grid_cols=3):
    """
    Generates all plots and combines them into a single grid image using Pillow,
    dynamically resizing to avoid decompression bomb warnings.
    """
    # --- Manual Caching Logic ---
    # Create a hashable key from all function parameters
    cache_key = (
        tuple(filenames), primary_dim, secondary_dim, y_axis, show_titles,
        comparison_mode, show_outliers, fig_size_mode, fig_width_cm,
        fig_height_cm, output_format, str(plot_titles), axes_label_mode,
        x_label, y_label, str(primary_bin_edges), str(secondary_bin_edges),
        combined_plot_title, grid_layout_mode, grid_rows, grid_cols
    )

    # If the key matches the cached one, display the cached content and exit
    if st.session_state.get('combined_plot_cache_key') == cache_key:
        cached_data = st.session_state.combined_plot_cache
        st.image(cached_data['image'], caption="Combined Plot")
        st.download_button(**cached_data['download_args'])
        return
    # --- End of Caching Logic ---

    with st.spinner("Generating combined plot... This may take a moment."):
        try:
            image_buffers = []
            # 1. Generate all plots and store them in memory
            for df, filename in zip(dataframes, filenames):
                modified_df = preprocess_data(df, primary_dim, secondary_dim, primary_bin_edges, secondary_bin_edges)
                
                title = plot_titles.get(filename, os.path.splitext(filename)[0]) if show_titles else None
                
                plot_func = create_side_by_side_plot if comparison_mode == "Side by Side" else create_stacked_plots
                
                fig = plot_func(modified_df, primary_dim, secondary_dim, y_axis, 
                                show_titles, title, show_outliers, 
                                fig_size_mode, fig_width_cm, fig_height_cm,
                                axes_label_mode, x_label, y_label)
                
                buffer = save_plot_to_buffer(fig, format='png')
                image_buffers.append(buffer)

            if not image_buffers:
                st.warning("No plots were generated to combine.")
                return

            # 2. Process images with Pillow
            images = [Image.open(buf) for buf in image_buffers]

            # 3. Calculate grid dimensions and determine max plot size to avoid decompression bomb warning
            num_images = len(images)
            if grid_layout_mode == "Manual":
                rows = grid_rows
                cols = grid_cols
                if rows * cols < num_images:
                    st.warning(f"Manual grid ({rows}x{cols}) is too small for {num_images} plots. Falling back to auto layout.")
                    cols = math.ceil(math.sqrt(num_images))
                    rows = math.ceil(num_images / cols)
            else: # Auto mode
                cols = math.ceil(math.sqrt(num_images))
                rows = math.ceil(num_images / cols)

            # Get the Pillow decompression bomb limit (use a safe default if the attribute doesn't exist)
            Pillow_limit = getattr(Image, 'MAX_IMAGE_PIXELS', 89478485)

            # Calculate the max dimension per plot to stay under the limit, with a small safety margin
            if cols * rows > 0:
                max_pixels_per_plot = (Pillow_limit / (cols * rows)) * 0.99
                max_dim_allowed_by_limit = math.floor(math.sqrt(max_pixels_per_plot))
            else:
                max_dim_allowed_by_limit = 1200 # Default value if no images

            # Also enforce a user-defined "reasonable" max size for a single plot
            USER_MAX_PLOT_DIM = 1200
            
            # The final dimension for each square plot is the minimum of these constraints
            max_dim = min(USER_MAX_PLOT_DIM, max_dim_allowed_by_limit)

            # Use Image.Resampling.LANCZOS for high quality downscaling (Pillow >= 9.1.0)
            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

            squared_images = []
            for img in images:
                # Resize the original image to fit within a max_dim box, preserving aspect ratio
                img.thumbnail((max_dim, max_dim), resample_filter)
                
                # Create a new white square canvas
                square_img = Image.new('RGB', (max_dim, max_dim), 'white')
                
                # Paste the (possibly resized) image onto the center of the square canvas
                paste_pos = ((max_dim - img.width) // 2, (max_dim - img.height) // 2)
                square_img.paste(img, paste_pos, img if img.mode == 'RGBA' else None)
                squared_images.append(square_img)

            # 4. Create final canvas and add title
            title_height = 0
            title_font_size = max(15, int(max_dim / 25))
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", title_font_size)
            except IOError:
                try: # Pillow >= 10.0.0
                    font = ImageFont.load_default().font_variant(size=title_font_size)
                except AttributeError: # Pillow < 10.0.0
                    font = ImageFont.load_default(size=title_font_size)

            if combined_plot_title:
                dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
                try: # Modern Pillow
                    bbox = dummy_draw.textbbox((0, 0), combined_plot_title, font=font)
                    title_width = bbox[2] - bbox[0]
                    title_height = (bbox[3] - bbox[1]) + int(title_font_size * 0.5) # Padding
                except AttributeError: # Older Pillow
                    title_width, legacy_h = dummy_draw.textsize(combined_plot_title, font=font)
                    title_height = legacy_h + int(title_font_size * 0.5)

            grid_width = cols * max_dim
            final_width = grid_width
            final_height = (rows * max_dim) + title_height

            final_image = Image.new('RGB', (final_width, final_height), 'white')
            draw = ImageDraw.Draw(final_image)

            if combined_plot_title:
                title_x = (final_width - title_width) / 2 if title_width < final_width else 0
                draw.text((title_x, title_height / 4), combined_plot_title, fill='black', font=font)

            # 5. Paste squared images onto the grid
            for i, img in enumerate(squared_images):
                row_idx = i // cols
                col_idx = i % cols
                paste_x = col_idx * max_dim
                paste_y = (row_idx * max_dim) + title_height
                final_image.paste(img, (paste_x, paste_y))

            # 6. Display and provide download
            st.image(final_image, caption="Combined Plot")

            final_buffer = io.BytesIO()
            final_format = output_format.lower()
            
            if final_format == 'pdf':
                final_image.save(final_buffer, format='PDF', resolution=100.0, save_all=True)
            else: # PNG
                final_image.save(final_buffer, format='PNG')
            
            final_buffer.seek(0)
            
            extension = "pdf" if final_format == "pdf" else "png"
            mime_type = "application/pdf" if final_format == "pdf" else "image/png"

            download_args = {
                "label": f"Download Combined Plot as {extension.upper()}",
                "data": final_buffer,
                "file_name": f"combined_plot.{extension}",
                "mime": mime_type,
                "key": "download_combined_plot"
            }
            st.download_button(**download_args)

            # Store results in the manual cache
            st.session_state.combined_plot_cache_key = cache_key
            st.session_state.combined_plot_cache = {
                'image': final_image,
                'download_args': download_args
            }

        except Exception as e:
            st.error(f"Error generating combined plot: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def preprocess_data(df, primary_dim, secondary_dim, primary_bin_edges=None, secondary_bin_edges=None):
    """Process data for visualization, handling numerical columns by binning."""
    modified_df = df.copy()

    def _apply_binning(series, bin_edges, column_name):
        if not bin_edges or len(bin_edges) < 2:
            return series

        # Ensure bins are unique and sorted
        bins = sorted(list(set(bin_edges)))
        if len(bins) < 2:
            return series

        # Create labels like [start-end) and [start-end] for the last one
        labels = []
        for i in range(len(bins) - 1):
            start, end = bins[i], bins[i+1]
            if i < len(bins) - 2:
                labels.append(f"[{start:g} - {end:g})")
            else:
                labels.append(f"[{start:g} - {end:g}]")
        
        try:
            # Use right=False for [left, right) intervals
            return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
        except ValueError as e:
            # This can happen if all values fall outside the bins
            st.warning(f"Could not apply binning on '{column_name}': {e}. Values might be outside the specified range.")
            return series

    if primary_dim and primary_bin_edges:
        modified_df[primary_dim] = _apply_binning(modified_df[primary_dim], primary_bin_edges, primary_dim)

    if secondary_dim and secondary_bin_edges:
        modified_df[secondary_dim] = _apply_binning(modified_df[secondary_dim], secondary_bin_edges, secondary_dim)
    
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