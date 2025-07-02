import pandas as pd
import math
import io
import os
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from matplotlib_interface import (
    create_side_by_side_plot,
    create_stacked_plots,
    save_plot_to_buffer
)

# --- Utility Functions ---

def check_columns_match(dataframes):
    """
    Check if all dataframes have exactly the same column names.
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
    Check if a column contains numerical data.
    """
    return pd.api.types.is_numeric_dtype(df[column])

# --- Data Processing ---

def preprocess_data(df, primary_dim, secondary_dim, primary_bin_edges=None, secondary_bin_edges=None):
    """Process data for visualization, handling numerical columns by binning."""
    modified_df = df.copy()

    def _apply_binning(series, bin_edges, column_name):
        if not bin_edges or len(bin_edges) < 2:
            return series
        # Ensure bins are sorted and unique to avoid errors with pd.cut
        bins = sorted(list(set(bin_edges)))
        if len(bins) < 2:
            return series
        
        # Generate labels that strictly reflect the provided bin edges, e.g., [a, b)
        labels = [f"[{bins[i]:g} - {bins[i+1]:g})" for i in range(len(bins) - 1)]
        
        try:
            # `right=False` creates intervals like [a, b), [b, c), ...
            # `include_lowest=True` ensures the first value is included in the first bin.
            return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)
        except ValueError as e:
            st.warning(f"Could not apply binning on '{column_name}': {e}. Values might be outside the specified range.")
            return series

    if primary_dim and primary_bin_edges:
        modified_df[primary_dim] = _apply_binning(modified_df[primary_dim], primary_bin_edges, primary_dim)
    if secondary_dim and secondary_bin_edges:
        modified_df[secondary_dim] = _apply_binning(modified_df[secondary_dim], secondary_bin_edges, secondary_dim)
    
    return modified_df

# --- Plot Generation Logic ---

def generate_plot_figures(dataframes, filenames, primary_dim, secondary_dim, y_axis,
                          show_titles, plot_titles, comparison_mode, show_outliers,
                          fig_size_mode, fig_width_cm, fig_height_cm,
                          axes_label_mode, x_label, y_label,
                          primary_bin_edges, secondary_bin_edges,
                          primary_dim_order, secondary_dim_order):
    """
    Generates Matplotlib figure objects for each dataframe.
    Returns a list of tuples: (figure, filename).
    """
    figures = []
    for df, filename in zip(dataframes, filenames):
        try:
            modified_df = preprocess_data(df, primary_dim, secondary_dim, primary_bin_edges, secondary_bin_edges)
            title = plot_titles.get(filename, os.path.splitext(filename)[0]) if show_titles else None
            
            plot_func = create_side_by_side_plot if comparison_mode == "Side by Side" else create_stacked_plots
            
            fig = plot_func(modified_df, primary_dim, secondary_dim, y_axis,
                            show_titles, title, show_outliers,
                            fig_size_mode, fig_width_cm, fig_height_cm,
                            axes_label_mode, x_label, y_label,
                            primary_dim_order=primary_dim_order,
                            secondary_dim_order=secondary_dim_order)
            figures.append((fig, filename))
        except Exception as e:
            st.error(f"Error generating plot figure for {filename}: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            figures.append((None, filename))
    return figures


def generate_combined_plot_logic(
    dataframes, filenames, primary_dim, secondary_dim, y_axis, 
    show_titles, comparison_mode, show_outliers, 
    fig_size_mode, fig_width_cm, fig_height_cm, output_format, 
    plot_titles, axes_label_mode, x_label, y_label, 
    primary_bin_edges, secondary_bin_edges, combined_plot_title,
    grid_layout_mode, grid_rows, grid_cols,
    primary_dim_order, secondary_dim_order
):
    """
    Generates all plots, combines them into a single grid image, and returns
    the final image object and arguments for a Streamlit download button.
    """
    try:
        image_buffers = []
        figures_with_filenames = generate_plot_figures(
            dataframes, filenames, primary_dim, secondary_dim, y_axis,
            show_titles, plot_titles, comparison_mode, show_outliers,
            fig_size_mode, fig_width_cm, fig_height_cm,
            axes_label_mode, x_label, y_label,
            primary_bin_edges, secondary_bin_edges,
            primary_dim_order, secondary_dim_order
        )

        for fig, _ in figures_with_filenames:
            if fig:
                buffer = save_plot_to_buffer(fig, format='png')
                image_buffers.append(buffer)

        if not image_buffers:
            st.warning("No plots were generated to combine.")
            return None, None

        # 2. Process images with Pillow
        images = [Image.open(buf) for buf in image_buffers]
        num_images = len(images)

        # 3. Calculate grid dimensions
        if grid_layout_mode == "Manual":
            rows, cols = grid_rows, grid_cols
            if rows * cols < num_images:
                st.warning(f"Manual grid ({rows}x{cols}) is too small for {num_images} plots. Falling back to auto layout.")
                cols = math.ceil(math.sqrt(num_images))
                rows = math.ceil(num_images / cols)
        else: # Auto mode
            cols = math.ceil(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)

        # 4. Determine max plot size to avoid decompression bomb warning
        Pillow_limit = getattr(Image, 'MAX_IMAGE_PIXELS', 89478485)
        if cols * rows > 0:
            max_pixels_per_plot = (Pillow_limit / (cols * rows)) * 0.99
            max_dim_allowed_by_limit = math.floor(math.sqrt(max_pixels_per_plot))
        else:
            max_dim_allowed_by_limit = 1200
        
        USER_MAX_PLOT_DIM = 1200
        max_dim = min(USER_MAX_PLOT_DIM, max_dim_allowed_by_limit)
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

        # 5. Resize and paste images onto a square canvas
        squared_images = []
        for img in images:
            img.thumbnail((max_dim, max_dim), resample_filter)
            square_img = Image.new('RGB', (max_dim, max_dim), 'white')
            paste_pos = ((max_dim - img.width) // 2, (max_dim - img.height) // 2)
            square_img.paste(img, paste_pos, img if img.mode == 'RGBA' else None)
            squared_images.append(square_img)

        # 6. Create final canvas and add title
        title_height = 0
        title_font_size = max(15, int(max_dim / 25))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", title_font_size)
        except IOError:
            try:
                font = ImageFont.load_default().font_variant(size=title_font_size)
            except AttributeError:
                font = ImageFont.load_default()

        if combined_plot_title:
            dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            try:
                bbox = dummy_draw.textbbox((0, 0), combined_plot_title, font=font)
                title_width = bbox[2] - bbox[0]
                title_height = (bbox[3] - bbox[1]) + int(title_font_size * 0.5)
            except AttributeError:
                title_width, legacy_h = dummy_draw.textsize(combined_plot_title, font=font)
                title_height = legacy_h + int(title_font_size * 0.5)

        final_width = cols * max_dim
        final_height = (rows * max_dim) + title_height
        final_image = Image.new('RGB', (final_width, final_height), 'white')
        draw = ImageDraw.Draw(final_image)

        if combined_plot_title:
            title_x = (final_width - title_width) / 2 if title_width < final_width else 0
            draw.text((title_x, title_height / 4), combined_plot_title, fill='black', font=font)

        # 7. Paste squared images onto the grid
        for i, img in enumerate(squared_images):
            row_idx = i // cols
            col_idx = i % cols
            paste_x = col_idx * max_dim
            paste_y = (row_idx * max_dim) + title_height
            final_image.paste(img, (paste_x, paste_y))

        # 8. Prepare for download
        final_buffer = io.BytesIO()
        final_format = output_format.lower()
        if final_format == 'pdf':
            final_image.save(final_buffer, format='PDF', resolution=100.0, save_all=True)
        else:
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
        return final_image, download_args

    except Exception as e:
        st.error(f"Error generating combined plot: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None