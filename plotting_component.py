import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import os
import io
import zipfile
import numpy as np

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
            st.session_state.plots_generated = False
            # Clear other selections
            if "group_by_cols" in st.session_state:
                del st.session_state.group_by_cols
            if "y_axis" in st.session_state:
                del st.session_state.y_axis
            if "show_titles" in st.session_state:
                del st.session_state.show_titles
            if "plot_params" in st.session_state:  # Ensure plot_params is cleared
                del st.session_state.plot_params
            st.rerun()
    
    # Get column names from first dataframe
    columns = list(dataframes[0].columns)
    
    # Rest of controls...
    st.selectbox("Plot Type", ["Boxplot"], key="plot_mode", disabled=True)
    
    col1, col2 = st.columns(2)
    with col1:
        group_by_cols = st.multiselect("Group by", columns, key="group_by_cols")
    with col2:
        default_y = "gflops" if "gflops" in columns else columns[0]
        y_axis = st.selectbox("Y-axis", columns, 
                             index=columns.index(default_y) if default_y in columns else 0, 
                             key="y_axis")
    
    # Add title checkbox
    show_titles = st.checkbox("add subset name as title", value=False, key="show_titles")
    
    # Generate plot button
    if st.button("Generate Plot", key="generate_plot"):
        if not group_by_cols:
            st.warning("Please select at least one column to group by")
            return
        
        st.session_state.plots_generated = True
        st.session_state.plot_params = { # plot_params is set here
            'dataframes': dataframes,
            'filenames': filenames,
            'group_by_cols': group_by_cols,
            'y_axis': y_axis,
            'show_titles': show_titles
        }
    
    # Show plots if they've been generated
    if st.session_state.plots_generated:
        generate_individual_boxplots(**st.session_state.plot_params)

def generate_individual_boxplots(dataframes, filenames, group_by_cols, y_axis, show_titles=False):
    """
    Generate separate boxplots for each dataframe
    """
    plot_buffers = []
    
    # Check if we need to create zip file
    need_zip = len(dataframes) > 1
    
    for i, (df, filename) in enumerate(zip(dataframes, filenames)):
        st.markdown(f"### {os.path.splitext(filename)[0]}")
        
        fig, ax = plt.subplots()
        
        # Handle numerical columns that may have too many unique values
        modified_df = df.copy()
        numerical_warning_shown = False
        
        for col in group_by_cols:
            if is_numerical_column(df, col) and df[col].nunique() > 10:  # Too many unique values for boxplot
                if not numerical_warning_shown:
                    st.warning("Numerical column with many unique values detected. Binning into groups (not fully implemented).")
                    numerical_warning_shown = True
                
                # Create a binned version of the numerical column
                modified_df[col] = bin_numerical_data(df, col)
        
        # Prepare data for boxplot
        grouped_data = []
        labels = []
        
        # Group the data and collect values for boxplot
        for name, group in modified_df.groupby(group_by_cols, observed=False):
            if isinstance(name, tuple):
                # Better formatting for combination values
                label = ' + '.join(str(x) for x in name)
            else:
                label = str(name)
            
            grouped_data.append(group[y_axis].values)
            labels.append(label)
        
        # Create boxplot
        ax.boxplot(grouped_data, labels=labels, patch_artist=True)
        
        # Customize plot
        if show_titles:
            ax.set_title(os.path.splitext(filename)[0])
        ax.set_ylabel(y_axis)
        ax.set_xlabel(' & '.join(group_by_cols))
        
        # Rotate x-axis labels if needed
        if len(labels) > 3 or any(len(str(label)) > 10 for label in labels):
            ax.set_xticklabels(labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Create buffer for this plot
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_buffers.append((buffer, os.path.splitext(filename)[0] + '.png'))
        
        # Individual download button
        st.download_button(
            label=f"download : {os.path.splitext(filename)[0]}",
            data=buffer,
            file_name=f"{os.path.splitext(filename)[0]}_plot.png",
            mime="image/png",
            key=f"download_plot_{i}"
        )
    
    # Add zip download option if multiple plots
    if need_zip:
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