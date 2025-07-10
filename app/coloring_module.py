import streamlit as st
import matplotlib.pyplot as plt

def render_color_picker_component():
    """
    Renders a component for selecting and customizing color palettes for plots.
    The chosen palette is stored in st.session_state.active_color_palette.
    """
    
    # List of official qualitative colormaps
    QUALITATIVE_CMAPS = [
        'Set1', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
        'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
    ]
    
    # --- Initialize Session State ---
    if 'color_palette_selection' not in st.session_state:
        st.session_state.color_palette_selection = "Set1"
    
    # Default custom colors (15 distinct colors)
    if 'custom_color_palette' not in st.session_state:
        st.session_state.custom_color_palette = [
            "#FF6347", "#4682B4", "#32CD32", "#FFD700", "#6A5ACD",
            "#40E0D0", "#FF69B4", "#8A2BE2", "#00BFFF", "#ADFF2F",
            "#DA70D6", "#FF4500", "#2E8B57", "#1E90FF", "#D2691E"
        ]

    def update_active_palette():
        """Updates the active color palette based on user selection."""
        selection = st.session_state.color_palette_selection
        if selection == "Custom":
            st.session_state.active_color_palette = st.session_state.custom_color_palette
        else:
            # Get colors from the selected matplotlib colormap
            cmap = plt.get_cmap(selection)
            st.session_state.active_color_palette = [plt.matplotlib.colors.to_hex(c) for c in cmap.colors]

    # Initialize the active palette on first run
    if 'active_color_palette' not in st.session_state:
        update_active_palette()

    # --- UI Component ---
    with st.expander("Color Palette Options", expanded=False):
        st.image(
            "./quantitative_color_map.png",
            caption="Matplotlib Qualitative Colormaps",
            use_container_width=True
        )

        st.selectbox(
            "Select a color palette",
            options=QUALITATIVE_CMAPS + ["Custom"],
            key='color_palette_selection',
            on_change=update_active_palette,
            help="Choose a predefined color palette or create a custom one below."
        )

        st.markdown("Custom Palette")

        # Display 15 color pickers in a row
        cols = st.columns(15)
        for i in range(15):
            with cols[i]:
                # The key is essential for Streamlit to track each widget individually
                new_color = st.color_picker(
                    f"Color {i+1}",
                    value=st.session_state.custom_color_palette[i],
                    key=f"custom_color_{i}",
                    label_visibility="hidden"
                )
                # If a color changes, update the list and the active palette if "Custom" is selected
                if new_color != st.session_state.custom_color_palette[i]:
                    st.session_state.custom_color_palette[i] = new_color
                    if st.session_state.color_palette_selection == "Custom":
                        update_active_palette()
                        # st.rerun()
