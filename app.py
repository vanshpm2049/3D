import streamlit as st
import pandas as pd
import pydeck as pdk
import re
import io
import base64
from folium.plugins import MarkerCluster, HeatMap
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="3D Waste Analytics Map (Enhanced)", layout="wide")

st.title("♻️ 3D Waste Analytics Map (Enhanced)")
st.markdown("Interactive 3D visualization with popups, heatmaps, and custom time periods")

st.sidebar.header("⚙️ Controls")

# ============ FILE UPLOAD ============
uploaded = st.sidebar.file_uploader("Upload CSV (Bintix format)", type="csv")

if uploaded is None:
    st.warning("📁 Upload the provided DATA.csv format")
    st.stop()

df = pd.read_csv(uploaded)

# ============ DETECT MONTHS ============
month_pattern = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}")
months = sorted({month_pattern.search(c).group(0) for c in df.columns if month_pattern.search(c)})

if not months:
    st.error("❌ No months detected in CSV columns. Ensure format: 'Metric Jan 2024'")
    st.stop()

# ============ TIME PERIOD SELECTION ============
st.sidebar.markdown("### 📅 Time Period")
time_period = st.sidebar.radio(
    "Select Data View:",
    options=["Cumulative (All Months)", "Single Month"],
    index=0,
    help="Cumulative: Sum across all months | Single: Select one month"
)

if time_period == "Cumulative (All Months)":
    selected_months = months
    period_label = "Cumulative"
else:
    selected_month = st.sidebar.selectbox("Select Month", months, index=len(months)-1)
    selected_months = [selected_month]
    period_label = selected_month

# ============ METRIC SELECTION ============
st.sidebar.markdown("### 📊 Display Metrics")

metric_map = {
    "Tonnage": "Tonnage",
    "Trees Saved": "Trees Saved",
    "CO2 Kgs Averted": "CO2 Kgs Averted",
    "Households Participation %": "Households Participation %",
    "Segregation Compliance %": "Segregation Compliance %",
}

metric_name = st.sidebar.selectbox("Extrude Metric (3D Height)", list(metric_map.keys()))

# ============ VISUALIZATION CONTROLS ============
st.sidebar.markdown("### 🎨 Visualization")

bar_scale = st.sidebar.slider(
    "3D Bar Height Scale",
    min_value=200,
    max_value=6000,
    value=2000,
    step=200,
    help="Controls the height multiplier for 3D bars"
)

heat_intensity = st.sidebar.slider(
    "Heatmap Intensity",
    min_value=0.5,
    max_value=6.0,
    value=2.0,
    step=0.1,
    help="Higher = more pronounced heat concentrations"
)

# ============ HEATMAP METRIC SELECTION ============
st.sidebar.markdown("### 🔥 Heatmap Configuration")

heatmap_metric = st.sidebar.selectbox(
    "Heatmap Metric",
    options=list(metric_map.keys()),
    index=0,
    help="Select which metric to visualize in the heatmap layer"
)

heatmap_color_scheme = st.sidebar.radio(
    "Color Scheme:",
    options=["Green (Efficient) → Red (Lagging)", "Blue → Red (Standard)", "Cool → Warm (Rainbow)"],
    index=0,
    help="Green=High Performance | Red=Low Performance"
)

# Define color gradients based on selection
if heatmap_color_scheme == "Green (Efficient) → Red (Lagging)":
    color_gradient = {
        0.0: '#d73027',    # Red - Lagging
        0.2: '#fc8d59',    # Orange
        0.4: '#fee090',    # Yellow
        0.6: '#91bfdb',    # Light Blue
        0.8: '#4575b4',    # Blue
        1.0: '#1a9850'     # Green - Efficient
    }
    color_desc = "🟢 Green = Highly Efficient | 🔴 Red = Lagging"
elif heatmap_color_scheme == "Blue → Red (Standard)":
    color_gradient = {
        0.2: 'blue',
        0.4: 'lime',
        0.6: 'yellow',
        0.8: 'orange',
        1.0: 'red'
    }
    color_desc = "🔵 Blue = Low | 🔴 Red = High"
else:  # Cool to Warm Rainbow
    color_gradient = {
        0.0: '#440154',    # Dark Purple
        0.2: '#31688e',    # Blue
        0.4: '#35b779',    # Green
        0.6: '#fde724',    # Yellow
        0.8: '#ff6e3a',    # Orange
        1.0: '#ff0000'     # Red
    }
    color_desc = "🌈 Cool → Warm intensity gradient"

st.sidebar.info(f"📊 Color scheme: {color_desc}")

show_heatmap = st.sidebar.checkbox("Show Heatmap Layer", value=True)
show_columns = st.sidebar.checkbox("Show 3D Columns", value=True)
show_folium_map = st.sidebar.checkbox("Show 2D Folium Map (Alternative)", value=False)

st.sidebar.info("💡 **Note:** Use Folium map to see custom color schemes. Pydeck heatmap uses default colors.")

# ============ DATA PROCESSING ============

def get_column_name(base_name, month):
    """Extract exact column name from CSV"""
    pattern = re.compile(f"^{re.escape(base_name)}\\s+{re.escape(month)}$")
    for col in df.columns:
        if pattern.match(col):
            return col
    # Fallback: fuzzy match
    for col in df.columns:
        if base_name.lower() in col.lower() and month in col:
            return col
    return None

# Build cumulative data
map_df = df.dropna(subset=["Latitude", "Longitude"]).copy()

if map_df.empty:
    st.error("❌ No location data (Latitude/Longitude) found in CSV")
    st.stop()

# Calculate aggregated values for main metric (3D bars)
cumulative_values = []
for idx, row in map_df.iterrows():
    total = 0
    if time_period == "Cumulative (All Months)":
        for month in selected_months:
            col = get_column_name(metric_map[metric_name], month)
            if col and col in df.columns:
                val = pd.to_numeric(df.loc[idx, col], errors="coerce")
                if pd.notna(val):
                    total += val
    else:
        # Single month
        col = get_column_name(metric_map[metric_name], selected_months[0])
        if col and col in df.columns:
            total = pd.to_numeric(df.loc[idx, col], errors="coerce")
            if pd.isna(total):
                total = 0
    
    cumulative_values.append(total)

map_df["metric_value"] = cumulative_values
map_df["metric_value"] = pd.to_numeric(map_df["metric_value"], errors="coerce").fillna(0)

# Calculate aggregated values for heatmap metric (may be different from display metric)
heatmap_values = []
for idx, row in map_df.iterrows():
    total = 0
    if time_period == "Cumulative (All Months)":
        for month in selected_months:
            col = get_column_name(metric_map[heatmap_metric], month)
            if col and col in df.columns:
                val = pd.to_numeric(df.loc[idx, col], errors="coerce")
                if pd.notna(val):
                    total += val
    else:
        # Single month
        col = get_column_name(metric_map[heatmap_metric], selected_months[0])
        if col and col in df.columns:
            total = pd.to_numeric(df.loc[idx, col], errors="coerce")
            if pd.isna(total):
                total = 0
    
    heatmap_values.append(total)

map_df["heatmap_value"] = heatmap_values
map_df["heatmap_value"] = pd.to_numeric(map_df["heatmap_value"], errors="coerce").fillna(0)

# Calculate heights for 3D bars
max_val = map_df["metric_value"].max()
map_df["height"] = (map_df["metric_value"] / max_val * bar_scale) if max_val > 0 else 10

# ============ BUILD POPUP HTML ============

def build_popup_html(row, metric_name, value, period_label):
    """Create rich popup HTML from row data"""
    community = row.get("Community", "Unknown")
    city = row.get("City", "Unknown")
    pincode = row.get("Pincode", "N/A")
    
    html = f"""
    <div style="font-family: Arial; width: 300px; padding: 14px; background: #f8f9fa; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <h3 style="margin: 0 0 10px 0; color: #36204D; font-size: 16px; font-weight: bold;">📍 {community}</h3>
        
        <div style="background: white; padding: 10px; border-radius: 4px; margin-bottom: 10px; border-left: 4px solid #7c3aed;">
            <p style="margin: 4px 0; color: #666; font-size: 12px;"><strong>🏙️ City:</strong> {city}</p>
            <p style="margin: 4px 0; color: #666; font-size: 12px;"><strong>📮 Pincode:</strong> {pincode}</p>
        </div>
        
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;">
        
        <div style="background: linear-gradient(135deg, #f5f3ff 0%, #faf9ff 100%); padding: 10px; border-radius: 6px; margin: 10px 0; border: 1px solid #e9d5ff;">
            <p style="margin: 0; font-size: 13px; color: #333;">
                <strong>{metric_name}:</strong><br/>
                <span style="color: #7c3aed; font-weight: bold; font-size: 18px;">
                    {value:,.1f}
                </span>
            </p>
        </div>
        
        <div style="background: #f0f4ff; padding: 8px; border-radius: 4px; margin-top: 10px;">
            <p style="margin: 0; font-size: 11px; color: #1e40af;">
                <strong>📊 Period:</strong> {period_label}
            </p>
        </div>
    </div>
    """
    return html

# ============ PYDECK LAYERS ============

if not show_folium_map:
    st.markdown("### 🗺️ 3D Interactive Map (Pydeck)")
    st.info("⚠️ Pydeck heatmap uses default color scheme. Switch to Folium map to see custom colors.")
    
    layers = []

    # Heatmap Layer with selected metric
    if show_heatmap:
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=map_df,
            get_position=["Longitude", "Latitude"],
            get_weight="heatmap_value",
            intensity=heat_intensity,
            threshold=0.03,
            radiusPixels=70,
        )
        layers.append(heatmap_layer)

    # Column Layer (3D bars)
    if show_columns:
        column_layer = pdk.Layer(
            "ColumnLayer",
            data=map_df,
            get_position=["Longitude", "Latitude"],
            get_elevation="height",
            radius=120,
            elevation_scale=1,
            extruded=True,
            pickable=True,
            auto_highlight=True,
            get_fill_color="[150, 100, 255, 200]",
            get_line_color="[80, 50, 200, 255]",
        )
        layers.append(column_layer)

    # ============ VIEW STATE & DECK ============

    view_state = pdk.ViewState(
        latitude=map_df["Latitude"].mean(),
        longitude=map_df["Longitude"].mean(),
        zoom=11,
        pitch=50,
        bearing=25,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="light",
        tooltip={
            "html": """
            <b>{Community}</b><br/>
            City: {City}<br/>
            """ + metric_name + f""": {{metric_value:.1f}}<br/>
            """ + heatmap_metric + f""": {{heatmap_value:.1f}}<br/>
            <small>Period: {period_label}</small>
            """,
            "style": {"backgroundColor": "white", "color": "black", "fontSize": "12px"},
        },
    )

    st.pydeck_chart(deck, use_container_width=True)

else:
    # ============ FOLIUM MAP WITH POPUPS ============
    st.markdown("### 🗺️ 2D Interactive Map (Folium with Rich Popups & Custom Colors)")
    st.success("✅ This map shows custom color schemes based on your selection.")
    
    center_lat = map_df["Latitude"].mean()
    center_lon = map_df["Longitude"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True
    )
    
    # Add heatmap layer with selected metric and color scheme
    if show_heatmap:
        heat_data = []
        for idx, row in map_df.iterrows():
            if pd.notna(row["heatmap_value"]) and row["heatmap_value"] > 0:
                weight = row["heatmap_value"] / map_df["heatmap_value"].max()
                heat_data.append([row["Latitude"], row["Longitude"], weight])
        
        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.2,
                max_opacity=0.8,
                radius=15,
                blur=20,
                gradient=color_gradient
            ).add_to(m)
    
    # Add markers with rich popups
    marker_cluster = MarkerCluster().add_to(m)
    
    for idx, row in map_df.iterrows():
        popup_html = build_popup_html(
            row=row,
            metric_name=metric_name,
            value=row["metric_value"],
            period_label=period_label
        )
        
        popup = folium.Popup(
            folium.Html(popup_html, script=True),
            max_width=350,
            min_width=300
        )
        
        # Color code by heatmap metric value
        heatmap_normalized = row["heatmap_value"] / map_df["heatmap_value"].max() if map_df["heatmap_value"].max() > 0 else 0
        
        if heatmap_color_scheme == "Green (Efficient) → Red (Lagging)":
            # Green for high, red for low
            if heatmap_normalized > 0.66:
                color = "green"
            elif heatmap_normalized > 0.33:
                color = "orange"
            else:
                color = "red"
        else:
            # Red for high, blue for low (standard)
            if heatmap_normalized > 0.66:
                color = "red"
            elif heatmap_normalized > 0.33:
                color = "orange"
            else:
                color = "blue"
        
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=8,
            popup=popup,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
            opacity=0.9
        ).add_to(marker_cluster)
    
    st_folium(m, width=1400, height=700)

# ============ STATISTICS DISPLAY ============

st.markdown("---")
st.markdown("### 📈 Cumulative Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total = map_df["metric_value"].sum()
    st.metric(
        f"Total {metric_name}",
        f"{total:,.0f}",
        help=f"Sum across all communities for {period_label}"
    )

with col2:
    avg = map_df["metric_value"].mean()
    st.metric(
        f"Average {metric_name}",
        f"{avg:,.1f}",
        help="Mean value per community"
    )

with col3:
    max_val_stat = map_df["metric_value"].max()
    max_community = map_df.loc[map_df["metric_value"].idxmax(), "Community"]
    st.metric(
        f"Peak Community",
        max_community,
        f"{max_val_stat:,.0f}",
        help="Highest performing community"
    )

with col4:
    coverage = len(map_df[map_df["metric_value"] > 0]) / len(map_df) * 100
    st.metric(
        f"Coverage",
        f"{coverage:.1f}%",
        help="% of communities with data"
    )

# ============ HEATMAP STATISTICS ============

st.markdown("---")
st.markdown(f"### 🔥 {heatmap_metric} Heatmap Statistics")

hcol1, hcol2, hcol3, hcol4 = st.columns(4)

with hcol1:
    heatmap_total = map_df["heatmap_value"].sum()
    st.metric(
        f"Total {heatmap_metric}",
        f"{heatmap_total:,.0f}",
        help=f"Sum of {heatmap_metric} across all communities"
    )

with hcol2:
    heatmap_avg = map_df["heatmap_value"].mean()
    st.metric(
        f"Average {heatmap_metric}",
        f"{heatmap_avg:,.1f}",
        help="Mean value per community"
    )

with hcol3:
    heatmap_max = map_df["heatmap_value"].max()
    heatmap_leader = map_df.loc[map_df["heatmap_value"].idxmax(), "Community"]
    st.metric(
        f"Top Community",
        heatmap_leader,
        f"{heatmap_max:,.0f}",
        help="Highest in heatmap metric"
    )

with hcol4:
    heatmap_min = map_df["heatmap_value"].min()
    heatmap_laggard = map_df.loc[map_df["heatmap_value"].idxmin(), "Community"]
    st.metric(
        f"Needs Improvement",
        heatmap_laggard,
        f"{heatmap_min:,.0f}",
        help="Lowest in heatmap metric"
    )

# ============ DETAILED TABLE VIEW ============

st.markdown("---")
st.markdown("### 📊 Community-Level Data")

if st.checkbox("View Detailed Community Data", value=True):
    # Prepare display dataframe with unique column names
    display_df = map_df[[
        "Community", "City", "Pincode", "metric_value", "heatmap_value", "height"
    ]].copy()
    display_df = display_df.sort_values("metric_value", ascending=False)
    display_df = display_df.reset_index(drop=True)
    
    # Create column names with differentiators if metrics are the same
    if metric_name == heatmap_metric:
        display_df.columns = ["Community", "City", "Pincode", f"{metric_name} (3D Bar)", f"{heatmap_metric} (Heatmap)", "3D Height"]
    else:
        display_df.columns = ["Community", "City", "Pincode", metric_name, heatmap_metric, "3D Height"]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        column_config={
            f"{metric_name} (3D Bar)" if metric_name == heatmap_metric else metric_name: st.column_config.NumberColumn(format="%.2f"),
            f"{heatmap_metric} (Heatmap)" if metric_name == heatmap_metric else heatmap_metric: st.column_config.NumberColumn(format="%.2f"),
            "3D Height": st.column_config.NumberColumn(format="%.1f")
        }
    )
    
    # Download option
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"waste_analytics_{period_label}_{heatmap_metric.replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ============ LEGEND & INFO ============

st.markdown("---")
st.markdown("### 📖 Legend & Information")

col_legend1, col_legend2 = st.columns(2)

with col_legend1:
    st.subheader("3D Bar Chart (Pydeck)")
    st.info(f"""
    - **Metric**: {metric_name}
    - **Height**: Proportional to {metric_name} value
    - **Color**: Purple gradient (fixed)
    - **Interaction**: Hover for tooltip
    """)

with col_legend2:
    st.subheader("Heatmap Layer (Folium)")
    st.info(f"""
    - **Metric**: {heatmap_metric}
    - **Intensity**: Controlled by slider
    - **Color Scheme**: {heatmap_color_scheme}
    - **Switch to Folium Map**: To see custom colors
    - **Interaction**: Hover over map for values
    """)

# ============ FOOTER ============
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #999; font-size: 12px; padding: 20px;">
        <p>✔️ Enhanced 3D Waste Analytics Map with Configurable Heatmaps & Custom Color Schemes</p>
        <p>💡 Tip: Use Folium map to see custom heatmap colors!</p>
        <p>Built with ❤️ using Streamlit, Pydeck & Folium</p>
    </div>
    """,
    unsafe_allow_html=True
)
