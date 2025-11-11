import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from pathlib import Path
from folium.plugins import MarkerCluster
from folium import Icon
from typing import Optional, Tuple

# Set page config
st.set_page_config(page_title="Teacher-School Manager", layout="wide")

# File paths
DATA_DIR = Path(__file__).parent / "data"
TEACHERS_FILE = DATA_DIR / "teachers_copy.csv"
SCHOOLS_FILE = DATA_DIR / "kidsduo_schools.csv"
TRAVEL_TIMES_FILE = DATA_DIR / "station_travel_times.csv"

@st.cache_data
def load_teachers():
    """Load teacher data from CSV"""
    df = pd.read_csv(TEACHERS_FILE)
    # Handle 'move' column with proper type conversion
    if 'move' in df.columns:
        # Convert to string first, then handle true/false values
        df['move'] = df['move'].astype(str).str.lower().replace({
            'true': True,
            'false': False,
            'nan': False,
            '': False
        }).astype(bool)
    else:
        # If 'move' column doesn't exist, create it with False as default
        df['move'] = False
    return df

@st.cache_data
def load_schools():
    """Load school data and return as a dictionary of id: name"""
    df = pd.read_csv(SCHOOLS_FILE)
    return df.set_index('id')['name'].to_dict()

def save_teachers(teachers_df):
    """Save teacher data back to CSV"""
    teachers_df.to_csv(TEACHERS_FILE, index=False)

# Load data
teachers_df = load_teachers()
schools = load_schools()

# Add a 'No School' option
school_options = [""] + list(schools.keys())
school_display = {"": "No School"}
school_display.update(schools)

# App title
st.title("Teacher-School Manager")

# Create a map to display teachers and schools
@st.cache_data
def load_travel_times():
    """Load and process travel times data"""
    try:
        return pd.read_csv(TRAVEL_TIMES_FILE)
    except Exception as e:
        st.error(f"Error loading travel times: {e}")
        return pd.DataFrame()

def get_travel_time(origin_id, destination_id, travel_times_df):
    """Get travel time between two stations in minutes."""
    if pd.isna(origin_id) or pd.isna(destination_id) or origin_id == '' or destination_id == '':
        return float('inf')
        
    # Check direct route
    direct = travel_times_df[
        (travel_times_df['origin_uuid'] == origin_id) & 
        (travel_times_df['destination_uuid'] == destination_id)
    ]
    
    if not direct.empty:
        return direct['travel_min'].iloc[0]
    
    # Check reverse route
    reverse = travel_times_df[
        (travel_times_df['origin_uuid'] == destination_id) & 
        (travel_times_df['destination_uuid'] == origin_id)
    ]
    
    if not reverse.empty:
        return reverse['travel_min'].iloc[0]
    
    return float('inf')  # No route found

def create_map(teachers_df, schools_df, selected_school_id=None, travel_times_df=None):
    # Create a base map centered on Tokyo with settings to prevent clustering
    m = folium.Map(
        location=[35.6895, 139.6917],
        zoom_start=11,
        min_zoom=10,
        max_zoom=18,
        min_lat=35.3,
        max_lat=35.9,
        min_lon=139.4,
        max_lon=140.0,
    )
    
    # Get selected school data if any
    selected_school = schools_df[schools_df['id'] == selected_school_id].iloc[0] if selected_school_id else None
    
    # Add markers for each teacher
    for _, teacher in teachers_df.iterrows():
        if pd.isna(teacher.get('station_lat')) or pd.isna(teacher.get('station_lon')):
            continue
            
        # Determine if teacher is within 60 minutes of selected school
        is_within_range = False
        if selected_school is not None and 'station_id' in teacher and 'station_id' in selected_school:
            travel_time = get_travel_time(
                teacher['station_id'], 
                selected_school['station_id'],
                travel_times_df
            )
            is_within_range = travel_time <= 60
        
        # Determine icon color based on status and range
        if is_within_range:
            color = 'purple'  # Within 60 minutes of selected school
        elif pd.isna(teacher.get('school_id')) or teacher.get('school_id') == '':
            color = 'blue'  # Available for assignment
        elif teacher.get('move', False):
            color = 'orange'  # Willing to move
        else:
            color = 'green'  # Stable assignment
        
        # Add popup with travel time if applicable
        popup_text = f"{teacher['name']} ({teacher['type']})\nStation: {teacher['station']}"
        if is_within_range and 'station' in selected_school:
            popup_text += f"\n{travel_time:.0f} min from {selected_school['name']}"
        
        # Create a circle marker
        folium.CircleMarker(
            location=[teacher['station_lat'], teacher['station_lon']],
            radius=7 if is_within_range else 5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2 if is_within_range else 1,
            popup=popup_text,
            tooltip=f"Teacher: {teacher['name']}"
        ).add_to(m)
    
    # Add markers for schools
    for _, school in schools_df.iterrows():
        if pd.notna(school.get('latitude')) and pd.notna(school.get('longitude')):
            is_selected = school['id'] == selected_school_id
            # Create a pin marker for the school
            folium.Marker(
                location=[school['latitude'], school['longitude']],
                popup=f"School: {school['name']}\n{school['station']}",
                tooltip=f"School: {school['name']}",
                icon=Icon(
                    color='red',
                    icon='info-sign',  # Using info-sign as it's a clear pin icon
                    prefix='glyphicon',  # Using glyphicon for the pin icon
                    icon_size=(20, 30) if is_selected else (16, 25),  # Larger if selected
                    icon_anchor=(10, 30) if is_selected else (8, 25)  # Adjust anchor for the pointy end
                )
            ).add_to(m)
    
    return m


# Load school data with locations
@st.cache_data
def load_schools_with_locations():
    df = pd.read_csv(SCHOOLS_FILE)
    return df

# Add station coordinates for teachers using tokyo_stations_gps.csv
@st.cache_data
def add_teacher_locations(teachers_df, _):
    # Load station coordinates from tokyo_stations_gps.csv
    stations_gps = pd.read_csv(DATA_DIR / 'tokyo_stations_gps.csv')
    
    # Create a mapping of station_id to coordinates
    station_coords = {}
    for _, row in stations_gps.iterrows():
        station_coords[row['id']] = (row['latitude'], row['longitude'])
    
    # Add coordinates to teachers using station_id
    teachers_df['station_lat'] = teachers_df['station_id'].map(lambda x: station_coords.get(x, (None, None))[0])
    teachers_df['station_lon'] = teachers_df['station_id'].map(lambda x: station_coords.get(x, (None, None))[1])
    
    return teachers_df

# Load data
schools_df = load_schools_with_locations()
teachers_df = load_teachers()
teachers_df = add_teacher_locations(teachers_df, schools_df)
travel_times_df = load_travel_times()

# School selection for filtering
st.sidebar.subheader("Filter by School")
selected_school_id = st.sidebar.selectbox(
    "Select a school to see teachers within 60 minutes",
    [""] + sorted(schools_df['id'].tolist()),
    format_func=lambda x: schools_df[schools_df['id'] == x]['name'].iloc[0] if x else "All Schools"
)

# Display the map
st.subheader("Teacher and School Locations")
map_fig = create_map(teachers_df, schools_df, selected_school_id if selected_school_id else None, travel_times_df)
folium_static(map_fig, width=1200, height=500)

# Show teachers within 60 minutes of selected school
if selected_school_id:
    selected_school = schools_df[schools_df['id'] == selected_school_id].iloc[0]
    st.subheader(f"Teachers within 60 minutes of {selected_school['name']}")
    
    # Filter teachers within 60 minutes
    teachers_within_range = []
    for _, teacher in teachers_df.iterrows():
        if 'station_id' not in teacher or pd.isna(teacher['station_id']) or teacher['station_id'] == '':
            continue
            
        travel_time = get_travel_time(
            teacher['station_id'],
            selected_school['station_uuid'],
            travel_times_df
        )
        
        if travel_time <= 60:  # 60 minutes threshold
            teacher_copy = teacher.copy()
            teacher_copy['travel_time'] = f"{int(travel_time)} min"
            teachers_within_range.append(teacher_copy)
    
    if teachers_within_range:
        # Display as a nice table
        result_df = pd.DataFrame(teachers_within_range)
        result_df = result_df[['name', 'type', 'station', 'travel_time']]
        result_df.columns = ['Name', 'Type', 'Station', 'Travel Time']
        st.dataframe(
            result_df,
            column_config={
                "Name": "Teacher Name",
                "Type": "Teacher Type",
                "Station": "Nearest Station",
                "Travel Time": "Travel Time"
            },
            use_container_width=True
        )
    else:
        st.info("No teachers found within 60 minutes of the selected school.")

st.write("Manage teacher assignments and move preferences")

# Create form for editing
temp_teachers = teachers_df.copy()

# Display teachers in a responsive grid of cards
st.subheader("Teacher Directory")

# Create a grid of cards (3 cards per row on large screens)
cols_per_row = 3
total_teachers = len(teachers_df)

for i in range(0, total_teachers, cols_per_row):
    # Create a row of cards
    cols = st.columns(cols_per_row)
    
    # Fill the current row with cards
    for col_idx in range(cols_per_row):
        teacher_idx = i + col_idx
        if teacher_idx >= total_teachers:
            break
            
        teacher = teachers_df.iloc[teacher_idx]
        current_school = teacher.get('school_id', '')
        can_move = pd.isna(teacher.get('school_id')) or teacher.get('school_id') == ''
        
        with cols[col_idx]:
            with st.container(border=True):
                # Header with name and type
                st.markdown(f"### {teacher['name']}")
                st.caption(f"{teacher['type']} | {teacher['station']}")
                
                # School selection
                new_school = st.selectbox(
                    "Assigned School",
                    options=school_options,
                    index=school_options.index(current_school) if pd.notna(current_school) and current_school in school_options else 0,
                    key=f"school_{teacher_idx}",
                    format_func=lambda x: school_display.get(x, "No School")
                )
                temp_teachers.at[teacher_idx, 'school_id'] = new_school if new_school else None
                
                # Move preference
                move = st.checkbox(
                    "Willing to Move",
                    value=can_move or teacher.get('move', False),
                    disabled=can_move,
                    key=f"move_{teacher_idx}",
                    help="Check if teacher is willing to move to a different school"
                )
                temp_teachers.at[teacher_idx, 'move'] = move
                
                # Status indicator
                if can_move:
                    st.info("Reassignable", icon="ℹ️")
                elif move:
                    st.warning("Open to relocation", icon="🔄")
                else:
                    st.success("Stable assignment", icon="✅")

# Save button
if st.button("Save Changes"):
    save_teachers(temp_teachers)
    st.success("Changes saved successfully!")
    st.rerun()

# Display current data
expander = st.expander("View Raw Data")
with expander:
    st.dataframe(teachers_df)