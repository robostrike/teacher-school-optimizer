import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from pathlib import Path
from folium.plugins import MarkerCluster

# Set page config
st.set_page_config(page_title="Teacher-School Manager", layout="wide")

# File paths
DATA_DIR = Path(__file__).parent / "data"
TEACHERS_FILE = DATA_DIR / "teachers_copy.csv"
SCHOOLS_FILE = DATA_DIR / "kidsduo_schools.csv"

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
def create_map(teachers_df, schools_df):
    # Create a base map centered on Tokyo
    m = folium.Map(location=[35.6895, 139.6917], zoom_start=11)
    
    # Add marker cluster for teachers
    teacher_cluster = MarkerCluster(name="Teachers").add_to(m)
    
    # Add markers for each teacher
    for _, teacher in teachers_df.iterrows():
        # Skip teachers without location data
        if pd.isna(teacher.get('station_lat')) or pd.isna(teacher.get('station_lon')):
            continue
            
        # Determine icon color based on status
        if pd.isna(teacher.get('school_id')) or teacher.get('school_id') == '':
            icon_color = 'blue'  # Available for assignment
        elif teacher.get('move', False):
            icon_color = 'orange'  # Willing to move
        else:
            icon_color = 'green'  # Stable assignment
            
        folium.Marker(
            location=[teacher['station_lat'], teacher['station_lon']],
            popup=f"{teacher['name']} ({teacher['type']})\nStation: {teacher['station']}",
            icon=folium.Icon(color=icon_color, icon='user', prefix='fa'),
            tooltip=f"Teacher: {teacher['name']}"
        ).add_to(teacher_cluster)
    
    # Add markers for schools
    school_cluster = MarkerCluster(name="Schools").add_to(m)
    for _, school in schools_df.iterrows():
        if pd.notna(school.get('latitude')) and pd.notna(school.get('longitude')):
            folium.Marker(
                location=[school['latitude'], school['longitude']],
                popup=f"{school['name']}\n{school['station']}",
                icon=folium.Icon(color='red', icon='school', prefix='fa'),
                tooltip=f"School: {school['name']}"
            ).add_to(school_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Load school data with locations
@st.cache_data
def load_schools_with_locations():
    df = pd.read_csv(SCHOOLS_FILE)
    return df

# Add station coordinates for teachers (in a real app, you'd want to add these to your data)
@st.cache_data
def add_teacher_locations(teachers_df, schools_df):
    # Create a mapping of station names to coordinates from schools data
    station_coords = {}
    for _, school in schools_df.iterrows():
        if pd.notna(school['station']) and pd.notna(school['latitude']) and pd.notna(school['longitude']):
            station_coords[school['station']] = (school['latitude'], school['longitude'])
    
    # Add coordinates to teachers
    teachers_df['station_lat'] = teachers_df['station'].map(lambda x: station_coords.get(x, {}).get(0, None))
    teachers_df['station_lon'] = teachers_df['station'].map(lambda x: station_coords.get(x, {}).get(1, None))
    
    return teachers_df

# Load data
schools_df = load_schools_with_locations()
teachers_df = load_teachers()
teachers_df = add_teacher_locations(teachers_df, schools_df)

# Display the map
st.subheader("Teacher and School Locations")
map_fig = create_map(teachers_df, schools_df)
folium_static(map_fig, width=1200, height=500)

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