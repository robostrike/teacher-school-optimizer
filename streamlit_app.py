import streamlit as st
import pandas as pd
import folium
import traceback
from streamlit_folium import folium_static
from pathlib import Path
from folium.plugins import MarkerCluster
from folium import Icon
from typing import Optional, Tuple

# Set page config
st.set_page_config(page_title="Teacher Optimizer", layout="wide")

# File paths
import os
DATA_DIR = Path(__file__).parent / "data"
TEACHERS_FILE = DATA_DIR / "teachers.csv"
SCHOOLS_FILE = DATA_DIR / "kidsduo_schools.csv"
TRAVEL_TIMES_FILE = DATA_DIR / "station_travel_times.csv"
ASSIGNMENTS_FILE = DATA_DIR / "teacher_school_assignments.csv"

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
def load_assignments():
    """Load teacher-school assignments from CSV"""
    if not os.path.exists(ASSIGNMENTS_FILE):
        return pd.DataFrame(columns=['teacher_id', 'school_id', 'assigned_date', 'is_current'])
    return pd.read_csv(ASSIGNMENTS_FILE)

def save_assignments(assignments_df):
    """Save teacher-school assignments to CSV"""
    assignments_df.to_csv(ASSIGNMENTS_FILE, index=False)

def get_current_assignments(assignments_df):
    """Get current assignments (where is_current is True)"""
    return assignments_df[assignments_df['is_current'] == True]

def assign_teacher_to_school(teacher_id, school_id, assignments_df):
    """Assign a teacher to a school, updating previous assignments if needed"""
    from datetime import datetime
    
    # Mark any existing current assignments as not current
    mask = (assignments_df['teacher_id'] == teacher_id) & (assignments_df['is_current'] == True)
    assignments_df.loc[mask, 'is_current'] = False
    
    # Add new assignment if school_id is not empty
    if pd.notna(school_id) and school_id != '':
        new_assignment = pd.DataFrame([{
            'teacher_id': teacher_id,
            'school_id': school_id,
            'assigned_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_current': True
        }])
        assignments_df = pd.concat([assignments_df, new_assignment], ignore_index=True)
    
    return assignments_df

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
assignments_df = load_assignments()
current_assignments = get_current_assignments(assignments_df)
schools = load_schools()

# Add a 'No School' option
school_options = [""] + list(schools.keys())
school_display = {"": "No School"}
school_display.update(schools)

# Add current school to teachers for backward compatibility
if not teachers_df.empty and not current_assignments.empty:
    teachers_df = teachers_df.merge(
        current_assignments[['teacher_id', 'school_id']],
        left_on='id',
        right_on='teacher_id',
        how='left'
    ).drop(columns=['teacher_id'])
else:
    teachers_df['school_id'] = None

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

def get_travel_time(origin_id, dest_id, travel_times_df):
    """Get travel time between two stations using their UUIDs"""
    debug_output = []
    
    def add_debug(msg):
        debug_output.append(msg)
        print(msg)
    
    try:
        # Debug: Log input parameters
        add_debug(f"\n=== get_travel_time called ===")
        add_debug(f"Origin ID: {origin_id} (type: {type(origin_id)})")
        add_debug(f"Dest ID:   {dest_id} (type: {type(dest_id)})")
        
        if travel_times_df is None or travel_times_df.empty:
            msg = "Error: travel_times_df is None or empty"
            add_debug(msg)
            return {'travel_time': float('inf'), 'debug': debug_output}
        
        if pd.isna(origin_id) or pd.isna(dest_id):
            msg = f"Error: Missing station IDs - origin: {origin_id}, dest: {dest_id}"
            add_debug(msg)
            return {'travel_time': float('inf'), 'debug': debug_output}
            
        # Check if columns exist in the DataFrame
        required_columns = ['origin_uuid', 'destination_uuid', 'travel_min']
        if not all(col in travel_times_df.columns for col in required_columns):
            msg = f"Error: Missing required columns in travel_times_df. Available columns: {travel_times_df.columns.tolist()}"
            add_debug(msg)
            return {'travel_time': float('inf'), 'debug': debug_output}
        
        # Check direct direction (origin -> destination)
        direct = travel_times_df[
            (travel_times_df['origin_uuid'] == origin_id) & 
            (travel_times_df['destination_uuid'] == dest_id)
        ]
        
        if not direct.empty:
            travel_time = direct['travel_min'].iloc[0]
            add_debug(f"Found direct route: {len(direct)} matches")
            add_debug(f"Travel time: {travel_time} min")
            return {'travel_time': travel_time, 'debug': debug_output}
        
        print("No direct route found, checking reverse direction...")
        
        # Check reverse direction (destination -> origin)
        reverse = travel_times_df[
            (travel_times_df['origin_uuid'] == dest_id) & 
            (travel_times_df['destination_uuid'] == origin_id)
        ]
        
        if not reverse.empty:
            travel_time = reverse['travel_min'].iloc[0]
            add_debug(f"Found reverse route: {len(reverse)} matches")
            add_debug(f"Travel time: {travel_time} min")
            return {'travel_time': travel_time, 'debug': debug_output}
            
        add_debug("No route found in either direction")
        
        # Debug: Check if either station exists in the travel times data
        origin_exists = (travel_times_df['origin_uuid'] == origin_id).any() or \
                       (travel_times_df['destination_uuid'] == origin_id).any()
        dest_exists = (travel_times_df['origin_uuid'] == dest_id).any() or \
                     (travel_times_df['destination_uuid'] == dest_id).any()
        
        add_debug(f"Origin station in data: {'Yes' if origin_exists else 'No'}")
        add_debug(f"Destination station in data: {'Yes' if dest_exists else 'No'}")
        
        return {'travel_time': float('inf'), 'debug': debug_output}  # No route found
        
    except Exception as e:
        error_msg = f"Error in get_travel_time: {str(e)}"
        add_debug(error_msg)
        add_debug(f"Origin ID: {origin_id}, Dest ID: {dest_id}")
        if 'travel_times_df' in locals():
            add_debug(f"DataFrame columns: {travel_times_df.columns.tolist()}")
            add_debug(f"DataFrame sample: {travel_times_df.head(1).to_dict()}")
        return {'travel_time': float('inf'), 'debug': debug_output, 'error': str(e)}

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
    
    # Filter schools if a specific school is selected
    if selected_school_id:
        schools_to_show = schools_df[schools_df['id'] == selected_school_id]
        selected_school = schools_to_show.iloc[0] if not schools_to_show.empty else None
    else:
        schools_to_show = schools_df
        selected_school = None
    
    # Add markers for each teacher
    for _, teacher in teachers_df.iterrows():
        if pd.isna(teacher.get('station_lat')) or pd.isna(teacher.get('station_lon')):
            continue
            
        # Determine if teacher is within 60 minutes of selected school
        is_within_range = False
        travel_time = float('inf')
        if selected_school is not None and 'station_uuid' in teacher and 'station_uuid' in selected_school:
            teacher_station = teacher['station_uuid']
            school_station = selected_school['station_uuid']
            
            # Debug: Print station information
            debug_info = f"""
            === DEBUG INFO ===
            Teacher: {teacher['name']}
            Teacher Station UUID: {teacher_station}
            School: {selected_school['name']}
            School Station UUID: {school_station}
            """
            print(debug_info)
            
            try:
                # Get travel time with debug info
                result = get_travel_time(
                    teacher_station,
                    school_station,
                    travel_times_df
                )
                
                # Extract travel time and debug info
                travel_time = result['travel_time']
                debug_info = result.get('debug', [])
                
                # Display debug info in an expander
                with st.expander(f"Debug: {teacher['name']} → {selected_school['name']}"):
                    st.code("\n".join(debug_info))
                
                st.info(f"Travel time: {travel_time} minutes")
                is_within_range = travel_time <= 60
                
                if is_within_range:
                    st.success(f"✅ Within 60 minutes ({travel_time:.0f} min)")
                else:
                    st.warning(f"❌ Not within 60 minutes ({travel_time:.0f} min)")
                    
            except Exception as e:
                error_msg = f"Error calculating travel time: {e}"
                st.error(error_msg)
                with st.expander("Debug Details"):
                    st.error(traceback.format_exc())
                is_within_range = False
        
        # Determine icon color based on status and range
        if not teacher.get('move', False) and pd.notna(teacher.get('school_id')) and teacher.get('school_id') != '':
            color = 'gray'  # Cannot move (has school and not willing to move)
        elif is_within_range:
            color = 'green'  # Within 60 minutes of selected school
        else:
            color = 'blue'  # Default color (available for assignment or willing to move)
        
        # Add popup with teacher info and travel time if applicable
        popup_text = f"{teacher['name']} ({teacher['type']})\nStation: {teacher['station']}"
        if selected_school is not None and 'station' in selected_school:
            if is_within_range:
                popup_text += f"\n✅ {travel_time:.0f} min from {selected_school['name']}"
            else:
                popup_text += f"\n❌ {travel_time:.0f} min from {selected_school['name']} (too far)"
        
        # Add school assignment info if any
        if pd.notna(teacher.get('school_id')) and teacher.get('school_id') != '':
            school_name = schools_df[schools_df['id'] == teacher['school_id']]['name'].iloc[0] \
                if not schools_df[schools_df['id'] == teacher['school_id']].empty else 'Unknown School'
            popup_text += f"\n🏫 Assigned to: {school_name}"
            if not teacher.get('move', False):
                popup_text += " (Not willing to move)"
        
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
    for _, school in schools_to_show.iterrows():
        if pd.notna(school.get('latitude')) and pd.notna(school.get('longitude')):
            is_selected = school['id'] == selected_school_id if selected_school_id else False
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

# Get current assignments
current_assignments = get_current_assignments(load_assignments())

# Count teachers per school
school_teacher_counts = current_assignments['school_id'].value_counts().to_dict()

# Format school names with teacher counts
def format_school_option(school_id):
    if not school_id:
        return "All Schools"
    school_name = schools_df[schools_df['id'] == school_id]['name'].iloc[0]
    teacher_count = school_teacher_counts.get(school_id, 0)
    return f"{school_name} ({teacher_count} teachers)"

# Create school options with teacher counts
school_options = [""] + sorted(schools_df['id'].tolist())
selected_school_id = st.sidebar.selectbox(
    "Select a school to see teachers within 60 minutes",
    school_options,
    format_func=format_school_option
)

# Show assigned teachers for the selected school
if selected_school_id:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Assigned Teachers")
    assigned_teachers = current_assignments[current_assignments['school_id'] == selected_school_id]
    
    if not assigned_teachers.empty:
        # Get teacher details for the assigned teachers
        school_teachers = teachers_df[teachers_df['id'].isin(assigned_teachers['teacher_id'])]
        
        # Display each teacher with their type and station
        for _, teacher in school_teachers.iterrows():
            gender_icon = '♂️' if teacher.get('gender') == 'Male' else '♀️'
            teacher_type = teacher.get('type', 'Unknown')
            station = teacher.get('station', 'Unknown')
            
            st.sidebar.markdown(
                f"- {gender_icon} **{teacher['name']}**  \n"
                f"  *{teacher_type}* at {station}"
            )
    else:
        st.sidebar.info("No teachers currently assigned to this school")

# Display the map
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
        # Create a DataFrame with all teachers and their travel times
        result_df = pd.DataFrame(teachers_within_range)
        
        # Convert travel_time to numeric for sorting
        result_df['travel_minutes'] = result_df['travel_time'].str.extract('(\d+)').astype(float)
        
        # Sort by travel time
        result_df = result_df.sort_values('travel_minutes')
        
        # Add gender icons
        result_df['Gender'] = result_df['gender'].apply(lambda x: '♂️' if x == 'Male' else '♀️')
        
        # Add a checkbox column for selection
        result_df['Select'] = [st.checkbox('', key=f'select_{i}') for i in result_df.index]
        
        # Prepare display columns
        display_cols = ['Select', 'name', 'Gender', 'type', 'station', 'travel_time']
        display_df = result_df[display_cols]
        display_df.columns = ['Select', 'Name', ' ', 'Type', 'Station', 'Travel Time']
        
        # Display styled DataFrame
        def color_row(row):
            color = '#e6f3ff' if row['Type'] == 'Native' else '#f3e6ff'  # Light blue for Native, light purple for Bilingual
            return ['background-color: ' + color] * len(row)
        
        # Display the table without the index
        st.dataframe(
            display_df.style.apply(color_row, axis=1),
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", width=10),
                "Name": "Teacher Name",
                "Type": "Teacher Type",
                "Station": "Nearest Station",
                "Travel Time": "Travel Time"
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Add assign button if any teachers are selected
        if st.button('Assign Selected Teachers to This School', type='primary'):
            selected_teachers = result_df[result_df['Select']]['id'].tolist()
            if selected_teachers:
                assignments_updated = False
                for teacher_id in selected_teachers:
                    # Only update if not already assigned to this school
                    current_assignment = current_assignments[
                        (current_assignments['teacher_id'] == teacher_id) & 
                        (current_assignments['is_current'] == True)
                    ]
                    if current_assignment.empty or current_assignment['school_id'].iloc[0] != selected_school_id:
                        assignments_df = assign_teacher_to_school(
                            teacher_id, 
                            selected_school_id,
                            assignments_df
                        )
                        assignments_updated = True
                
                if assignments_updated:
                    save_assignments(assignments_df)
                    st.success(f"Successfully assigned {len(selected_teachers)} teachers to {selected_school['name']}!")
                    st.rerun()
                else:
                    st.warning("Selected teachers are already assigned to this school.")
            else:
                st.warning("Please select at least one teacher to assign.")
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

# Create a copy of teachers for editing
temp_teachers = teachers_df.copy()

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
        can_move = pd.isna(current_school) or current_school == ''
        
        with cols[col_idx]:
            with st.container(border=True):
                # Header with name, gender, and type
                gender_icon = '♂️' if teacher.get('gender') == 'Male' else '♀️'
                st.markdown(f"### {teacher['name']} {gender_icon}")
                st.caption(f"{teacher['type']} | {teacher['station']}")
                
                # School selection
                current_school_idx = school_options.index(current_school) if pd.notna(current_school) and current_school in school_options else 0
                new_school = st.selectbox(
                    "Assigned School",
                    options=school_options,
                    index=current_school_idx,
                    key=f"school_{teacher['id']}",
                    format_func=lambda x: school_display.get(x, "No School")
                )
                temp_teachers.at[teacher_idx, 'school_id'] = new_school if new_school else None
                
                # Move preference
                move = st.checkbox(
                    "Willing to Move",
                    value=teacher.get('move', False),
                    disabled=can_move,
                    key=f"move_{teacher['id']}",
                    help="Check if teacher is willing to move to a different school"
                )
                temp_teachers.at[teacher_idx, 'move'] = move
                
                # Status indicator with color coding
                status_container = st.container()
                with status_container:
                    if can_move:
                        st.info("Reassignable", icon="ℹ️")
                    elif move:
                        st.warning("Open to relocation", icon="🔄")
                    else:
                        st.success("Stable assignment", icon="✅")

# Save button
if st.button("Save Changes"):
    # Save teacher data
    save_teachers(temp_teachers)
    
    # Update assignments
    for _, teacher in temp_teachers.iterrows():
        current_school = teacher.get('school_id', '')
        if pd.notna(current_school) and current_school != '':
            assignments_df = assign_teacher_to_school(
                teacher['id'], 
                current_school,
                assignments_df
            )
    
    # Save assignments
    save_assignments(assignments_df)
    st.success("Changes saved successfully!")
    st.rerun()

# Display current data
expander = st.expander("View Raw Data")
with expander:
    st.dataframe(teachers_df)