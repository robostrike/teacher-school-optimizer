import streamlit as st
import pandas as pd
import folium
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
    """Load teacher data from CSV and merge with assignments"""
    # Load teachers
    df = pd.read_csv(TEACHERS_FILE)
    
    # Handle 'move' column with proper type conversion
    if 'move' in df.columns:
        df['move'] = df['move'].astype(str).str.lower().replace({
            'true': True,
            'false': False,
            'nan': False,
            '': False
        }).astype(bool)
    else:
        df['move'] = False
    
    # Load current assignments
    if os.path.exists(ASSIGNMENTS_FILE):
        assignments = pd.read_csv(ASSIGNMENTS_FILE)
        current_assignments = assignments[assignments['is_current'] == True]
        
        # Merge with teachers to get school assignments
        df = pd.merge(
            df,
            current_assignments[['teacher_id', 'school_id']],
            left_on='id',
            right_on='teacher_id',
            how='left'
        )
        
        # Clean up the merged columns
        df.drop(columns=['teacher_id'], inplace=True, errors='ignore')
        
        # Ensure school_id is string and handle NaN values
        if 'school_id' in df.columns:
            df['school_id'] = df['school_id'].fillna('').astype(str)
        else:
            df['school_id'] = ''
    
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
    try:
        # Debug: Print input parameters
        print(f"\n=== get_travel_time called ===")
        print(f"Origin ID: {origin_id} (type: {type(origin_id)})")
        print(f"Dest ID:   {dest_id} (type: {type(dest_id)})")
        
        if travel_times_df is None or travel_times_df.empty:
            print("Error: travel_times_df is None or empty")
            return float('inf')
        
        if pd.isna(origin_id) or pd.isna(dest_id):
            print(f"Error: Missing station IDs - origin: {origin_id}, dest: {dest_id}")
            return float('inf')
            
        # Check if columns exist in the DataFrame
        required_columns = ['origin_uuid', 'destination_uuid', 'travel_min']
        if not all(col in travel_times_df.columns for col in required_columns):
            print(f"Error: Missing required columns in travel_times_df. Available columns: {travel_times_df.columns.tolist()}")
            return float('inf')
        
        # Check direct direction (origin -> destination)
        direct = travel_times_df[
            (travel_times_df['origin_uuid'] == origin_id) & 
            (travel_times_df['destination_uuid'] == dest_id)
        ]
        
        if not direct.empty:
            print(f"Found direct route: {len(direct)} matches")
            print(f"Sample travel time: {direct['travel_min'].iloc[0]} min")
            return direct['travel_min'].iloc[0]
        
        print("No direct route found, checking reverse direction...")
        
        # Check reverse direction (destination -> origin)
        reverse = travel_times_df[
            (travel_times_df['origin_uuid'] == dest_id) & 
            (travel_times_df['destination_uuid'] == origin_id)
        ]
        
        if not reverse.empty:
            print(f"Found reverse route: {len(reverse)} matches")
            print(f"Sample travel time: {reverse['travel_min'].iloc[0]} min")
            return reverse['travel_min'].iloc[0]
            
        print("No route found in either direction")
        
        # Debug: Check if either station exists in the travel times data
        origin_exists = (travel_times_df['origin_uuid'] == origin_id).any() or \
                       (travel_times_df['destination_uuid'] == origin_id).any()
        dest_exists = (travel_times_df['origin_uuid'] == dest_id).any() or \
                     (travel_times_df['destination_uuid'] == dest_id).any()
        
        print(f"Origin station in data: {'Yes' if origin_exists else 'No'}")
        print(f"Destination station in data: {'Yes' if dest_exists else 'No'}")
        
        return float('inf')  # No route found
        
    except Exception as e:
        print(f"Error in get_travel_time: {str(e)}")
        print(f"Origin ID: {origin_id}, Dest ID: {dest_id}")
        if 'travel_times_df' in locals():
            print(f"DataFrame columns: {travel_times_df.columns.tolist()}")
            print(f"DataFrame sample: {travel_times_df.head(1).to_dict()}")
        return float('inf')

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
        color = 'blue'
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
                travel_time = get_travel_time(
                    teacher_station,
                    school_station,
                    travel_times_df
                )
                print(f"Travel time: {travel_time}")
                is_within_range = travel_time <= 60
            except Exception as e:
                error_msg = f"Error calculating travel time: {e}"
                print(error_msg)
                st.warning(error_msg)
                is_within_range = False
        
        # Reset color for each teacher
        color = 'blue'  # Default color
        
        # Determine icon color based on status and range
        if not teacher.get('move', False) and pd.notna(teacher.get('school_id')) and teacher.get('school_id') != '':
            color = 'gray'  # Cannot move (has school and not willing to move)
        elif is_within_range:
            color = 'green'  # Within 60 minutes of selected school
        
        print(f"Color for {teacher['name']}: {color}")

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
        
        # Prepare display columns with the correct final names
        display_cols = ['name', 'Gender', 'type', 'station', 'travel_time']
        new_columns = ['Name', ' ', 'Type', 'Station', 'Travel Time']
        display_df = result_df[display_cols].copy()
        display_df.columns = new_columns
        
        # Debug: Print DataFrame info and sample data
        if not display_df.empty:
            st.write("Debug - First row 'Type' value:", display_df.iloc[0]['Type'])
            
            # Convert DataFrame to HTML with custom styling
            def get_row_style(row):
                color = '#e6f3ff' if row['Type'] == 'Native' else '#f3e6ff'
                return f'style="background-color: {color};"'
                
            # Create HTML table
            html = "<div style='overflow-x:auto;'><table style='width:100%; border-collapse: collapse;'>"
            
            # Add header
            html += "<tr>"
            for col in display_df.columns:
                html += f"<th style='text-align: left; padding: 8px; border: 1px solid #ddd;'>{col}</th>"
            html += "</tr>"
            
            # Add rows
            for _, row in display_df.iterrows():
                row_style = get_row_style(row)
                html += f"<tr {row_style}>"
                for col in display_df.columns:
                    html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{row[col]}</td>"
                html += "</tr>"
                
            html += "</table></div>"
            
            # Display the HTML table
            st.markdown(html, unsafe_allow_html=True)
            
            # Also show the count of teachers
            st.info(f"Found {len(display_df)} teachers within 60 minutes of {selected_school['name']}.")
        else:
            st.info("No teachers found within 60 minutes of the selected school.")
        
        
    else:
        st.info("No teachers found within 60 minutes of the selected school.")

# Add custom CSS to make the directory more compact
st.markdown("""
    <style>
    /* Make the teacher cards more compact */
    .stContainer {
        padding: 0.25rem !important;
    }
    
    /* Remove all vertical gaps between elements */
    .stMarkdown h3, .stMarkdown p, .stMarkdown div, 
    .stVerticalBlock, .stElementContainer, .stHeading, 
    .stSelectbox, .stAlert, .stMarkdown {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.1 !important;
    }
    
    /* Teacher name and type */
    .stMarkdown h3 {
        font-size: 0.95rem !important;
        line-height: 1 !important;
    }
    
    /* Caption text (type and station) */
    .stMarkdown .stCaption {
        font-size: 0.75rem !important;
        line-height: 1 !important;
        margin-top: 0.1rem !important;
    }
    
    /* Make the selectbox and checkbox more compact */
    .stSelectbox, .stCheckbox {
        margin: 0.1rem 0 !important;
        font-size: 0.85rem !important;
    }
    
    /* Reduce padding and height in the selectbox */
    .stSelectbox > div > div {
        padding: 0.15rem 0.75rem 0.15rem 0.5rem !important;
        min-height: 1rem !important;
    }
    
    /* Space between cards */
    .stContainer > div {
        margin: 0 0.1rem 0.2rem 0.1rem !important;
    }
    
    /* Make sure columns have the right spacing */
    .stHorizontal > div[data-testid="column"] {
        padding: 0 0.1rem !important;
    }
    
    /* Reduce space around status messages */
    .stAlert {
        padding: 0.1rem 0.3rem !important;
        margin: 0 !important;
        min-height: 1rem !important;
    }
    
    /* School filter section specific styles */
    .stVerticalBlock.st-emotion-cache-tn0cau {
        gap: 0 !important;
    }
    
    .st-emotion-cache-1vo6xi6 {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stHeading h3 {
        margin: 0 !important;
        padding: 0.2rem 0 !important;
    }
    
    /* Make form elements more compact */
    .stTextInput, .stSelectbox, .stCheckbox {
        margin-bottom: 0.15rem !important;
    }
    
    /* Reduce space between form elements */
    .element-container {
        padding: 0.1rem 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.write("Manage teacher assignments and move preferences")

# Create form for editing
temp_teachers = teachers_df.copy()

# Display teachers in a responsive grid of cards
st.subheader("Teacher Directory")

# Create a grid of cards (4 cards per row on large screens)
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
                # Empty label with CSS to hide it but keep it accessible
                st.markdown("<style>.school-select-label { display: none; }</style>", unsafe_allow_html=True)
                new_school = st.selectbox(
                    " ",  # Empty space for label (required by Streamlit)
                    options=school_options,
                    index=current_school_idx,
                    key=f"school_{teacher['id']}",
                    format_func=lambda x: school_display.get(x, "No School"),
                    label_visibility="collapsed"  # This hides the label
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
                        st.warning("Open", icon="🔄")
                    else:
                        st.success("Stable", icon="✅")

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