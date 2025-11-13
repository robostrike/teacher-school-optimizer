import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from pathlib import Path
from folium.plugins import MarkerCluster
from folium import Icon
from typing import Optional, Tuple
import display_utils
import css_app
from data_loader import (
    load_teachers, load_assignments, save_assignments, get_current_assignments,
    assign_teacher_to_school, load_schools, save_teachers, load_schools_with_locations,
    add_teacher_locations, load_travel_times, get_travel_time, get_school_options
)

# Set page config
st.set_page_config(page_title="Teacher Optimizer", layout="wide")

# Data loading functions have been moved to data_loader.py

# Load data
teachers_df = load_teachers()
assignments_df = load_assignments()
current_assignments = get_current_assignments(assignments_df)
schools_df = load_schools_with_locations()
travel_times_df = load_travel_times()

# Get school options for dropdowns
school_options, school_display = get_school_options(schools_df)
schools = load_schools()  # Keep for backward compatibility

# Ensure we have a school_id column (set to None if missing)
if 'school_id' not in teachers_df.columns:
    teachers_df['school_id'] = None

# Add current school to teachers for backward compatibility
if not teachers_df.empty and not current_assignments.empty:
    # First drop any existing school_id column to avoid duplicate column issues
    if 'school_id' in teachers_df.columns:
        teachers_df = teachers_df.drop(columns=['school_id'], errors='ignore')
    
    # Perform the merge with explicit suffixes
    teachers_df = teachers_df.merge(
        current_assignments[['teacher_id', 'school_id']],
        left_on='id',
        right_on='teacher_id',
        how='left',
        suffixes=('', '_drop')
    )
    
    # Drop any columns that end with _drop
    drop_cols = [col for col in teachers_df.columns if col.endswith('_drop')]
    if drop_cols:
        teachers_df = teachers_df.drop(columns=drop_cols)

# Ensure teachers without a school have move=True and update assignments
if 'move' in teachers_df.columns and 'school_id' in teachers_df.columns:
    # Set move=True for teachers without a school assignment
    teachers_df.loc[pd.isna(teachers_df['school_id']) | (teachers_df['school_id'] == ''), 'move'] = True
    
    # Ensure school_id is properly formatted as string and handle NaN/None
    teachers_df['school_id'] = teachers_df['school_id'].fillna('')
    
    # Ensure move is boolean type
    teachers_df['move'] = teachers_df['move'].astype(bool)
    
    # Update current_assignments to match the teachers_df
    updated_assignments = teachers_df[['id', 'school_id']].copy()
    updated_assignments = updated_assignments[updated_assignments['school_id'] != '']
    updated_assignments = updated_assignments.rename(columns={'id': 'teacher_id'})
    updated_assignments['assigned_date'] = pd.Timestamp.now()
    updated_assignments['is_current'] = True
    
    # Update the current_assignments in session state
    if 'current_assignments' not in st.session_state:
        st.session_state.current_assignments = updated_assignments
    else:
        # Keep only the most recent assignment for each teacher
        st.session_state.current_assignments = pd.concat([
            st.session_state.current_assignments[~st.session_state.current_assignments['teacher_id'].isin(updated_assignments['teacher_id'])],
            updated_assignments
        ])
    
    # Clean up the teacher_id column if it was added
    if 'teacher_id' in teachers_df.columns and 'id' in teachers_df.columns:
        teachers_df = teachers_df.drop(columns=['teacher_id'])
else:
    teachers_df['school_id'] = None

# App title
st.title("Teacher-School Manager")

# Travel time functions have been moved to data_loader.py

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


# School and teacher location functions have been moved to data_loader.py

# School data and teacher locations are already loaded above
travel_times_df = load_travel_times()

# Initialize session state for temporary assignments
if 'temp_assignments' not in st.session_state:
    st.session_state.temp_assignments = {}

# School selection for filtering
st.sidebar.subheader("Filter by School")

# Get current assignments
current_assignments = get_current_assignments(load_assignments())

def get_current_teacher_counts():
    """Get current teacher counts including pending changes"""
    # Start with current assignments
    counts = current_assignments['school_id'].value_counts().to_dict()
    
    # Update with pending changes if they exist
    if hasattr(st.session_state, 'temp_assignments') and st.session_state.temp_assignments:
        for teacher_id, school_id in st.session_state.temp_assignments.items():
            # Remove from old school if teacher was assigned
            old_school = current_assignments[
                (current_assignments['teacher_id'] == teacher_id) & 
                (current_assignments['is_current'] == True)
            ]
            if not old_school.empty and old_school['school_id'].iloc[0] in counts:
                counts[old_school['school_id'].iloc[0]] -= 1
                if counts[old_school['school_id'].iloc[0]] == 0:
                    del counts[old_school['school_id'].iloc[0]]
            
            # Add to new school if there is one
            if school_id:
                counts[school_id] = counts.get(school_id, 0) + 1
    
    return counts

# Format school names with teacher counts
def format_school_option(school_id):
    if not school_id:
        return "All Schools"
    
    # Get current counts including pending changes
    current_counts = get_current_teacher_counts()
    count = current_counts.get(school_id, 0)
    school_name = schools_df[schools_df['id'] == school_id]['name'].iloc[0]
    return f"{school_name} ({count} teachers)"

# Create school options with teacher counts
school_options = [""] + sorted(schools_df['id'].tolist())
selected_school_id = st.sidebar.selectbox(
    "Select a school...",
    school_options,
    format_func=format_school_option
)

# Add custom CSS
st.markdown(css_app.get_custom_css(), unsafe_allow_html=True)

# Show school details and assigned teachers
if selected_school_id:
    # Create a container for school details that we can update
    school_details_container = st.sidebar.container()
    
    # Get the most up-to-date teacher data including pending changes
    current_teachers = teachers_df.copy()
    for tid, sid in st.session_state.temp_assignments.items():
        mask = current_teachers['id'] == tid
        if mask.any():
            current_teachers.loc[mask, 'school_id'] = sid if sid else None
    
    # Get detailed school information with the latest data
    school_details = display_utils.get_school_details(
        selected_school_id,
        current_teachers,  # Use the updated teachers data
        schools_df,
        travel_times_df
    )
    
    with school_details_container:
        if school_details:
            st.markdown("---")
            st.subheader("School Details")
            
            # School name and station
            st.markdown(f"**{school_details['name']}**")
            st.caption(f"🚉 {school_details['station']}")
            
            # Student count
            st.markdown("---")
            st.metric("👥 Students", f"{school_details['student_count']:,}")
            
            # Teacher count and gender ratio
            st.metric("👨‍🏫 Teachers", 
                     f"{school_details['teacher_count']}",
                     f"{school_details['gender_ratio']} (♂:♀)")
            
            # Travel time statistics
            st.markdown("---")
            st.markdown("**Travel Times**")
            st.metric("🚆 Total Travel Time", 
                     f"{school_details['total_travel_time']:,.1f} min")
            
            # Add a refresh button to force update the display
            if st.button("↻ Refresh School Details"):
                st.rerun()
            
            # Assigned teachers list
            st.markdown("---")
            st.subheader("Assigned Teachers")
        
        # Get filtered teachers assigned to this school
        school_teachers = st.session_state.filtered_teachers[
            (st.session_state.filtered_teachers['school_id'] == selected_school_id) &
            (st.session_state.filtered_teachers['school_id'].notna()) &
            (st.session_state.filtered_teachers['school_id'] != '')
        ].copy()
        
        if not school_teachers.empty:
            # Add travel time to school for each teacher
            if 'station_id' in school_teachers.columns:
                school_station = schools_df[schools_df['id'] == selected_school_id]['station_uuid'].iloc[0]
                school_teachers['travel_time'] = school_teachers['station_id'].apply(
                    lambda x: get_travel_time(x, school_station, travel_times_df) if pd.notna(x) else float('inf')
                )
            
            # Create a list to store teachers and their sort keys
            teachers_list = []
            
            for _, teacher in school_teachers.iterrows():
                # Determine if teacher is assigned and not willing to move
                has_school = pd.notna(teacher.get('school_id')) and teacher.get('school_id') != ''
                is_assigned_not_moving = has_school and not teacher.get('move', True)
                
                # Create sort key: (is_assigned_not_moving, is_female, is_bilingual, travel_time)
                sort_key = (
                    is_assigned_not_moving,  # False (0) comes before True (1)
                    0 if teacher.get('gender') == 'Female' else 1,  # Female first
                    0 if teacher.get('type') == 'Bilingual' else 1,  # Bilingual first
                    teacher.get('travel_time', float('inf'))  # Shorter travel time first
                )
                
                teachers_list.append((sort_key, teacher))
            
            # Sort teachers: assigned+not_moving at bottom, then by gender, type, travel time
            teachers_list.sort(key=lambda x: x[0])
            
            # Display sorted teachers
            for sort_key, teacher in teachers_list:
                gender_icon = '♀️' if teacher.get('gender') == 'Female' else '♂️'
                teacher_type = teacher.get('type', 'Teacher')
                station = teacher.get('station', 'Unknown station')
                
                st.sidebar.markdown(
                    f"<div>"
                    f"{gender_icon} <strong>{teacher['name']}</strong><br>"
                    f"<span style='font-style: italic;'>{teacher_type}</span> at {station}"
                    "</div>",
                    unsafe_allow_html=True
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
    
    # Filter teachers within 60 minutes using the filtered teachers from session state
    teachers_within_range = []
    filtered_teachers = st.session_state.get('filtered_teachers', teachers_df)
    
    for _, teacher in filtered_teachers.iterrows():
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
        
        # Prepare sorting columns
        sort_columns = []
        
        # Add bilingual column if it exists
        if 'bilingual' in result_df.columns:
            sort_columns.append('bilingual')
            
        # Create sort columns
        result_df['sort_gender'] = result_df['gender'].apply(lambda x: 0 if x == 'Female' else 1)
        
        # Create a priority column: 1=assigned+not willing to move (will be sorted to bottom), 0=others
        result_df['sort_priority'] = 0  # Default priority (lower is higher priority)
        if 'move' in result_df.columns:
            result_df['sort_priority'] = result_df.apply(
                lambda x: 1 if (pd.notna(x['school_id']) and not x['move']) else 0, 
                axis=1
            )
        
        # Define the sort order - first by priority (to put assigned+not_willing_to_move at bottom)
        # Then for the rest: female first, then bilingual, then travel time
        sort_columns = ['sort_priority']  # Sort by priority first (0=normal, 1=assigned+not_willing_to_move)
        sort_columns.extend(['sort_gender'])  # Then by gender (female first)
        
        # Add bilingual status if column exists
        if 'bilingual' in result_df.columns:
            sort_columns.append('bilingual')
            
        # Add travel time
        sort_columns.append('travel_minutes')
        
        # Sort the DataFrame
        result_df = result_df.sort_values(by=sort_columns)
        
        # Clean up temporary columns
        result_df = result_df.drop(columns=['sort_gender', 'has_school'], errors='ignore')
        
        # Add gender icons
        result_df['Gender'] = result_df['gender'].apply(lambda x: '♂️' if x == 'Male' else '♀️')
        
        # Get school names for assigned teachers
        school_names = schools_df.set_index('id')['name'].to_dict()
        result_df['assigned_school'] = result_df['school_id'].map(school_names).fillna('Unassigned')
        
        # Prepare display columns with the correct final names
        display_cols = ['name', 'Gender', 'type', 'assigned_school', 'travel_time']
        new_columns = ['Name', ' ', 'Type', 'Assigned School', 'Travel Time']
        display_df = result_df[display_cols].copy()
        display_df.columns = new_columns
        
        # Debug: Print DataFrame info and sample data
        if not display_df.empty:
            # Convert DataFrame to HTML with custom styling
            def get_row_style(row):
                # Get the original row index from the result_df
                idx = result_df[result_df['name'] == row['Name']].index[0]
                
                # Check if teacher is willing to move (using 'move' column from teachers.csv)
                if 'move' in result_df.columns and not result_df.loc[idx, 'move']:
                    return 'style="background-color: #f0f0f0; color: #999;"'  # Grey background for not willing to move
                
                # Default coloring based on teacher type
                color = '#e6f3ea' if row['Type'] == 'Native' else '#f3e6ff'
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
st.markdown(css_app.get_custom_css(), unsafe_allow_html=True)

st.write("Manage teacher assignments and move preferences")

# Create form for editing
temp_teachers = teachers_df.copy()

# Display teachers in a responsive grid of cards
st.subheader("Teacher Directory")

# Create a 3-column layout for filters
search_col, type_col, gender_col = st.columns([2, 1, 1])

# Search bar in the first column
with search_col:
    search_term = st.text_input(
        "Search teachers by name, type, or station", 
        "",
        help="Search by teacher name, type (Bilingual/Native), or station name"
    )

# Teacher type filter in the second column
with type_col:
    teacher_type = st.radio(
        "**Teacher Type**",
        ["All", "Bilingual", "Native"],
        index=0,
        key="teacher_type_filter",
        horizontal=True
    )

# Gender filter in the third column
with gender_col:
    gender = st.radio(
        "**Gender**",
        ["All", "Male", "Female"],
        index=0,
        key="gender_filter",
        horizontal=True
    )

# Clear filters button below the columns
if st.button("Clear All Filters"):
    st.session_state.teacher_type_filter = "All"
    st.session_state.gender_filter = "All"
    st.rerun()

# Initialize session state variables
if 'filtered_teachers' not in st.session_state:
    st.session_state.filtered_teachers = teachers_df.copy()

# Initialize temporary storage for teacher-school assignments
if 'temp_assignments' not in st.session_state:
    st.session_state.temp_assignments = teachers_df[['id', 'school_id']].set_index('id')['school_id'].to_dict()

# Function to update a teacher's school assignment
def update_teacher_school(teacher_id, school_id):
    """Update a teacher's school assignment in the temporary storage"""
    st.session_state.temp_assignments[teacher_id] = school_id if school_id else None
    
    # Also update the filtered_teachers dataframe
    mask = st.session_state.filtered_teachers['id'] == teacher_id
    if mask.any():
        st.session_state.filtered_teachers.loc[mask, 'school_id'] = school_id if school_id else None

# Always apply filters to ensure the display is up to date
filtered = teachers_df.copy()

# Apply search term filter
if search_term:
    search_term = search_term.lower().strip()
    if search_term:
        mask = (
            filtered['name'].str.lower().str.contains(search_term, na=False) |
            filtered['type'].str.lower().str.contains(search_term, na=False) |
            filtered['station'].str.lower().str.contains(search_term, na=False)
        )
        filtered = filtered[mask].copy()

# Apply type filter
if teacher_type != "All":
    filtered = filtered[filtered['type'] == teacher_type].copy()

# Apply gender filter
if gender != "All":
    filtered = filtered[filtered['gender'] == gender].copy()

# Update session state with filtered results
filtered_teachers = filtered.reset_index(drop=True)

# Display message if no teachers match
if filtered_teachers.empty:
    st.info("No teachers match the current filters. Try adjusting your search criteria.")

# Create a grid of cards (3 cards per row)
cols_per_row = 3
total_teachers = len(filtered_teachers)

# Display the teachers with full functionality
for i in range(0, total_teachers, cols_per_row):
    # Create a row of cards
    cols = st.columns(cols_per_row)
    
    # Fill the current row with cards
    for col_idx in range(cols_per_row):
        teacher_idx = i + col_idx
        if teacher_idx >= total_teachers:
            break
            
        # Get teacher from the filtered list and get the original index
        filtered_teacher = filtered_teachers.iloc[teacher_idx]
        teacher_id = filtered_teacher['id']
        
        # Find the teacher in the original DataFrame
        teacher_mask = teachers_df['id'] == teacher_id
        teacher_idx_original = teachers_df[teacher_mask].index[0]
        teacher = teachers_df.loc[teacher_idx_original].copy()
        
        current_school = teacher.get('school_id', '')
        can_move = pd.isna(current_school) or current_school == ''
        
        with cols[col_idx]:
            with st.container(border=True):
                # Header with name, gender, and type
                gender_icon = '♂️' if teacher.get('gender') == 'Male' else '♀️'
                st.markdown(f"### {teacher['name']} {gender_icon}")
                st.caption(f"{teacher['type']} | {teacher['station']}")
                
                # School selection
                teacher_id = teacher['id']
                current_school = st.session_state.temp_assignments.get(teacher_id, '')
                
                # Create a callback to handle school selection changes
                def on_school_change(teacher_id=teacher_id):
                    new_school = st.session_state[f'school_{teacher_id}']
                    update_teacher_school(teacher_id, new_school)
                
                # Create the dropdown with the current assignment
                school_index = 0  # Default to "No School"
                if current_school and current_school in school_options:
                    school_index = school_options.index(current_school)
                
                st.selectbox(
                    "Assign to school",
                    school_options,
                    index=school_index,
                    key=f"school_{teacher_id}",
                    format_func=lambda x: school_display.get(x, "No School") if x else "No School",
                    label_visibility="collapsed",
                    on_change=on_school_change,
                    args=(teacher_id,)
                )
                
                # Move preference checkbox
                # If no school is selected, force move=True and disable the checkbox
                if not current_school:
                    move = True
                    move_disabled = True
                    # Update the move status in the teachers_df and filtered_teachers
                    teachers_df.loc[teacher_mask, 'move'] = True
                    if teacher_id in st.session_state.temp_assignments:
                        st.session_state.filtered_teachers.loc[teacher_mask, 'move'] = True
                else:
                    # Enable checkbox if teacher has a school assignment
                    move_disabled = False
                    move = st.checkbox(
                        "Willing to Move",
                        value=teacher.get('move', True),  # Default to True if not set
                        disabled=move_disabled,
                        key=f"move_{teacher_id}",
                        help="Check if teacher is willing to move to a different school"
                    )
                    # Update the move status in the teachers_df and filtered_teachers
                    teachers_df.loc[teacher_mask, 'move'] = move
                    if teacher_id in st.session_state.temp_assignments:
                        st.session_state.filtered_teachers.loc[teacher_mask, 'move'] = move
                
                # Always show the checkbox, but it might be disabled
                if not current_school:
                    st.checkbox(
                        "Willing to Move",
                        value=True,
                        disabled=True,
                        key=f"disabled_move_{teacher_id}"
                    )
                
                # Status indicator with color coding - check the latest state
                status_container = st.container()
                with status_container:
                    # Get the most current school assignment
                    current_school = st.session_state.temp_assignments.get(teacher_id, teacher.get('school_id', ''))
                    can_move_now = not current_school or current_school == ''
                    
                    # Get the most current move status
                    current_move = teacher.get('move', True)
                    if f"move_{teacher_id}" in st.session_state:
                        current_move = st.session_state[f"move_{teacher_id}"]
                    
                    # Determine status
                    if can_move_now or current_move:
                        st.info("Assignable", icon="📅")
                    else:
                        st.success("Stable", icon="✅")

# Save button
if st.button("Save Changes"):
    try:
        # Update the main teachers_df with all temporary assignments
        for teacher_id, school_id in st.session_state.temp_assignments.items():
            mask = teachers_df['id'] == teacher_id
            if mask.any():
                teachers_df.loc[mask, 'school_id'] = school_id if school_id else None
        
        # Save teacher data
        save_teachers(teachers_df)
        
        # Update assignments
        for teacher_id, school_id in st.session_state.temp_assignments.items():
            if school_id:  # Only update if there's a school assignment
                assignments_df = assign_teacher_to_school(
                    teacher_id, 
                    school_id,
                    assignments_df
                )
        
        # Save assignments
        save_assignments(assignments_df)
        
        # Update the current assignments in session state
        st.session_state.current_assignments = get_current_assignments(assignments_df)
        
        st.success("Changes saved successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error saving changes: {str(e)}")

def get_current_assignments_with_pending(assignments_df, temp_assignments):
    """Get current assignments including pending changes"""
    # Create a copy of assignments
    current_assignments = assignments_df.copy()
    
    # Update with any pending assignments
    for teacher_id, school_id in temp_assignments.items():
        # Remove any existing assignments for this teacher
        current_assignments = current_assignments[current_assignments['teacher_id'] != teacher_id]
        
        # Add the new assignment if there is one
        if school_id:
            new_assignment = pd.DataFrame([{
                'teacher_id': teacher_id,
                'school_id': school_id,
                'assigned_at': pd.Timestamp.now(),
                'is_current': True
            }])
            current_assignments = pd.concat([current_assignments, new_assignment], ignore_index=True)
    
    return current_assignments

# Display school statistics
st.header("School Statistics")

# Get current assignments including pending changes
current_assignments = get_current_assignments_with_pending(assignments_df, st.session_state.temp_assignments)

# Calculate and display occupancy using filtered teachers
occupancy = display_utils.display_school_occupancy(
    current_assignments[current_assignments['teacher_id'].isin(filtered_teachers['id'])], 
    schools_df
)
balance = display_utils.check_school_balance(current_assignments, filtered_teachers)
balance_summary = display_utils.get_school_balance_summary(balance)

# Calculate basic metrics
unassigned_teacher_ids = [tid for tid, sid in st.session_state.temp_assignments.items() if not sid]
unassigned_count = len(unassigned_teacher_ids)
total_teachers = len(teachers_df)
unassigned_pct = (unassigned_count / total_teachers * 100) if total_teachers > 0 else 0

# Calculate travel time statistics with current assignments
temp_teachers = teachers_df.copy()
for teacher_id, school_id in st.session_state.temp_assignments.items():
    mask = temp_teachers['id'] == teacher_id
    if mask.any():
        temp_teachers.loc[mask, 'school_id'] = school_id if school_id else None

travel_stats = display_utils.calculate_total_travel_time(temp_teachers, schools_df, travel_times_df)

# Calculate metrics and percentages
total_schools = len(schools_df)
no_teacher_schools = int(occupancy.sum())
no_teacher_pct = (no_teacher_schools / total_schools * 100) if total_schools > 0 else 0

# Create columns for metrics
col1, col2, col3 = st.columns(3)

# Column 1: Teacher Statistics
with col1:
    st.metric("Total Teachers", f"{total_teachers:,}")
    st.metric(
        "Unassigned Teachers", 
        f"{unassigned_count:,}",
        f"{unassigned_pct:.1f}% of teachers"
    )

# Column 2: Travel Time Statistics
with col2:
    st.metric(
        "Total Travel Time (min)",
        f"{travel_stats['total_travel_time']:,.1f}"
    )
    st.metric(
        "Avg. Travel (min/teacher)",
        f"{travel_stats['average_travel_time']:,.1f}"
    )

# Column 3: School Statistics
with col3:
    st.metric("Total Schools", f"{total_schools:,}")
    st.metric(
        "Schools with No Teachers", 
        f"{no_teacher_schools:,}",
        f"{no_teacher_pct:.1f}% of schools"
    )
    
# Display detailed balance information
with st.expander("View Detailed School Balance"):
    if not balance.empty:
        # Merge with school names for better display
        display_balance = balance.merge(
            schools_df[['id', 'name']], 
            left_on='school_id', 
            right_on='id',
            how='left'
        )
        display_balance = display_balance[[
            'school_id', 'name', 'teacher_count'
        ]]
        display_balance = display_balance.rename(columns={
            'school_id': 'School ID',
            'name': 'School Name',
            'teacher_count': 'Teacher Count'
        })
        
        st.dataframe(display_balance, use_container_width=True)
    else:
        st.info("No school balance data available.")

# Display current data
expander = st.expander("View Raw Data")
with expander:
    st.dataframe(teachers_df)