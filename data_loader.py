import os
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import streamlit as st

# File paths
DATA_DIR = Path(__file__).parent / "data"
TEACHERS_FILE = DATA_DIR / "teachers.csv"
SCHOOLS_FILE = DATA_DIR / "kidsduo_schools.csv"
TRAVEL_TIMES_FILE = DATA_DIR / "station_travel_times.csv"
ASSIGNMENTS_FILE = DATA_DIR / "teacher_school_assignments.csv"

@st.cache_data
def load_teachers() -> pd.DataFrame:
    """Load teacher data from CSV and merge with current assignments"""
    # Load teachers
    df = pd.read_csv(TEACHERS_FILE)
    
    # Initialize 'move' column if it doesn't exist
    if 'move' not in df.columns:
        df['move'] = True
    
    # Convert 'move' to boolean, defaulting to True if not specified
    df['move'] = df['move'].astype(str).str.lower().replace({
        'true': True,
        'false': False,
        'nan': True,
        '': True
    }).astype(bool)
    
    # Ensure school_id is properly initialized and clean
    if 'school_id' not in df.columns:
        df['school_id'] = ''
    df['school_id'] = df['school_id'].fillna('').astype(str)
    
    # Load current assignments
    if os.path.exists(ASSIGNMENTS_FILE):
        try:
            assignments = load_assignments()
            if not assignments.empty:
                # Get only current assignments
                current_assignments = get_current_assignments(assignments)
                
                # Ensure teacher_id is string type in both dataframes
                df['id'] = df['id'].astype(str).str.strip()
                current_assignments['teacher_id'] = current_assignments['teacher_id'].astype(str).str.strip()
                
                # Create a mapping of teacher_id to school_id from current assignments
                assignment_map = current_assignments.set_index('teacher_id')['school_id'].to_dict()
                
                # Update school_id from assignments, only for teachers who have current assignments
                df['assigned_school'] = df['id'].map(assignment_map)
                df['school_id'] = df['assigned_school'].fillna(df['school_id'])
                df = df.drop(columns=['assigned_school'], errors='ignore')
                
                # Debug info
                print(f"Updated {len(current_assignments)} teacher assignments")
                print(f"Teachers with assignments: {df[df['school_id'] != ''].shape[0]}")
                
        except Exception as e:
            print(f"Error processing assignments: {str(e)}")
    
    # Ensure school_id is clean and handle NaN/None values
    df['school_id'] = df['school_id'].fillna('').astype(str)
    
    # Update move status: if teacher has no school, they should be movable
    # Otherwise, respect their current move status
    df['move'] = (df['school_id'] == '') | (df['move'] == True)
    
    return df

@st.cache_data
def load_assignments() -> pd.DataFrame:
    """Load teacher-school assignments from CSV"""
    if not os.path.exists(ASSIGNMENTS_FILE):
        return pd.DataFrame(columns=['teacher_id', 'school_id', 'assigned_date', 'is_current'])
    return pd.read_csv(ASSIGNMENTS_FILE)

def save_assignments(assignments_df: pd.DataFrame) -> None:
    """Save teacher-school assignments to CSV"""
    assignments_df.to_csv(ASSIGNMENTS_FILE, index=False)

def get_current_assignments(assignments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the most recent current assignment for each teacher.
    If multiple current assignments exist for a teacher, keep the most recent one.
    """
    if assignments_df.empty:
        return pd.DataFrame()
        
    # Ensure we have the required columns
    required_columns = ['teacher_id', 'school_id', 'assigned_date', 'is_current']
    if not all(col in assignments_df.columns for col in required_columns):
        return pd.DataFrame()
    
    # Convert assigned_date to datetime for proper sorting
    assignments_df = assignments_df.copy()
    assignments_df['assigned_date'] = pd.to_datetime(assignments_df['assigned_date'])
    
    # Get only current assignments
    current = assignments_df[assignments_df['is_current'] == True].copy()
    
    if current.empty:
        return pd.DataFrame()
    
    # Sort by assigned_date in descending order to get the most recent first
    current = current.sort_values('assigned_date', ascending=False)
    
    # Keep only the most recent assignment per teacher
    current = current.drop_duplicates('teacher_id', keep='first')
    
    return current

def assign_teacher_to_school(teacher_id: str, school_id: str, assignments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a teacher to a school, updating previous assignments if needed
    
    Args:
        teacher_id: ID of the teacher (string)
        school_id: ID of the school to assign (empty string to unassign)
        assignments_df: DataFrame containing all assignments
        
    Returns:
        Updated assignments DataFrame
    """
    from datetime import datetime
    
    # Ensure teacher_id is string and stripped of whitespace
    teacher_id = str(teacher_id).strip()
    
    # Create a copy to avoid SettingWithCopyWarning
    assignments_df = assignments_df.copy()
    
    # Mark any existing current assignments as not current
    mask = (assignments_df['teacher_id'].astype(str).str.strip() == teacher_id) & \
           (assignments_df['is_current'] == True)
           
    if mask.any():
        # Only update if there are current assignments to update
        assignments_df.loc[mask, 'is_current'] = False
        assignments_df.loc[mask, 'unassigned_date'] = datetime.now()
    
    # Add new assignment if school_id is not empty
    if pd.notna(school_id) and str(school_id).strip() != '':
        school_id = str(school_id).strip()
        new_assignment = pd.DataFrame([{
            'teacher_id': teacher_id,
            'school_id': school_id,
            'assigned_date': datetime.now(),
            'is_current': True,
            'assigned_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        
        # Ensure all required columns exist in the DataFrame
        for col in ['unassigned_date']:
            if col not in assignments_df.columns:
                assignments_df[col] = None
        
        # Ensure new assignment has all columns from the original DataFrame
        for col in assignments_df.columns:
            if col not in new_assignment.columns:
                new_assignment[col] = None
        
        assignments_df = pd.concat([assignments_df, new_assignment], ignore_index=True)
    
    return assignments_df

@st.cache_data
def load_schools() -> Dict[str, str]:
    """Load school data and return as a dictionary of id: name"""
    df = pd.read_csv(SCHOOLS_FILE)
    return df.set_index('id')['name'].to_dict()

def save_teachers(teachers_df: pd.DataFrame) -> None:
    """Save teacher data back to CSV"""
    teachers_df.to_csv(TEACHERS_FILE, index=False)

@st.cache_data
def load_schools_with_locations() -> pd.DataFrame:
    """Load school data with location information"""
    return pd.read_csv(SCHOOLS_FILE)

def add_teacher_locations(teachers_df: pd.DataFrame, schools_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add station coordinates for teachers using tokyo_stations_gps.csv
    
    Args:
        teachers_df: DataFrame containing teacher information
        schools_df: DataFrame containing school information with location data
        
    Returns:
        Updated teachers DataFrame with location information
    """
    # Load station coordinates
    stations_file = DATA_DIR / "tokyo_stations_gps.csv"
    if os.path.exists(stations_file):
        stations = pd.read_csv(stations_file)
        # Merge station coordinates with teachers
        teachers_df = pd.merge(
            teachers_df,
            stations[['station_name', 'latitude', 'longitude']],
            left_on='station',
            right_on='station_name',
            how='left'
        ).drop(columns=['station_name'])
    
    return teachers_df

@st.cache_data
def load_travel_times() -> pd.DataFrame:
    """Load and process travel times data"""
    try:
        return pd.read_csv(TRAVEL_TIMES_FILE)
    except Exception as e:
        st.error(f"Error loading travel times: {e}")
        return pd.DataFrame()

def get_travel_time(origin_id: str, dest_id: str, travel_times_df: pd.DataFrame) -> float:
    """
    Get travel time between two stations using their UUIDs
    
    Args:
        origin_id: UUID of the origin station
        dest_id: UUID of the destination station
        travel_times_df: DataFrame containing travel time information
        
    Returns:
        Travel time in minutes, or infinity if no route found
    """
    if travel_times_df is None or travel_times_df.empty:
        return float('inf')
    
    if pd.isna(origin_id) or pd.isna(dest_id):
        return float('inf')
        
    # Check if columns exist in the DataFrame
    required_columns = ['origin_uuid', 'destination_uuid', 'travel_min']
    if not all(col in travel_times_df.columns for col in required_columns):
        return float('inf')
    
    # Check direct direction (origin -> destination)
    direct = travel_times_df[
        (travel_times_df['origin_uuid'] == origin_id) & 
        (travel_times_df['destination_uuid'] == dest_id)
    ]
    
    if not direct.empty:
        return direct['travel_min'].iloc[0]
    
    # Check reverse direction (destination -> origin)
    reverse = travel_times_df[
        (travel_times_df['origin_uuid'] == dest_id) & 
        (travel_times_df['destination_uuid'] == origin_id)
    ]
    
    if not reverse.empty:
        return reverse['travel_min'].iloc[0]
        
    return float('inf')

def get_school_options(schools_df: pd.DataFrame) -> Tuple[list, dict]:
    """
    Generate school options for dropdowns and display
    
    Args:
        schools_df: DataFrame containing school information
        
    Returns:
        Tuple of (school_options, school_display) where:
        - school_options: List of school IDs (with empty string for 'No School')
        - school_display: Dictionary mapping school IDs to display names
    """
    school_options = [""] + sorted(schools_df['id'].tolist())
    school_display = {"": "No School"}
    school_display.update(schools_df.set_index('id')['name'].to_dict())
    
    return school_options, school_display
