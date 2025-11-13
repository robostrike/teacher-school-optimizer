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
    
    # Load current assignments
    if os.path.exists(ASSIGNMENTS_FILE):
        assignments = load_assignments()
        current_assignments = get_current_assignments(assignments)
        
        # Merge with teachers to get school assignments
        df = pd.merge(
            df,
            current_assignments[['teacher_id', 'school_id']],
            left_on='id',
            right_on='teacher_id',
            how='left',
            suffixes=('', '_assignment')
        )
        
        # Clean up columns
        drop_cols = ['teacher_id']
        if 'school_id' in df.columns and 'school_id_assignment' in df.columns:
            drop_cols.append('school_id')
            df.rename(columns={'school_id_assignment': 'school_id'}, inplace=True)
        
        df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        # Ensure school_id is string and handle NaN values
        df['school_id'] = df['school_id'].fillna('').astype(str)
        
        # Update move status for unassigned teachers
        df.loc[df['school_id'] == '', 'move'] = True
    
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
    """Get current assignments (where is_current is True)"""
    return assignments_df[assignments_df['is_current'] == True]

def assign_teacher_to_school(teacher_id: str, school_id: str, assignments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a teacher to a school, updating previous assignments if needed
    
    Args:
        teacher_id: ID of the teacher
        school_id: ID of the school to assign (empty string to unassign)
        assignments_df: DataFrame containing all assignments
        
    Returns:
        Updated assignments DataFrame
    """
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
