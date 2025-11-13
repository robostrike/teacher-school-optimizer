import pandas as pd
from data_loader import get_travel_time

def display_school_occupancy(assignments_df, schools_df):
    """
    Identify schools with no teachers assigned.
    
    Args:
        assignments_df (pd.DataFrame): DataFrame containing teacher-school assignments
        schools_df (pd.DataFrame): DataFrame containing school information
        
    Returns:
        pd.Series: Series with school IDs as index and occupancy status as values 
                  (1 for no teachers, 0 for at least one teacher)
    """
    # Get all school IDs from the schools dataframe
    all_schools = pd.Series(1, index=schools_df['id'].unique())
    
    # If no assignments, all schools have no teachers
    if assignments_df.empty or 'school_id' not in assignments_df.columns:
        return all_schools
    
    # Get schools with at least one teacher (non-null school_id)
    valid_assignments = assignments_df[assignments_df['school_id'].notna()]
    schools_with_teachers = valid_assignments['school_id'].unique()
    
    # Create a series with 1 for all schools, then set 0 for schools with teachers
    occupancy = all_schools.copy()
    if len(schools_with_teachers) > 0:
        occupancy[schools_with_teachers] = 0
    
    return occupancy

def check_school_balance(assignments_df, teachers_df):
    """
    Check the number of teachers assigned to each school.
    
    Args:
        assignments_df: DataFrame containing teacher-school assignments
        teachers_df: DataFrame containing teacher information (unused in this simplified version)
        
    Returns:
        DataFrame with school_id and teacher_count
    """
    result_columns = ['school_id', 'teacher_count']
    
    # Handle empty assignments
    if assignments_df.empty:
        return pd.DataFrame(columns=result_columns)
        
    # Filter out assignments without a school_id
    valid_assignments = assignments_df[assignments_df['school_id'].notna()].copy()
    
    # Return empty if no valid assignments
    if valid_assignments.empty:
        return pd.DataFrame(columns=result_columns)
    
    # Count teachers per school
    teacher_counts = valid_assignments['school_id'].value_counts().reset_index()
    teacher_counts.columns = ['school_id', 'teacher_count']
    
    return teacher_counts[result_columns]

def count_unassigned_teachers(teachers_df, assignments_df):
    """
    Count the number of teachers without school assignments.
    
    Args:
        teachers_df (pd.DataFrame): DataFrame containing all teachers
        assignments_df (pd.DataFrame): DataFrame containing teacher-school assignments
        
    Returns:
        int: Number of teachers without school assignments
    """
    if teachers_df.empty:
        return 0
        
    # Get all teacher IDs that have assignments
    assigned_teacher_ids = set(assignments_df['teacher_id'].unique())
    
    # Count teachers who don't have assignments
    unassigned_count = teachers_df[~teachers_df['id'].isin(assigned_teacher_ids)].shape[0]
    
    return unassigned_count

def get_school_balance_summary(balance_df):
    """
    Generate a summary of school balance statistics.
    
    Args:
        balance_df (pd.DataFrame): DataFrame from check_school_balance function
        
    Returns:
        dict: Dictionary containing the total number of schools
    """
    return {
        'total_schools': len(balance_df) if not balance_df.empty else 0
    }

def get_school_details(school_id, teachers_df, schools_df, travel_times_df):
    """
    Get detailed information about a specific school.
    
    Args:
        school_id: ID of the school to get details for
        teachers_df: DataFrame containing teacher information
        schools_df: DataFrame containing school information
        travel_times_df: DataFrame containing travel times between stations
        
    Returns:
        Dictionary containing school details or None if school not found
    """
    # Find the school
    school = schools_df[schools_df['id'] == school_id].iloc[0] if not schools_df[schools_df['id'] == school_id].empty else None
    if school is None:
        return None
    
    # Get all teachers assigned to this school
    school_teachers = teachers_df[teachers_df['school_id'] == school_id]
    
    # Calculate gender ratio
    gender_counts = school_teachers['gender'].value_counts()
    male_count = gender_counts.get('Male', 0)
    female_count = gender_counts.get('Female', 0)
    total_teachers = len(school_teachers)
    
    # Calculate total and average travel time for teachers
    total_travel_time = 0
    valid_teachers = 0
    
    for _, teacher in school_teachers.iterrows():
        if pd.isna(teacher.get('station_id')) or 'station_uuid' not in school:
            continue
            
        travel_time = get_travel_time(
            origin_id=teacher['station_id'],
            dest_id=school['station_uuid'],
            travel_times_df=travel_times_df
        )
        
        if pd.notna(travel_time) and travel_time < float('inf'):
            total_travel_time += travel_time
            valid_teachers += 1
    
    avg_travel_time = total_travel_time / valid_teachers if valid_teachers > 0 else 0
    
    return {
        'name': school.get('name', 'Unknown School'),
        'student_count': school.get('num_students', 0),  # Updated to use 'num_students' to match CSV column
        'teacher_count': total_teachers,
        'male_teachers': male_count,
        'female_teachers': female_count,
        'gender_ratio': f"{male_count}:{female_count}",
        'total_travel_time': total_travel_time,
        'avg_travel_time': avg_travel_time,
        'station': school.get('station', 'Unknown')
    }

def calculate_total_travel_time(teachers_df, schools_df, travel_times_df):
    """
    Calculate total travel time for all teachers to their assigned schools.
    
    Args:
        teachers_df: DataFrame containing teacher information with school assignments
        schools_df: DataFrame containing school information with station UUIDs
        travel_times_df: DataFrame containing travel times between stations
        
    Returns:
        Dictionary with total travel time statistics
    """
    if teachers_df.empty or schools_df.empty or travel_times_df.empty:
        print("One or more DataFrames are empty")
        return {
            'total_travel_time': 0,
            'average_travel_time': 0,
            'teachers_with_assignment': 0,
            'total_teachers': 0
        }
    
    # Create a mapping of school IDs to station UUIDs
    school_to_station = schools_df.set_index('id')['station_uuid'].to_dict()
    
    total_time = 0
    teachers_with_valid_travel = 0
    
    # Debug: Print column names
    print("Teachers columns:", teachers_df.columns.tolist())
    print("Schools columns:", schools_df.columns.tolist())
    print("Travel times columns:", travel_times_df.columns.tolist())
    
    for idx, teacher in teachers_df.iterrows():
        # Skip if no school assignment or station info
        if (pd.isna(teacher.get('school_id')) or 
            teacher.get('school_id') == '' or 
            pd.isna(teacher.get('station_id')) or
            teacher['school_id'] not in school_to_station):
            print(f"Skipping teacher {teacher.get('name')} - missing school or station info")
            continue
            
        # Get station UUIDs
        teacher_station = teacher['station_id']
        school_station = school_to_station[teacher['school_id']]
        
        # Debug info
        print(f"\nProcessing teacher: {teacher.get('name')}")
        print(f"Teacher station: {teacher_station}")
        print(f"School station: {school_station}")
        
        # Get travel time
        travel_time = get_travel_time(
            origin_id=teacher_station,
            dest_id=school_station,
            travel_times_df=travel_times_df
        )
        
        print(f"Travel time: {travel_time} minutes")
        
        # Add to total if valid travel time
        if pd.notna(travel_time) and travel_time < float('inf'):
            total_time += travel_time
            teachers_with_valid_travel += 1
    
    print(f"\nTotal travel time: {total_time} minutes")
    print(f"Teachers with valid travel: {teachers_with_valid_travel}")
    print(f"Total teachers: {len(teachers_df)}")
    
    return {
        'total_travel_time': total_time,
        'average_travel_time': total_time / teachers_with_valid_travel if teachers_with_valid_travel > 0 else 0,
        'teachers_with_assignment': teachers_with_valid_travel,
        'total_teachers': len(teachers_df)
    }
