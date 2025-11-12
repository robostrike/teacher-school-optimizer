import pandas as pd

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
    Check school balance based on teacher assignments.
    
    Rules:
    - Only schools with at least 1 teacher are considered
    - If a school has at least 2 teachers, it should have at least one male and one female teacher
    - If a school has more than 2 teachers, it should have at least one bilingual teacher
    
    Args:
        assignments_df (pd.DataFrame): DataFrame containing teacher-school assignments
        teachers_df (pd.DataFrame): DataFrame containing teacher information
        
    Returns:
        pd.DataFrame: DataFrame with school_id and balance status columns for schools with teachers
    """
    # Define default columns for the result
    result_columns = ['school_id', 'teacher_count', 'has_both_genders', 'has_bilingual']
    
    # Check for required columns
    if assignments_df.empty or 'school_id' not in assignments_df.columns:
        return pd.DataFrame(columns=result_columns)
        
    # Filter out assignments without a school_id
    valid_assignments = assignments_df[assignments_df['school_id'].notna()].copy()
    
    # Return empty if no valid assignments
    if valid_assignments.empty:
        return pd.DataFrame(columns=result_columns)
    
    # Check if we have teacher information
    if teachers_df.empty or 'teacher_id' not in teachers_df.columns:
        # If no teacher info, just return teacher counts
        teacher_counts = valid_assignments['school_id'].value_counts().reset_index()
        teacher_counts.columns = ['school_id', 'teacher_count']
        teacher_counts['has_both_genders'] = 1  # Default pass
        teacher_counts['has_bilingual'] = 1      # Default pass
        return teacher_counts[result_columns]
    
    # Get teacher counts per school
    teacher_counts = valid_assignments['school_id'].value_counts().reset_index()
    teacher_counts.columns = ['school_id', 'teacher_count']
    
    # Only keep schools with at least 1 teacher
    schools_with_teachers = teacher_counts[teacher_counts['teacher_count'] > 0]
    
    if schools_with_teachers.empty:
        return pd.DataFrame(columns=result_columns)
    
    try:
        # Ensure teacher_id is in both dataframes and has the same type
        if 'teacher_id' not in valid_assignments.columns:
            return pd.DataFrame(columns=result_columns)
            
        # Convert teacher_id to string in both dataframes to ensure consistent types
        valid_assignments['teacher_id'] = valid_assignments['teacher_id'].astype(str)
        teachers_df['teacher_id'] = teachers_df['teacher_id'].astype(str)
        
        # Merge with teacher info for schools that have teachers
        merged = valid_assignments.merge(
            teachers_df[['teacher_id', 'gender', 'bilingual']], 
            on='teacher_id', 
            how='left'
        )
        
        # Handle missing values after merge
        merged['gender'] = merged['gender'].fillna('Unknown')
        if 'bilingual' not in merged.columns:
            merged['bilingual'] = False
        else:
            merged['bilingual'] = merged['bilingual'].fillna(False)
        
        # Group by school and check conditions
        def check_balance(group):
            school_id = group.name
            teacher_count = len(group)
            
            # Initialize result with default values
            result = {
                'school_id': school_id,
                'teacher_count': teacher_count,
                'has_both_genders': 0,
                'has_bilingual': 0
            }
            
            if teacher_count >= 2:
                # Check for at least one male and one female teacher
                genders = group['gender'].dropna().unique()
                has_male = any(g.lower() == 'male' for g in genders)
                has_female = any(g.lower() == 'female' for g in genders)
                result['has_both_genders'] = 1 if (has_male and has_female) else 0
                
                # If more than 2 teachers, check for at least one bilingual
                if teacher_count > 2:
                    result['has_bilingual'] = 1 if group['bilingual'].any() else 0
                else:
                    result['has_bilingual'] = 1  # Not required, so pass the check
            else:
                # If less than 2 teachers, pass both checks (not applicable)
                result['has_both_genders'] = 1
                result['has_bilingual'] = 1
                
            return pd.Series(result)
        
        # Only process schools that have teachers and have valid data after merge
        if not merged.empty:
            balance = merged.groupby('school_id').apply(check_balance).reset_index(drop=True)
            return balance[result_columns]  # Ensure consistent column order
            
    except Exception as e:
        print(f"Error in check_school_balance: {str(e)}")
    
    # Return empty result if anything went wrong
    return pd.DataFrame(columns=result_columns)

def get_school_balance_summary(balance_df):
    """
    Generate a summary of school balance statistics.
    
    Args:
        balance_df (pd.DataFrame): DataFrame from check_school_balance function
        
    Returns:
        dict: Dictionary containing balance statistics
    """
    if balance_df.empty:
        return {
            'total_schools': 0,
            'schools_with_2plus_teachers': 0,
            'schools_with_gender_balance': 0,
            'schools_with_bilingual': 0
        }
    
    schools_with_2plus = balance_df[balance_df['teacher_count'] >= 2]
    
    return {
        'total_schools': len(balance_df),
        'schools_with_2plus_teachers': len(schools_with_2plus),
        'schools_with_gender_balance': int(schools_with_2plus['has_both_genders'].sum()),
        'schools_with_bilingual': int(balance_df[balance_df['teacher_count'] > 2]['has_bilingual'].sum())
    }
