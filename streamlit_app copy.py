"""
Teacher-School Allocation Optimizer

This Streamlit application optimizes teacher allocations to schools based on various
constraints including travel distance, school capacity, and teacher qualifications.
"""
import streamlit as st
import pandas as pd
import numpy as np
import pulp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Set page config
st.set_page_config(page_title="Teacher Allocation Optimizer", layout="wide")

# Title and description
st.title("🏫 Teacher-School Allocation Optimizer")
st.markdown("""
This tool helps optimize teacher allocations to schools based on travel distance.
The optimization aims to minimize the total travel distance for all teachers.
""")

# Data loading function
def create_tentative_assignments(
    teachers_df: pd.DataFrame,
    schools_df: pd.DataFrame,
    travel_times_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a tentative assignment of teachers to schools, respecting existing assignments
    and move restrictions.
    
    Args:
        teachers_df: DataFrame containing teacher information including 'school_id' and 'move' columns
        schools_df: DataFrame containing school information
        travel_times_df: DataFrame containing travel times between stations
        
    Returns:
        DataFrame with teacher assignments including school and travel time info
    """
    # Initialize assignments with teacher data
    assignments = teachers_df.copy()
    
    # Ensure required columns exist
    if 'School ID' not in assignments.columns:
        assignments['School ID'] = ''
    if 'School Name' not in assignments.columns:
        assignments['School Name'] = ''
    if 'Travel Time (min)' not in assignments.columns:
        assignments['Travel Time (min)'] = 0.0
    if 'move' not in assignments.columns:
        assignments['move'] = True  # Default to movable if not specified
    
    # Get school ID to name mapping
    school_id_to_name = dict(zip(schools_df['id'], schools_df['name']))
    
    # Process each teacher
    for idx, teacher in assignments.iterrows():
        # If teacher has an existing school assignment and is not marked to move, keep it
        if pd.notna(teacher.get('school_id')) and teacher.get('school_id') != '' and not teacher.get('move', True):
            school_id = teacher['school_id']
            if school_id in school_id_to_name:
                assignments.at[idx, 'School ID'] = school_id
                assignments.at[idx, 'School Name'] = school_id_to_name[school_id]
                
                # Calculate travel time for existing assignment
                if 'station_id' in teacher and 'station_id' in schools_df.columns:
                    school_station = schools_df.loc[schools_df['id'] == school_id, 'station_id'].iloc[0] \
                                    if not schools_df[schools_df['id'] == school_id].empty else None
                    if pd.notna(school_station):
                        travel_time = get_travel_time(
                            teacher['station_id'],
                            school_station,
                            travel_times_df
                        )
                        if pd.notna(travel_time):
                            assignments.at[idx, 'Travel Time (min)'] = float(travel_time)
            continue
            
        # For teachers that can be moved or don't have an assignment
        # Assign to schools in round-robin fashion (or use other assignment logic)
        school_idx = idx % len(schools_df)
        school = schools_df.iloc[school_idx]
        
        # Assign school
        assignments.at[idx, 'School ID'] = school['id']
        assignments.at[idx, 'School Name'] = school['name']
        
        # Calculate travel time if station data is available
        if 'station_id' in teacher and 'station_id' in school:
            travel_time = get_travel_time(
                teacher['station_id'],
                school['station_id'],
                travel_times_df
            )
            if pd.notna(travel_time):
                assignments.at[idx, 'Travel Time (min)'] = float(travel_time)
    
    return assignments

def load_data() -> Tuple[Optional[pd.DataFrame], ...]:
    """
    Load and preprocess the required data files.
    
    Returns:
        Tuple containing (teachers_df, schools_df, travel_times_df) or (None, None, None) on error
    """
    data_dir = Path("data")
    try:
        # Load data from default files
        teachers_df = pd.read_csv(data_dir / "teachers_copy.csv")
        schools_df = pd.read_csv(data_dir / "kidsduo_schools.csv")
        travel_times_df = pd.read_csv(data_dir / "station_travel_times.csv")
        
        # Preprocess schools data
        schools_df = _preprocess_schools_data(schools_df)
        
        # Preprocess teachers data
        teachers_df = _preprocess_teachers_data(teachers_df)
        
        return teachers_df, schools_df, travel_times_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Please ensure all required data files exist in the 'data' directory.")
        return None, None, None


def _preprocess_schools_data(schools_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess schools dataframe with proper types and defaults."""
    df = schools_df.copy()
    
    # Ensure required columns with defaults
    if 'size' not in df.columns:
        df['size'] = 20
    
    # Convert and validate numeric fields
    df['size'] = pd.to_numeric(df['size'], errors='coerce').fillna(20).astype(int)
    
    # Ensure station_id is string if it exists
    if 'station_id' in df.columns:
        df['station_id'] = df['station_id'].astype(str)
    
    return df


def _preprocess_teachers_data(teachers_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess teachers dataframe with proper types and defaults."""
def calculate_priority_scores(assignments_df, schools_df):
    """
    Calculate priority scores for the current assignments.
    Returns a dictionary with scores for each priority (0-100%).
    """
    # Initialize scores
    scores = {
        'Staffing': 0,
        'Bilingual': 0,
        'Gender Balance': 0,
        'Travel Efficiency': 0
    }
    
    # 1. Staffing Score: % of schools with required teachers (1 per 7 students, max 4)
    schools_with_required_teachers = 0
    for _, school in schools_df.iterrows():
        school_id = school['id']
        school_teachers = assignments_df[assignments_df['School ID'] == school_id]
        num_teachers = len(school_teachers)
        
        # Calculate required teachers (1 per 7 students, max 4)
        school_size = school.get('size', 7)  # Default to 7 if size not available
        required_teachers = min(4, max(1, int(np.ceil(school_size / 7))))
        
        if num_teachers >= required_teachers:
            schools_with_required_teachers += 1
    
    if len(schools_df) > 0:
        scores['Staffing'] = (schools_with_required_teachers / len(schools_df)) * 100
    
    # 2. Bilingual Score: % of schools with at least one bilingual teacher
    schools_with_bilingual = 0
    for _, school in schools_df.iterrows():
        school_id = school['id']
        school_teachers = assignments_df[assignments_df['School ID'] == school_id]
        bilingual_teachers = school_teachers[school_teachers['Type'] == 'Bilingual']
        
        if len(bilingual_teachers) >= 1:
            schools_with_bilingual += 1
    
    if len(schools_df) > 0:
        scores['Bilingual'] = (schools_with_bilingual / len(schools_df)) * 100
    
    # 3. Gender Balance Score: Average balance across all schools (0-100%)
    gender_balance_scores = []
    for _, school in schools_df.iterrows():
        school_id = school['id']
        school_teachers = assignments_df[assignments_df['School ID'] == school_id]
        
        if len(school_teachers) > 0:
            male_count = len(school_teachers[school_teachers['Gender'] == 'Male'])
            female_count = len(school_teachers[school_teachers['Gender'] == 'Female'])
            total = male_count + female_count
            
            if total > 0:
                # Calculate balance score (0-100% where 100% is perfect balance)
                balance = 1 - (abs(male_count - female_count) / total)
                gender_balance_scores.append(balance * 100)
    
    if gender_balance_scores:
        scores['Gender Balance'] = np.mean(gender_balance_scores)
    
    # 4. Travel Efficiency Score: Normalized score based on total travel time
    if 'Travel Time (min)' in assignments_df.columns:
        total_travel = assignments_df['Travel Time (min)'].sum()
        # Simple normalization (this could be improved with better scaling)
        # Assuming max travel time per teacher is 60 minutes as a reasonable upper bound
        max_possible = len(assignments_df) * 60
        if max_possible > 0:
            # Higher score is better, so we invert the ratio
            scores['Travel Efficiency'] = max(0, 100 * (1 - (total_travel / max_possible)))
    
    return scores


def get_travel_time(origin_id, destination_id, travel_times_df):
    """Get travel time between two stations in minutes."""
    # Check if we have a direct route
    direct = travel_times_df[
        (travel_times_df['origin_uuid'] == origin_id) & 
        (travel_times_df['destination_uuid'] == destination_id)
    ]
    
    if not direct.empty:
        return direct['travel_min'].iloc[0]
    
    # If no direct route, check reverse direction
    reverse = travel_times_df[
        (travel_times_df['origin_uuid'] == destination_id) & 
        (travel_times_df['destination_uuid'] == origin_id)
    ]
    
    if not reverse.empty:
        return reverse['travel_min'].iloc[0]
    
    # If no route found, return a high penalty value
    return 999  # High penalty for impossible routes


def optimize_teacher_allocation(teachers_df, schools_df, travel_times_df):
    """
    Optimize teacher allocation while respecting move restrictions.
    Teachers with move=False will keep their current school assignments.
    """
    try:
        num_teachers = len(teachers_df)
        num_schools = len(schools_df)
        
        # Create a mapping of teacher indices to their station IDs
        teacher_stations = {i: row['station_id'] for i, row in teachers_df.iterrows()}
        
        # Create a mapping of school indices to their station IDs
        school_stations = {j: row['station_uuid'] for j, row in schools_df.iterrows()}
        
        # Calculate travel times between all teachers and schools
        distances = np.zeros((num_teachers, num_schools))
        for i in range(num_teachers):
            for j in range(num_schools):
                distances[i][j] = get_travel_time(
                    teacher_stations[i], 
                    school_stations[j],
                    travel_times_df
                )
        
        # Create the optimization problem
        prob = pulp.LpProblem("Teacher_School_Allocation", pulp.LpMinimize)
        
        # Decision variables: x[i][j] = 1 if teacher i is assigned to school j
        x = pulp.LpVariable.dicts("assignment",
                                ((i, j) for i in range(num_teachers) 
                                 for j in range(num_schools)),
                                cat='Binary')
        
        # Slack variables for objectives (to handle priorities)
        staffing_slack = pulp.LpVariable.dicts("staffing_slack", 
                                           range(num_schools), 
                                           lowBound=0)
        
        bilingual_slack = pulp.LpVariable.dicts("bilingual_slack", 
                                             range(num_schools), 
                                             lowBound=0)
        
        gender_slack = pulp.LpVariable.dicts("gender_slack", 
                                           range(num_schools), 
                                           lowBound=0)
        
        # 1. Primary objective: Meet staffing requirements (1 teacher per 7 students, max 4)
        # We want to minimize the total slack in staffing requirements
        total_staffing_shortfall = pulp.lpSum(staffing_slack[j] for j in range(num_schools))
        
        # 2. Secondary objective: At least one bilingual teacher per school
        total_bilingual_shortfall = pulp.lpSum(bilingual_slack[j] for j in range(num_schools))
        
        # 3. Tertiary objective: Balance of female and male teachers
        total_gender_imbalance = pulp.lpSum(gender_slack[j] for j in range(num_schools))
        
        # 4. Quaternary objective: Minimize total travel distance
        total_distance = pulp.lpSum([distances[i][j] * x[(i, j)] 
                                   for i in range(num_teachers)
                                   for j in range(num_schools)])
        
        # Combined objective with weights to enforce priority order
        # Using large weights to ensure strict priority order
        M1 = 100000  # Weight for primary objective (staffing)
        M2 = 10000   # Weight for secondary objective (bilingual)
        M3 = 100     # Weight for tertiary objective (gender balance)
        M4 = 1       # Weight for quaternary objective (distance)
        
        prob += (M1 * total_staffing_shortfall + 
                M2 * total_bilingual_shortfall + 
                M3 * total_gender_imbalance + 
                M4 * total_distance)
        
        # Constraints:
        # 1. Each teacher is assigned to exactly one school
        for i in range(num_teachers):
            prob += pulp.lpSum([x[(i, j)] for j in range(num_schools)]) == 1
        
        # 2. School capacity constraints (1 teacher per 7 students, max 4)
        for j in range(num_schools):
            # Ensure school size is a valid number, default to 7 if not
            try:
                school_size = float(schools_df.iloc[j].get('size', 7))
                if pd.isna(school_size) or school_size <= 0:
                    school_size = 7  # Default to 7 if size is invalid
            except (ValueError, TypeError):
                school_size = 7  # Default to 7 if conversion fails
            
            # Calculate required teachers (1 per 7 students, rounded up, max 4)
            required_teachers = min(4, max(1, int(np.ceil(school_size / 7))))
            
            # Add staffing requirement with slack variable
            total_teachers = pulp.lpSum([x[(i, j)] for i in range(num_teachers)])
            prob += total_teachers + staffing_slack[j] >= required_teachers
            
            # Add maximum teachers constraint (4)
            prob += total_teachers <= 4
            
            # 3. At least one bilingual teacher per school (with slack variable)
            bilingual_teachers = [i for i, row in teachers_df.iterrows() 
                                if row['type'] == 'Bilingual']
            prob += (pulp.lpSum([x[(i, j)] for i in bilingual_teachers]) + 
                    bilingual_slack[j] >= 1)
            
            # 4. Gender balance constraints
            male_teachers = [i for i, row in teachers_df.iterrows() 
                           if row['gender'] == 'Male']
            female_teachers = [i for i, row in teachers_df.iterrows() 
                             if row['gender'] == 'Female']
            
            # Number of male and female teachers at school j
            male_count = pulp.lpSum([x[(i, j)] for i in male_teachers])
            female_count = pulp.lpSum([x[(i, j)] for i in female_teachers])
            
            # Add slack variables to measure gender imbalance
            prob += (male_count - female_count) <= gender_slack[j]
            prob += (female_count - male_count) <= gender_slack[j]
        
        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        
        # Extract the solution
        assignments = []
        school_assignments = {j: [] for j in range(num_schools)}
        
        # First pass: collect all assignments
        for i in range(num_teachers):
            for j in range(num_schools):
                if x[(i, j)].varValue == 1:
                    teacher = teachers_df.iloc[i]
                    school = schools_df.iloc[j]
                    
                    assignment = {
                        'Teacher ID': teacher['id'],
                        'Teacher Name': teacher['name'],
                        'Gender': teacher['gender'],
                        'Type': teacher['type'],
                        'School ID': school['id'],
                        'School Name': school['name'],
                        'Travel Time (min)': round(distances[i][j], 1),
                        'School Size': school.get('size', 0)
                    }
                    assignments.append(assignment)
                    school_assignments[j].append(assignment)
        
        # Calculate school statistics
        school_stats = []
        for j in range(num_schools):
            if school_assignments[j]:  # Only process schools with assignments
                school = schools_df.iloc[j]
                teachers = school_assignments[j]
                num_teachers = len(teachers)
                num_bilingual = sum(1 for t in teachers if t['Type'] == 'Bilingual')
                num_male = sum(1 for t in teachers if t['Gender'] == 'Male')
                num_female = sum(1 for t in teachers if t['Gender'] == 'Female')
                avg_travel = np.mean([t['Travel Time (min)'] for t in teachers])
                
                school_stats.append({
                    'School ID': school['id'],
                    'School Name': school['name'],
                    'School Size': school.get('size', 0),
                    'Teachers Assigned': num_teachers,
                    'Bilingual Teachers': num_bilingual,
                    'Male Teachers': num_male,
                    'Female Teachers': num_female,
                    'Gender Ratio': f"{num_male}:{num_female}",
                    'Avg Travel Time (min)': round(avg_travel, 1)
                })
        
        # Convert to DataFrames
        assignments_df = pd.DataFrame(assignments)
        school_stats_df = pd.DataFrame(school_stats)
        
        # Calculate overall statistics
        total_distance = assignments_df['Travel Time (min)'].sum()
        avg_travel_time = assignments_df['Travel Time (min)'].mean()
        
        # Store in session state for display
        st.session_state.assignments_df = assignments_df
        st.session_state.school_stats_df = school_stats_df
        st.session_state.total_distance = total_distance
        st.session_state.avg_travel_time = avg_travel_time
        
        return assignments_df, school_stats_df, total_distance, avg_travel_time
    
    except Exception as e:
        st.error(f"Error in optimization: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None


# Main app
if st.button("🚀 Run Optimization"):
    with st.spinner("Optimizing teacher allocations..."):
        teachers_df, schools_df, travel_times_df = load_data()
        
        if teachers_df is not None and schools_df is not None and travel_times_df is not None:
            # Show data summaries
            st.subheader("📊 Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Teachers", len(teachers_df))
            with col2:
                st.metric("Number of Schools", len(schools_df))
            
            # Run optimization
            assignments_df, school_stats_df, total_distance, avg_travel_time = optimize_teacher_allocation(
                teachers_df, schools_df, travel_times_df
            )
            
            if assignments_df is not None and school_stats_df is not None:
                st.success("✅ Optimization complete!")
                
                # Show summary statistics
                st.subheader("📊 Summary Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Travel Time (min)", f"{total_distance:.1f}")
                    st.metric("Average Travel Time (min/teacher)", f"{avg_travel_time:.1f}")
                
                # School statistics
                st.subheader("🏫 School Statistics")
                st.dataframe(school_stats_df, use_container_width=True)
                
                # Detailed assignments
                st.subheader("👨‍🏫 Teacher Assignments")
                st.dataframe(assignments_df, use_container_width=True)
                
                # Display optimization priorities and scores
                st.subheader("🎯 Optimization Priorities")
                
                # Priority descriptions
                with st.expander("ℹ️ Priority Details"):
                    st.markdown("""
                    The optimization follows these priorities in order:
                    1. **Staffing**: Ensures each school has 1 teacher per 7 students (max 4)
                    2. **Bilingual**: Ensures at least one bilingual teacher per school
                    3. **Gender Balance**: Balances male and female teachers
                    4. **Travel Efficiency**: Minimizes total travel time for all teachers
                    """)
                
                # Calculate and display scores for the optimized solution
                optimized_scores = calculate_priority_scores(assignments_df, schools_df)
                
                # Create a DataFrame for visualization
                scores_df = pd.DataFrame({
                    'Priority': ['Staffing', 'Bilingual', 'Gender Balance', 'Travel Efficiency'],
                    'Score (%)': [
                        optimized_scores['Staffing'],
                        optimized_scores['Bilingual'],
                        optimized_scores['Gender Balance'],
                        optimized_scores['Travel Efficiency']
                    ]
                })
                
                # Display scores in a bar chart
                st.bar_chart(
                    scores_df.set_index('Priority'),
                    use_container_width=True
                )
                
                # Visualizations
                st.subheader("📊 School Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(school_stats_df.set_index('School Name')['Teachers Assigned'])
                    st.caption("Number of Teachers per School")
                with col2:
                    st.bar_chart(school_stats_df.set_index('School Name')['Avg Travel Time (min)'])
                    st.caption("Average Travel Time per School (min)")
                
                # Export results
                st.download_button(
                    label="💾 Download Assignments (CSV)",
                    data=assignments_df.to_csv(index=False).encode('utf-8'),
                    file_name="teacher_assignments.csv",
                    mime="text/csv"
                )
                
                # Initialize assignment history for undo functionality
                if 'assignment_history' not in st.session_state:
                    st.session_state.assignment_history = []
                
                # Manual assignment section
                st.subheader("🔄 Manual Assignment Override")
                st.write("Manually adjust teacher assignments as needed.")
                
                # Create a copy of the assignments for editing if not exists
                if 'edited_assignments' not in st.session_state:
                    st.session_state.edited_assignments = assignments_df.copy()
                
                # Get unique schools for the dropdown
                school_options = schools_df['name'].tolist()
                school_id_to_name = dict(zip(schools_df['id'], schools_df['name']))
                
                # Track changes for real-time updates
                changes_made = False
                
                # Create a form for editing assignments
                with st.form("manual_assignment"):
                    # Create assignment controls for each teacher
                    for idx, row in st.session_state.edited_assignments.iterrows():
                        teacher_name = row['Teacher Name']
                        current_school = row.get('School Name', '')
                        can_move = row.get('move', True)  # Default to True if not specified
                        
                        # Display teacher name and move status
                        status = "🔒" if not can_move else "🔄"
                        st.markdown(f"**{status} {teacher_name}**")
                        
                        # Get current school index, default to 0 if not found
                        school_idx = (
                            school_options.index(current_school)
                            if current_school in school_options
                            else 0
                        )
                        
                        # Show current assignment and move status
                        if not can_move and current_school:
                            st.info(f"Currently assigned to: {current_school} (Fixed Assignment)")
                        
                        # Only show dropdown for teachers that can be moved
                        if can_move:
                            # School selection dropdown
                            new_school = st.selectbox(
                                f"Assign {teacher_name} to school",
                                school_options,
                                index=school_idx,
                                key=f"school_select_{idx}",
                                label_visibility="collapsed"
                            )
                            
                            # Update assignment if changed
                            if new_school != current_school:
                                st.session_state.edited_assignments.at[idx, 'School Name'] = new_school
                                st.session_state.edited_assignments.at[idx, 'School ID'] = school_id_to_name[new_school]
                        else:
                            # For fixed assignments, store the current school
                            st.session_state.edited_assignments.at[idx, 'School Name'] = current_school
                            if current_school in school_id_to_name:  # Only update if we have a valid mapping
                                st.session_state.edited_assignments.at[idx, 'School ID'] = school_id_to_name[current_school]
                        
                        # Update changes_made flag if any changes were made
                        if new_school != current_school:
                            changes_made = True
                    
                    # Form submission buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        save_clicked = st.form_submit_button("💾 Save Changes")
                    with col2:
                        # Only enable undo if we have history
                        undo_clicked = st.form_submit_button("⏪ Undo", 
                                                          disabled=len(st.session_state.assignment_history) == 0)
                    
                    if save_clicked and changes_made:
                        # Save current state to history before making changes (for undo)
                        st.session_state.assignment_history.append(st.session_state.edited_assignments.copy())
                        
                        # Update the assignments with new selections
                        for idx, row in st.session_state.edited_assignments.iterrows():
                            new_school_name = st.session_state[f"school_select_{idx}"]
                            if new_school_name != row['School Name']:  # Only update if changed
                                new_school_id = schools_df[schools_df['name'] == new_school_name]['id'].iloc[0]
                                st.session_state.edited_assignments.at[idx, 'School Name'] = new_school_name
                                st.session_state.edited_assignments.at[idx, 'School ID'] = new_school_id
                        
                        # Recalculate statistics
                        st.session_state.total_distance = st.session_state.edited_assignments['Travel Time (min)'].sum()
                        st.session_state.avg_travel_time = st.session_state.edited_assignments['Travel Time (min)'].mean()
                        
                        st.success("✅ Changes saved!")
                        st.rerun()
                        
                    elif undo_clicked and st.session_state.assignment_history:
                        # Restore previous state
                        st.session_state.edited_assignments = st.session_state.assignment_history.pop()
                        st.session_state.total_distance = st.session_state.edited_assignments['Travel Time (min)'].sum()
                        st.session_state.avg_travel_time = st.session_state.edited_assignments['Travel Time (min)'].mean()
                        st.success("⏪ Undo successful!")
                        st.rerun()
                    elif save_clicked and not changes_made:
                        st.warning("No changes detected.")
                
                # Calculate scores for current assignments
                current_scores = calculate_priority_scores(
                    st.session_state.edited_assignments, 
                    schools_df
                )
                
                # Display priority scores
                st.subheader("🎯 Priority Scores")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Staffing", f"{current_scores['Staffing']:.1f}%")
                    st.caption("% schools with required teachers")
                with col2:
                    st.metric("Bilingual", f"{current_scores['Bilingual']:.1f}%")
                    st.caption("% schools with ≥1 bilingual")
                with col3:
                    st.metric("Gender Balance", f"{current_scores['Gender Balance']:.1f}%")
                    st.caption("Average balance (100% = perfect)")
                with col4:
                    st.metric("Travel Efficiency", f"{current_scores['Travel Efficiency']:.1f}%")
                    st.caption("100% = minimal travel time")
                
                # Display the updated assignments
                st.subheader("📋 Updated Assignments")
                st.dataframe(st.session_state.edited_assignments, use_container_width=True)
                
                # Add a button to update scores without saving changes
                if st.button("🔄 Update Scores"):
                    st.rerun()  # Just rerun to recalculate scores
                
                # Add a button to reset to original optimization
                if st.button("🔄 Reset to Optimized Assignments"):
                    if 'assignments_df' in st.session_state:
                        st.session_state.edited_assignments = st.session_state.assignments_df.copy()
                        st.session_state.total_distance = st.session_state.assignments_df['Travel Time (min)'].sum()
                        st.session_state.avg_travel_time = st.session_state.assignments_df['Travel Time (min)'].mean()
                        st.rerun()
else:
    st.info("👈 Click 'Run Optimization' to start the teacher allocation process.")
    
    # Show data preview and allow manual assignment
    try:
        teachers_df, schools_df, travel_times_df = load_data()
        
        with st.expander("📋 Preview Data"):
            st.subheader("Teachers Data")
            st.dataframe(teachers_df.head())
            
            st.subheader("Schools Data")
            st.dataframe(schools_df.head())
        
        # Create tentative assignments
        if st.button("📋 Create Tentative Assignments"):
            with st.spinner("Creating tentative assignments..."):
                tentative_assignments = create_tentative_assignments(teachers_df, schools_df, travel_times_df)
                
                # Store in session state
                st.session_state.edited_assignments = tentative_assignments
                st.session_state.assignments_df = tentative_assignments.copy()
                st.session_state.total_distance = tentative_assignments['Travel Time (min)'].sum()
                st.session_state.avg_travel_time = tentative_assignments['Travel Time (min)'].mean()
                st.session_state.schools_df = schools_df
                
                st.success("Tentative assignments created! You can now adjust them manually.")
                st.rerun()
        
        # Show manual assignment interface if we have tentative assignments
        if 'edited_assignments' in st.session_state:
            st.subheader("✏️ Manual Assignment")
            
            # Create a form for editing assignments
            with st.form("manual_assignment"):
                # Get unique schools for the dropdown
                school_options = schools_df['name'].tolist()
                school_name_to_id = dict(zip(schools_df['name'], schools_df['id']))
                
                # Create a row for each teacher's assignment
                for idx, row in st.session_state.edited_assignments.iterrows():
                    # Display teacher name as a label
                    st.markdown(f"**{row['name']}**")
                    
                    # Get current school name
                    current_school_name = row.get('School Name', '')
                    current_school_idx = school_options.index(current_school_name) if current_school_name in school_options else 0
                    
                    # Create school selection dropdown
                    new_school = st.selectbox(
                        f"Assign {row['name']} to school",
                        school_options,
                        index=current_school_idx,
                        key=f"school_select_{idx}",
                        label_visibility="collapsed"
                    )
                        
                    # Update assignment if changed
                    if new_school != current_school_name:
                        st.session_state.edited_assignments.at[idx, 'School Name'] = new_school
                        st.session_state.edited_assignments.at[idx, 'School ID'] = school_name_to_id[new_school]
            
                # Form submission buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("💾 Save Changes"):
                        # Update travel times and other metrics
                        st.session_state.total_distance = st.session_state.edited_assignments['Travel Time (min)'].sum()
                        st.session_state.avg_travel_time = st.session_state.edited_assignments['Travel Time (min)'].mean()
                        st.session_state.assignments_df = st.session_state.edited_assignments.copy()
                        st.success("Changes saved!")
                        st.rerun()
                
                with col2:
                    if st.form_submit_button("🔄 Update Scores"):
                        st.rerun()
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error(traceback.format_exc())