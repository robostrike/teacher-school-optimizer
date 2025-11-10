import streamlit as st
import pandas as pd
import numpy as np
import pulp
from pathlib import Path

# Set page config
st.set_page_config(page_title="Teacher Allocation Optimizer", layout="wide")

# Title and description
st.title("🏫 Teacher-School Allocation Optimizer")
st.markdown("""
This tool helps optimize teacher allocations to schools based on travel distance.
The optimization aims to minimize the total travel distance for all teachers.
""")

# Data loading function
def load_data():
    data_dir = Path("data")
    try:
        # Load data
        teachers_df = pd.read_csv(teachers if teachers else data_dir / "teachers_copy.csv")
        schools_df = pd.read_csv(schools if schools else data_dir / "kidsduo_schools.csv")
        travel_times_df = pd.read_csv(stations if stations else data_dir / "station_travel_times.csv")
        
        # Ensure required columns exist and have proper types
        if 'size' not in schools_df.columns:
            schools_df['size'] = 20  # Default size if not present
        
        # Convert size to numeric, coerce errors to NaN, then fill with default
        schools_df['size'] = pd.to_numeric(schools_df['size'], errors='coerce').fillna(20).astype(int)
        
        # Ensure station IDs are strings
        if 'station_id' in teachers_df.columns:
            teachers_df['station_id'] = teachers_df['station_id'].astype(str)
        
        return teachers_df, schools_df, travel_times_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

# File uploaders in sidebar
with st.sidebar:
    st.header("📂 Data Upload")
    st.caption("Upload custom files or use the default dataset")
    teachers = st.file_uploader("Teachers Data (CSV)", type="csv")
    schools = st.file_uploader("Schools Data (CSV)", type="csv")
    stations = st.file_uploader("Travel Times (CSV)", type="csv")
    
    # Fitness value display
    if 'total_distance' in locals() or 'total_distance' in globals():
        st.metric("Current Fitness (Total Distance)", 
                 f"{total_distance:.2f} km" if 'total_distance' in locals() or 'total_distance' in globals() else "N/A",
                 delta=None)
    
    if st.button("🔍 View Sample Data"):
        teachers_df, schools_df, _ = load_data()
        if teachers_df is not None and schools_df is not None:
            st.subheader("Sample Teachers Data")
            st.dataframe(teachers_df.head())
            st.subheader("Sample Schools Data")
            st.dataframe(schools_df.head())

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
                
                # Visualizations
                st.subheader("📈 Assignment Distribution")
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
                
                # Manual assignment section
                st.subheader("🔄 Manual Assignment Override")
                st.write("Manually adjust teacher assignments as needed.")
                
                # Create a copy of the assignments for editing
                if 'edited_assignments' not in st.session_state:
                    st.session_state.edited_assignments = assignments_df.copy()
                
                # Create a form for editing assignments
                with st.form("manual_assignment"):
                    # Get unique schools for the dropdown
                    school_options = schools_df['name'].tolist()
                    school_id_to_name = dict(zip(schools_df['id'], schools_df['name']))
                    
                    # Create a row for each teacher's assignment
                    for idx, row in st.session_state.edited_assignments.iterrows():
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            st.text_input("Teacher", 
                                        value=f"{row['Teacher Name']} ({row['Teacher ID']})", 
                                        key=f"teacher_{idx}", 
                                        disabled=True)
                        with col2:
                            # Get current school name, default to first school if not found
                            current_school_name = school_id_to_name.get(row['School ID'], school_options[0])
                            current_school_idx = school_options.index(current_school_name) if current_school_name in school_options else 0
                            
                            # Create school selection dropdown
                            new_school = st.selectbox(
                                "Assigned School",
                                school_options,
                                index=current_school_idx,
                                key=f"school_{idx}",
                                label_visibility="collapsed"
                            )
                    
                    # Add submit button for the form
                    submitted = st.form_submit_button("💾 Save Manual Assignments")
                    
                    if submitted:
                        # Update the edited assignments
                        for idx, row in st.session_state.edited_assignments.iterrows():
                            # Get the new school ID from the selected school name
                            new_school_name = st.session_state[f"school_{idx}"]
                            new_school_id = schools_df[schools_df['name'] == new_school_name]['id'].iloc[0]
                            
                            # Update the assignment
                            st.session_state.edited_assignments.at[idx, 'School Name'] = new_school_name
                            st.session_state.edited_assignments.at[idx, 'School ID'] = new_school_id
                        
                        # Update the main assignments dataframe
                        assignments_df = st.session_state.edited_assignments
                        
                        # Recalculate statistics
                        st.session_state.total_distance = assignments_df['Travel Time (min)'].sum()
                        st.session_state.avg_travel_time = assignments_df['Travel Time (min)'].mean()
                        
                        st.success("✅ Manual assignments saved!")
                        st.rerun()
                
                # Display the updated assignments with current fitness
                st.subheader("📋 Updated Assignments")
                
                # Show summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Travel Time", f"{st.session_state.get('total_distance', 0):.1f} min")
                with col2:
                    st.metric("Avg Travel Time", f"{st.session_state.get('avg_travel_time', 0):.1f} min/teacher")
                
                # Display the updated assignments
                st.dataframe(st.session_state.edited_assignments, use_container_width=True)
                
                # Add a button to reset to original optimization
                if st.button("🔄 Reset to Optimized Assignments"):
                    if 'assignments_df' in st.session_state:
                        st.session_state.edited_assignments = st.session_state.assignments_df.copy()
                        st.session_state.total_distance = st.session_state.assignments_df['Travel Time (min)'].sum()
                        st.session_state.avg_travel_time = st.session_state.assignments_df['Travel Time (min)'].mean()
                        st.rerun()
else:
    st.info("👈 Upload your data files or use the default dataset in the sidebar, then click 'Run Optimization'.")
    
    # Show data preview if available
    teachers_df, schools_df, _ = load_data()
    if teachers_df is not None and schools_df is not None:
        with st.expander("📋 Preview Data"):
            st.subheader("Teachers Data")
            st.dataframe(teachers_df.head())
            st.subheader("Schools Data")
            st.dataframe(schools_df.head())