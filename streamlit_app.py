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
        teachers_df = pd.read_csv(teachers if teachers else data_dir / "teachers_copy.csv")
        schools_df = pd.read_csv(schools if schools else data_dir / "kidsduo_schools.csv")
        travel_times_df = pd.read_csv(stations if stations else data_dir / "station_travel_times.csv")
        return teachers_df, schools_df, travel_times_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
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
        
        # Slack variables for gender balance
        gender_slack = pulp.LpVariable.dicts("gender_slack", 
                                           range(num_schools), 
                                           lowBound=0)
        
        # 1. Primary objective: Minimize total travel time (weighted highest)
        total_distance = pulp.lpSum([distances[i][j] * x[(i, j)] 
                                   for i in range(num_teachers)
                                   for j in range(num_schools)])
        
        # 2. Secondary objective: Minimize gender imbalance
        total_gender_imbalance = pulp.lpSum([gender_slack[j] for j in range(num_schools)])
        
        # Combined objective with weights (travel time is 10x more important than gender balance)
        prob += total_distance * 10 + total_gender_imbalance
        
        # Constraints:
        # 1. Each teacher is assigned to exactly one school
        for i in range(num_teachers):
            prob += pulp.lpSum([x[(i, j)] for j in range(num_schools)]) == 1
        
        # 2. School capacity constraints (4 teachers per 20-30 students)
        for j in range(num_schools):
            school_size = schools_df.iloc[j].get('size', 20)  # Default to 20 if size not specified
            min_teachers = (school_size // 30) * 4
            max_teachers = ((school_size + 29) // 20) * 4  # Round up to nearest 20
            
            # Ensure at least min_teachers and at most max_teachers per school
            prob += pulp.lpSum([x[(i, j)] for i in range(num_teachers)]) >= min_teachers
            prob += pulp.lpSum([x[(i, j)] for i in range(num_teachers)]) <= max_teachers
            
            # 3. At least one bilingual teacher per school
            bilingual_teachers = [i for i, row in teachers_df.iterrows() 
                                if row['type'] == 'Bilingual']
            prob += pulp.lpSum([x[(i, j)] for i in bilingual_teachers]) >= 1
            
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
                    school_options = schools_df['name'].tolist() if 'name' in schools_df.columns else [f"School {i+1}" for i in range(len(schools_df))]
                    
                    # Create a row for each teacher's assignment
                    for idx, row in st.session_state.edited_assignments.iterrows():
                        col1, col2 = st.columns([2, 3])
                        with col1:
                            st.text_input("Teacher", 
                                        value=row['Teacher'], 
                                        key=f"teacher_{idx}", 
                                        disabled=True)
                        with col2:
                            # Default to current school, fallback to first school if not found
                            current_school_idx = school_options.index(row['School']) if row['School'] in school_options else 0
                            st.selectbox("Assigned School", 
                                        school_options, 
                                        index=current_school_idx,
                                        key=f"school_{idx}")
                    
                    # Form submission
                    if st.form_submit_button("💾 Save Manual Assignments"):
                        # Update the edited assignments
                        for idx in range(len(st.session_state.edited_assignments)):
                            teacher = st.session_state[f"teacher_{idx}"]
                            school = st.session_state[f"school_{idx}"]
                            st.session_state.edited_assignments.at[idx, 'School'] = school
                        
                        # Update the main assignments dataframe
                        assignments_df = st.session_state.edited_assignments
                        st.success("✅ Manual assignments saved!")
                        
                        # Recalculate total distance (placeholder - you might want to update this based on actual distances)
                        if 'Distance (km)' in assignments_df.columns:
                            # Update the total_distance in session state to trigger a rerender
                            st.session_state.total_distance = assignments_df['Distance (km)'].sum()
                            total_distance = st.session_state.total_distance
                            
                            # Force a rerun to update the sidebar
                            st.rerun()
                
                # Display the updated assignments with current fitness
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("📋 Updated Assignments")
                with col2:
                    if 'total_distance' in st.session_state:
                        st.metric("Current Fitness", f"{st.session_state.total_distance:.2f} km")
                
                st.dataframe(st.session_state.edited_assignments, use_container_width=True)
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