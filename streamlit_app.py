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

# Main optimization function
def optimize_teacher_allocation(teachers_df, schools_df, travel_times_df):
    try:
        # Create a distance matrix (simple example - you'll need to adapt this)
        # This is a placeholder - you'll need to implement the actual distance calculation
        # based on your travel_times_df structure
        num_teachers = len(teachers_df)
        num_schools = len(schools_df)
        
        # Simple distance matrix (replace with actual calculation from travel_times_df)
        distances = np.random.rand(num_teachers, num_schools) * 100  # Random distances for demo
        
        # Create the optimization problem
        prob = pulp.LpProblem("Teacher_School_Allocation", pulp.LpMinimize)
        
        # Decision variables
        x = pulp.LpVariable.dicts("assignment",
                                ((i, j) for i in range(num_teachers) 
                                 for j in range(num_schools)),
                                cat='Binary')
        
        # Objective function: minimize total distance
        prob += pulp.lpSum([distances[i][j] * x[(i, j)] 
                          for i in range(num_teachers)
                          for j in range(num_schools)])
        
        # Constraints:
        # 1. Each teacher is assigned to exactly one school
        for i in range(num_teachers):
            prob += pulp.lpSum([x[(i, j)] for j in range(num_schools)]) == 1
            
        # 2. School capacity constraints (assuming each school can handle all teachers for now)
        #    You can add specific school capacities here if available
        
        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        
        # Extract the solution
        assignments = []
        for i in range(num_teachers):
            for j in range(num_schools):
                if x[(i, j)].varValue == 1:
                    assignments.append({
                        'Teacher': teachers_df.iloc[i]['name'] if 'name' in teachers_df.columns else f'Teacher {i+1}',
                        'School': schools_df.iloc[j]['name'] if 'name' in schools_df.columns else f'School {j+1}',
                        'Distance (km)': round(distances[i][j], 2)
                    })
        
        return pd.DataFrame(assignments), prob.objective.value()
    
    except Exception as e:
        st.error(f"Error in optimization: {e}")
        return None, None

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
            assignments_df, total_distance = optimize_teacher_allocation(
                teachers_df, schools_df, travel_times_df
            )
            
            if assignments_df is not None:
                st.success("✅ Optimization complete!")
                
                # Show results
                st.subheader("📋 Optimal Assignments")
                st.dataframe(assignments_df, use_container_width=True)
                
                st.subheader("📈 Summary Statistics")
                st.metric("Total Travel Distance (km)", f"{total_distance:.2f}" if total_distance else "N/A")
                
                # Visualizations
                st.subheader("📊 Assignment Distribution")
                school_counts = assignments_df['School'].value_counts()
                st.bar_chart(school_counts)
                
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