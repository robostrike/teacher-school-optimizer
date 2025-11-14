import pandas as pd
import pulp
from typing import Dict, List, Tuple
from data_loader import load_teachers, load_schools_with_locations, load_assignments, get_current_assignments, save_assignments
from datetime import datetime

def get_required_teachers(num_students: int) -> int:
    """Determine required number of teachers based on student count."""
    if num_students >= 25:
        return 4
    elif num_students >= 15:
        return 3
    elif num_students >= 5:
        return 2
    else:
        return 1

def optimize_teacher_assignments() -> Dict[str, List[str]]:
    """
    Optimize teacher assignments to schools based on student counts.
    
    Returns:
        Dictionary mapping school IDs to lists of assigned teacher IDs
    """
    # Load data
    teachers_df = load_teachers()
    schools_df = load_schools_with_locations()
    
    # Filter teachers who are willing to move
    movable_teachers = teachers_df[teachers_df['move'] == True].copy()
    
    # Get current assignments for teachers who aren't moving
    fixed_assignments = {}
    for _, row in teachers_df[teachers_df['move'] == False].iterrows():
        school_id = row.get('school_id')
        if pd.notna(school_id) and school_id != '':
            fixed_assignments.setdefault(school_id, []).append(row['id'])
    
    # Create the optimization problem
    prob = pulp.LpProblem("Teacher_School_Assignment", pulp.LpMaximize)
    
    # Decision variables: x_ij = 1 if teacher i is assigned to school j
    teacher_ids = movable_teachers['id'].tolist()
    school_ids = schools_df['id'].tolist()
    
    # Create variables for each teacher-school pair
    x = pulp.LpVariable.dicts("assign", 
                            ((t, s) for t in teacher_ids for s in school_ids),
                            cat='Binary')
    
    # Add constraints
    
    # 1. Each teacher can be assigned to at most one school
    for t in teacher_ids:
        prob += pulp.lpSum(x[(t, s)] for s in school_ids) <= 1
    
    # 2. Each school must have at least the required number of teachers
    for _, school in schools_df.iterrows():
        school_id = school['id']
        num_students = school.get('num_students', 0)
        required = get_required_teachers(num_students)
        
        # Count fixed assignments for this school
        fixed_count = len(fixed_assignments.get(school_id, []))
        
        # Add constraint for movable teachers
        prob += pulp.lpSum(x[(t, school_id)] for t in teacher_ids) >= max(0, required - fixed_count)
    
    # 3. Objective: Maximize the number of students covered by assigned teachers
    # (This prioritizes assigning to schools with more students first)
    prob += pulp.lpSum(
        (schools_df.loc[schools_df['id'] == s, 'num_students'].iloc[0] or 0) * x[(t, s)]
        for t in teacher_ids
        for s in school_ids
    )
    
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Process results
    assignments = {s: [] for s in school_ids}
    
    # Add fixed assignments
    for school_id, teacher_list in fixed_assignments.items():
        if school_id in assignments:
            assignments[school_id].extend(teacher_list)
    
    # Add optimized assignments
    for t in teacher_ids:
        for s in school_ids:
            if pulp.value(x[(t, s)]) == 1:
                assignments[s].append(t)
    
    return assignments

def update_assignments(assignments: Dict[str, List[str]]) -> None:
    """Update the assignments in the database."""
    assignments_df = load_assignments()
    
    # Create a set of all teacher IDs in the new assignments
    newly_assigned = set()
    for teachers in assignments.values():
        newly_assigned.update(teachers)
    
    # Mark old assignments as not current
    current_assignments = get_current_assignments(assignments_df)
    for _, row in current_assignments.iterrows():
        if row['teacher_id'] in newly_assigned:
            assignments_df = assign_teacher_to_school(
                teacher_id=row['teacher_id'],
                school_id='',  # Unassign
                assignments_df=assignments_df
            )
    
    # Add new assignments
    for school_id, teachers in assignments.items():
        for teacher_id in teachers:
            if teacher_id in newly_assigned:  # Only assign movable teachers
                assignments_df = assign_teacher_to_school(
                    teacher_id=teacher_id,
                    school_id=school_id,
                    assignments_df=assignments_df
                )
    
    # Save the updated assignments
    save_assignments(assignments_df)

def run_optimization():
    """Run the optimization and return the results."""
    try:
        # Run optimization
        optimized_assignments = optimize_teacher_assignments()
        
        # Update the assignments in the database
        update_assignments(optimized_assignments)
        
        # Prepare results for display
        schools_df = load_schools_with_locations()
        results = []
        
        for school_id, teachers in optimized_assignments.items():
            if not teachers:
                continue
                
            school_info = schools_df[schools_df['id'] == school_id].iloc[0]
            num_students = school_info.get('num_students', 0)
            required = get_required_teachers(num_students)
            
            results.append({
                'school_id': school_id,
                'school_name': school_info.get('name', 'Unknown'),
                'num_students': num_students,
                'required_teachers': required,
                'assigned_teachers': len(teachers),
                'teacher_ids': ', '.join(teachers)
            })
        
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        raise

if __name__ == "__main__":
    # This allows the script to be run directly for testing
    results = run_optimization()
    if results is not None:
        print("Optimization completed successfully!")
        print("\nAssignment Summary:")
        print(results[['school_name', 'num_students', 'required_teachers', 'assigned_teachers']])
