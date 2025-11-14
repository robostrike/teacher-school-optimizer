import pandas as pd
import pulp
import numpy as np
from typing import Dict, List, Tuple, Optional
from data_loader import (
    load_teachers, 
    load_schools_with_locations, 
    load_assignments, 
    get_current_assignments, 
    save_assignments,
    assign_teacher_to_school,
    load_travel_times,
    get_travel_time
)
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

def get_teacher_station(teacher_id: str, teachers_df: pd.DataFrame) -> Optional[str]:
    """Get the station UUID for a teacher."""
    teacher = teachers_df[teachers_df['id'] == teacher_id]
    if not teacher.empty and 'station_uuid' in teacher.columns:
        return teacher['station_uuid'].iloc[0]
    return None

def get_school_station(school_id: str, schools_df: pd.DataFrame) -> Optional[str]:
    """Get the station UUID for a school."""
    school = schools_df[schools_df['id'] == school_id]
    if not school.empty and 'station_uuid' in school.columns:
        return school['station_uuid'].iloc[0]
    return None

def optimize_teacher_assignments() -> Dict[str, List[str]]:
    """
    Optimize teacher assignments to schools based on student counts and travel time.
    
    Returns:
        Dictionary mapping school IDs to lists of assigned teacher IDs
    """
    # Load data
    teachers_df = load_teachers()
    schools_df = load_schools_with_locations()
    travel_times_df = load_travel_times()
    
    # Filter teachers who are willing to move
    movable_teachers = teachers_df[teachers_df['move'] == True].copy()
    
    # Get current assignments for teachers who aren't moving
    fixed_assignments = {}
    for _, row in teachers_df[teachers_df['move'] == False].iterrows():
        school_id = row.get('school_id')
        if pd.notna(school_id) and school_id != '':
            fixed_assignments.setdefault(school_id, []).append(row['id'])
    
    # Create the optimization problem
    prob = pulp.LpProblem("Teacher_School_Assignment", pulp.LpMinimize)  # Minimize total travel time
    
    # Decision variables: x_ij = 1 if teacher i is assigned to school j
    teacher_ids = movable_teachers['id'].tolist()
    school_ids = schools_df['id'].tolist()
    
    # Create variables for each teacher-school pair
    x = pulp.LpVariable.dicts(
        "assign", 
        ((t, s) for t in teacher_ids for s in school_ids),
        cat='Binary'
    )
    
    # Create a dictionary to store travel times
    travel_costs = {}
    
    # Pre-calculate travel times
    for t in teacher_ids:
        teacher_station = get_teacher_station(t, teachers_df)
        if not teacher_station:
            continue
            
        for s in school_ids:
            school_station = get_school_station(s, schools_df)
            if not school_station:
                continue
                
            # Get travel time in minutes (or a large number if no route exists)
            travel_time = get_travel_time(teacher_station, school_station, travel_times_df)
            travel_costs[(t, s)] = travel_time if travel_time < float('inf') else 1000  # Large penalty for unreachable
    
    # Add constraints
    
    # 1. Each teacher can be assigned to at most one school
    for t in teacher_ids:
        prob += pulp.lpSum(x[(t, s)] for s in school_ids) <= 1
    
    # 2. Each school must have at least the required number of teachers and at least one teacher
    for _, school in schools_df.iterrows():
        school_id = school['id']
        num_students = school.get('num_students', 0)
        required = get_required_teachers(num_students)
        
        # Count fixed assignments for this school
        fixed_count = len(fixed_assignments.get(school_id, []))
        
        # Constraint 2a: Each school must have at least the required number of teachers
        prob += pulp.lpSum(x[(t, school_id)] for t in teacher_ids) >= max(0, required - fixed_count)
        
        # Constraint 2b: Each school must have at least one teacher (either fixed or assigned)
        if fixed_count == 0:  # Only add this constraint if there are no fixed assignments
            prob += pulp.lpSum(x[(t, school_id)] for t in teacher_ids) >= 1
    
    # 3. Create binary variables for school coverage
    school_covered = pulp.LpVariable.dicts("covered", school_ids, cat='Binary')
    for s in school_ids:
        fixed_count = len(fixed_assignments.get(s, []))
        if fixed_count == 0:
            # School is covered if it has at least one assigned teacher
            prob += school_covered[s] <= pulp.lpSum(x[(t, s)] for t in teacher_ids)
        else:
            # Schools with fixed assignments are always considered covered
            prob += school_covered[s] == 1

    # 4. Three-tiered objective
    # Priority 1: Maximize number of schools with at least one teacher (highest weight)
    # Priority 2: Minimize total travel time (medium weight)
    # Priority 3: Maximize student coverage (lowest weight)

    # Weights (must be ordered: w1 >> w2 >> w3)
    w1 = 100000  # Weight for school coverage (highest priority)
    w2 = 1000    # Weight for travel time
    w3 = 1       # Weight for student coverage

    # Calculate maximum possible values for normalization
    max_students = schools_df['num_students'].max() if not schools_df.empty else 1
    max_travel = max(travel_costs.values()) if travel_costs else 1

    # Three-tiered objective
    prob += (
        # Priority 1: Maximize number of schools with at least one teacher
        -w1 * pulp.lpSum(1 - school_covered[s] for s in school_ids) +
        
        # Priority 2: Minimize total travel time
        w2 * pulp.lpSum(
            (travel_costs.get((t, s), 1000) / max_travel) * x[(t, s)]
            for t in teacher_ids 
            for s in school_ids
            if (t, s) in travel_costs
        ) +
        
        # Priority 3: Maximize student coverage (negative because we're minimizing)
        -w3 * pulp.lpSum(
            (schools_df.loc[schools_df['id'] == s, 'num_students'].iloc[0] / max_students) * x[(t, s)]
            for t in teacher_ids
            for s in school_ids
        )
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
    """Run the optimization and return the results with travel time information."""
    try:
        # Load data needed for travel time calculation
        teachers_df = load_teachers()
        schools_df = load_schools_with_locations()
        travel_times_df = load_travel_times()
        
        # Run optimization
        optimized_assignments = optimize_teacher_assignments()
        
        # Update the assignments in the database
        update_assignments(optimized_assignments)
        
        # Prepare results for display
        results = []
        
        # Get all school IDs from the schools DataFrame
        all_school_ids = schools_df['id'].unique()
        
        # Process all schools, not just those with assignments
        for school_id in all_school_ids:
            school_info = schools_df[schools_df['id'] == school_id].iloc[0]
            num_students = school_info.get('num_students', 0)
            required = get_required_teachers(num_students)
            
            # Get teachers assigned to this school (if any)
            teachers = optimized_assignments.get(school_id, [])
            
            # Calculate travel times for assigned teachers
            travel_times = []
            for teacher_id in teachers:
                teacher_station = get_teacher_station(teacher_id, teachers_df)
                school_station = get_school_station(school_id, schools_df)
                
                if teacher_station and school_station:
                    travel_time = get_travel_time(teacher_station, school_station, travel_times_df)
                    if travel_time < float('inf'):
                        travel_times.append(travel_time)
            
            # Calculate average travel time (0 if no teachers assigned)
            avg_travel_time = sum(travel_times) / len(travel_times) if travel_times else 0
            
            results.append({
                'school_id': school_id,
                'school_name': school_info.get('name', 'Unknown'),
                'num_students': num_students,
                'required_teachers': required,
                'assigned_teachers': len(teachers),
                'teacher_ids': ', '.join(teachers) if teachers else '',
                'avg_travel_time': avg_travel_time,
                'total_travel_time': sum(travel_times) if travel_times else 0
            })
        
        # Calculate overall statistics
        if results:
            total_travel = sum(r['total_travel_time'] for r in results)
            avg_travel = total_travel / sum(r['assigned_teachers'] for r in results) if any(r['assigned_teachers'] > 0 for r in results) else 0
            
            # Add summary row
            results.append({
                'school_id': 'TOTAL',
                'school_name': 'ALL SCHOOLS',
                'num_students': sum(r['num_students'] for r in results if r['school_id'] != 'TOTAL'),
                'required_teachers': sum(r['required_teachers'] for r in results if r['school_id'] != 'TOTAL'),
                'assigned_teachers': sum(r['assigned_teachers'] for r in results if r['school_id'] != 'TOTAL'),
                'teacher_ids': '',
                'avg_travel_time': avg_travel,
                'total_travel_time': total_travel
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
