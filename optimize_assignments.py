import pandas as pd
import pulp
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Optional
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
    print("\n" + "="*80)
    print("STARTING OPTIMIZATION")
    print("="*80)
    # Load data
    print("\n[1/6] Loading data...")
    teachers_df = load_teachers()
    schools_df = load_schools_with_locations()
    travel_times_df = load_travel_times()
    
    # Print data summary
    print(f"  • Loaded {len(teachers_df)} teachers ({len(teachers_df[teachers_df['move'] == False])} fixed, {len(teachers_df[teachers_df['move'] == True])} movable)")
    print(f"  • Loaded {len(schools_df)} schools with student counts: {schools_df['num_students'].sum()} total students")
    print(f"  • Loaded travel times between {len(travel_times_df)} station pairs")
    
    # Filter teachers who are willing to move
    movable_teachers = teachers_df[teachers_df['move'] == True].copy()
    teacher_ids = movable_teachers['id'].tolist()
    school_ids = schools_df['id'].tolist()
    
    print(f"\n[2/6] Processing {len(teacher_ids)} movable teachers and {len(school_ids)} schools")
    
    # Get current assignments for teachers who aren't moving
    fixed_assignments = {}
    for _, row in teachers_df[teachers_df['move'] == False].iterrows():
        school_id = row.get('school_id')
        if pd.notna(school_id) and school_id != '':
            fixed_assignments.setdefault(school_id, []).append(row['id'])
    
    # Print fixed assignments
    if fixed_assignments:
        print(f"\n[3/6] Found {sum(len(v) for v in fixed_assignments.values())} fixed assignments:")
        for school_id, teachers in fixed_assignments.items():
            school_name = schools_df[schools_df['id'] == school_id]['name'].iloc[0] if not schools_df[schools_df['id'] == school_id].empty else 'Unknown'
            print(f"  • {school_name} ({school_id}): {len(teachers)} teachers")
    else:
        print("\n[3/6] No fixed assignments found")
    
    # Create the optimization problem
    print("\n[4/6] Setting up optimization problem...")
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
    
    # 1. Each teacher must be assigned to exactly one school
    for t in teacher_ids:
        prob += pulp.lpSum(x[(t, s)] for s in school_ids) == 1
    
    # 2. School assignment constraints
    for _, school in schools_df.iterrows():
        school_id = school['id']
        num_students = school.get('num_students', 0)
        required = get_required_teachers(num_students)
        
        # Count fixed assignments for this school
        fixed_count = len(fixed_assignments.get(school_id, []))
        
        # Total teachers (fixed + assigned)
        total_teachers = pulp.lpSum(x[(t, school_id)] for t in teacher_ids) + fixed_count
        
        # School must have at least one teacher (hard constraint)
        prob += total_teachers >= 1
        
        # School cannot have more than 4 teachers
        prob += total_teachers <= 4
        
        # If there are fixed assignments, ensure those teachers stay assigned
        for t in fixed_assignments.get(school_id, []):
            if t in teacher_ids:  # If this is a movable teacher (shouldn't happen, but just in case)
                prob += x[(t, school_id)] == 1  # Force the assignment
    
    # 3. Create binary variables for school coverage
    school_covered = pulp.LpVariable.dicts("covered", school_ids, cat='Binary')
    for s in school_ids:
        fixed_count = len(fixed_assignments.get(s, []))
        if fixed_count > 0:
            # Schools with fixed assignments are always considered covered
            prob += school_covered[s] == 1
        else:
            # School is covered if it has at least one assigned teacher
            prob += school_covered[s] <= pulp.lpSum(x[(t, s)] for t in teacher_ids)
            # Ensure at least one teacher is assigned if the school is covered
            prob += school_covered[s] * len(teacher_ids) >= pulp.lpSum(x[(t, s)] for t in teacher_ids)

    # 4. Three-tiered objective
    # Priority 1: Maximize number of schools with at least one teacher (highest weight)
    # Priority 2: Minimize total travel time (medium weight)
    # Priority 3: Maximize student coverage (lowest weight)

    # Weights (must be ordered: w1 >> w2 >> w3 >> w4)
    # Higher weights mean higher priority
    w1 = 1000000   # Weight for school coverage (highest priority)
    w2 = 10000     # Weight for teacher distribution (second priority)
    w3 = 1         # Weight for travel time (lowest priority)
    w4 = 100000    # Weight for respecting fixed assignments (very high priority)

    # Calculate maximum possible values for normalization
    max_students = schools_df['num_students'].max() if not schools_df.empty else 1
    max_travel = max(travel_costs.values()) if travel_costs else 1

    # Calculate required teachers for each school
    school_requirements = {}
    for school_id in school_ids:
        num_students = schools_df[schools_df['id'] == school_id]['num_students'].iloc[0]
        school_requirements[school_id] = get_required_teachers(num_students)
    
    # Create objective function components
    objective = []
    
    # Priority 1: Maximize number of schools with at least one teacher
    # This is now a hard constraint, but we'll keep it in the objective with high weight
    objective.append(-w1 * pulp.lpSum(1 - school_covered[s] for s in school_ids))
    
    # Priority 2: Balance teacher distribution (minimize deviation from required)
    # Create deviation variables for each school
    deviation_pos = pulp.LpVariable.dicts("dev_pos", school_ids, lowBound=0, cat='Continuous')
    deviation_neg = pulp.LpVariable.dicts("dev_neg", school_ids, lowBound=0, cat='Continuous')
    
    for school_id in school_ids:
        fixed_count = len(fixed_assignments.get(school_id, []))
        required = school_requirements[school_id]
        
        # Total teachers (fixed + assigned)
        total_teachers = fixed_count + pulp.lpSum(x[(t, school_id)] for t in teacher_ids)
        
        # Deviation constraints
        prob += (total_teachers - required) == (deviation_pos[school_id] - deviation_neg[school_id])
    
    # Add the balance term to the objective (minimize total absolute deviation)
    # Using absolute deviation instead of squared to avoid non-linear terms
    objective.append(w2 * pulp.lpSum(deviation_pos[s] + deviation_neg[s] for s in school_ids))
    
    # Priority 3: Respect fixed assignments
    # This adds a penalty for moving teachers with fixed assignments
    for school_id, teachers in fixed_assignments.items():
        for t in teachers:
            # If this is a movable teacher (shouldn't happen, but just in case)
            if t in teacher_ids:
                # Add a large penalty if the teacher is not assigned to their fixed school
                objective.append(w4 * (1 - x[(t, school_id)]))
    
    # Priority 4: Minimize total travel time
    objective.append(w3 * pulp.lpSum(
        (travel_costs.get((t, s), 1000) / max_travel) * x[(t, s)]
        for t in teacher_ids 
        for s in school_ids
        if (t, s) in travel_costs
    ))
    
    # Set the complete objective
    prob += pulp.lpSum(objective)
    
    # Solve the problem
    print("\n[5/6] Solving optimization problem...")
    start_time = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=True))  # Enable solver messages
    solve_time = time.time() - start_time
    
    # Print solver status
    print(f"\n[6/6] Optimization completed in {solve_time:.2f} seconds")
    print(f"Solver status: {pulp.LpStatus[prob.status]}")
    print(f"Objective value: {pulp.value(prob.objective):.2f}")
    
    # Print objective components
    print("\nObjective components:")
    try:
        print(f"  • School coverage: {pulp.value(-w1 * pulp.lpSum(1 - school_covered[s] for s in school_ids)):,.0f}")
        print(f"  • Teacher distribution: {pulp.value(w2 * pulp.lpSum(deviation_pos[s] + deviation_neg[s] for s in school_ids)):,.0f}")
        print(f"  • Fixed assignment penalties: {pulp.value(sum(w4 * (1 - x[(t, school_id)]) for school_id, teachers in fixed_assignments.items() for t in teachers if t in teacher_ids)):,.0f}")
        print(f"  • Travel time: {pulp.value(w3 * pulp.lpSum((travel_costs.get((t, s), 1000) / max_travel) * x[(t, s)] for t in teacher_ids for s in school_ids if (t, s) in travel_costs)):,.2f}")
    except Exception as e:
        print(f"  • Could not calculate all objective components: {str(e)}")
    
    # Process results
    print("\nProcessing results...")
    assignments = {s: [] for s in school_ids}
    
    # Add fixed assignments
    for school_id, teacher_list in fixed_assignments.items():
        if school_id in assignments:
            assignments[school_id].extend(teacher_list)
    
    # Add optimized assignments
    teacher_assignments = {t: None for t in teacher_ids}
    for t in teacher_ids:
        for s in school_ids:
            if pulp.value(x[(t, s)]) == 1:
                assignments[s].append(t)
                teacher_assignments[t] = s
    
    # Print assignment summary
    print("\nAssignment Summary:")
    print("-" * 100)
    print(f"{'School ID':<10} {'School Name':<30} {'Students':>8} {'Req.':>5} {'Assigned':>8} {'Fixed':>6} {'New':>5} {'Status':<15}")
    print("-" * 100)
    
    for school_id in school_ids:
        school_info = schools_df[schools_df['id'] == school_id].iloc[0] if not schools_df[schools_df['id'] == school_id].empty else None
        school_name = school_info['name'] if school_info is not None else 'Unknown'
        num_students = school_info.get('num_students', 0) if school_info is not None else 0
        required = get_required_teachers(num_students)
        assigned = len(assignments.get(school_id, []))
        fixed = len(fixed_assignments.get(school_id, []))
        new = assigned - fixed
        
        # Check constraints
        status = []
        if assigned < 1:
            status.append("NO TEACHERS")
        elif assigned > 4:
            status.append("TOO MANY")
        elif assigned < required:
            status.append(f"UNDER {required}")
        elif assigned > required:
            status.append(f"OVER {required}")
        else:
            status.append("OK")
            
        print(f"{school_id:<10} {school_name[:28]:<30} {num_students:>8} {required:>5} {assigned:>8} {fixed:>6} {new:>5} {'/'.join(status):<15}")
    
    # Print unassigned teachers (shouldn't happen with current constraints)
    unassigned = [t for t, s in teacher_assignments.items() if s is None]
    if unassigned:
        print(f"\nWARNING: {len(unassigned)} teachers were not assigned to any school!")
        print("Unassigned teachers:", ", ".join(unassigned))
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80 + "\n")
    
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
