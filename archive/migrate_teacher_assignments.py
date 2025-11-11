import pandas as pd
from datetime import datetime
import os

# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEACHERS_FILE = os.path.join(DATA_DIR, 'teachers_copy.csv')
ASSIGNMENTS_FILE = os.path.join(DATA_DIR, 'teacher_school_assignments.csv')
UPDATED_TEACHERS_FILE = os.path.join(DATA_DIR, 'teachers.csv')

def migrate_assignments():
    # Read the current teachers data
    teachers_df = pd.read_csv(TEACHERS_FILE)
    
    # Create assignments dataframe
    assignments = []
    
    # Get current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Process each teacher with a school assignment
    for _, row in teachers_df[teachers_df['school_id'].notna()].iterrows():
        assignments.append({
            'teacher_id': row['id'],
            'school_id': row['school_id'],
            'assigned_date': current_time,
            'is_current': True
        })
    
    # Create assignments dataframe
    if assignments:
        assignments_df = pd.DataFrame(assignments)
        # Save to CSV
        assignments_df.to_csv(ASSIGNMENTS_FILE, index=False)
        print(f"Created {len(assignments)} assignments in {ASSIGNMENTS_FILE}")
    
    # Create updated teachers file without school_id
    updated_teachers = teachers_df.drop(columns=['school_id'], errors='ignore')
    updated_teachers.to_csv(UPDATED_TEACHERS_FILE, index=False)
    print(f"Created updated teachers file at {UPDATED_TEACHERS_FILE}")
    
    # Show summary
    print("\nMigration Summary:")
    print(f"- Total teachers processed: {len(teachers_df)}")
    print(f"- Teachers with school assignments: {len(assignments)}")
    
    return assignments_df, updated_teachers

if __name__ == "__main__":
    print("Starting teacher-school assignments migration...")
    migrate_assignments()
    print("\nMigration completed successfully!")
