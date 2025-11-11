import streamlit as st
import pandas as pd
import json
from pathlib import Path
from streamlit_draggable_list import draggable_list

# Set page config
st.set_page_config(
    page_title="Teacher-School Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths
DATA_DIR = Path(__file__).parent / "data"
TEACHERS_FILE = DATA_DIR / "teachers_copy.csv"
SCHOOLS_FILE = DATA_DIR / "kidsduo_schools.csv"

@st.cache_data
def load_teachers():
    """Load teacher data from CSV"""
    df = pd.read_csv(TEACHERS_FILE)
    df['move'] = df['move'].fillna('false').str.lower() == 'true'
    return df

@st.cache_data
def load_schools():
    """Load school data"""
    df = pd.read_csv(SCHOOLS_FILE)
    return df

def save_teachers(teachers_df):
    """Save teacher data back to CSV"""
    teachers_df.to_csv(TEACHERS_FILE, index=False)

# Load data
teachers_df = load_teachers()
schools_df = load_schools()

# Convert to list of dicts for draggable components
teachers = teachers_df.to_dict('records')
schools = schools_df.to_dict('records')

# Add a 'No School' option
no_school = {"id": "", "name": "Unassigned Teachers"}
schools.insert(0, no_school)

# Initialize session state for drag and drop
if 'teacher_assignments' not in st.session_state:
    st.session_state.teacher_assignments = {}
    for teacher in teachers:
        st.session_state.teacher_assignments[teacher['id']] = teacher.get('school_id', '')

# App title
st.title("🎓 Teacher-School Manager")
st.write("Drag and drop teachers to assign them to schools")

# Split the screen into two columns
left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("🏫 Schools")
    
    # Create a container for each school
    for school in schools:
        school_id = school['id']
        school_name = school.get('name', 'Unassigned')
        
        # Filter teachers assigned to this school
        assigned_teachers = [
            t for t in teachers 
            if st.session_state.teacher_assignments.get(t['id'], '') == school_id
        ]
        
        with st.expander(f"{school_name} ({len(assigned_teachers)} teachers)", expanded=True):
            # Create a list of teacher cards for this school
            teacher_cards = []
            for teacher in assigned_teachers:
                card = {
                    'id': teacher['id'],
                    'title': teacher['name'],
                    'content': f"{teacher['type']} | {teacher['station']}",
                    'data': json.dumps(teacher)
                }
                teacher_cards.append(card)
            
            # Create a drop zone for this school
            if teacher_cards:
                result = draggable_list(
                    items=teacher_cards,
                    on_drag_end=lambda x: None,  # We'll handle updates in the callback
                    key=f"school_{school_id}"
                )

with right_col:
    st.header("👥 Unassigned Teachers")
    
    # Show unassigned teachers
    unassigned_teachers = [
        t for t in teachers 
        if not st.session_state.teacher_assignments.get(t['id']) 
        and pd.isna(t.get('school_id'))
    ]
    
    if not unassigned_teachers:
        st.info("All teachers have been assigned to schools!")
    else:
        for teacher in unassigned_teachers:
            with st.container(border=True):
                st.markdown(f"**{teacher['name']}**")
                st.caption(f"{teacher['type']} | {teacher['station']}")
                
                # Add a button to assign to school
                selected_school = st.selectbox(
                    "Assign to school",
                    options=[s['id'] for s in schools if s['id'] != ''],
                    format_func=lambda x: next((s['name'] for s in schools if s['id'] == x), "Select a school"),
                    key=f"assign_{teacher['id']}"
                )
                
                if st.button("Assign", key=f"btn_assign_{teacher['id']}"):
                    st.session_state.teacher_assignments[teacher['id']] = selected_school
                    st.rerun()

# Save changes
if st.button("💾 Save All Assignments"):
    # Update the teachers dataframe with new assignments
    for idx, teacher in teachers_df.iterrows():
        teacher_id = teacher['id']
        if teacher_id in st.session_state.teacher_assignments:
            teachers_df.at[idx, 'school_id'] = st.session_state.teacher_assignments[teacher_id]
    
    # Save to CSV
    save_teachers(teachers_df)
    st.success("Teacher assignments saved successfully!")
    st.rerun()

# Display current assignments
expander = st.expander("📊 Current Assignments Overview")
with expander:
    st.dataframe(
        teachers_df[['name', 'type', 'station', 'school_id']].merge(
            schools_df[['id', 'name']],
            left_on='school_id',
            right_on='id',
            how='left'
        ).rename(columns={'name_y': 'school_name', 'name_x': 'teacher_name'})
        [['teacher_name', 'type', 'station', 'school_name']],
        hide_index=True
    )