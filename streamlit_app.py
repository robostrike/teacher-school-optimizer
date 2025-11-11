import streamlit as st
import pandas as pd
from pathlib import Path

# Set page config
st.set_page_config(page_title="Teacher-School Manager", layout="wide")

# File paths
DATA_DIR = Path(__file__).parent / "data"
TEACHERS_FILE = DATA_DIR / "teachers_copy.csv"
SCHOOLS_FILE = DATA_DIR / "kidsduo_schools.csv"

@st.cache_data
def load_teachers():
    """Load teacher data from CSV"""
    df = pd.read_csv(TEACHERS_FILE)
    # Ensure 'move' column is boolean, defaulting to False if empty
    df['move'] = df['move'].fillna('false').str.lower() == 'true'
    return df

@st.cache_data
def load_schools():
    """Load school data and return as a dictionary of id: name"""
    df = pd.read_csv(SCHOOLS_FILE)
    return df.set_index('id')['name'].to_dict()

def save_teachers(teachers_df):
    """Save teacher data back to CSV"""
    teachers_df.to_csv(TEACHERS_FILE, index=False)

# Load data
teachers_df = load_teachers()
schools = load_schools()

# Add a 'No School' option
school_options = [""] + list(schools.keys())
school_display = {"": "No School"}
school_display.update(schools)

# App title
st.title("Teacher-School Manager")
st.write("Manage teacher assignments and move preferences")

# Create form for editing
temp_teachers = teachers_df.copy()

# Display teachers in a table with editable fields
for idx, teacher in teachers_df.iterrows():
    with st.container():
        cols = st.columns([1, 2, 2, 1])
        with cols[0]:
            st.write(f"**{teacher['name']}**")
            st.caption(f"{teacher['type']} - {teacher['station']}")
        
        with cols[1]:
            current_school = teacher.get('school_id', '')
            new_school = st.selectbox(
                "School",
                options=school_options,
                index=school_options.index(current_school) if pd.notna(current_school) and current_school in school_options else 0,
                key=f"school_{idx}",
                format_func=lambda x: school_display.get(x, "No School")
            )
            temp_teachers.at[idx, 'school_id'] = new_school if new_school else None
        
        with cols[2]:
            # If teacher has no school, they can always move
            can_move = pd.isna(teacher.get('school_id')) or teacher.get('school_id') == ''
            move_help = "Teacher has no school - can be assigned to any location" if can_move else "Check if teacher is willing to move"
            
            move = st.checkbox(
                "Willing to Move",
                value=can_move or teacher.get('move', False),
                disabled=can_move,  # Disable if teacher has no school
                key=f"move_{idx}",
                help=move_help
            )
            temp_teachers.at[idx, 'move'] = move
            
            # If teacher has no school, show a note
            if can_move:
                st.caption("Can be assigned to any location")
        
        st.divider()

# Save button
if st.button("Save Changes"):
    save_teachers(temp_teachers)
    st.success("Changes saved successfully!")
    st.rerun()

# Display current data
expander = st.expander("View Raw Data")
with expander:
    st.dataframe(teachers_df)