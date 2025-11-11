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
    # Handle 'move' column with proper type conversion
    if 'move' in df.columns:
        # Convert to string first, then handle true/false values
        df['move'] = df['move'].astype(str).str.lower().replace({
            'true': True,
            'false': False,
            'nan': False,
            '': False
        }).astype(bool)
    else:
        # If 'move' column doesn't exist, create it with False as default
        df['move'] = False
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

# Display teachers in a responsive grid of cards
st.subheader("Teacher Directory")

# Create a grid of cards (3 cards per row on large screens)
cols_per_row = 3
total_teachers = len(teachers_df)

for i in range(0, total_teachers, cols_per_row):
    # Create a row of cards
    cols = st.columns(cols_per_row)
    
    # Fill the current row with cards
    for col_idx in range(cols_per_row):
        teacher_idx = i + col_idx
        if teacher_idx >= total_teachers:
            break
            
        teacher = teachers_df.iloc[teacher_idx]
        current_school = teacher.get('school_id', '')
        can_move = pd.isna(teacher.get('school_id')) or teacher.get('school_id') == ''
        
        with cols[col_idx]:
            with st.container(border=True):
                # Header with name and type
                st.markdown(f"### {teacher['name']}")
                st.caption(f"{teacher['type']} | {teacher['station']}")
                
                # School selection
                new_school = st.selectbox(
                    "Assigned School",
                    options=school_options,
                    index=school_options.index(current_school) if pd.notna(current_school) and current_school in school_options else 0,
                    key=f"school_{teacher_idx}",
                    format_func=lambda x: school_display.get(x, "No School")
                )
                temp_teachers.at[teacher_idx, 'school_id'] = new_school if new_school else None
                
                # Move preference
                move = st.checkbox(
                    "Willing to Move",
                    value=can_move or teacher.get('move', False),
                    disabled=can_move,
                    key=f"move_{teacher_idx}",
                    help="Check if teacher is willing to move to a different school"
                )
                temp_teachers.at[teacher_idx, 'move'] = move
                
                # Status indicator
                if can_move:
                    st.info("Available for assignment", icon="ℹ️")
                elif move:
                    st.warning("Open to relocation", icon="🔄")
                else:
                    st.success("Stable assignment", icon="✅")

# Save button
if st.button("Save Changes"):
    save_teachers(temp_teachers)
    st.success("Changes saved successfully!")
    st.rerun()

# Display current data
expander = st.expander("View Raw Data")
with expander:
    st.dataframe(teachers_df)