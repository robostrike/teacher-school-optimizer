import streamlit as st
import pandas as pd
from pathlib import Path
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Set page config
st.set_page_config(
    page_title="Teacher-School Manager",
    layout="wide"
)

# File paths
DATA_DIR = Path(__file__).parent / "data"
TEACHERS_FILE = DATA_DIR / "teachers_copy.csv"
SCHOOLS_FILE = DATA_DIR / "kidsduo_schools.csv"

@st.cache_data
def load_teachers():
    """Load teacher data from CSV"""
    df = pd.read_csv(TEACHERS_FILE)
    # Convert 'move' column to string, handle NaN, then to boolean
    df['move'] = df['move'].astype(str).str.lower() == 'true'
    # Fill other columns with empty string for display
    return df.fillna('')

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

# Prepare school options for dropdown
school_options = [{'label': row['name'], 'value': row['id']} for _, row in schools_df.iterrows()]
school_dict = {row['id']: row['name'] for _, row in schools_df.iterrows()}

# Add a 'No School' option
school_options.insert(0, {'label': 'Unassigned', 'value': ''})
school_dict[''] = 'Unassigned'

# App title
st.title("🎓 Teacher-School Manager")
st.write("Assign teachers to schools using the interactive table below")

# Create a copy of the teachers dataframe for editing
if 'edited_teachers' not in st.session_state:
    st.session_state.edited_teachers = teachers_df.copy()

# Add a button to reset changes
if st.button("🔄 Reset Changes"):
    st.session_state.edited_teachers = teachers_df.copy()
    st.rerun()

# Configure the grid
gb = GridOptionsBuilder.from_dataframe(st.session_state.edited_teachers)

# Define column definitions
school_editor = {
    'field': 'school_id',
    'cellEditor': 'agSelectCellEditor',
    'cellEditorParams': {
        'values': [s['value'] for s in school_options],
        'valueFormatter': "(params) => params.value ? params.value : 'Unassigned'"
    },
    'valueFormatter': "(params) => params.value ? params.value : 'Unassigned'"
}

gb.configure_column('id', headerName='ID', editable=False)
gb.configure_column('name', headerName='Name', editable=False)
gb.configure_column('type', headerName='Type', editable=False)
gb.configure_column('station', headerName='Station', editable=False)
gb.configure_column('school_id', headerName='School', **school_editor)
gb.configure_column('move', headerName='Willing to Move', editable=True, cellRenderer='agCheckboxCellRenderer')

gb.configure_default_column(editable=True, filterable=True, sortable=True, resizable=True)

gb.configure_grid_options(
    enableRangeSelection=True,
    rowSelection='multiple',
    suppressRowClickSelection=True,
    pagination=True,
    paginationPageSize=20,
    domLayout='autoHeight',
    defaultColDef={
        'editable': False,
        'filter': True,
        'sortable': True,
        'resizable': True,
        'floatingFilter': True
    }
)

grid_options = gb.build()

# Display the grid
response = AgGrid(
    st.session_state.edited_teachers,
    gridOptions=grid_options,
    height=600,
    width='100%',
    theme='streamlit',
    update_mode='VALUE_CHANGED',
    allow_unsafe_jscode=True,
    enable_enterprise_modules=False
)

# Update the dataframe with any changes
if response['data'] is not None:
    st.session_state.edited_teachers = response['data']

# Add a save button
if st.button("💾 Save Changes"):
    # Update the original dataframe
    teachers_df = st.session_state.edited_teachers
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