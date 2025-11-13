"""
CSS styles for the Teacher-School Optimizer application.
"""

def get_custom_css():
    """
    Returns the custom CSS for the application.
    """
    return """
    <style>
    /* Make the teacher cards more compact */
    .stContainer {
        padding: 0.25rem !important;
        gap: 0 !important;
    }
    
    /* Remove all vertical gaps between elements */
    .stVerticalBlock {
        gap: 0.5rem !important;
    }
    
    /* Specific class for teacher cards */
    .st-emotion-cache-1ne20ew {
        gap: 0 !important;
    }
    
    .stMarkdown h3, .stMarkdown p, .stMarkdown div {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.1 !important;
    }
    
    /* Teacher name and type */
    .stMarkdown h3 {
        font-size: 0.95rem !important;
        line-height: 1.1 !important;
    }
    
    /* Caption text (type and station) */
    .stMarkdown .stCaption {
        font-size: 0.75rem !important;
        line-height: 1 !important;
        margin-top: 0.1rem !important;
    }
    
    /* Make the selectbox and checkbox more compact */
    .stSelectbox, .stCheckbox {
        margin: 0.1rem 0 !important;
        font-size: 0.85rem !important;
    }
    
    /* Sidebar styles */
    .st-emotion-cache-1fwbbrh hr {
        margin: 0.1rem 0 !important;
        height: 0.5px !important;
        background-color: #e0e0e0 !important;
        border: none !important;
    }
    
    /* Reduce spacing around section headers in sidebar */
    .st-emotion-cache-1fwbbrh h3 {
        margin: 0.25rem 0 0.5rem 0 !important;
        font-size: 0.95rem !important;
    }
    
    /* Make sidebar content more compact */
    .st-emotion-cache-1fwbbrh {
        gap: 0.5rem !important;
    }
    
    /* Reduce padding and height in the selectbox */
    .stSelectbox > div > div {
        padding: 0.15rem 0.75rem 0.15rem 0.5rem !important;
        min-height: 1.5rem !important;
    }
    
    /* Space between cards */
    .stContainer > div {
        margin: 0 !important;
    }
    
    /* Make sure columns have the right spacing */
    .stHorizontal > div[data-testid="column"] {
        padding: 0 !important;
    }
    
    /* Reduce space around status messages */
    .stAlert {
        padding: 0.25rem 0.5rem !important;
        margin: 0.15rem 0 !important;
        min-height: 1.5rem !important;
    }
    
    /* Make form elements more compact */
    .stTextInput, .stSelectbox, .stCheckbox {
        margin-bottom: 0.15rem !important;
    }
    
    /* Reduce space between form elements */
    .element-container {
        padding: 0.1rem 0 !important;
    }
    
    /* Table styles */
    .stTable {
        width: 100%;
        border-collapse: collapse;
    }
    
    .stTable th, .stTable td {
        padding: 8px;
        border: 1px solid #ddd;
        text-align: left;
    }
    
    .stTable th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    
    .stTable tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    .stTable tr:hover {
        background-color: #f1f1f1;
    }
    
    /* Card styles */
    .stCard {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Status indicators */
    .status-reassignable {
        color: #1e88e5;
    }
    
    .status-open {
        color: #ff9800;
    }
    
    .status-stable {
        color: #4caf50;
    }
    
    /* Utility classes */
    .text-center {
        text-align: center;
    }
    
    .text-right {
        text-align: right;
    }
    
    .mt-1 {
        margin-top: 0.25rem;
    }
    
    .mb-1 {
        margin-bottom: 0.25rem;
    }
    
    .ml-1 {
        margin-left: 0.25rem;
    }
    
    .mr-1 {
        margin-right: 0.25rem;
    }
    
    .p-1 {
        padding: 0.25rem;
    }
    
    .p-2 {
        padding: 0.5rem;
    }
    
    .p-3 {
        padding: 1rem;
    }
    </style>
    """

def get_table_styles():
    """
    Returns styles for the teachers table.
    """
    return """
    <style>
    /* Make the table scrollable */
    .table-container {
        overflow-x: auto;
        margin: 1rem 0;
    }
    
    /* Table styles */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9em;
        min-width: 400px;
        border-radius: 5px 5px 0 0;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    
    .dataframe thead tr {
        background-color: #1e88e5;
        color: #ffffff;
        text-align: left;
        font-weight: bold;
    }
    
    .dataframe th,
    .dataframe td {
        padding: 12px 15px;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .dataframe tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    
    .dataframe tbody tr:last-of-type {
        border-bottom: 2px solid #1e88e5;
    }
    
    .dataframe tbody tr.active-row {
        font-weight: bold;
        color: #1e88e5;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        text-align: center;
    }
    
    .status-available {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    
    .status-assigned {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    
    .status-unavailable {
        background-color: #ffebee;
        color: #c62828;
    }
    </style>
    """
