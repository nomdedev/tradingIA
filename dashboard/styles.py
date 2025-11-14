"""
CSS personalizado para el dashboard.
Inyectar con st.markdown(..., unsafe_allow_html=True)
"""

CSS_MINIMALISTA = """
<style>

/* Reset y base */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: #FFFFFF;
    color: #0F172A;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    border: none;
}

/* Main content */
[data-testid="stAppViewContainer"] > section {
    padding: 2rem 1.5rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* Títulos */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    color: #0F172A;
    letter-spacing: -0.01em;
    font-weight: 600;
}

h1 {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 1.5rem;
}

h2 {
    font-size: 22px;
    margin-bottom: 1rem;
    margin-top: 2rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #E2E8F0;
}

h3 {
    font-size: 18px;
    margin-bottom: 0.75rem;
}

/* Texto */
p {
    color: #64748B;
    line-height: 1.6;
    font-size: 14px;
}

/* Métricos (Cards) */
[data-testid="metric-container"] {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.25rem;
    transition: all 300ms ease-in-out;
}

[data-testid="metric-container"]:hover {
    background-color: #F1F5F9;
    border-color: #CBD5E1;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

/* Botones */
button {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 14px;
    border-radius: 6px;
    border: none;
    cursor: pointer;
    transition: all 300ms ease-in-out;
    padding: 0.5rem 1rem;
}

button[kind="primary"] {
    background-color: #0F172A;
    color: white;
}

button[kind="primary"]:hover {
    background-color: #1E293B;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

button[kind="secondary"] {
    background-color: #F8FAFC;
    color: #0F172A;
    border: 1px solid #E2E8F0;
}

button[kind="secondary"]:hover {
    background-color: #F1F5F9;
    border-color: #CBD5E1;
}

/* Inputs */
input, select, textarea {
    font-family: 'Inter', sans-serif;
    border-radius: 6px;
    border: 1px solid #E2E8F0;
    padding: 0.5rem 0.75rem;
    font-size: 14px;
    transition: all 300ms ease-in-out;
}

input:focus, select:focus, textarea:focus {
    outline: none;
    border-color: #0F172A;
    box-shadow: 0 0 0 3px rgba(15, 23, 42, 0.1);
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #E2E8F0;
    gap: 2rem;
}

[data-testid="stTabs"] button[role="tab"] {
    color: #64748B;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    padding: 0.75rem 0;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #0F172A;
    border-bottom-color: #0F172A;
}

/* Expandable sections */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0;
    border-radius: 6px;
}

[data-testid="stExpander"] > button {
    background-color: #F8FAFC;
    color: #0F172A;
    padding: 1rem;
}

[data-testid="stExpander"] > button:hover {
    background-color: #F1F5F9;
}

/* Data tables */
[data-testid="stDataFrame"] {
    font-size: 14px;
}

[data-testid="stDataFrame"] tr {
    border-color: #E2E8F0;
}

[data-testid="stDataFrame"] tr:hover {
    background-color: #F8FAFC;
}

[data-testid="stDataFrame"] th {
    background-color: #F8FAFC;
    color: #0F172A;
    font-weight: 600;
    border-color: #E2E8F0;
}

/* Selectbox y multiselect */
[data-baseweb="select"] {
    font-family: 'Inter', sans-serif;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.5rem 0;
}

/* Alert boxes */
.stAlert {
    border-radius: 6px;
    border: 1px solid;
    font-size: 14px;
}

.stAlert > div:first-child {
    padding: 0;
    margin-right: 0.75rem;
}

/* Success alert */
[data-testid="stAlert"] {
    border-radius: 6px;
}

/* Cards personalizadas */
.card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.5rem;
    transition: all 300ms ease-in-out;
}

.card:hover {
    background-color: #F1F5F9;
    border-color: #CBD5E1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* Badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
}

.badge-success {
    background-color: rgba(16, 185, 129, 0.1);
    color: #10B981;
}

.badge-warning {
    background-color: rgba(245, 158, 11, 0.1);
    color: #F59E0B;
}

.badge-danger {
    background-color: rgba(239, 68, 68, 0.1);
    color: #EF4444;
}

.badge-info {
    background-color: rgba(59, 130, 246, 0.1);
    color: #3B82F6;
}

/* Gráficos */
[data-testid="plotly"] {
    border-radius: 8px;
    background-color: #F8FAFC;
    padding: 1rem;
}

/* Scrollbar personalizado */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #F8FAFC;
}

::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94A3B8;
}

/* Responsive */
@media (max-width: 768px) {
    [data-testid="stAppViewContainer"] > section {
        padding: 1rem;
    }

    h1 {
        font-size: 24px;
    }

    h2 {
        font-size: 20px;
    }
}

</style>
"""