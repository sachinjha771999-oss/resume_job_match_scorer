import streamlit as st
import pandas as pd
import os
import nltk
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Download NLTK data at app startup
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Call the download function
download_nltk_data()

from analyzer import process_resume, GEMINI_AVAILABLE
from database import ResumeDatabase

# Initialize database
db = ResumeDatabase()

# Page configuration
st.set_page_config(page_title="Resume vs Job Match Scorer", layout="wide", initial_sidebar_state="collapsed")

# Professional CSS with Advanced Design
st.markdown("""
    <style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    :root {
        --primary: #4361ee;  /* Modern Blue */
        --primary-dark: #3a0ca3;
        --secondary: #4cc9f0;  /* Light Blue */
        --accent: #f72585;     /* Pink */
        --warning: #f8961e;   /* Orange */
        --dark: #03071e;
        --light: #f8f9fa;
        --gray: #6c757d;
        --card-bg: rgba(255, 255, 255, 0.95);
        --card-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --gradient-1: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
        --gradient-2: linear-gradient(135deg, #f72585 0%, #b5179e 100%);
        --gradient-3: linear-gradient(135deg, #4cc9f0 0%, #4361ee 100%);
        --gradient-4: linear-gradient(135deg, #7209b7 0%, #560bad 100%);
        --success-bg: rgba(40, 167, 69, 0.15); /* Green background */
        --success-border: rgba(40, 167, 69, 0.3); /* Green border */
        --success-text: #155724; /* Dark green text */
        --error-bg: rgba(220, 53, 69, 0.15); /* Red background */
        --error-border: rgba(220, 53, 69, 0.3); /* Red border */
        --error-text: #721c24; /* Dark red text */
        --info-bg: rgba(23, 162, 184, 0.15); /* Blue background */
        --info-border: rgba(23, 162, 184, 0.3); /* Blue border */
        --info-text: #0c5460; /* Dark blue text */
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, rgba(3, 7, 30, 0.95), rgba(7, 11, 41, 0.98)), 
                    url('https://images.unsplash.com/photo-1552664730-d307ca884978?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #ffffff;
        line-height: 1.6;
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
        padding: 0;
        margin: 0;
        overflow-x: hidden;
    }
    
    /* Header and Navigation */
    .main-header {
        background: rgba(3, 7, 30, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border-radius: 0 0 16px 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 10;
        border-left: 4px solid var(--primary);
    }
    
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 0;
        letter-spacing: -0.025em;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-fill-color: transparent;
        display: inline-block;
        position: relative;
        padding-bottom: 15px; /* Added padding to create space for the line */
    }
    
    .main-title::after {
        content: "";
        position: absolute;
        bottom: 0; /* Changed from -8px to 0 to place the line under the text */
        left: 0;
        width: 100%; /* Changed from 60px to 100% to make the line span the entire title */
        height: 4px;
        background: var(--accent);
        border-radius: 2px;
    }
    
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1.2rem;
        gap: 1rem;
    }
    
    .nav-button {
        background: var(--gradient-1);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(67, 97, 238, 0.4);
        position: relative;
        overflow: hidden;
        z-index: 1;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .nav-button::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
        transform: translateX(-100%);
        transition: transform 0.3s ease;
        z-index: -1;
    }
    
    .nav-button:hover::before {
        transform: translateX(0);
    }
    
    .nav-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(67, 97, 238, 0.5);
    }
    
    .nav-button.active {
        background: var(--gradient-2);
        box-shadow: 0 4px 6px -1px rgba(247, 37, 133, 0.4);
    }
    
    /* Content Section */
    .content-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
        color: #000000;
        border-left: 4px solid var(--primary);
    }
    
    .content-section::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: var(--gradient-3);
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        position: relative;
    }
    
    .section-title::after {
        content: "";
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 40px;
        height: 3px;
        background: var(--primary);
        border-radius: 1.5px;
    }
    
    .section-icon {
        width: 40px;
        height: 40px;
        background: var(--gradient-1);
        color: white;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(67, 97, 238, 0.4);
    }
    
    /* Subsection dividers within content sections */
    .subsection {
        margin: 2rem 0;
        padding: 1.5rem 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        position: relative;
    }
    
    .subsection:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    
    .subsection-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .subsection-icon {
        font-size: 1.4rem;
        color: var(--primary);
    }
    
    /* Form Elements */
    .stTextArea, .stFileUploader, .stSelectbox, .stSlider {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.15);
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: #000000;
    }
    
    .stTextArea:focus, .stFileUploader:focus, .stSelectbox:focus, .stSlider:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.2);
        outline: none;
    }
    
    .stTextArea > div > textarea {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        line-height: 1.5;
        color: #000000;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--gradient-1);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(67, 97, 238, 0.4);
        width: 100%;
        margin-top: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-100%);
        transition: transform 0.3s ease;
        z-index: -1;
    }
    
    .stButton > button:hover::before {
        transform: translateY(0);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(67, 97, 238, 0.5);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: var(--gradient-3);
        border-radius: 10px;
        height: 10px;
    }
    
    /* Metrics */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        flex: 1;
        min-width: 150px;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        margin: 1rem 0;
        color: #000000;
        border-top: 4px solid var(--primary);
    }
    
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--gradient-4);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #000000;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #000000;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Data Display */
    .dataframe-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: var(--card-shadow);
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #000000;
        border-left: 4px solid var(--primary);
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame > div > div > table {
        border-collapse: collapse;
        width: 100%;
    }
    
    .stDataFrame > div > div > table th {
        background: var(--gradient-1);
        color: #ffffff;
        font-weight: 700;
        text-align: left;
        padding: 1rem;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stDataFrame > div > div > table td {
        padding: 1rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        font-size: 1rem;
        color: #000000;
    }
    
    /* Status Messages - FIXED */
    .status-message {
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: var(--card-shadow);
        display: flex;
        align-items: center;
        gap: 1rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        text-align: left;
    }
    
    .status-icon {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        flex-shrink: 0;
        color: white;
    }
    
    .status-text {
        flex: 1;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
        padding: 0;
        text-align: left;
    }
    
    .success-message {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
        border-left: 4px solid #28a745;
    }
    
    .success-message .status-icon {
        background: #28a745;
    }
    
    .success-message .status-text {
        color: var(--success-text);
    }
    
    .error-message {
        background: var(--error-bg);
        border: 1px solid var(--error-border);
        border-left: 4px solid #dc3545;
    }
    
    .error-message .status-icon {
        background: #dc3545;
    }
    
    .error-message .status-text {
        color: var(--error-text);
    }
    
    .info-message {
        background: var(--info-bg);
        border: 1px solid var(--info-border);
        border-left: 4px solid #17a2b8;
    }
    
    .info-message .status-icon {
        background: #17a2b8;
    }
    
    .info-message .status-text {
        color: var(--info-text);
    }
    
    /* Verdict Badge */
    .verdict-badge {
        display: inline-block;
        padding: 0.6rem 1.5rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0.8rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .verdict-high {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
        color: var(--success-text);
    }
    
    .verdict-medium {
        background: rgba(248, 150, 30, 0.15);
        border: 1px solid rgba(248, 150, 30, 0.3);
        color: #9c4a1a;
    }
    
    .verdict-low {
        background: var(--info-bg);
        border: 1px solid var(--info-border);
        color: var(--info-text);
    }
    
    /* Hide Streamlit Elements */
    #MainMenu, header, footer, [data-testid="stDeployButton"], [data-testid="collapsedControl"] {
        visibility: hidden;
        height: 0;
    }
    
    /* API Key Status - FIXED */
    .api-status {
        background: var(--success-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--success-border);
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 4px 6px -1px rgba(40, 167, 69, 0.2);
        border-left: 4px solid #28a745;
        text-align: left;
    }
    
    .api-status-icon {
        width: 30px;
        height: 30px;
        background: #28a745;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 4px 6px -1px rgba(40, 167, 69, 0.3);
    }
    
    .api-status-text {
        font-size: 1rem;
        color: var(--success-text);
        font-weight: 600;
        margin: 0;
        padding: 0;
        text-align: left;
    }
    
    .api-error {
        background: var(--error-bg);
        border: 1px solid var(--error-border);
        box-shadow: 0 4px 6px -1px rgba(220, 53, 69, 0.2);
        border-left: 4px solid #dc3545;
    }
    
    .api-error .api-status-icon {
        background: #dc3545;
    }
    
    .api-error .api-status-text {
        color: var(--error-text);
    }
    
    /* Suggestions */
    .suggestion-item {
        background: rgba(67, 97, 238, 0.1);
        border-left: 4px solid var(--primary);
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        border-radius: 0 12px 12px 0;
        font-size: 1rem;
        color: #000000;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .suggestion-item:hover {
        background: rgba(67, 97, 238, 0.15);
        transform: translateX(8px);
        box-shadow: 0 10px 15px -3px rgba(67, 97, 238, 0.3);
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary);
        animation: spin 1s ease-in-out infinite;
        margin-bottom: 1.5rem;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .loading-text {
        font-size: 1.2rem;
        color: #ffffff;
        font-weight: 600;
        text-align: center;
    }
    
    /* Fix for text area height */
    .stTextArea > div > div > textarea {
        min-height: 180px;
    }
    
    /* Custom text colors */
    .upload-resume-text {
        color: var(--accent) !important;
        font-weight: 600;
    }
    
    .detailed-analysis-text {
        color: var(--primary) !important;
        font-weight: 600;
    }
    
    /* Selectbox with scrollbar */
    .custom-selectbox {
        max-height: 200px;
        overflow-y: auto;
    }
    
    /* Blue recommendation box */
    .blue-recommendation {
        background: var(--info-bg);
        border: 1px solid var(--info-border);
        border-left: 4px solid #17a2b8;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: var(--card-shadow);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .blue-recommendation .status-icon {
        background: #17a2b8;
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        flex-shrink: 0;
    }
    
    .blue-recommendation .status-text {
        color: var(--info-text);
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
        padding: 0;
        text-align: left;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem;
        }
        
        .nav-container {
            flex-direction: column;
            gap: 0.8rem;
        }
        
        .metric-container {
            flex-direction: column;
            gap: 1rem;
        }
        
        .content-section {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .section-title {
            font-size: 1.3rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .status-message {
            padding: 1.2rem;
        }
        
        .status-icon {
            width: 40px;
            height: 40px;
            font-size: 1.2rem;
        }
    }
    
    /* Advanced Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .content-section {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Glassmorphism Effect */
    .glass-effect {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gradient-1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

# Call the initialization function
init_session_state()

# Get Gemini API key from environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Navigation
def navigation():
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">Resume vs Job Match Scorer</h1>
            <div class="nav-container">
    """, unsafe_allow_html=True)
    
    # Navigation buttons using Streamlit
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Home", key="nav_home", help="Go to Home Page"):
            st.session_state.page = "home"
            st.query_params["page"] = "home"
            st.rerun()
    
    with col2:
        if st.button("📊 Dashboard", key="nav_dashboard", help="Go to Dashboard"):
            st.session_state.page = "dashboard"
            st.query_params["page"] = "dashboard"
            st.rerun()
    
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Handle navigation via query params
    if "page" in st.query_params:
        st.session_state.page = st.query_params["page"]

# Home page
def show_home():
    # Show API key status
    if gemini_api_key:
        st.markdown("""
            <div class="api-status">
                <div class="api-status-icon">✓</div>
                <div class="api-status-text">Gemini API key is configured and ready to use</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="api-status api-error">
                <div class="api-status-icon">!</div>
                <div class="api-status-text">Gemini API key is not configured. Please add it to your .env file.</div>
            </div>
        """, unsafe_allow_html=True)
        st.error("""
            **How to set up your Gemini API key:**
            
            1. Create a `.env` file in your project root directory
            2. Add the following line to the file:
               ```
               GEMINI_API_KEY=your_gemini_api_key_here
               ```
            3. Replace `your_gemini_api_key_here` with your actual API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
            4. Restart the application
        """)
    
    # Combined Job Description and Resume Upload Section
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📝</div>
                Resume Analysis Setup
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">📄</span>
                    Job Description
                </div>
                <p style="color: #000000; margin-bottom: 1rem;">
                    Enter the complete job description including required skills, qualifications, and responsibilities.
                </p>
    """, unsafe_allow_html=True)
    
    job_desc = st.text_area("Paste the job description here", height=180, 
                            placeholder="Enter the complete job description including required skills, qualifications, and responsibilities...")
    
    st.markdown("""
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">🔄</span>
                    <span class="upload-resume-text">Upload Resume</span>
                </div>
                <p style="color: #000000; margin-bottom: 1rem;">
                    Upload your resume in PDF or DOCX format for analysis.
                </p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Show Gemini availability status
    if not GEMINI_AVAILABLE:
        st.error("⚠️ Gemini integration not available. Install google-generativeai package.")
    
    if uploaded_file and job_desc:
        if not gemini_api_key:
            st.error("🔑 Please configure your Gemini API key in the .env file.")
        elif not GEMINI_AVAILABLE:
            st.error("⚠️ Gemini integration is not available. Please install the required packages.")
        else:
            if st.button("🚀 Analyze Resume", key="analyze_button"):
                with st.spinner("Analyzing your resume with Gemini AI..."):
                    # Save the uploaded file temporarily
                    temp_dir = "temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Initialize Gemini analyzer
                    from gemini_integration import GeminiAnalyzer
                    gemini_analyzer = GeminiAnalyzer(api_key=gemini_api_key)
                    
                    # Process the resume
                    results = process_resume(file_path, job_desc, gemini_analyzer=gemini_analyzer)
                    
                    # Save to database
                    db.save_evaluation(uploaded_file.name, job_desc, results)
                    
                    # Store results in session state
                    st.session_state.analysis_results = results
                    st.session_state.analyzed = True
                    st.session_state.page = "results"
                    st.query_params["page"] = "results"
                    st.rerun()
    else:
        # Custom styled info message
        st.markdown("""
            <div class="info-message status-message">
                <div class="status-icon">📥</div>
                <div class="status-text">Please upload a resume and enter the job description to begin analysis.</div>
            </div>
        """, unsafe_allow_html=True)

# Results page
def show_results():
    # Check if analysis results exist
    if st.session_state.analysis_results is None:
        st.markdown("""
            <div class="content-section">
                <div class="section-title">
                    <div class="section-icon">⚠️</div>
                    No Analysis Results Found
                </div>
                <div class="error-message status-message">
                    <div class="status-icon">⚠️</div>
                    <div class="status-text">
                        It looks like you haven't analyzed any resume yet. Please go back to the home page to upload a resume and job description.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔙 Back to Home", key="no_results_back_button"):
            st.session_state.page = "home"
            st.query_params["page"] = "home"
            st.rerun()
        return
    
    results = st.session_state.analysis_results
    
    # Combined Results and Score Breakdown Section
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📊</div>
                Resume Match Results
            </div>
            <p style="color: #000000; margin-bottom: 1.5rem;">
                Your resume has been analyzed against the job description. Here are the comprehensive results:
            </p>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">🎯</span>
                    Overall Performance
                </div>
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-label">Matching Score</div>
                        <div class="metric-value">{}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Verdict</div>
                        <div class="verdict-badge verdict-{}">{}</div>
                    </div>
                </div>
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">📈</span>
                    Score Breakdown
                </div>
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-label">Hard Match Score</div>
                        <div class="metric-value">{}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Semantic Match Score</div>
                        <div class="metric-value">{}%</div>
                    </div>
                </div>
            </div>
        </div>
    """.format(results['score'], results['verdict'].lower(), results['verdict'], results['hard_score'], results['semantic_score']), unsafe_allow_html=True)
    
    # Progress bar
    st.progress(results["score"] / 100)
    
    # Combined Skills Analysis Section
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">🎯</div>
                Skills Analysis & Recommendations
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">✅</span>
                    Matched Skills
                </div>
                <p style="color: #000000; margin-bottom: 1rem;">
                    Skills from your resume that match the job requirements:
                </p>
                <div class="success-message status-message">
                    <div class="status-icon">✅</div>
                    <div class="status-text">{}</div>
                </div>
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">⚠️</span>
                    Missing Skills
                </div>
                <p style="color: #000000; margin-bottom: 1rem;">
                    Skills mentioned in the job description that are not present in your resume:
                </p>
                <div class="error-message status-message">
                    <div class="status-icon">⚠️</div>
                    <div class="status-text">{}</div>
                </div>
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">💡</span>
                    Improvement Suggestions
                </div>
                <p style="color: #000000; margin-bottom: 1.5rem;">
                    Here are some suggestions to improve your resume based on the analysis:
                </p>
    """.format(", ".join(results["matched_skills"]), ", ".join(results["missing_skills"])), unsafe_allow_html=True)
    
    for suggestion in results["suggestions"]:
        st.markdown(f'<div class="suggestion-item">🔹 {suggestion}</div>', unsafe_allow_html=True)
    
    st.markdown("""
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Priority table
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📌</div>
                Priority Improvements
            </div>
            <p style="color: #000000; margin-bottom: 1rem;">
                Focus on these high-priority improvements to increase your match score:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    df = pd.DataFrame(results["priority_table"])
    st.dataframe(df, use_container_width=True)
    
    # Back button
    if st.button("🔙 Back to Home", key="back_button"):
        st.session_state.page = "home"
        st.query_params["page"] = "home"
        st.rerun()

# Dashboard page
def show_dashboard():
    # Combined Filter Section
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📊</div>
                Dashboard Controls & Filters
            </div>
            <p style="color: #000000; margin-bottom: 1.5rem;">
                View and manage all resume evaluations with advanced filtering options:
            </p>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">🔍</span>
                    Filter Evaluations
                </div>
                <p style="color: #000000; margin-bottom: 1rem;">
                    Filter evaluations by score and verdict to find specific results:
                </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Minimum Score", 0, 100, 0)
    with col2:
        verdict_filter = st.selectbox("Verdict", ["All", "High", "Medium", "Low"])
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Apply filters
    filters = {"min_score": min_score}
    if verdict_filter != "All":
        filters["verdict"] = verdict_filter
    
    # Get evaluations
    evaluations = db.get_evaluations(filters)
    
    # Combined Evaluations List and Details Section
    st.markdown("""
        <div class="content-section">
            <div class="section-title">
                <div class="section-icon">📋</div>
                Resume Evaluations & Analysis
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">📊</span>
                    Evaluations List ({} Records Found)
                </div>
                <p style="color: #000000; margin-bottom: 1rem;">
                    Complete list of resume evaluations matching your filter criteria:
                </p>
    """.format(len(evaluations)), unsafe_allow_html=True)
    
    if evaluations:
        # Create a dataframe for display
        data = []
        for eval in evaluations:
            data.append({
                "ID": eval["id"],
                "Resume": eval["resume_name"],
                "Score": eval["score"],
                "Verdict": eval["verdict"],
                "Date": eval["evaluation_date"]
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("""
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">🔍</span>
                    <span class="detailed-analysis-text">Detailed Analysis View</span>
                </div>
                <p style="color: #000000; margin-bottom: 1rem;">
                    Select any evaluation ID from the table above to view comprehensive analysis details:
                </p>
        """, unsafe_allow_html=True)
        
        # Create options for selectbox with ID and resume name
        options = []
        for eval in evaluations:
            options.append(f"{eval['id']} - {eval['resume_name']}")
        
        # Allow viewing details with custom selectbox
        selected_option = st.selectbox("Choose evaluation ID for detailed analysis", options, key="eval_selector")
        
        if selected_option:
            # Extract ID from selected option
            selected_id = int(selected_option.split(" - ")[0])
            selected_eval = next(eval for eval in evaluations if eval["id"] == selected_id)
            display_evaluation_details_combined(selected_eval)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("""
            </div>
            <div class="info-message status-message">
                <div class="status-icon">📊</div>
                <div class="status-text">No evaluations found matching the selected criteria.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_evaluation_details_combined(evaluation):
    # Generate HTML for suggestions
    suggestions_html = ""
    suggestions = evaluation["suggestions"].split('|')
    for suggestion in suggestions:
        suggestions_html += f'<div class="suggestion-item">🔹 {suggestion}</div>'
    
    # Create a single HTML structure for the entire section
    st.markdown(f"""
        <div class="content-section" style="margin-top: 1.5rem;">
            <div class="section-title">
                <div class="section-icon">📊</div>
                Comprehensive Evaluation Report
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">📊</span>
                    Performance Metrics
                </div>
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-label">Score</div>
                        <div class="metric-value">{evaluation['score']}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Verdict</div>
                        <div class="verdict-badge verdict-{evaluation['verdict'].lower()}">{evaluation['verdict']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Hard Match</div>
                        <div class="metric-value">{evaluation['hard_score']}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Semantic Match</div>
                        <div class="metric-value">{evaluation['semantic_score']}%</div>
                    </div>
                </div>
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">✅</span>
                    Skills Assessment
                </div>
                <p style="color: #000000; margin-bottom: 1rem;">
                    Skills from the resume that match the job requirements:
                </p>
                <div class="success-message status-message">
                    <div class="status-icon">✅</div>
                    <div class="status-text">{", ".join(evaluation["matched_skills"].split(','))}</div>
                </div>
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">⚠️</span>
                    Gap Analysis
                </div>
                <p style="color: #000000; margin-bottom: 1rem;">
                    Skills mentioned in the job description that are not present in the resume:
                </p>
                <div class="error-message status-message">
                    <div class="status-icon">⚠️</div>
                    <div class="status-text">{", ".join(evaluation["missing_skills"].split(','))}</div>
                </div>
            </div>
            <div class="subsection">
                <div class="subsection-title">
                    <span class="subsection-icon">💡</span>
                    Recommendations
                </div>
                <p style="color: #000000; margin-bottom: 1.5rem;">
                    Actionable suggestions to improve this resume:
                </p>
                {suggestions_html}
                <div class="blue-recommendation">
                    <div class="status-icon">💡</div>
                    <div class="status-text">
                        <strong>AI Recommendation:</strong> Based on the analysis, consider adding the missing skills and highlighting your matched skills more prominently in your resume.
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    navigation()
    
    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "results":
        show_results()
    elif st.session_state.page == "dashboard":
        show_dashboard()

if __name__ == "__main__":
    main()