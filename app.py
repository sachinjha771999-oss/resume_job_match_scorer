import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scoring import ResumeJobScorer
import docx
import PyPDF2
import io

# Page configuration
st.set_page_config(
    page_title="Resume-Job Match Scorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 10px 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .skill-match {
        background-color: #d4edda;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #28a745;
    }
    .skill-missing {
        background-color: #f8d7da;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #dc3545;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def read_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        return ""

def read_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"❌ Error reading DOCX: {str(e)}")
        return ""

def read_txt(file):
    """Extract text from TXT file"""
    try:
        text = str(file.read(), "utf-8")
        return text.strip()
    except Exception as e:
        st.error(f"❌ Error reading TXT file: {str(e)}")
        return ""

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Resume-Job Match Scorer</h1>', unsafe_allow_html=True)
    st.markdown("### *Intelligent Resume Analysis & Job Matching*")
    
    # Initialize scorer in session state
    if 'scorer' not in st.session_state:
        st.session_state.scorer = ResumeJobScorer()
    
    # Sidebar
    st.sidebar.title("🚀 Navigation")
    app_mode = st.sidebar.selectbox("Choose Analysis Mode", 
                                   ["Single Analysis", "Batch Analysis", "About"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **How to use:**
    1. Upload or paste your resume
    2. Paste the job description
    3. Click 'Analyze Match'
    4. Get detailed insights and recommendations
    """)
    
    if app_mode == "Single Analysis":
        single_analysis()
    elif app_mode == "Batch Analysis":
        batch_analysis()
    else:
        about_page()

def single_analysis():
    """Single resume-job analysis mode"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Resume Input")
        resume_option = st.radio("Choose input method:", 
                               ["Text Input", "File Upload"],
                               horizontal=True)
        
        resume_text = ""
        if resume_option == "Text Input":
            resume_text = st.text_area(
                "Paste your resume text here:", 
                height=250,
                placeholder="Copy and paste your resume content here...\n\nInclude:\n• Work experience\n• Skills\n• Education\n• Projects\n• Certifications",
                key="resume_text"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload Resume File", 
                type=['txt', 'pdf', 'docx'],
                help="Supported formats: TXT, PDF, DOCX",
                key="resume_upload"
            )
            if uploaded_file is not None:
                with st.spinner("Reading your resume file..."):
                    if uploaded_file.type == "text/plain":
                        resume_text = read_txt(uploaded_file)
                    elif uploaded_file.type == "application/pdf":
                        resume_text = read_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        resume_text = read_docx(uploaded_file)
                
                if resume_text:
                    st.success(f"✅ Successfully loaded {uploaded_file.name}")
                    st.text_area("Resume Preview", resume_text[:500] + "..." if len(resume_text) > 500 else resume_text, height=100)
    
    with col2:
        st.subheader("💼 Job Description")
        job_description = st.text_area(
            "Paste job description here:",
            height=250,
            placeholder="Copy and paste the job description here...\n\nInclude:\n• Job requirements\n• Required skills\n• Responsibilities\n• Qualifications",
            key="job_text"
        )
    
    st.markdown("---")
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Match Score", type="primary", use_container_width=True):
            if not resume_text.strip():
                st.error("❌ Please provide your resume content")
                return
            if not job_description.strip():
                st.error("❌ Please provide the job description")
                return
            
            with st.spinner("🤖 Analyzing your resume and job description... This may take a few seconds."):
                try:
                    results = st.session_state.scorer.analyze_match(resume_text, job_description)
                    display_results(results)
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Please try again with different text or check your file format.")

def display_results(results):
    """Display analysis results"""
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Determine score color
    score = results['overall_score']
    if score < 40:
        score_color = "#ff4b4b"
        score_emoji = "🔴"
    elif score < 60:
        score_color = "#ffa500"
        score_emoji = "🟡"
    elif score < 80:
        score_color = "#4CAF50"
        score_emoji = "🟢"
    else:
        score_color = "#2196F3"
        score_emoji = "🔵"
    
    # Score cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="score-card">
            <h3>{score_emoji} Overall Match</h3>
            <h1 style="color: white; text-align: center; font-size: 2.8rem; margin: 10px 0;">{results['overall_score']}%</h1>
            <p style="margin: 0; opacity: 0.9;">Comprehensive Compatibility</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <h3>📈 Content Similarity</h3>
            <h1 style="color: white; text-align: center; font-size: 2.8rem; margin: 10px 0;">{results['similarity_score']}%</h1>
            <p style="margin: 0; opacity: 0.9;">Text & Keyword Match</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="score-card">
            <h3>🔧 Skill Match</h3>
            <h1 style="color: white; text-align: center; font-size: 2.8rem; margin: 10px 0;">{results['skill_match_score']}%</h1>
            <p style="margin: 0; opacity: 0.9;">Technical Skills Alignment</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization
    st.subheader("📈 Detailed Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Score comparison chart
        categories = ['Overall Match', 'Content Similarity', 'Skill Match']
        scores = [results['overall_score'], results['similarity_score'], results['skill_match_score']]
        
        fig = go.Figure(data=[
            go.Bar(name='Scores', x=categories, y=scores,
                  marker_color=['#667eea', '#764ba2', '#f093fb'])
        ])
        fig.update_layout(
            title="Match Score Breakdown",
            yaxis_title="Score (%)",
            yaxis_range=[0, 100],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Skills distribution
        if results['job_skills']:
            matched_count = len(results['matched_skills'])
            missing_count = len(results['missing_skills'])
            
            fig = px.pie(
                values=[matched_count, missing_count],
                names=['Matched Skills', 'Missing Skills'],
                title=f"Skills Match: {matched_count}/{matched_count + missing_count}",
                color=['Matched Skills', 'Missing Skills'],
                color_discrete_map={'Matched Skills': '#28a745', 'Missing Skills': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No specific skills detected in the job description.")
    
    # Skills Analysis
    st.subheader("🔧 Skills Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Matched Skills")
        if results['matched_skills']:
            for skill in sorted(results['matched_skills']):
                st.markdown(f'<div class="skill-match">✓ {skill}</div>', unsafe_allow_html=True)
        else:
            st.info("No matching skills found. Consider adding relevant technical skills to your resume.")
    
    with col2:
        st.markdown("### ❌ Missing Skills")
        if results['missing_skills']:
            for skill in sorted(results['missing_skills']):
                st.markdown(f'<div class="skill-missing">✗ {skill}</div>', unsafe_allow_html=True)
        else:
            st.success("🎉 Excellent! Your resume contains all the skills mentioned in the job description.")
    
    # Recommendations
    st.subheader("💡 Improvement Recommendations")
    for i, recommendation in enumerate(results['recommendations'], 1):
        st.markdown(f"""
        <div class="recommendation-box">
            <strong>Recommendation #{i}:</strong> {recommendation}
        </div>
        """, unsafe_allow_html=True)

def batch_analysis():
    """Batch analysis mode for multiple resumes"""
    st.subheader("📊 Batch Resume Analysis")
    
    st.info("""
    **For Recruiters & Hiring Managers:** 
    Upload multiple resumes and compare them against a single job description to identify the best candidates.
    """)
    
    # Job description input
    job_description = st.text_area(
        "Job Description for Batch Analysis:",
        height=200,
        placeholder="Paste the job description you're hiring for...",
        key="batch_job"
    )
    
    # Multiple resume upload
    uploaded_files = st.file_uploader(
        "Upload Multiple Resumes", 
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="Select multiple resume files to compare",
        key="batch_upload"
    )
    
    if st.button("🚀 Analyze All Resumes", type="primary") and job_description and uploaded_files:
        if not job_description.strip():
            st.error("Please provide a job description")
            return
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            # Extract text based on file type
            resume_text = ""
            try:
                if uploaded_file.type == "text/plain":
                    resume_text = read_txt(uploaded_file)
                elif uploaded_file.type == "application/pdf":
                    resume_text = read_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = read_docx(uploaded_file)
                
                if resume_text:
                    # Analyze
                    analysis = st.session_state.scorer.analyze_match(resume_text, job_description)
                    results.append({
                        'Filename': uploaded_file.name,
                        'Overall Score': analysis['overall_score'],
                        'Similarity Score': analysis['similarity_score'],
                        'Skill Match %': analysis['skill_match_score'],
                        'Matched Skills': len(analysis['matched_skills']),
                        'Missing Skills': len(analysis['missing_skills']),
                        'Total Job Skills': len(analysis['job_skills'])
                    })
                else:
                    st.warning(f"Could not read {uploaded_file.name}")
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ Analysis complete!")
        
        # Display batch results
        if results:
            df = pd.DataFrame(results)
            st.subheader("📋 Batch Analysis Results")
            
            # Sort by overall score
            df = df.sort_values('Overall Score', ascending=False)
            
            # Display table with formatting
            st.dataframe(
                df.style.background_gradient(
                    subset=['Overall Score', 'Skill Match %'], 
                    cmap='RdYlGn'
                ).format({
                    'Overall Score': '{:.1f}%',
                    'Similarity Score': '{:.1f}%', 
                    'Skill Match %': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df, 
                    x='Filename', 
                    y='Overall Score',
                    title="Candidate Ranking by Overall Score",
                    color='Overall Score',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(xaxis_title="Candidate", yaxis_title="Overall Score (%)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    df, 
                    x='Overall Score', 
                    y='Skill Match %',
                    size='Matched Skills',
                    color='Filename',
                    title="Overall Score vs Skill Match",
                    hover_data=['Filename', 'Matched Skills', 'Missing Skills']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Export results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="resume_analysis_results.csv",
                mime="text/csv"
            )

def about_page():
    """About page with project information"""
    st.subheader("ℹ️ About Resume-Job Match Scorer")
    
    st.markdown("""
    ## 🎯 What is this tool?
    
    The **Resume-Job Match Scorer** is an intelligent application that uses advanced Natural Language Processing (NLP) 
    and machine learning to analyze the compatibility between resumes and job descriptions.
    
    ## 🛠️ How it works
    
    1. **Text Processing**: Cleans and normalizes text from resumes and job descriptions
    2. **TF-IDF Vectorization**: Converts text into numerical representations
    3. **Cosine Similarity**: Calculates semantic similarity between documents
    4. **Skill Extraction**: Identifies and matches technical and soft skills
    5. **Comprehensive Scoring**: Provides weighted scores and detailed analysis
    
    ## 📊 What gets analyzed?
    
    - **Content Similarity**: How well your resume text matches the job description
    - **Skill Alignment**: Matching of technical skills and qualifications  
    - **Keyword Optimization**: Use of relevant keywords from the job description
    - **Overall Compatibility**: Combined score considering all factors
    
    ## 🚀 Features
    
    | Feature | Description |
    |---------|-------------|
    | ✅ Single Analysis | Analyze one resume against one job description |
    | ✅ Batch Analysis | Compare multiple resumes for the same job |
    | ✅ File Support | TXT, PDF, and DOCX file formats |
    | ✅ Skill Matching | Automatic technical skill identification |
    | ✅ Visual Analytics | Interactive charts and graphs |
    | ✅ Actionable Insights | Specific improvement recommendations |
    
    ## 💡 Tips for best results
    
    1. **Use complete resumes** with detailed work experience and skills
    2. **Provide detailed job descriptions** with specific requirements
    3. **Review all recommendations** for actionable improvements
    4. **Compare multiple versions** of your resume to optimize
    
    ## 🔧 Technology Stack
    
    - **Streamlit** - Web application framework
    - **Scikit-learn** - Machine learning and NLP
    - **NLTK** - Natural language processing
    - **Plotly** - Interactive visualizations
    - **Pandas** - Data analysis and manipulation
    
    ---
    
    *Built with ❤️ for job seekers and recruiters*
    """)

if __name__ == "__main__":
    main()