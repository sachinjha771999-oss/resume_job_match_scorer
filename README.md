Automated Resume Relevance Check System
🚀 What is this?
A smart tool that analyzes your resume against job descriptions to show how well you match and how to improve.

✨ Key Features
Resume Analysis: Extracts text from PDF/DOCX files
Job Matching: Compares your skills with job requirements
Relevance Scoring: Gives you a match percentage (0-100%)
Skill Gap Detection: Shows which skills you're missing
Improvement Tips: Suggests how to enhance your resume
Team Dashboard: For placement teams to manage evaluations
🛠️ Technologies Used
Streamlit - Web interface
Python - Core programming
PDF/DOCX Processing - File extraction
Scikit-learn - Text analysis
Gemini AI - Enhanced analysis
SQLite - Data storage
Custom CSS - Professional styling
📦 Installation
1.Clone the repo
git clone https://github.com/yourusername/resume-relevance-check-system.git
cd resume-relevance-check-system
2.Install dependencies
pip install -r requirements.txt
3.Add API keys (create .env file)
GEMINI_API_KEY=your_gemini_api_key_here
🚀 How to Use
1.Start the app
streamlit run app.py
2.Open browser
Go to http://localhost:8501
3.Analyze your resume
Upload your resume (PDF/DOCX)
Paste job description
Click "Analyze Resume"
View your results
📊 What You Get
Match Score: How well your resume fits (0-100%)
Matched Skills: Skills you have that match the job
Missing Skills: Skills the job wants that you don't have
Improvement Tips: Specific ways to boost your resume
Priority Actions: Most important improvements to make
🌐 Deployment Options
Streamlit Cloud (easiest): Connect GitHub and deploy
Docker: Use provided Dockerfile
AWS EC2: Follow setup guide
Heroku: With Procfile configuration
📁 Project Structure
├── analyzer.py          # Core analysis logic
├── app.py              # Streamlit web app
├── database.py         # Database operations
├── gemini_integration.py # AI analysis
├── requirements.txt    # Dependencies
└── .env               # API keys
🤝 Contributing
Fork the repo
Create a feature branch
Make your changes
Submit a pull request
📄 License
MIT License - free to use and modify

Note: Works without API keys, but Gemini integration provides enhanced analysis.