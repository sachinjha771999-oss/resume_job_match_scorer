import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

# Download NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class ResumeJobScorer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        # Comprehensive skills database
        self.skills_db = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'swift', 'kotlin', 'typescript', 'r', 'matlab'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'cassandra'],
            'cloud': ['aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'tableau', 'power bi', 'excel'],
            'soft_skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'analytical', 'creativity', 'time management']
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers but keep words
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and short tokens, then lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Text preprocessing error: {e}")
            return ""
    
    def extract_skills(self, text):
        """Extract skills from text using comprehensive skills database"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_skills = []
        
        # Check each skill category
        for category, skills in self.skills_db.items():
            for skill in skills:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    found_skills.append(skill.title())
        
        return list(set(found_skills))  # Remove duplicates
    
    def calculate_similarity(self, resume_text, job_description):
        """Calculate cosine similarity between resume and job description"""
        processed_resume = self.preprocess_text(resume_text)
        processed_job = self.preprocess_text(job_description)
        
        if not processed_resume or not processed_job:
            return 0.0
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform([processed_resume, processed_job])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            score = min(similarity[0][0] * 100, 100)  # Cap at 100%
            return max(score, 0)  # Ensure non-negative
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0
    
    def analyze_match(self, resume_text, job_description):
        """Comprehensive analysis of resume-job match"""
        try:
            # Calculate basic similarity
            similarity_score = self.calculate_similarity(resume_text, job_description)
            
            # Extract skills
            resume_skills = self.extract_skills(resume_text)
            job_skills = self.extract_skills(job_description)
            
            # Calculate skill match
            if job_skills:
                matched_skills = set(resume_skills) & set(job_skills)
                skill_match_percentage = (len(matched_skills) / len(job_skills)) * 100
            else:
                skill_match_percentage = 0
                matched_skills = set()
            
            # Missing skills
            missing_skills = set(job_skills) - set(resume_skills)
            
            # Overall score (weighted average)
            overall_score = (similarity_score * 0.6) + (skill_match_percentage * 0.4)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(missing_skills, overall_score, len(matched_skills))
            
            return {
                'overall_score': round(overall_score, 2),
                'similarity_score': round(similarity_score, 2),
                'skill_match_score': round(skill_match_percentage, 2),
                'resume_skills': resume_skills,
                'job_skills': job_skills,
                'matched_skills': list(matched_skills),
                'missing_skills': list(missing_skills),
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Analysis error: {e}")
            # Return default structure in case of error
            return {
                'overall_score': 0,
                'similarity_score': 0,
                'skill_match_score': 0,
                'resume_skills': [],
                'job_skills': [],
                'matched_skills': [],
                'missing_skills': [],
                'recommendations': ["Error in analysis. Please check your input texts."]
            }
    
    def generate_recommendations(self, missing_skills, overall_score, matched_count):
        """Generate improvement recommendations"""
        recommendations = []
        
        # Score-based recommendations
        if overall_score < 30:
            recommendations.append("🔴 **Low Match**: Significant improvements needed. Consider completely rewriting your resume to match the job requirements.")
        elif overall_score < 50:
            recommendations.append("🟡 **Moderate Match**: Your resume needs substantial improvements to better align with this role.")
        elif overall_score < 70:
            recommendations.append("🟢 **Good Match**: Solid alignment, but there's room for optimization.")
        elif overall_score < 85:
            recommendations.append("🔵 **Strong Match**: Well done! Your resume aligns well with the job requirements.")
        else:
            recommendations.append("🎉 **Excellent Match**: Outstanding! Your resume is highly tailored to this position.")
        
        # Skill-based recommendations
        if missing_skills:
            skills_to_show = list(missing_skills)[:5]  # Show max 5 skills
            skills_text = ", ".join(skills_to_show)
            if len(missing_skills) > 5:
                skills_text += f" and {len(missing_skills) - 5} more..."
            recommendations.append(f"📚 **Develop These Skills**: {skills_text}")
        
        if matched_count == 0:
            recommendations.append("💡 **Add Relevant Skills**: Include more technical skills from the job description in your resume.")
        elif matched_count < 3:
            recommendations.append("💡 **Expand Skill Set**: Consider adding more relevant skills to increase your match score.")
        
        # General recommendations
        if overall_score < 60:
            recommendations.append("🎯 **Use Keywords**: Incorporate more keywords from the job description throughout your resume.")
            recommendations.append("📊 **Quantify Achievements**: Add numbers and metrics to demonstrate your impact (e.g., 'increased efficiency by 20%').")
        
        return recommendations