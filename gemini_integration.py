import google.generativeai as genai
import numpy as np
import google.generativeai as genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class GeminiAnalyzer:
    def __init__(self, api_key=None):
        if not api_key:
            raise ValueError("Gemini API key is required")
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def calculate_semantic_similarity(self, resume_text, job_desc):
        """Calculate semantic similarity between resume and job description"""
        # Use TF-IDF to create consistent embeddings
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            # Create TF-IDF vectors for both texts
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
            
            # Calculate cosine similarity between the two vectors
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity = similarity_matrix[0][0]
            
            return similarity * 100  # Convert to percentage
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            # Fallback to simple word overlap
            return self._simple_word_overlap(resume_text, job_desc)
    
    def _simple_word_overlap(self, resume_text, job_desc):
        """Calculate simple word overlap as fallback"""
        # Clean and tokenize texts
        resume_words = set(resume_text.lower().split())
        job_words = set(job_desc.lower().split())
        
        # Calculate overlap
        if len(job_words) == 0:
            return 0
        
        overlap = len(resume_words.intersection(job_words))
        return (overlap / len(job_words)) * 100
    
    def generate_suggestions(self, resume_text, job_desc, missing_skills):
        """Generate personalized suggestions using Gemini"""
        prompt = f"""
        Based on the following resume and job description, provide specific suggestions 
        to improve the resume for this position:
        
        Resume: {resume_text[:2000]}...
        
        Job Description: {job_desc[:2000]}...
        
        Missing Skills: {', '.join(missing_skills)}
        
        Please provide 5-7 specific, actionable suggestions to improve this resume.
        Format each suggestion as a separate point starting with "- ".
        """
        
        try:
            response = self.model.generate_content(prompt)
            suggestions_text = response.text
            
            # Parse the response into a list of suggestions
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    suggestions.append(line[2:])
                elif line and not suggestions:
                    # Handle case where first line doesn't have the dash
                    suggestions.append(line)
            
            return suggestions
        except Exception as e:
            # Fallback to generic suggestions if API call fails
            print(f"Error generating suggestions with Gemini: {e}")
            suggestions = []
            for skill in missing_skills:
                suggestions.append(f"Consider adding experience with {skill}.")
            
            if len(missing_skills) > 5:
                suggestions.append("Your resume appears to be missing several key skills. Consider tailoring it more to the job description.")
            
            return suggestions