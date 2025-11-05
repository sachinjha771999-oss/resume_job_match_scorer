import sqlite3
import sqlite3
from datetime import datetime

class ResumeDatabase:
    def __init__(self, db_path="resume_evaluations.db"):
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_name TEXT,
                    job_description TEXT,
                    score REAL,
                    hard_score REAL,
                    semantic_score REAL,
                    verdict TEXT,
                    matched_skills TEXT,
                    missing_skills TEXT,
                    suggestions TEXT,
                    evaluation_date TIMESTAMP
                )
            """)
            conn.commit()
    
    def save_evaluation(self, resume_name, job_desc, results):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluations 
                (resume_name, job_description, score, hard_score, semantic_score, verdict, 
                matched_skills, missing_skills, suggestions, evaluation_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                resume_name,
                job_desc,
                results.get('score', 0),
                results.get('hard_score', 0),
                results.get('semantic_score', 0),
                results.get('verdict', 'Low'),
                ','.join(results.get('matched_skills', [])),
                ','.join(results.get('missing_skills', [])),
                '|'.join(results.get('suggestions', [])),
                datetime.now()
            ))
            conn.commit()
            return cursor.lastrowid
            cursor.execute("""
                INSERT INTO evaluations 
                (resume_name, job_description, score, hard_score, semantic_score, verdict, matched_skills, missing_skills, suggestions, evaluation_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                resume_name,
                job_desc,
                results["score"],
                results["hard_score"],
                results["semantic_score"],
                results["verdict"],
                ",".join(results["matched_skills"]),
                ",".join(results["missing_skills"]),
                "|".join(results["suggestions"]),
                datetime.now()
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_evaluations(self, filters=None):
        query = "SELECT * FROM evaluations"
        params = []
        
        if filters:
            conditions = []
            if "min_score" in filters:
                conditions.append("score >= ?")
                params.append(filters["min_score"])
            if "verdict" in filters and filters["verdict"] != "All":
                conditions.append("verdict = ?")
                params.append(filters["verdict"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY evaluation_date DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]