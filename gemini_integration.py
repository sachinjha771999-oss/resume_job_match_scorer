"""Lightweight Gemini (Google Generative AI) wrapper with safe fallbacks.

This module provides GeminiAnalyzer which uses the `google.generativeai` SDK
when available. If calls fail, it falls back to a fast TF-IDF based similarity
and heuristic suggestions.
"""

from __future__ import annotations

import os
import logging
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


class GeminiAnalyzer:
    """Wrapper for Google Gemini (google-generativeai).

    Usage:
        gem = GeminiAnalyzer(api_key=os.getenv('GEMINI_API_KEY'))
        score = gem.calculate_semantic_similarity(resume_text, job_desc)
        suggestions = gem.generate_suggestions(resume_text, job_desc, missing_skills)
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:  # pragma: no cover - environment dependent
            raise ImportError("google-generativeai package is required for Gemini integration") from e

        if not api_key:
            # allow reading from env if not provided
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("Gemini API key is required (set GEMINI_API_KEY)")

        # configure
        genai.configure(api_key=api_key)
        self._genai = genai
        # allow model override via env or constructor
        self.model = model or os.getenv("GEMINI_MODEL", "models/text-bison-001")

    def _safe_generate(self, prompt: str, max_output_chars: int = 2000) -> str:
        """Call the Gemini generate API and return a text result, with multiple
        response shape handling for different SDK versions.
        """
        try:
            resp = self._genai.generate(model=self.model, input=prompt)
            # Newer SDKs set 'text' on response, older ones may have 'candidates'
            if hasattr(resp, "text") and resp.text:
                return str(resp.text)[:max_output_chars]
            # handle candidates
            if hasattr(resp, "candidates") and resp.candidates:
                first = resp.candidates[0]
                if hasattr(first, "content"):
                    return str(first.content)[:max_output_chars]
                if hasattr(first, "text"):
                    return str(first.text)[:max_output_chars]
            # Fallback to string representation
            return str(resp)[:max_output_chars]
        except Exception as e:  # pragma: no cover - depends on network
            log.exception("Gemini generate failed: %s", e)
            raise

    def calculate_semantic_similarity(self, resume_text: str, job_desc: str) -> float:
        """Return a semantic similarity score between 0 and 100.

        This implementation prefers using the Gemini model to estimate similarity
        (by asking the model to provide a single numeric percentage). If the API
        call fails, it falls back to TF-IDF cosine similarity.
        """
        prompt = (
            "On a scale from 0 to 100, where 100 means the resume fully matches the job "
            "description and 0 means no relevance, provide only a single numeric value "
            f"(no text) representing the semantic similarity between these two texts.\n\n"
            f"Job Description:\n{job_desc[:2500]}\n\nResume:\n{resume_text[:2500]}\n\n"
            "Only reply with a number between 0 and 100."
        )

        try:
            out = self._safe_generate(prompt)
            # extract first number from the output
            import re

            m = re.search(r"(\d{1,3}(?:\.\d+)?)", out)
            if m:
                val = float(m.group(1))
                return max(0.0, min(100.0, val))
        except Exception:
            log.info("Gemini similarity call failed; falling back to TF-IDF")

        # TF-IDF fallback
        try:
            vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
            tfidf = vect.fit_transform([resume_text, job_desc])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(max(0.0, min(100.0, sim * 100)))
        except Exception as e:
            log.exception("TF-IDF fallback failed: %s", e)
            return 0.0

    def generate_suggestions(self, resume_text: str, job_desc: str, missing_skills: List[str]) -> List[str]:
        """Ask Gemini to generate 3-7 specific, actionable resume suggestions.

        Returns a list of suggestion strings. Falls back to heuristic suggestions on error.
        """
        prompt = (
            "You are an expert career coach. Given a resume and a job description, "
            "provide 5 concise, actionable suggestions to improve the resume's fit for the job. "
            "Return each suggestion on its own line starting with a dash ('-').\n\n"
            f"Job Description:\n{job_desc[:2500]}\n\n"
            f"Resume:\n{resume_text[:2500]}\n\n"
            f"Missing Skills: {', '.join(missing_skills[:30])}\n\n"
            "Limit suggestions to practical edits (sections to update, phrasing, projects to highlight)."
        )

        try:
            out = self._safe_generate(prompt)
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            suggestions = []
            for line in lines:
                if line.startswith("- "):
                    suggestions.append(line[2:].strip())
                elif line.startswith("-"):
                    suggestions.append(line[1:].strip())
                else:
                    # include lines that look like suggestions
                    if len(line) > 20:
                        suggestions.append(line)
                if len(suggestions) >= 7:
                    break
            if suggestions:
                return suggestions
        except Exception:
            log.info("Gemini suggestion call failed; using heuristic fallback")

        # Heuristic fallback
        suggestions = []
        for skill in missing_skills[:5]:
            suggestions.append(f"Add or highlight a project demonstrating experience with {skill} (include metrics/outcomes).")
        suggestions.append("Add a concise summary at the top that mirrors the job's key skills.")
        suggestions.append("Quantify achievements where possible (numbers, percentages, outcomes).")
        # dedupe and limit
        out = []
        seen = set()
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= 7:
                break
        return out