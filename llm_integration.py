"""LLM integration with safe fallbacks.

This module provides a small wrapper around an LLM-based analyzer (OpenAI via
`langchain_community`) but degrades gracefully to a TF-IDF based similarity and
simple suggestion generator when the LLM packages or keys are not available.
"""

import logging
from typing import List

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


class LLMAnalyzer:
    """Analyzer that uses LLM embeddings/small LLMs when available.

    If langchain_community (or related adapters) are not installed or fail at
    runtime, this class will fallback to TF-IDF based similarity and a simple
    suggestions generator.
    """

    def __init__(self, openai_api_key: str | None = None):
        self.openai_api_key = openai_api_key
        # Try to import langchain-based embeddings; if they are unavailable,
        # we'll mark that and use TF-IDF fallbacks.
        try:
            from langchain_community.embeddings import OpenAIEmbeddings  # type: ignore
            from langchain_text_splitters import CharacterTextSplitter  # type: ignore

            self._has_langchain = True
            self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        except Exception as e:  # pragma: no cover - environment dependent
            log.info("LangChain or OpenAI embeddings not available: %s", e)
            self._has_langchain = False
            self.text_splitter = None
            self.embeddings = None

    def calculate_semantic_similarity(self, resume_text: str, job_desc: str) -> float:
        """Return a similarity score between 0 and 100.

        Priority: OpenAI embeddings via langchain (if available) -> TF-IDF fallback.
        """
        try:
            if self._has_langchain and self.embeddings is not None:
                # Use embeddings from provider
                res_emb = np.array(self.embeddings.embed_query(resume_text))
                job_emb = np.array(self.embeddings.embed_query(job_desc))
                if res_emb.size == 0 or job_emb.size == 0:
                    raise ValueError("Empty embeddings returned")
                sim = cosine_similarity(res_emb.reshape(1, -1), job_emb.reshape(1, -1))[0][0]
                return float(max(0.0, min(100.0, sim * 100)))

        except Exception as e:  # fallback to TF-IDF
            log.info("Embedding-based similarity failed, falling back to TF-IDF: %s", e)

        # TF-IDF fallback (fast, local)
        try:
            vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
            tfidf = vect.fit_transform([resume_text, job_desc])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(max(0.0, min(100.0, sim * 100)))
        except Exception as e:  # last resort
            log.error("TF-IDF similarity failed: %s", e)
            return 0.0

    def generate_suggestions(self, resume_text: str, job_desc: str, missing_skills: List[str]) -> List[str]:
        """Generate 3-7 actionable suggestions.

        If an LLM is available, prefer it; otherwise return heuristic suggestions.
        """
        # Try LLM generation if available
        if self._has_langchain and self.openai_api_key:
            try:
                from langchain_community.llms import OpenAI  # type: ignore

                llm = OpenAI(temperature=0, openai_api_key=self.openai_api_key)
                prompt = (
                    "Based on the resume and job description, provide 5 concise, actionable suggestions "
                    "to improve the resume's fit. Include phrasing examples and sections to update.\n\n"
                    f"Job Description:\n{job_desc[:2000]}...\n\n"
                    f"Resume:\n{resume_text[:2000]}...\n\n"
                    f"Missing Skills: {', '.join(missing_skills)}\n"
                )
                response = llm(prompt)
                text = str(response)
                # Split on newlines and bullets
                items = [s.strip('- •* \t') for s in text.split('\n') if s.strip()]
                # Keep up to 7 cleaned items
                suggestions = []
                for it in items:
                    if len(suggestions) >= 7:
                        break
                    if len(it) > 10:
                        suggestions.append(it)
                if suggestions:
                    return suggestions
            except Exception as e:  # pragma: no cover
                log.info("LLM suggestion generation failed: %s", e)

        # Heuristic fallback suggestions
        suggestions = []
        for skill in missing_skills[:3]:
            suggestions.append(f"Add or highlight a project showing experience with {skill} (briefly describe scope and outcome).")
        suggestions.append("Quantify achievements (e.g., reduced X by Y% or supported N users).")
        suggestions.append("Include a concise summary at top matching the job's key skills and role.")
        suggestions.append("Move technical skills to a clear 'Skills' section and list tools/versions.")
        # Deduplicate and limit
        seen = set()
        out = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= 7:
                break
        return out
