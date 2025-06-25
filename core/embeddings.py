import numpy as np
import re
from typing import List, Dict

class SimpleEmbeddings:
    def get_embedding(self, text: str) -> List[float]:
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        all_words = list(set(words))
        embedding = [word_freq.get(word, 0) for word in all_words]
        if embedding:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = [x / norm for x in embedding]
        return embedding

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        max_len = max(len(embedding1), len(embedding2))
        emb1 = embedding1 + [0] * (max_len - len(embedding1))
        emb2 = embedding2 + [0] * (max_len - len(embedding2))
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def calculate_match_percentage(self, resume_text: str, job_description: str) -> int:
        resume_embedding = self.get_embedding(resume_text)
        job_embedding = self.get_embedding(job_description)
        similarity = self.calculate_similarity(resume_embedding, job_embedding)
        match_score = (similarity + 1) / 2 * 100
        return int(max(50, min(match_score, 100)))

# Global instance
resume_embeddings = SimpleEmbeddings()

def calculate_match_percentage(resume_text: str, job_description: str) -> int:
    """Calculate match percentage between resume and job description."""
    return resume_embeddings.calculate_match_percentage(resume_text, job_description)
