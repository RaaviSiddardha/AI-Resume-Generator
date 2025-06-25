"""
Advanced features for the Resume Generator.
Includes resume scoring, A/B testing, and analytics capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import json
import os
from pathlib import Path
from core.llm import ResumeLLM
from core.embeddings import calculate_match_percentage
from core.config import get_st_config

class ResumeScorer:
    """Advanced resume scoring and analysis system."""
    
    def __init__(self):
        """Initialize the resume scorer."""
        self.llm = ResumeLLM()
        self.config = get_st_config()
    
    def score_resume(self, resume_text: str, job_description: str = None) -> Dict[str, Any]:
        """
        Comprehensive resume scoring.
        
        Args:
            resume_text: The resume text to score
            job_description: Optional job description for targeted scoring
            
        Returns:
            Dictionary with detailed scoring results
        """
        scores = {}
        
        # Basic metrics
        scores["length_score"] = self._score_length(resume_text)
        scores["readability_score"] = self._score_readability(resume_text)
        scores["keyword_density"] = self._score_keyword_density(resume_text)
        
        # Content quality scores
        scores["action_verbs_score"] = self._score_action_verbs(resume_text)
        scores["quantification_score"] = self._score_quantification(resume_text)
        scores["professional_tone_score"] = self._score_professional_tone(resume_text)
        
        # Structure scores
        scores["structure_score"] = self._score_structure(resume_text)
        scores["formatting_score"] = self._score_formatting(resume_text)
        
        # Job-specific scores (if job description provided)
        if job_description:
            scores["job_match_score"] = self._score_job_match(resume_text, job_description)
            scores["keyword_alignment"] = self._score_keyword_alignment(resume_text, job_description)
            scores["skill_relevance"] = self._score_skill_relevance(resume_text, job_description)
        
        # Overall score
        scores["overall_score"] = self._calculate_overall_score(scores)
        
        # Detailed feedback
        scores["feedback"] = self._generate_feedback(scores, resume_text)
        
        return scores
    
    def _score_length(self, resume_text: str) -> float:
        """Score resume length (optimal: 400-800 words)."""
        word_count = len(resume_text.split())
        if 400 <= word_count <= 800:
            return 100.0
        elif 300 <= word_count <= 1000:
            return 80.0
        elif 200 <= word_count <= 1200:
            return 60.0
        else:
            return 40.0
    
    def _score_readability(self, resume_text: str) -> float:
        """Score readability using Flesch Reading Ease."""
        sentences = resume_text.split('.')
        words = resume_text.split()
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return 50.0
        
        flesch_score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))
        return max(0, min(100, flesch_score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(1, count)
    
    def _score_keyword_density(self, resume_text: str) -> float:
        """Score keyword density and variety."""
        words = resume_text.lower().split()
        unique_words = set(words)
        
        # Calculate type-token ratio
        if len(words) == 0:
            return 0.0
        
        ttr = len(unique_words) / len(words)
        return min(100, ttr * 200)  # Scale to 0-100
    
    def _score_action_verbs(self, resume_text: str) -> float:
        """Score usage of action verbs."""
        action_verbs = [
            "achieved", "developed", "implemented", "managed", "created", "designed",
            "led", "improved", "increased", "decreased", "optimized", "streamlined",
            "coordinated", "facilitated", "established", "launched", "delivered",
            "executed", "maintained", "produced", "reduced", "enhanced", "expanded"
        ]
        
        words = resume_text.lower().split()
        action_verb_count = sum(1 for word in words if word in action_verbs)
        
        if len(words) == 0:
            return 0.0
        
        density = action_verb_count / len(words)
        return min(100, density * 1000)  # Scale appropriately
    
    def _score_quantification(self, resume_text: str) -> float:
        """Score quantification of achievements."""
        import re
        
        # Look for numbers, percentages, and quantifiable metrics
        number_patterns = [
            r'\d+%',  # Percentages
            r'\$\d+',  # Dollar amounts
            r'\d+ people',  # Team sizes
            r'\d+ customers',  # Customer counts
            r'increased by \d+',  # Improvements
            r'reduced by \d+',  # Reductions
            r'\d+ years',  # Time periods
        ]
        
        matches = 0
        for pattern in number_patterns:
            matches += len(re.findall(pattern, resume_text.lower()))
        
        # Score based on number of quantifiable statements
        if matches >= 5:
            return 100.0
        elif matches >= 3:
            return 80.0
        elif matches >= 1:
            return 60.0
        else:
            return 20.0
    
    def _score_professional_tone(self, resume_text: str) -> float:
        """Score professional tone using LLM."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            prompt_template = PromptTemplate(
                input_variables=["resume_text"],
                template="""
                Rate the professional tone of this resume text on a scale of 0-100.
                Consider: formality, clarity, confidence, and appropriateness.
                
                Resume text: {resume_text}
                
                Return only a number between 0 and 100.
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=prompt_template)
            response = chain.run(resume_text=resume_text[:500] + "...")
            
            # Extract number from response
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                return min(100, max(0, float(numbers[0])))
            return 70.0  # Default score
        except:
            return 70.0
    
    def _score_structure(self, resume_text: str) -> float:
        """Score resume structure and organization."""
        sections = ["summary", "experience", "education", "skills"]
        section_count = sum(1 for section in sections if section.lower() in resume_text.lower())
        
        if section_count >= 4:
            return 100.0
        elif section_count >= 3:
            return 80.0
        elif section_count >= 2:
            return 60.0
        else:
            return 40.0
    
    def _score_formatting(self, resume_text: str) -> float:
        """Score formatting consistency."""
        lines = resume_text.split('\n')
        bullet_points = sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '○')))
        
        if bullet_points >= 5:
            return 100.0
        elif bullet_points >= 3:
            return 80.0
        elif bullet_points >= 1:
            return 60.0
        else:
            return 40.0
    
    def _score_job_match(self, resume_text: str, job_description: str) -> float:
        """Score job match using embeddings."""
        return calculate_match_percentage(resume_text, job_description)
    
    def _score_keyword_alignment(self, resume_text: str, job_description: str) -> float:
        """Score keyword alignment between resume and job description."""
        try:
            job_keywords = self.llm.extract_skills_from_job_description(job_description)
            resume_words = set(resume_text.lower().split())
            
            matches = sum(1 for keyword in job_keywords if keyword.lower() in resume_words)
            
            if len(job_keywords) == 0:
                return 50.0
            
            return (matches / len(job_keywords)) * 100
        except:
            return 50.0
    
    def _score_skill_relevance(self, resume_text: str, job_description: str) -> float:
        """Score skill relevance using semantic similarity."""
        try:
            # Extract skills from both texts
            resume_skills = self.llm.extract_skills_from_job_description(resume_text)
            job_skills = self.llm.extract_skills_from_job_description(job_description)
            
            if not resume_skills or not job_skills:
                return 50.0
            
            # Calculate similarity between skill sets
            resume_skills_text = " ".join(resume_skills)
            job_skills_text = " ".join(job_skills)
            
            return calculate_match_percentage(resume_skills_text, job_skills_text)
        except:
            return 50.0
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall score from individual scores."""
        weights = {
            "length_score": 0.1,
            "readability_score": 0.1,
            "action_verbs_score": 0.15,
            "quantification_score": 0.15,
            "professional_tone_score": 0.1,
            "structure_score": 0.1,
            "formatting_score": 0.05,
            "job_match_score": 0.15,
            "keyword_alignment": 0.05,
            "skill_relevance": 0.05
        }
        
        overall_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in scores:
                overall_score += scores[metric] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 50.0
        
        return overall_score / total_weight
    
    def _generate_feedback(self, scores: Dict[str, float], resume_text: str) -> List[str]:
        """Generate actionable feedback based on scores."""
        feedback = []
        
        if scores.get("length_score", 0) < 70:
            feedback.append("Consider adjusting resume length for optimal impact")
        
        if scores.get("action_verbs_score", 0) < 70:
            feedback.append("Use more strong action verbs to describe achievements")
        
        if scores.get("quantification_score", 0) < 70:
            feedback.append("Add specific numbers and metrics to quantify achievements")
        
        if scores.get("professional_tone_score", 0) < 70:
            feedback.append("Review tone for professionalism and clarity")
        
        if scores.get("job_match_score", 0) < 70:
            feedback.append("Tailor resume more closely to the target job description")
        
        if scores.get("structure_score", 0) < 70:
            feedback.append("Ensure all key sections are present and well-organized")
        
        return feedback

class ABTester:
    """A/B testing system for resume variations."""
    
    def __init__(self):
        """Initialize the A/B tester."""
        self.scorer = ResumeScorer()
        self.test_results = {}
    
    def create_variations(self, base_resume: str, job_description: str = None) -> Dict[str, str]:
        """Create different variations of a resume for A/B testing."""
        variations = {
            "original": base_resume,
            "action_verbs": self._enhance_action_verbs(base_resume),
            "quantified": self._enhance_quantification(base_resume),
            "keyword_optimized": self._optimize_keywords(base_resume, job_description),
            "concise": self._make_concise(base_resume),
            "detailed": self._add_details(base_resume)
        }
        
        return variations
    
    def _enhance_action_verbs(self, resume_text: str) -> str:
        """Create variation with enhanced action verbs."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            prompt_template = PromptTemplate(
                input_variables=["resume_text"],
                template="""
                Rewrite this resume with stronger action verbs while keeping the same content.
                
                Resume: {resume_text}
                
                Use powerful action verbs like: achieved, developed, implemented, managed, created, designed, led, improved, increased, optimized, streamlined, coordinated, facilitated, established, launched, delivered, executed, maintained, produced, reduced, enhanced, expanded.
                """
            )
            
            chain = LLMChain(llm=self.scorer.llm.llm, prompt=prompt_template)
            return chain.run(resume_text=resume_text)
        except:
            return resume_text
    
    def _enhance_quantification(self, resume_text: str) -> str:
        """Create variation with enhanced quantification."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            prompt_template = PromptTemplate(
                input_variables=["resume_text"],
                template="""
                Add specific numbers and metrics to quantify achievements in this resume.
                
                Resume: {resume_text}
                
                Add percentages, dollar amounts, time periods, team sizes, customer counts, etc.
                """
            )
            
            chain = LLMChain(llm=self.scorer.llm.llm, prompt=prompt_template)
            return chain.run(resume_text=resume_text)
        except:
            return resume_text
    
    def _optimize_keywords(self, resume_text: str, job_description: str) -> str:
        """Create variation optimized for job keywords."""
        if not job_description:
            return resume_text
        
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            keywords = self.scorer.llm.extract_skills_from_job_description(job_description)
            prompt_template = PromptTemplate(
                input_variables=["resume_text", "keywords"],
                template="""
                Optimize this resume to naturally include these keywords: {keywords}
                
                Resume: {resume_text}
                
                Integrate keywords naturally without keyword stuffing.
                """
            )
            
            chain = LLMChain(llm=self.scorer.llm.llm, prompt=prompt_template)
            return chain.run(resume_text=resume_text, keywords=keywords)
        except:
            return resume_text
    
    def _make_concise(self, resume_text: str) -> str:
        """Create a more concise variation."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            prompt_template = PromptTemplate(
                input_variables=["resume_text"],
                template="""
                Make this resume more concise while preserving key information.
                
                Resume: {resume_text}
                
                Reduce length by 20-30% while maintaining impact.
                """
            )
            
            chain = LLMChain(llm=self.scorer.llm.llm, prompt=prompt_template)
            return chain.run(resume_text=resume_text)
        except:
            return resume_text
    
    def _add_details(self, resume_text: str) -> str:
        """Create a more detailed variation."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            prompt_template = PromptTemplate(
                input_variables=["resume_text"],
                template="""
                Add more details and context to this resume.
                
                Resume: {resume_text}
                
                Expand on achievements and responsibilities while maintaining clarity.
                """
            )
            
            chain = LLMChain(llm=self.scorer.llm.llm, prompt=prompt_template)
            return chain.run(resume_text=resume_text)
        except:
            return resume_text
    
    def test_variations(self, variations: Dict[str, str], job_description: str = None) -> Dict[str, Any]:
        """Test all variations and return comparative results."""
        results = {}
        
        for name, variation in variations.items():
            scores = self.scorer.score_resume(variation, job_description)
            results[name] = {
                "scores": scores,
                "overall_score": scores["overall_score"],
                "word_count": len(variation.split()),
                "variation": variation
            }
        
        # Sort by overall score
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["overall_score"], reverse=True))
        
        return {
            "variations": sorted_results,
            "best_variation": list(sorted_results.keys())[0],
            "comparison": self._create_comparison_table(sorted_results)
        }
    
    def _create_comparison_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a comparison table for all variations."""
        comparison_data = []
        
        for name, result in results.items():
            scores = result["scores"]
            comparison_data.append({
                "Variation": name,
                "Overall Score": f"{result['overall_score']:.1f}",
                "Word Count": result["word_count"],
                "Length Score": f"{scores.get('length_score', 0):.1f}",
                "Action Verbs": f"{scores.get('action_verbs_score', 0):.1f}",
                "Quantification": f"{scores.get('quantification_score', 0):.1f}",
                "Job Match": f"{scores.get('job_match_score', 0):.1f}"
            })
        
        return pd.DataFrame(comparison_data)

class ResumeAnalytics:
    """Analytics and insights for resume performance."""
    
    def __init__(self):
        """Initialize analytics system."""
        self.scorer = ResumeScorer()
        self.data_file = "resume_analytics.json"
        self.load_data()
    
    def load_data(self):
        """Load existing analytics data."""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {"resumes": [], "tests": [], "insights": []}
    
    def save_data(self):
        """Save analytics data."""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def track_resume(self, resume_text: str, job_description: str = None, metadata: Dict = None):
        """Track a resume for analytics."""
        scores = self.scorer.score_resume(resume_text, job_description)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "scores": scores,
            "word_count": len(resume_text.split()),
            "metadata": metadata or {}
        }
        
        self.data["resumes"].append(entry)
        self.save_data()
    
    def get_insights(self) -> Dict[str, Any]:
        """Generate insights from tracked data."""
        if not self.data["resumes"]:
            return {"message": "No data available for insights"}
        
        df = pd.DataFrame(self.data["resumes"])
        
        insights = {
            "total_resumes": len(df),
            "average_overall_score": df["scores"].apply(lambda x: x["overall_score"]).mean(),
            "score_distribution": {
                "excellent": len(df[df["scores"].apply(lambda x: x["overall_score"]) >= 80]),
                "good": len(df[(df["scores"].apply(lambda x: x["overall_score"]) >= 60) & 
                              (df["scores"].apply(lambda x: x["overall_score"]) < 80)]),
                "needs_improvement": len(df[df["scores"].apply(lambda x: x["overall_score"]) < 60])
            },
            "common_issues": self._identify_common_issues(df),
            "trends": self._analyze_trends(df)
        }
        
        return insights
    
    def _identify_common_issues(self, df: pd.DataFrame) -> List[str]:
        """Identify common issues across resumes."""
        issues = []
        
        # Check for common low scores
        avg_scores = {}
        for score_type in ["action_verbs_score", "quantification_score", "professional_tone_score"]:
            avg_scores[score_type] = df["scores"].apply(lambda x: x.get(score_type, 0)).mean()
        
        if avg_scores.get("action_verbs_score", 0) < 60:
            issues.append("Low action verb usage")
        if avg_scores.get("quantification_score", 0) < 60:
            issues.append("Lack of quantified achievements")
        if avg_scores.get("professional_tone_score", 0) < 60:
            issues.append("Professional tone needs improvement")
        
        return issues
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in resume performance."""
        if len(df) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        # Convert timestamps to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("datetime")
        
        # Calculate trend in overall scores
        scores = df["scores"].apply(lambda x: x["overall_score"]).values
        if len(scores) > 1:
            trend = np.polyfit(range(len(scores)), scores, 1)[0]
            trend_direction = "improving" if trend > 0 else "declining" if trend < 0 else "stable"
        else:
            trend_direction = "insufficient data"
        
        return {
            "score_trend": trend_direction,
            "average_word_count": df["word_count"].mean(),
            "score_volatility": df["scores"].apply(lambda x: x["overall_score"]).std()
        }

# Global instances
resume_scorer = ResumeScorer()
ab_tester = ABTester()
resume_analytics = ResumeAnalytics()

# Convenience functions
def score_resume(resume_text: str, job_description: str = None) -> Dict[str, Any]:
    """Score a resume."""
    return resume_scorer.score_resume(resume_text, job_description)

def create_ab_test(base_resume: str, job_description: str = None) -> Dict[str, Any]:
    """Create and test resume variations."""
    variations = ab_tester.create_variations(base_resume, job_description)
    return ab_tester.test_variations(variations, job_description)

def track_resume_analytics(resume_text: str, job_description: str = None, metadata: Dict = None):
    """Track resume for analytics."""
    resume_analytics.track_resume(resume_text, job_description, metadata)

def get_analytics_insights() -> Dict[str, Any]:
    """Get analytics insights."""
    return resume_analytics.get_insights()

if __name__ == "__main__":
    print("Testing advanced features...")
    
    # Test resume scoring
    test_resume = """
    John Doe
    Software Engineer
    
    SUMMARY
    Experienced software engineer with 5 years in Python development.
    
    EXPERIENCE
    - Developed web applications using Python and Django
    - Managed team of 3 developers
    - Improved system performance by 25%
    """
    
    scores = score_resume(test_resume)
    print(f"Resume scores: {scores['overall_score']:.1f}/100")
    
    # Test A/B testing
    variations = create_ab_test(test_resume)
    print(f"Best variation: {variations['best_variation']}")
    
    print("Advanced features working!") 