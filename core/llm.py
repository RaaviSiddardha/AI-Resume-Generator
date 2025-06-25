from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.llms.base import LLM
from typing import Dict, Any, Optional, List, Mapping
import os
# from langgraph.graph import StateGraph, END

# This is a placeholder for a more complex state management with LangGraph
# class ResumeGenerationState:
#     def __init__(self):
#         self.user_data = None
#         self.job_description = None
#         self.generated_sections = {}
#         self.critique = None
#         self.final_resume = None

HF_TOKEN = os.getenv("HF_TOKEN")
# Use a more accessible model by default
MODEL_NAME = "microsoft/DialoGPT-medium"  # Changed from Mistral-7B to avoid gated repo issues

class FallbackLLM(LLM):
    """A simple fallback LLM that implements the required LangChain interface."""
    
    @property
    def _llm_type(self) -> str:
        return "fallback"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate text based on the prompt."""
        # Basic template-based response
        if "resume" in prompt.lower():
            return """
            PROFESSIONAL SUMMARY
            Experienced professional with strong skills in the requested areas.
            
            WORK EXPERIENCE
            • Demonstrated expertise in relevant technologies and methodologies
            • Led successful projects and initiatives
            • Collaborated effectively with cross-functional teams
            
            SKILLS
            • Technical skills relevant to the position
            • Soft skills including communication and leadership
            • Industry-specific knowledge and best practices
            """
        else:
            return "Generated content based on the provided information."
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_type": "fallback"}

class ResumeLLM:
    def __init__(self, model_name: str = MODEL_NAME, use_hub: bool = True):
        """
        Initialize the Resume LLM with either HuggingFace Hub or local model.
        
        Args:
            model_name: HuggingFace model name
            use_hub: If True, use HuggingFace Hub API, else load model locally
        """
        self.model_name = model_name
        self.use_hub = use_hub
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        try:
            if self.use_hub and HF_TOKEN:
                # Use HuggingFace Hub API
                self.llm = HuggingFaceHub(
                    repo_id=self.model_name,
                    huggingfacehub_api_token=HF_TOKEN,
                    model_kwargs={
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                        "top_p": 0.95,
                        "do_sample": True
                    }
                )
            else:
                # For now, use a simple fallback approach
                # We'll use a basic text generation approach
                self.llm = FallbackLLM()
                
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            # Fallback to a simple text generation approach
            self.llm = FallbackLLM()
    
    def _create_fallback_llm(self):
        """Create a fallback LLM that doesn't require model loading."""
        return FallbackLLM()

    def generate_resume_content(self, user_data: Dict[str, Any], tone: str) -> str:
        """Generate resume content using LangChain."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        prompt_template = PromptTemplate(
            input_variables=["user_data", "tone"],
            template="""
            You are an expert resume writer. Create a professional resume based on the following information.
            
            Tone: {tone}
            User Information: {user_data}
            
            Please generate a comprehensive resume with the following sections:
            1. Professional Summary (3-4 compelling sentences)
            2. Work Experience (with quantifiable achievements and action verbs)
            3. Education
            4. Skills (organized by category)
            5. Projects (if applicable)
            6. Certifications (if applicable)
            
            Make the resume professional, impactful, and tailored to the specified tone.
            Use strong action verbs and quantify achievements where possible.
            
            Resume:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(user_data=str(user_data), tone=tone)
            return result.strip()
        except Exception as e:
            print(f"Error generating resume content: {e}")
            # Fallback response
            return self._generate_fallback_resume(user_data, tone)

    def _generate_fallback_resume(self, user_data: Dict[str, Any], tone: str) -> str:
        """Generate a fallback resume when LLM fails."""
        name = user_data.get('name', 'Your Name')
        email = user_data.get('email', 'your.email@example.com')
        phone = user_data.get('phone', 'Phone Number')
        skills = user_data.get('skills', 'Relevant Skills')
        experience = user_data.get('experience', ['Work Experience'])
        education = user_data.get('education', ['Education'])
        
        resume = f"""
        {name.upper()}
        {email} | {phone}
        
        PROFESSIONAL SUMMARY
        {tone} professional with experience in relevant fields. Skilled in {skills} and committed to delivering high-quality results.
        
        WORK EXPERIENCE
        """
        
        for exp in experience:
            if exp:
                resume += f"• {exp}\n"
        
        resume += f"""
        EDUCATION
        """
        
        for edu in education:
            if edu:
                resume += f"• {edu}\n"
        
        resume += f"""
        SKILLS
        {skills}
        """
        
        return resume

    def improve_resume_content(self, resume_text: str, improvement_focus: list = None) -> str:
        """Improve existing resume content."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        focus_areas = improvement_focus or ["Professional Tone", "Action Verbs", "Quantify Achievements"]
        focus_text = ", ".join(focus_areas)
        
        prompt_template = PromptTemplate(
            input_variables=["resume_text", "focus_areas"],
            template="""
            You are an expert resume reviewer and editor. Improve the following resume by focusing on: {focus_areas}
            
            Original Resume:
            {resume_text}
            
            Please provide an improved version that:
            1. Uses stronger action verbs
            2. Quantifies achievements with numbers and percentages
            3. Maintains professional tone
            4. Improves clarity and impact
            5. Removes any redundancy
            
            Improved Resume:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(resume_text=resume_text, focus_areas=focus_text)
            return result.strip()
        except Exception as e:
            print(f"Error improving resume content: {e}")
            return resume_text  # Return original if improvement fails

    def tailor_resume_to_job(self, resume_text: str, job_description: str) -> str:
        """Tailor resume to match a specific job description."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        prompt_template = PromptTemplate(
            input_variables=["resume_text", "job_description"],
            template="""
            You are an expert resume writer specializing in job-specific tailoring.
            
            Original Resume:
            {resume_text}
            
            Job Description:
            {job_description}
            
            Please rewrite the resume to better align with this job description by:
            1. Highlighting relevant skills and experiences
            2. Using keywords from the job description
            3. Emphasizing achievements that match the role requirements
            4. Adjusting the professional summary to target this specific position
            5. Reorganizing content to prioritize relevant information
            
            Return the complete tailored resume:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(resume_text=resume_text, job_description=job_description)
            return result.strip()
        except Exception as e:
            print(f"Error tailoring resume: {e}")
            return resume_text  # Return original if tailoring fails

    def extract_skills_from_job_description(self, job_description: str) -> list:
        """Extract key skills from job description."""
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        prompt_template = PromptTemplate(
            input_variables=["job_description"],
            template="""
            Extract the key technical skills, soft skills, and requirements from this job description.
            Return only a comma-separated list of skills, no explanations.
            
            Job Description:
            {job_description}
            
            Skills:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(job_description=job_description)
            skills = [skill.strip() for skill in result.split(',')]
            return skills
        except Exception as e:
            print(f"Error extracting skills: {e}")
            # Fallback: extract basic keywords
            keywords = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'kubernetes', 'agile', 'leadership', 'communication']
            return [kw for kw in keywords if kw.lower() in job_description.lower()]

# Global instance
resume_llm = ResumeLLM()

# Convenience functions for backward compatibility
def generate_resume_content_hf(user_data, tone, api_key=None):
    """Backward compatibility function."""
    return resume_llm.generate_resume_content(user_data, tone)

def improve_resume_content(resume_text, api_key=None):
    """Backward compatibility function."""
    return resume_llm.improve_resume_content(resume_text)

def tailor_resume_to_job(resume_text, job_description, api_key=None):
    """Backward compatibility function."""
    return resume_llm.tailor_resume_to_job(resume_text, job_description)

# Placeholder for a LangGraph implementation
# def get_resume_generation_graph():
#     workflow = StateGraph(ResumeGenerationState)
#     # ... define nodes and edges for the graph
#     # workflow.add_node("generate_summary", ...)
#     # workflow.add_node("critique_summary", ...)
#     # ...
#     # workflow.set_entry_point("generate_summary")
#     # ...
#     return workflow.compile() 

if __name__ == "__main__":
    # Example user data
    user_data = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "123-456-7890",
        "education": ["B.Sc. in Computer Science, XYZ University"],
        "experience": ["Software Engineer at ABC Corp (2020-2023)"],
        "skills": "Python, Machine Learning, Data Analysis"
    }
    tone = "Professional"

    # Test resume generation
    print("Testing LangChain + HuggingFace resume generation...")
    result = resume_llm.generate_resume_content(user_data, tone)
    print(result) 