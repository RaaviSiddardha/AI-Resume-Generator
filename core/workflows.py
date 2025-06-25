"""
LangGraph workflows for complex resume generation pipelines.
Implements state management and multi-step resume generation processes.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any, List, TypedDict, Annotated
import operator
from core.llm import ResumeLLM
from core.config import get_st_config
import streamlit as st
from core.embeddings import calculate_match_percentage

# State definitions
class ResumeState(TypedDict):
    """State for resume generation workflow."""
    user_data: Dict[str, Any]
    job_description: str
    tone: str
    template: str
    generated_sections: Dict[str, str]
    critique: str
    improvements: List[str]
    final_resume: str
    match_score: float
    keywords: List[str]
    step: str
    errors: List[str]

class TailorState(TypedDict):
    """State for resume tailoring workflow."""
    original_resume: str
    job_description: str
    extracted_skills: List[str]
    tailored_sections: Dict[str, str]
    match_analysis: Dict[str, Any]
    final_resume: str
    improvements: List[str]
    step: str

class ImproveState(TypedDict):
    """State for resume improvement workflow."""
    original_resume: str
    improvement_focus: List[str]
    analysis: Dict[str, Any]
    improved_sections: Dict[str, str]
    final_resume: str
    step: str

class ResumeWorkflows:
    """LangGraph workflows for resume generation and processing."""
    
    def __init__(self):
        """Initialize workflows with LLM and embeddings."""
        self.llm = ResumeLLM()
        self.config = get_st_config()
    
    def create_resume_generation_workflow(self) -> StateGraph:
        """Create a comprehensive resume generation workflow."""
        
        # Define the graph
        workflow = StateGraph(ResumeState)
        
        # Add nodes
        workflow.add_node("extract_keywords", self._extract_keywords)
        workflow.add_node("generate_summary", self._generate_summary)
        workflow.add_node("generate_experience", self._generate_experience)
        workflow.add_node("generate_skills", self._generate_skills)
        workflow.add_node("critique_resume", self._critique_resume)
        workflow.add_node("improve_resume", self._improve_resume)
        workflow.add_node("finalize_resume", self._finalize_resume)
        
        # Define edges
        workflow.set_entry_point("extract_keywords")
        workflow.add_edge("extract_keywords", "generate_summary")
        workflow.add_edge("generate_summary", "generate_experience")
        workflow.add_edge("generate_experience", "generate_skills")
        workflow.add_edge("generate_skills", "critique_resume")
        workflow.add_conditional_edges(
            "critique_resume",
            self._should_improve,
            {
                "improve": "improve_resume",
                "finalize": "finalize_resume"
            }
        )
        workflow.add_edge("improve_resume", "finalize_resume")
        workflow.add_edge("finalize_resume", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def create_tailoring_workflow(self) -> StateGraph:
        """Create a resume tailoring workflow."""
        
        workflow = StateGraph(TailorState)
        
        # Add nodes
        workflow.add_node("extract_job_skills", self._extract_job_skills)
        workflow.add_node("analyze_match", self._analyze_match)
        workflow.add_node("tailor_summary", self._tailor_summary)
        workflow.add_node("tailor_experience", self._tailor_experience)
        workflow.add_node("optimize_keywords", self._optimize_keywords)
        workflow.add_node("finalize_tailored", self._finalize_tailored)
        
        # Define edges
        workflow.set_entry_point("extract_job_skills")
        workflow.add_edge("extract_job_skills", "analyze_match")
        workflow.add_edge("analyze_match", "tailor_summary")
        workflow.add_edge("tailor_summary", "tailor_experience")
        workflow.add_edge("tailor_experience", "optimize_keywords")
        workflow.add_edge("optimize_keywords", "finalize_tailored")
        workflow.add_edge("finalize_tailored", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def create_improvement_workflow(self) -> StateGraph:
        """Create a resume improvement workflow."""
        
        workflow = StateGraph(ImproveState)
        
        # Add nodes
        workflow.add_node("analyze_resume", self._analyze_resume)
        workflow.add_node("identify_improvements", self._identify_improvements)
        workflow.add_node("improve_sections", self._improve_sections)
        workflow.add_node("validate_improvements", self._validate_improvements)
        workflow.add_node("finalize_improved", self._finalize_improved)
        
        # Define edges
        workflow.set_entry_point("analyze_resume")
        workflow.add_edge("analyze_resume", "identify_improvements")
        workflow.add_edge("identify_improvements", "improve_sections")
        workflow.add_edge("improve_sections", "validate_improvements")
        workflow.add_conditional_edges(
            "validate_improvements",
            self._should_continue_improving,
            {
                "continue": "improve_sections",
                "finalize": "finalize_improved"
            }
        )
        workflow.add_edge("finalize_improved", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    # Node functions for resume generation workflow
    def _extract_keywords(self, state: ResumeState) -> ResumeState:
        """Extract keywords from job description."""
        try:
            if state.get("job_description"):
                keywords = self.llm.extract_skills_from_job_description(state["job_description"])
                state["keywords"] = keywords
            else:
                state["keywords"] = []
            state["step"] = "extract_keywords"
        except Exception as e:
            state["errors"].append(f"Error extracting keywords: {str(e)}")
        return state
    
    def _generate_summary(self, state: ResumeState) -> ResumeState:
        """Generate professional summary."""
        try:
            user_data = state["user_data"]
            tone = state["tone"]
            keywords = state.get("keywords", [])
            
            # Create enhanced prompt with keywords
            enhanced_data = {**user_data, "target_keywords": keywords}
            summary = self.llm.generate_resume_content(enhanced_data, tone)
            
            # Extract just the summary section
            state["generated_sections"]["summary"] = summary
            state["step"] = "generate_summary"
        except Exception as e:
            state["errors"].append(f"Error generating summary: {str(e)}")
        return state
    
    def _generate_experience(self, state: ResumeState) -> ResumeState:
        """Generate work experience section."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            user_data = state["user_data"]
            keywords = state.get("keywords", [])
            
            experience_prompt_template = PromptTemplate(
                input_variables=["user_data", "keywords"],
                template="""
                Generate a compelling work experience section for this resume.
                User data: {user_data}
                
                Focus on:
                1. Quantifiable achievements
                2. Action verbs
                3. Relevant skills and technologies
                4. Impact on business outcomes
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=experience_prompt_template)
            experience = chain.run(user_data=str(user_data), keywords=str(keywords))
            state["generated_sections"]["experience"] = experience
            state["step"] = "generate_experience"
        except Exception as e:
            state["errors"].append(f"Error generating experience: {str(e)}")
        return state
    
    def _generate_skills(self, state: ResumeState) -> ResumeState:
        """Generate skills section."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            user_data = state["user_data"]
            keywords = state.get("keywords", [])
            
            # Organize skills by relevance
            skills_prompt_template = PromptTemplate(
                input_variables=["skills", "keywords"],
                template="""
                Organize and enhance the skills section for this resume.
                User skills: {skills}
                Target job keywords: {keywords}
                
                Organize into:
                1. Technical Skills
                2. Soft Skills
                3. Tools & Technologies
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=skills_prompt_template)
            skills = chain.run(skills=user_data.get('skills', ''), keywords=str(keywords))
            state["generated_sections"]["skills"] = skills
            state["step"] = "generate_skills"
        except Exception as e:
            state["errors"].append(f"Error generating skills: {str(e)}")
        return state
    
    def _critique_resume(self, state: ResumeState) -> ResumeState:
        """Critique the generated resume."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            sections = state["generated_sections"]
            resume_text = "\n\n".join(sections.values())
            
            critique_prompt_template = PromptTemplate(
                input_variables=["resume_text"],
                template="""
                Critically analyze this resume and provide specific improvement suggestions.
                Resume: {resume_text}
                
                Focus on:
                1. Clarity and impact
                2. Quantification of achievements
                3. Relevance to target role
                4. Professional tone
                5. Action verbs usage
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=critique_prompt_template)
            critique = chain.run(resume_text=resume_text)
            state["critique"] = critique
            state["step"] = "critique_resume"
        except Exception as e:
            state["errors"].append(f"Error critiquing resume: {str(e)}")
        return state
    
    def _should_improve(self, state: ResumeState) -> str:
        """Decide whether to improve the resume based on critique."""
        critique = state.get("critique", "")
        # Simple heuristic: if critique mentions specific improvements, continue
        improvement_keywords = ["improve", "enhance", "strengthen", "better", "stronger"]
        if any(keyword in critique.lower() for keyword in improvement_keywords):
            return "improve"
        return "finalize"
    
    def _improve_resume(self, state: ResumeState) -> ResumeState:
        """Improve the resume based on critique."""
        try:
            sections = state["generated_sections"]
            critique = state["critique"]
            resume_text = "\n\n".join(sections.values())
            
            improved = self.llm.improve_resume_content(resume_text, [critique])
            state["generated_sections"]["improved"] = improved
            state["step"] = "improve_resume"
        except Exception as e:
            state["errors"].append(f"Error improving resume: {str(e)}")
        return state
    
    def _finalize_resume(self, state: ResumeState) -> ResumeState:
        """Finalize the resume."""
        try:
            sections = state["generated_sections"]
            if "improved" in sections:
                final_resume = sections["improved"]
            else:
                final_resume = "\n\n".join(sections.values())
            
            state["final_resume"] = final_resume
            state["step"] = "finalize_resume"
        except Exception as e:
            state["errors"].append(f"Error finalizing resume: {str(e)}")
        return state
    
    # Node functions for tailoring workflow
    def _extract_job_skills(self, state: TailorState) -> TailorState:
        """Extract skills from job description."""
        try:
            skills = self.llm.extract_skills_from_job_description(state["job_description"])
            state["extracted_skills"] = skills
            state["step"] = "extract_job_skills"
        except Exception as e:
            state["errors"] = [f"Error extracting job skills: {str(e)}"]
        return state
    
    def _analyze_match(self, state: TailorState) -> TailorState:
        """Analyze match between resume and job."""
        try:
            resume_text = state["original_resume"]
            job_desc = state["job_description"]
            match_score = calculate_match_percentage(resume_text, job_desc)
            state["match_analysis"] = {
                "score": match_score,
            }
            state["step"] = "analyze_match"
        except Exception as e:
            state["errors"] = [f"Error analyzing match: {str(e)}"]
        return state
    
    def _tailor_summary(self, state: TailorState) -> TailorState:
        """Tailor the summary section."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # Extract and tailor summary
            summary_prompt_template = PromptTemplate(
                input_variables=["original_resume", "job_description", "extracted_skills"],
                template="""
                Rewrite the summary section of this resume to better match the job description.
                
                Original Resume: {original_resume}
                Job Description: {job_description}
                Target Skills: {extracted_skills}
                
                Focus on highlighting relevant experience and skills.
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=summary_prompt_template)
            tailored_summary = chain.run(
                original_resume=state["original_resume"],
                job_description=state["job_description"],
                extracted_skills=str(state["extracted_skills"])
            )
            state["tailored_sections"]["summary"] = tailored_summary
            state["step"] = "tailor_summary"
        except Exception as e:
            state["errors"] = [f"Error tailoring summary: {str(e)}"]
        return state
    
    def _tailor_experience(self, state: TailorState) -> TailorState:
        """Tailor the experience section."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            experience_prompt_template = PromptTemplate(
                input_variables=["original_resume", "job_description", "extracted_skills"],
                template="""
                Rewrite the work experience section to better align with the job requirements.
                
                Original Resume: {original_resume}
                Job Description: {job_description}
                Target Skills: {extracted_skills}
                
                Emphasize relevant achievements and skills.
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=experience_prompt_template)
            tailored_experience = chain.run(
                original_resume=state["original_resume"],
                job_description=state["job_description"],
                extracted_skills=str(state["extracted_skills"])
            )
            state["tailored_sections"]["experience"] = tailored_experience
            state["step"] = "tailor_experience"
        except Exception as e:
            state["errors"] = [f"Error tailoring experience: {str(e)}"]
        return state
    
    def _optimize_keywords(self, state: TailorState) -> TailorState:
        """Optimize keyword usage."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            # Ensure target keywords are present
            keywords = state["extracted_skills"]
            resume_text = state["original_resume"]
            
            optimization_prompt_template = PromptTemplate(
                input_variables=["keywords", "resume_text"],
                template="""
                Optimize this resume to include the following keywords naturally:
                Keywords: {keywords}
                
                Resume: {resume_text}
                
                Ensure keywords are integrated naturally without keyword stuffing.
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=optimization_prompt_template)
            optimized = chain.run(keywords=str(keywords), resume_text=resume_text)
            state["tailored_sections"]["optimized"] = optimized
            state["step"] = "optimize_keywords"
        except Exception as e:
            state["errors"] = [f"Error optimizing keywords: {str(e)}"]
        return state
    
    def _finalize_tailored(self, state: TailorState) -> TailorState:
        """Finalize the tailored resume."""
        try:
            # Combine all tailored sections
            sections = state["tailored_sections"]
            final_resume = "\n\n".join(sections.values())
            state["final_resume"] = final_resume
            state["step"] = "finalize_tailored"
        except Exception as e:
            state["errors"] = [f"Error finalizing tailored resume: {str(e)}"]
        return state
    
    # Node functions for improvement workflow
    def _analyze_resume(self, state: ImproveState) -> ImproveState:
        """Analyze the resume for improvement opportunities."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            resume_text = state["original_resume"]
            focus_areas = state["improvement_focus"]
            
            analysis_prompt_template = PromptTemplate(
                input_variables=["resume_text", "focus_areas"],
                template="""
                Analyze this resume for improvement opportunities in: {focus_areas}
                
                Resume: {resume_text}
                
                Provide specific analysis of each area.
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=analysis_prompt_template)
            analysis = chain.run(resume_text=resume_text, focus_areas=str(focus_areas))
            state["analysis"] = {"text": analysis, "focus_areas": focus_areas}
            state["step"] = "analyze_resume"
        except Exception as e:
            state["errors"] = [f"Error analyzing resume: {str(e)}"]
        return state
    
    def _identify_improvements(self, state: ImproveState) -> ImproveState:
        """Identify specific improvements needed."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            analysis = state["analysis"]["text"]
            focus_areas = state["analysis"]["focus_areas"]
            
            improvements_prompt_template = PromptTemplate(
                input_variables=["analysis", "focus_areas"],
                template="""
                Based on this analysis, identify specific improvements needed:
                Analysis: {analysis}
                Focus Areas: {focus_areas}
                
                List specific actionable improvements.
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=improvements_prompt_template)
            improvements = chain.run(analysis=analysis, focus_areas=str(focus_areas))
            state["improvements"] = [improvements]
            state["step"] = "identify_improvements"
        except Exception as e:
            state["errors"] = [f"Error identifying improvements: {str(e)}"]
        return state
    
    def _improve_sections(self, state: ImproveState) -> ImproveState:
        """Improve specific sections."""
        try:
            resume_text = state["original_resume"]
            improvements = state["improvements"]
            
            improved = self.llm.improve_resume_content(resume_text, improvements)
            state["improved_sections"]["main"] = improved
            state["step"] = "improve_sections"
        except Exception as e:
            state["errors"] = [f"Error improving sections: {str(e)}"]
        return state
    
    def _validate_improvements(self, state: ImproveState) -> ImproveState:
        """Validate the improvements made."""
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
            
            original = state["original_resume"]
            improved = state["improved_sections"]["main"]
            
            validation_prompt_template = PromptTemplate(
                input_variables=["original", "improved"],
                template="""
                Compare the original and improved resumes:
                
                Original: {original}
                Improved: {improved}
                
                Are the improvements significant and address the focus areas?
                """
            )
            
            chain = LLMChain(llm=self.llm.llm, prompt=validation_prompt_template)
            validation = chain.run(original=original, improved=improved)
            state["analysis"]["validation"] = validation
            state["step"] = "validate_improvements"
        except Exception as e:
            state["errors"] = [f"Error validating improvements: {str(e)}"]
        return state
    
    def _should_continue_improving(self, state: ImproveState) -> str:
        """Decide whether to continue improving."""
        validation = state["analysis"].get("validation", "")
        if "significant" in validation.lower() and "yes" in validation.lower():
            return "finalize"
        return "continue"
    
    def _finalize_improved(self, ImproveState) -> ImproveState:
        """Finalize the improved resume."""
        try:
            improved = state["improved_sections"]["main"]
            state["final_resume"] = improved
            state["step"] = "finalize_improved"
        except Exception as e:
            state["errors"] = [f"Error finalizing improved resume: {str(e)}"]
        return state

# Global workflows instance
resume_workflows = ResumeWorkflows()

# Convenience functions
def get_resume_generation_workflow():
    """Get the resume generation workflow."""
    return resume_workflows.create_resume_generation_workflow()

def get_tailoring_workflow():
    """Get the resume tailoring workflow."""
    return resume_workflows.create_tailoring_workflow()

def get_improvement_workflow():
    """Get the resume improvement workflow."""
    return resume_workflows.create_improvement_workflow()

if __name__ == "__main__":
    print("Testing LangGraph workflows...")
    
    # Test workflow creation
    workflows = ResumeWorkflows()
    gen_workflow = workflows.create_resume_generation_workflow()
    tailor_workflow = workflows.create_tailoring_workflow()
    improve_workflow = workflows.create_improvement_workflow()
    
    print("Workflows created successfully!")
    print(f"Generation workflow nodes: {list(gen_workflow.nodes.keys())}")
    print(f"Tailoring workflow nodes: {list(tailor_workflow.nodes.keys())}")
    print(f"Improvement workflow nodes: {list(improve_workflow.nodes.keys())}") 