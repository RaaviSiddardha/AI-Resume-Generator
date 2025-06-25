from typing import Dict, Any, Tuple
from pathlib import Path
from io import BytesIO

from .llm import (
    generate_resume_content_hf,
    tailor_resume_to_job,
    improve_resume_content
)
from .embeddings import calculate_match_percentage
from .pdf import parse_pdf, generate_pdf_from_html, save_generated_pdf

def process_generate_resume(user_data: Dict[str, Any], tone: str, template: str, api_key: str) -> Path:
    """
    Orchestrates the resume generation process.
    1. Generates content with LLM.
    2. Formats it into HTML using a template.
    3. Converts the HTML to a PDF.
    """
    # 1. Generate resume content
    ai_generated_text = generate_resume_content_hf(user_data, tone, api_key)
    
    # Combine user data with AI generated content
    full_resume_data = {**user_data, "full_text": ai_generated_text}

    # 2. Generate PDF from HTML template
    template_name = f"{template.lower()}.html"
    pdf_bytes = generate_pdf_from_html(full_resume_data, template_name)

    # 3. Save the PDF
    pdf_path = save_generated_pdf(pdf_bytes, filename=f"{user_data.get('name', 'resume').replace(' ', '_')}_resume.pdf")
    
    return pdf_path

def process_improve_resume(uploaded_file: BytesIO, api_key: str) -> Path:
    """
    Orchestrates the resume improvement process.
    1. Parses text from uploaded PDF.
    2. Sends text to LLM for improvement.
    3. (For a full implementation) Parses the improved text back into a structured format.
    4. Generates a new PDF.
    """
    # 1. Parse PDF
    resume_text = parse_pdf(uploaded_file)
    
    # 2. Improve content
    improved_text = improve_resume_content(resume_text, api_key)
    
    # Placeholder: In a real app, you'd parse 'improved_text' into a structured
    # dict and use a template. For now, we'll make a simple PDF.
    # This part would require more sophisticated parsing logic.
    resume_data = {"full_text": improved_text}
    pdf_bytes = generate_pdf_from_html(resume_data, 'modern.html') # Using a default template

    # 3. Save PDF
    pdf_path = save_generated_pdf(pdf_bytes, filename="improved_resume.pdf")

    return pdf_path


def process_tailor_resume(uploaded_file: BytesIO, job_description: str, api_key: str) -> Tuple[Path, int]:
    """
    Orchestrates the resume tailoring process.
    1. Parses text from the PDF.
    2. Calculates an initial match score.
    3. Sends resume and job description to LLM for tailoring.
    4. Recalculates match score on the new resume.
    5. Generates the final PDF.
    """
    # 1. Parse PDF
    resume_text = parse_pdf(uploaded_file)
    
    # 2. Tailor content
    tailored_text = tailor_resume_to_job(resume_text, job_description, api_key)
    
    # 3. Calculate new match score
    match_score = calculate_match_percentage(tailored_text, job_description)
    
    # 4. Generate PDF
    # As with improve, this would ideally be parsed and templated.
    resume_data = {"full_text": tailored_text}
    pdf_bytes = generate_pdf_from_html(resume_data, 'modern.html') # Default template

    # 5. Save PDF
    pdf_path = save_generated_pdf(pdf_bytes, filename="tailored_resume.pdf")

    return pdf_path, match_score 