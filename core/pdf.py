from io import StringIO, BytesIO
from pathlib import Path
from typing import Dict, Any
import base64

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from jinja2 import Environment, FileSystemLoader

# Try to import PyPDF2, fallback to pdfminer if not available
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

def parse_pdf(file: BytesIO) -> str:
    """
    Extracts text from an uploaded PDF file.
    """
    resource_manager = PDFResourceManager()
    string_io = StringIO()
    converter = TextConverter(resource_manager, string_io, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    for page in PDFPage.get_pages(file, check_extractable=True):
        page_interpreter.process_page(page)

    text = string_io.getvalue()

    converter.close()
    string_io.close()

    return text

def generate_pdf_from_html(resume_data: Dict[str, Any], template_name: str = "modern.html") -> bytes:
    """
    Generates a PDF from HTML template with resume data.
    Returns HTML content as bytes (cloud-compatible approach).
    """
    try:
        # Load the template
        env = Environment(loader=FileSystemLoader('templates/'))
        template = env.get_template(template_name)
        
        # Render the template with resume data
        html_content = template.render(**resume_data)
        
        # Add CSS for better styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Resume - {resume_data.get('name', 'Generated Resume')}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .resume-container {{
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    border-radius: 8px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                    margin-top: 0;
                }}
                h1 {{
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    border-bottom: 2px solid #ecf0f1;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }}
                .section {{
                    margin-bottom: 25px;
                }}
                .contact-info {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .experience-item, .education-item {{
                    margin-bottom: 15px;
                    padding-left: 15px;
                    border-left: 3px solid #3498db;
                }}
                .skills {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }}
                .skill-tag {{
                    background-color: #3498db;
                    color: white;
                    padding: 5px 12px;
                    border-radius: 15px;
                    font-size: 0.9em;
                }}
                @media print {{
                    body {{ background-color: white; }}
                    .resume-container {{ box-shadow: none; }}
                }}
            </style>
        </head>
        <body>
            <div class="resume-container">
                {html_content}
            </div>
        </body>
        </html>
        """
        
        return styled_html.encode('utf-8')
        
    except Exception as e:
        print(f"An error occurred during HTML generation: {e}")
        # Fallback: return simple HTML content as bytes
        fallback_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Resume - {resume_data.get('name', 'Generated Resume')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
            <h1>{resume_data.get('name', 'Resume')}</h1>
            <div>{resume_data.get('full_text', 'Resume content')}</div>
        </body>
        </html>
        """
        return fallback_html.encode('utf-8')

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file using PyPDF2 or pdfminer as fallback.
    """
    if PYPDF2_AVAILABLE:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}, falling back to pdfminer")
    
    # Fallback to pdfminer
    try:
        return parse_pdf(uploaded_file)
    except Exception as e:
        print(f"PDF text extraction failed: {e}")
        return ""

def save_generated_pdf(pdf_bytes: bytes, filename: str = "generated_resume.html") -> Path:
    """Saves the generated HTML bytes to a file."""
    output_path = Path(filename)
    output_path.write_bytes(pdf_bytes)
    return output_path

def get_download_link(html_bytes: bytes, filename: str = "resume.html") -> str:
    """Creates a download link for the HTML file."""
    b64 = base64.b64encode(html_bytes).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}">Download Resume</a>' 