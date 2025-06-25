from io import StringIO, BytesIO
from pathlib import Path
from typing import Dict, Any

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
import pdfkit
from jinja2 import Environment, FileSystemLoader
import PyPDF2

# Explicitly configure the path to the wkhtmltopdf executable.
# This is a robust way to avoid PATH issues.
try:
    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
except FileNotFoundError:
    config = None

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
    Returns PDF bytes.
    """
    try:
        # Load the template
        env = Environment(loader=FileSystemLoader('templates/'))
        template = env.get_template(template_name)
        
        # Render the template with resume data
        html_content = template.render(**resume_data)
        
        if config is None:
            print("[WARNING] wkhtmltopdf not found. Creating HTML file instead.")
            # Fallback: save as HTML
            html_path = Path("resume.html")
            html_path.write_text(html_content, encoding='utf-8')
            return html_content.encode('utf-8')

        options = {
            'enable-local-file-access': None,
            'page-size': 'A4',
            'encoding': "UTF-8",
            'quiet': ''
        }
        
        # Generate PDF bytes
        pdf_bytes = pdfkit.from_string(html_content, False, configuration=config, options=options)
        return pdf_bytes
        
    except Exception as e:
        print(f"An error occurred during PDF generation: {e}")
        # Fallback: return HTML content as bytes
        fallback_html = f"""
        <html>
        <body>
        <h1>{resume_data.get('name', 'Resume')}</h1>
        <p>{resume_data.get('full_text', 'Resume content')}</p>
        </body>
        </html>
        """
        return fallback_html.encode('utf-8')

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file using PyPDF2.
    """
    reader = PyPDF2.PdfReader(uploaded_file)
    return "\n".join([page.extract_text() for page in reader.pages])

def save_generated_pdf(pdf_bytes: bytes, filename: str = "generated_resume.pdf") -> Path:
    """Saves the generated PDF bytes to a file."""
    output_path = Path(filename)
    output_path.write_bytes(pdf_bytes)
    return output_path 