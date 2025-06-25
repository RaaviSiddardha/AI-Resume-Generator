def build_html_resume(data):
    """
    Builds a more robust HTML resume, correctly handling lists of items
    and including new fields like LinkedIn and GitHub.
    """
    # Helper to convert lists of items into HTML list items
    def to_li(items):
        return "".join(f"<li>{item}</li>" for item in items if item and item.strip())

    # Build contact string, including links if available
    contact_info = [
        f"<strong>Email:</strong> {data.get('email', 'N/A')}",
        f"<strong>Phone:</strong> {data.get('phone', 'N/A')}"
    ]
    if data.get('linkedin'):
        contact_info.append(f'<strong>LinkedIn:</strong> <a href="{data["linkedin"]}">{data["linkedin"]}</a>')
    if data.get('github'):
        contact_info.append(f'<strong>GitHub:</strong> <a href="{data["github"]}">{data["github"]}</a>')

    return f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: sans-serif; line-height: 1.6; }}
            h1, h2 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            ul {{ padding-left: 20px; }}
            p.contact-info {{ display: flex; flex-wrap: wrap; gap: 15px; }}
        </style>
    </head>
    <body>
        <h1>{data.get('name', 'N/A')}</h1>
        <p class="contact-info">
            {' | '.join(contact_info)}
        </p>

        <h2>Education</h2>
        <ul>{to_li(data.get('education', []))}</ul>

        <h2>Experience</h2>
        <ul>{to_li(data.get('experience', []))}</ul>
        
        <h2>Projects</h2>
        <ul>{to_li(data.get('projects', []))}</ul>

        <h2>Certifications</h2>
        <ul>{to_li(data.get('certifications', []))}</ul>

        <h2>Skills</h2>
        <ul>{''.join(f"<li>{skill.strip()}</li>" for skill in data.get('skills', '').split(','))}</ul>
    </body>
    </html>
    """

def modify_resume_text(text, prompt):
    """
    Simple text replacement based on a prompt.
    """
    if "replace" in prompt and "with" in prompt:
        try:
            _, remaining = prompt.split("replace", 1)
            old, new = remaining.split("with", 1)
            return text.replace(old.strip(), new.strip())
        except Exception:
            return text
    return text

def plain_text_to_html(text):
    """
    Wraps plain text in HTML <pre> tags for basic formatting.
    """
    return f"<html><body><pre>{text}</pre></body></html>" 