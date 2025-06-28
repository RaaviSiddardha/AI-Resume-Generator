import streamlit as st
import os
from dotenv import load_dotenv
from core.pdf import generate_pdf_from_html, extract_text_from_pdf
from core.utils import build_html_resume, modify_resume_text, plain_text_to_html
from core.logic import (
    process_generate_resume,
    process_improve_resume,
    process_tailor_resume
)
from core.config import get_st_config, update_st_config, llm_models, embedding_models, default_llm, default_embedding, get_llm_config, get_embedding_config
from core.llm import resume_llm
from core.embeddings import resume_embeddings
import pandas as pd
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Initialize configuration
config = get_st_config()

# Configure API keys
api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not api_key and not hf_token:
    st.error("No API keys found. Please set OPENAI_API_KEY or HF_TOKEN in your .env file.")
    st.stop()

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="AI Resume Generator", layout="wide")
st.title("🤖 AI-Powered Resume Generator with LangChain & HuggingFace")

st.markdown("""
<div style='
    background: linear-gradient(to right, #6a11cb, #2575fc);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    font-size: 1.1rem;
    margin-bottom: 25px;
'>
    <h2 style='margin-bottom: 10px;'>🧠 Smart Resumes Start Here</h2>
    <p>
        Powered by LangChain and HuggingFace - Generate, improve, and tailor resumes with advanced AI.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    st.subheader("Model Settings")
    llm_model = st.selectbox(
        "LLM Model",
        list(llm_models.keys()),
        index=list(llm_models.keys()).index(default_llm)
    )
    
    embedding_model = st.selectbox(
        "Embedding Model",
        list(embedding_models.keys()),
        index=list(embedding_models.keys()).index(default_embedding)
    )
    
    # Update configuration
    if st.button("Update Configuration"):
        update_st_config({
            "llm": {"name": llm_model, "provider": llm_models[llm_model]["provider"]},
            "embedding": {"name": embedding_model, "provider": embedding_models[embedding_model]["provider"]}
        })
        st.success("Configuration updated!")
    
    # API Status
    st.subheader("API Status")
    if api_key:
        st.success("✅ OpenAI API Key")
    if hf_token:
        st.success("✅ HuggingFace Token")
    
    # Advanced features toggle
    st.subheader("Advanced Features")
    enable_scoring = st.checkbox("Enable Resume Scoring", value=True)
    enable_ab_testing = st.checkbox("Enable A/B Testing", value=False)
    enable_analytics = st.checkbox("Enable Analytics", value=False)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🆕 Generate New Resume",
    "🛠️ Improve Existing Resume", 
    "🎯 Tailor Resume",
    "📊 Advanced Features"
])

# ----------------------------
# Generate New Resume
# ----------------------------
with tab1:
    st.subheader("Generate New Resume")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone Number")
        linkedin = st.text_input("LinkedIn Profile URL")
        github = st.text_input("GitHub Profile URL")
    
    with col2:
        # Dynamic sections
        for section, key_prefix in [
            ("Education", "edu"), 
            ("Experience", "exp"), 
            ("Projects", "proj"), 
            ("Certifications", "cert")
        ]:
            st.markdown(f"### {section}")
            if f"{key_prefix}_count" not in st.session_state:
                st.session_state[f"{key_prefix}_count"] = 1
            
            for i in range(st.session_state[f"{key_prefix}_count"]):
                if section == "Experience":
                    st.text_area(f"{section} #{i + 1}", key=f"{key_prefix}_{i}")
                else:
                    st.text_input(f"{section} #{i + 1}", key=f"{key_prefix}_{i}")
            
            if st.button(f"➕ Add {section}", key=f"add_{key_prefix}"):
                st.session_state[f"{key_prefix}_count"] += 1
                st.rerun()
    
    skills = st.text_input("Skills (comma-separated)")
    
    # Generation options
    col1, col2 = st.columns(2)
    with col1:
        tone = st.selectbox(
            "Select Resume Tone",
            ["Professional", "Casual", "Technical", "Executive"],
            index=0
        )
    
    with col2:
        template = st.selectbox(
            "Select Resume Template",
            ["Modern", "Formal", "Creative"],
            index=0
        )
    
    # Advanced generation options
    with st.expander("Advanced Generation Options"):
        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider("Max Resume Length (words)", 300, 1500, 800)
            temperature = st.slider("Creativity Level", 0.1, 1.0, 0.7)
        
        with col2:
            focus_areas = st.multiselect(
                "Focus Areas",
                ["Technical Skills", "Leadership", "Quantifiable Achievements", "Industry-Specific"],
                default=["Technical Skills", "Quantifiable Achievements"]
            )
    
    if st.button("🚀 Generate Resume", type="primary"):
        with st.spinner("🤖 AI is crafting your professional resume..."):
            # Gather all the multi-entry fields
            education_list = [st.session_state.get(f"edu_{i}", "") for i in range(st.session_state.edu_count)]
            experience_list = [st.session_state.get(f"exp_{i}", "") for i in range(st.session_state.exp_count)]
            projects_list = [st.session_state.get(f"proj_{i}", "") for i in range(st.session_state.proj_count)]
            certs_list = [st.session_state.get(f"cert_{i}", "") for i in range(st.session_state.cert_count)]

            user_data = {
                "name": name, "email": email, "phone": phone,
                "linkedin": linkedin, "github": github,
                "education": education_list, "experience": experience_list,
                "projects": projects_list, "certifications": certs_list,
                "skills": skills, "focus_areas": focus_areas
            }
            
            try:
                # Use enhanced LLM with LangChain
                ai_generated_text = resume_llm.generate_resume_content(user_data, tone.lower())
                
                # Combine user data with AI generated content
                full_resume_data = {**user_data, "full_text": ai_generated_text}
                
                # Generate PDF
                pdf_path = process_generate_resume(user_data, tone.lower(), template.lower(), api_key)
                
                st.success("✨ Resume generated successfully with LangChain & HuggingFace!")
                
                # Show generation details
                with st.expander("Generation Details"):
                    st.write("**Model Used:**", get_llm_config()['name'])
                    st.write("**Tone:**", tone)
                    st.write("**Template:**", template)
                    st.write("**Word Count:**", len(ai_generated_text.split()))
                
                # Download button
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download AI-Enhanced Resume",
                        data=f,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf"
                    )
                
                # Show generated content
                with st.expander("Generated Content Preview"):
                    st.text_area("AI Generated Content", ai_generated_text, height=300)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# ----------------------------
# Improve Existing Resume
# ----------------------------
with tab2:
    st.subheader("Improve an Existing Resume")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    
    col1, col2 = st.columns(2)
    with col1:
        improvement_type = st.multiselect(
            "What aspects would you like to improve?",
            ["Professional Tone", "Action Verbs", "Quantify Achievements", "Technical Language", "Leadership Focus", "Clarity", "Impact"],
            default=["Professional Tone", "Action Verbs"]
        )
    
    with col2:
        specific_instructions = st.text_area(
            "Any specific improvements you'd like? (Optional)",
            placeholder="E.g., 'Focus more on leadership skills' or 'Emphasize technical achievements'"
        )
    
    # Advanced improvement options
    with st.expander("Advanced Improvement Options"):
        col1, col2 = st.columns(2)
        with col1:
            improvement_strength = st.slider("Improvement Strength", 1, 5, 3)
            preserve_style = st.checkbox("Preserve Original Style", value=True)
        
        with col2:
            target_length = st.selectbox("Target Length", ["Keep Original", "Make Shorter", "Add More Detail"])
            focus_sections = st.multiselect(
                "Focus on Sections",
                ["Summary", "Experience", "Skills", "Education", "All"],
                default=["All"]
            )

    if st.button("🛠️ Improve Resume"):
        if uploaded_file:
            with st.spinner("🤖 AI is enhancing your resume..."):
                try:
                    # Process the resume with enhanced LLM
                    pdf_path = process_improve_resume(uploaded_file, api_key)
                    
                    st.success("✨ Resume improved successfully!")
                    
                    # Show improvement details
                    with st.expander("Improvement Details"):
                        st.write("**Improvement Focus:**", ", ".join(improvement_type))
                        st.write("**Model Used:**", get_llm_config()['name'])
                    
                    # Download button
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📄 Download Improved Resume",
                            data=f,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload a resume to improve.")

# ----------------------------
# Tailor Resume Tab
# ----------------------------
with tab3:
    st.subheader("Tailor Resume for a Job Description")

    uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf", key="tailor_pdf")
    job_desc = st.text_area(
        "Paste the Job Description",
        placeholder="Paste the full job description here. The AI will analyze it and tailor your resume accordingly."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        highlight_keywords = st.checkbox("Highlight Matching Keywords", value=True)
        show_match_score = st.checkbox("Show Match Score", value=True)
    
    with col2:
        tailoring_strength = st.slider("Tailoring Strength", 1, 5, 3)
        preserve_original = st.checkbox("Preserve Original Content", value=False)
    
    # Advanced tailoring options
    with st.expander("Advanced Tailoring Options"):
        col1, col2 = st.columns(2)
        with col1:
            focus_sections = st.multiselect(
                "Tailor These Sections",
                ["Summary", "Experience", "Skills", "All"],
                default=["All"]
            )
            keyword_optimization = st.checkbox("Optimize Keywords", value=True)
        
        with col2:
            skill_alignment = st.checkbox("Align Skills", value=True)
            experience_relevance = st.checkbox("Prioritize Relevant Experience", value=True)

    if st.button("🎯 Tailor Resume"):
        if uploaded_file and job_desc:
            with st.spinner("🤖 AI is tailoring your resume to the job description..."):
                try:
                    # Process the resume with enhanced tailoring
                    pdf_path, match_score = process_tailor_resume(uploaded_file, job_desc, api_key)
                    
                    # Show match score if requested
                    if show_match_score:
                        score_color = "green" if match_score >= 70 else "orange" if match_score >= 50 else "red"
                        st.markdown(f"""
                        <div style='padding: 20px; border-radius: 10px; background-color: #f0f2f6;'>
                            <h3 style='margin: 0;'>Resume-Job Match Score</h3>
                            <p style='font-size: 24px; margin: 10px 0; color: {score_color};'>{match_score}%</p>
                            <p style='margin: 0; font-size: 14px;'>Based on skill matching and content relevance</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show tailoring details
                    with st.expander("Tailoring Details"):
                        st.write("**Match Score:**", f"{match_score}%")
                        st.write("**Model Used:**", get_llm_config()['name'])
                        st.write("**Embedding Model:**", get_embedding_config()['name'])
                    
                    # Download button
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📄 Download Tailored Resume",
                            data=f,
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf"
                        )
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload a resume and provide a job description.")

# ----------------------------
# Advanced Features Tab
# ----------------------------
with tab4:
    st.subheader("📊 Advanced Features")
    
    if not enable_scoring and not enable_ab_testing and not enable_analytics:
        st.info("Enable advanced features in the sidebar to access these tools.")
    
    if enable_scoring:
        st.markdown("### 📈 Resume Scoring")
        
        col1, col2 = st.columns(2)
        with col1:
            scoring_resume = st.text_area(
                "Paste resume text for scoring",
                height=200,
                placeholder="Paste your resume text here..."
            )
        
        with col2:
            scoring_job = st.text_area(
                "Job description (optional)",
                height=200,
                placeholder="Paste job description for targeted scoring..."
            )
        
        if st.button("📊 Score Resume") and scoring_resume:
            with st.spinner("Analyzing resume..."):
                try:
                    # Import scoring function
                    from core.advanced_features import score_resume
                    scores = score_resume(scoring_resume, scoring_job)
                    
                    # Display scores
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Score", f"{scores['overall_score']:.1f}/100")
                    with col2:
                        st.metric("Job Match", f"{scores.get('job_match_score', 0):.1f}/100")
                    with col3:
                        st.metric("Action Verbs", f"{scores.get('action_verbs_score', 0):.1f}/100")
                    
                    # Detailed scores
                    with st.expander("Detailed Scores"):
                        score_df = pd.DataFrame([
                            {"Metric": k.replace("_", " ").title(), "Score": f"{v:.1f}/100"}
                            for k, v in scores.items() if isinstance(v, (int, float)) and "score" in k.lower()
                        ])
                        st.dataframe(score_df, use_container_width=True)
                    
                    # Feedback
                    if scores.get("feedback"):
                        st.markdown("### 💡 Improvement Suggestions")
                        for feedback in scores["feedback"]:
                            st.write(f"• {feedback}")
                
                except Exception as e:
                    st.error(f"Error scoring resume: {str(e)}")
    
    if enable_ab_testing:
        st.markdown("### 🔬 A/B Testing")
        
        ab_resume = st.text_area(
            "Paste resume for A/B testing",
            height=150,
            placeholder="Paste your resume text here..."
        )
        
        ab_job = st.text_area(
            "Job description for testing",
            height=100,
            placeholder="Paste job description..."
        )
        
        if st.button("🔬 Run A/B Test") and ab_resume:
            with st.spinner("Creating and testing variations..."):
                try:
                    from core.advanced_features import create_ab_test
                    results = create_ab_test(ab_resume, ab_job)
                    
                    st.success(f"Best variation: {results['best_variation']}")
                    
                    # Show comparison
                    if "comparison" in results:
                        st.dataframe(results["comparison"], use_container_width=True)
                    
                    # Show best variation
                    best_variation = results["variations"][results["best_variation"]]
                    with st.expander("Best Variation Preview"):
                        st.text_area("Best Variation", best_variation["variation"], height=200)
                
                except Exception as e:
                    st.error(f"Error in A/B testing: {str(e)}")
    
    if enable_analytics:
        st.markdown("### 📊 Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 View Analytics"):
                try:
                    from core.advanced_features import get_analytics_insights
                    insights = get_analytics_insights()
                    
                    if "message" in insights:
                        st.info(insights["message"])
                    else:
                        st.metric("Total Resumes", insights.get("total_resumes", 0))
                        st.metric("Average Score", f"{insights.get('average_overall_score', 0):.1f}")
                        
                        # Score distribution
                        distribution = insights.get("score_distribution", {})
                        st.markdown("**Score Distribution:**")
                        for category, count in distribution.items():
                            st.write(f"• {category.title()}: {count}")
                
                except Exception as e:
                    st.error(f"Error loading analytics: {str(e)}")
        
        with col2:
            if st.button("🗑️ Clear Analytics"):
                try:
                    from core.advanced_features import resume_analytics
                    resume_analytics.data = {"resumes": [], "tests": [], "insights": []}
                    resume_analytics.save_data()
                    st.success("Analytics cleared!")
                except Exception as e:
                    st.error(f"Error clearing analytics: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using LangChain, HuggingFace, and Streamlit</p>
    <p>Advanced AI-powered resume generation and optimization</p>
</div>
""", unsafe_allow_html=True)

def generate_pdf_from_html(resume_data, template_name="modern.html"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add Name
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, resume_data.get("name", "Name"), ln=True)
    pdf.set_font("Arial", size=12)

    # Add Contact Info
    pdf.cell(0, 10, f"Email: {resume_data.get('email', '')}", ln=True)
    pdf.cell(0, 10, f"Phone: {resume_data.get('phone', '')}", ln=True)
    pdf.cell(0, 10, f"LinkedIn: {resume_data.get('linkedin', '')}", ln=True)
    pdf.cell(0, 10, f"GitHub: {resume_data.get('github', '')}", ln=True)
    pdf.ln(5)

    # Add Sections
    def add_section(title, items):
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", size=12)
        for item in items:
            pdf.multi_cell(0, 10, f"- {item}")
        pdf.ln(2)

    add_section("Education", resume_data.get("education", []))
    add_section("Experience", resume_data.get("experience", []))
    add_section("Projects", resume_data.get("projects", []))
    add_section("Certifications", resume_data.get("certifications", []))

    # Add Skills
    skills = resume_data.get("skills", "")
    if skills:
        add_section("Skills", [skills])

    # Add AI-generated text if present
    if "full_text" in resume_data:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "AI Generated Content", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, resume_data["full_text"])

    return pdf.output(dest="S").encode("latin1")