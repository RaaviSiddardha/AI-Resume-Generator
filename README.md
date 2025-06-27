![CI/CD](https://github.com/RaaviSiddardha/AI-Resume-Generator/actions/workflows/ci-cd.yml/badge.svg)

# 🤖 AI-Powered Resume Generator with LangChain & HuggingFace

A sophisticated resume generation and optimization platform powered by LangChain and HuggingFace, featuring advanced AI capabilities for creating, improving, and tailoring professional resumes.

## ✨ Features

### 🚀 Core Capabilities
- **AI-Powered Resume Generation**: Create professional resumes from scratch using advanced LLMs
- **Smart Resume Improvement**: Enhance existing resumes with AI-driven suggestions
- **Job-Specific Tailoring**: Optimize resumes for specific job descriptions
- **Multiple Templates**: Choose from Modern, Formal, and Creative templates
- **PDF Export**: Generate professional PDF resumes

### 🔧 Advanced Features
- **LangChain Integration**: Leverage LangChain's powerful LLM workflows
- **HuggingFace Models**: Support for multiple HuggingFace models (local and cloud)
- **Resume Scoring**: Comprehensive scoring system with detailed feedback
- **A/B Testing**: Test multiple resume variations to find the best version
- **Analytics Dashboard**: Track resume performance and improvements over time
- **Configuration Management**: Easy model switching and settings management

### 🎯 Smart Capabilities
- **Keyword Optimization**: Automatically align resume with job requirements
- **Skill Extraction**: Extract and match skills from job descriptions
- **Semantic Matching**: Advanced embedding-based job-resume matching
- **Professional Tone Analysis**: Ensure appropriate professional language
- **Quantification Suggestions**: Add measurable achievements to resumes

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Framework**: LangChain
- **Models**: HuggingFace Transformers
- **Embeddings**: Sentence Transformers
- **PDF Generation**: pdfkit
- **Configuration**: JSON-based config system

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Resume-Generator
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

4. **Run the application**:
```bash
streamlit run app.py
```

## 🔧 Configuration

### Model Selection
The application supports multiple LLM and embedding models:

**LLM Models**:
- Mistral 7B (HuggingFace)
- Llama 2 7B (HuggingFace)
- GPT-3.5 Turbo (OpenAI)
- GPT-4 (OpenAI)

**Embedding Models**:
- All-MiniLM-L6 (HuggingFace)
- All-MPNet-Base (HuggingFace)
- OpenAI Ada (OpenAI)

### Configuration Management
- Use the sidebar to switch between models
- Configuration is automatically saved
- Support for both HuggingFace Hub and local models

## 🚀 Usage

### 1. Generate New Resume
1. Fill in your personal information
2. Add education, experience, projects, and certifications
3. Select tone and template
4. Configure advanced options (optional)
5. Generate your AI-enhanced resume

### 2. Improve Existing Resume
1. Upload your current resume (PDF)
2. Select improvement focus areas
3. Add specific instructions (optional)
4. Get an improved version with detailed feedback

### 3. Tailor Resume for Job
1. Upload your resume
2. Paste the job description
3. Configure tailoring options
4. Get a job-specific version with match score

### 4. Advanced Features
- **Resume Scoring**: Get detailed analysis and improvement suggestions
- **A/B Testing**: Create and compare multiple resume variations
- **Analytics**: Track your resume performance over time

## 📊 Advanced Features

### Resume Scoring
The scoring system evaluates:
- **Length Score**: Optimal resume length (400-800 words)
- **Readability Score**: Flesch Reading Ease analysis
- **Action Verbs Score**: Usage of strong action verbs
- **Quantification Score**: Presence of measurable achievements
- **Professional Tone Score**: Language appropriateness
- **Job Match Score**: Alignment with job description
- **Structure Score**: Organization and completeness

### A/B Testing
Create multiple variations of your resume:
- **Action Verbs Enhancement**: Stronger action verb usage
- **Quantification**: Add specific metrics and numbers
- **Keyword Optimization**: Natural keyword integration
- **Concise Version**: Shorter, more focused content
- **Detailed Version**: Expanded descriptions and context

### Analytics Dashboard
Track your resume performance:
- **Score Trends**: Monitor improvement over time
- **Common Issues**: Identify recurring problems
- **Performance Metrics**: Average scores and distributions
- **Usage Statistics**: Track feature utilization

## 🔧 Customization

### Adding New Models
1. Update `core/config.py` with new model configurations
2. Add model details to the `llm_models` or `embedding_models` dictionaries
3. Restart the application

### Custom Templates
1. Add new HTML templates to the `templates/` directory
2. Update the template selection in the UI
3. Ensure templates follow the existing structure

### Advanced Workflows
The application includes LangGraph workflows for complex resume generation:
- **Resume Generation Workflow**: Multi-step generation with critique and improvement
- **Tailoring Workflow**: Job-specific optimization pipeline
- **Improvement Workflow**: Iterative enhancement process

## 🏗️ Architecture

```
Resume-Generator/
├── app.py                 # Main Streamlit application
├── core/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── llm.py           # LangChain LLM integration
│   ├── embeddings.py    # Embedding models and similarity
│   ├── logic.py         # Business logic and orchestration
│   ├── pdf.py           # PDF generation utilities
│   ├── utils.py         # Utility functions
│   ├── workflows.py     # LangGraph workflows
│   └── advanced_features.py # Scoring, A/B testing, analytics
├── templates/            # HTML resume templates
├── tests/               # Unit tests
└── requirements.txt     # Python dependencies
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For the powerful LLM framework
- **HuggingFace**: For the excellent model ecosystem
- **Streamlit**: For the intuitive web interface
- **Sentence Transformers**: For embedding capabilities

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**Built with ❤️ using LangChain, HuggingFace, and Streamlit** 
