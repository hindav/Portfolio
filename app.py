from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import re
import os
from datetime import date
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================
# FASTAPI INIT
# =========================
app = FastAPI(title="Hindav Profile Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mount will be added at the end of file to ensure API routes take precedence

# =========================
# OPENROUTER CONFIGURATION
# =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Choose between the two free models:
# - google/gemini-2.0-flash-exp:free
# - meta-llama/llama-3.2-3b-instruct:free
SELECTED_MODEL = "meta-llama/llama-3.2-3b-instruct:free"

# =========================
# EXPANDED PROFILE DATA
# =========================
PROFILE = {
    "identity": {
        "name": "Hindav Deshmukh",
        "date_of_birth": "2002-11-15",
        "pronouns": "He/Him",
        "location": "Pune, Maharashtra, India",
        "nationality": "Indian",
        "languages": ["English", "Hindi", "Marathi"]
    },
    "education": {
        "degree": "Bachelor of Engineering in Computer Science",
        "graduation_year": 2024,
        "college": "Jawaharlal Darda Institute of Engineering and Technology",
        "university": "Sant Gadge Baba Amravati University",
        "cgpa": 7.23,
        "highlights": [
            "Focused on Machine Learning and Deep Learning coursework",
            "Completed projects in Computer Vision and Time Series Forecasting",
            "Strong foundation in Data Structures, Algorithms, and System Design"
        ]
    },
    "career": {
        "status": "Machine Learning Engineer",
        "experience_years": 1.5,
        "current_company": "Newgen Technomate LLP",
        "current_role": "Machine Learning Engineer",
        "start_date": "2024-09-02",
        "end_date": "2026-02-21",
        "promotion_date": "2025-07-01",
        "previous_title": "Junior Machine Learning Engineer",
        "domain": "Healthcare",
        "primary_focus": "Machine Learning and Healthcare Analytics",
        "secondary_focus": "Predictive Modeling and Web Integration",
        "target_roles": [
            "Senior Machine Learning Engineer",
            "ML Engineer - Healthcare",
            "AI/ML Engineer",
            "Data Scientist"
        ]
    },
    "experience": [
        {
            "company": "Newgen Technomate LLP",
            "role": "Machine Learning Engineer",
            "duration": "September 2024 - February 2026",
            "previous_role": "Junior Machine Learning Engineer (September 2024 - June 2025)",
            "domain": "Healthcare",
            "key_project": {
                "name": "Predictive Adverse Drug Reaction (ADR) Management System",
                "description": "ML-based system to predict adverse drug reactions in patients using ensemble algorithms",
                "technologies": ["Python", "Random Forest", "Gradient Boosting", "Scikit-learn", "Pandas", "NumPy", "Flask"],
                "responsibilities": [
                    "Data preprocessing and feature engineering on patient and drug datasets",
                    "Model development using ensemble algorithms (Random Forest, Gradient Boosting)",
                    "Model training, evaluation, and hyperparameter tuning achieving 92% accuracy",
                    "Integration of ML models into web applications for real-time predictions",
                    "Performance optimization and model deployment"
                ],
                "impact": "Enhanced patient safety through accurate ADR prediction system"
            }
        }
    ],
    "skills": {
        "languages": ["Python", "SQL", "HTML", "CSS"],
        "ml": [
            "Machine Learning", "Ensemble Learning", "Random Forest", "Gradient Boosting",
            "Deep Learning", "LSTM", "RNNs", "Time Series Forecasting", "Neural Networks",
            "Predictive Modeling", "Healthcare Analytics", "LLM Integration", "Sentiment Analysis",
            "Computer Vision", "OpenCV", "EasyOCR"
        ],
        "backend": ["FastAPI", "Flask", "REST APIs", "Streamlit", "Plotly"],
        "databases": ["SQL", "PostgreSQL", "MongoDB"],
        "data_science": ["Pandas", "NumPy", "Scikit-learn", "Data Preprocessing", "Feature Engineering", "Model Evaluation", "Geospatial Analysis"],
        "tools": ["Git", "GitHub", "Docker", "VS Code", "Jupyter Notebooks", "PyMuPDF"],
        "other": ["Problem Solving", "System Design", "ML Model Deployment"]
    },
    "projects": [
        {
            "name": "Predictive ADR Management System",
            "type": "Professional",
            "description": "Healthcare ML system for predicting adverse drug reactions using ensemble algorithms (92% accuracy).",
            "technologies": ["Python", "Random Forest", "Gradient Boosting", "Flask", "Scikit-learn"]
        },
        {
            "name": "Excel to Visualise Data",
            "type": "Data Visualization",
            "description": "Streamlit app to transform Excel data into interactive Plotly visualizations and dashboards with PPT export.",
            "technologies": ["Python", "Streamlit", "Plotly", "Pandas"]
        },
        {
            "name": "Digital Toolbox",
            "type": "Utility Web App",
            "description": "Versatile web suite for image processing, OCR text extraction (EasyOCR), and PDF management.",
            "technologies": ["Python", "Streamlit", "EasyOCR", "PyMuPDF", "OpenCV"]
        },
        {
            "name": "StoxAi: Hybrid Stock Prediction System",
            "type": "AI Portfolio",
            "description": "Hybrid stock predictor using LSTM and LLM sentiment analysis (GPT-4) with 98% accuracy.",
            "technologies": ["Python", "TensorFlow", "FastAPI", "GPT-4", "Streamlit", "Plotly"]
        },
        {
            "name": "Movie Character Recognition System",
            "type": "Computer Vision",
            "description": "AI-powered system to identify movie characters in photos/videos (Research paper published).",
            "technologies": ["Python", "Flask", "OpenCV", "Deep Learning", "CNN"]
        },
        {
            "name": "Mumbai Housing Prices 2025",
            "type": "Data Science",
            "description": "Geospatial analysis and price prediction on 20K+ Mumbai listings.",
            "technologies": ["Python", "Pandas", "Scikit-learn", "Geospatial Analysis"]
        }
    ],
    "certifications": [
        "Machine Learning Specialization",
        "Python for Data Science",
        "Deep Learning Fundamentals"
    ],
    "soft_skills": [
        "Quick Learner", "Problem Solver", "Team Collaboration",
        "Effective Communication", "Self-Motivated", "Adaptable",
        "Healthcare Domain Knowledge"
    ]
}

# =========================
# AGE & EXP CALCULATION
# =========================
def calculate_age(dob_str):
    dob = date.fromisoformat(dob_str)
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def calculate_experience():
    # Job ended Feb 2026, fixed duration
    return "1.5 years (Sep 2024 - Feb 2026)"

AGE = calculate_age(PROFILE["identity"]["date_of_birth"])
EXP_DURATION = calculate_experience()

# Update profile with dynamic value (just for reference if needed)
PROFILE['career']['experience_years'] = EXP_DURATION 

# =========================
# SYSTEM PROMPT FOR AI
# =========================
def create_system_prompt():
    return f"""You are Hindav Deshmukh's AI assistant. Answer questions about his professional profile accurately and concisely.

KEY INFORMATION:
- Name: {PROFILE['identity']['name']}
- Age: {AGE} years old
- Location: {PROFILE['identity']['location']}
- Languages: {', '.join(PROFILE['identity']['languages'])}

PROFESSIONAL EXPERIENCE:
- Role: {PROFILE['career']['current_role']} at {PROFILE['career']['current_company']}
- Career Path: Started as {PROFILE['career']['previous_title']} (Sep 2024), promoted to {PROFILE['career']['current_role']} on {PROFILE['career']['promotion_date']}.
- Duration: {EXP_DURATION}
- Domain: {PROFILE['career']['domain']}
- Key Project: Predictive Adverse Drug Reaction (ADR) Management System
  * Used ensemble algorithms: Random Forest and Gradient Boosting
  * Worked with patient and drug-related data
  * Responsibilities: data preprocessing, feature engineering, model training, evaluation, web integration
  * Impact: Improved patient safety through accurate ADR prediction

EDUCATION:
- {PROFILE['education']['degree']} (Graduated {PROFILE['education']['graduation_year']})
- {PROFILE['education']['college']}
- CGPA: {PROFILE['education']['cgpa']}

TECHNICAL SKILLS:
- Programming: {', '.join(PROFILE['skills']['languages'])}
- ML/AI: {', '.join(PROFILE['skills']['ml'])}
- Backend: {', '.join(PROFILE['skills']['backend'])}
- Databases: {', '.join(PROFILE['skills']['databases'])}
- Data Science: {', '.join(PROFILE['skills']['data_science'])}
- Tools: {', '.join(PROFILE['skills']['tools'])}

PROJECTS:
1. Predictive ADR Management System (Professional) - Healthcare ML using ensemble algorithms
2. Movie Character Recognition System (Final Year Project) - Computer Vision with Deep Learning
3. Mumbai Housing Prices 2025 - Geospatial Analysis & Price Prediction
4. StoxAi: AI Stock Prediction System - Hybrid LSTM/LLM Prediction (98% accuracy)

CERTIFICATIONS:
{', '.join(PROFILE['certifications'])}

CAREER GOALS:
- Seeking roles as: {', '.join(PROFILE['career']['target_roles'])}
- Goal: Become Senior ML Engineer/Lead in 2-3 years

SOFT SKILLS: {', '.join(PROFILE['soft_skills'])}

INSTRUCTIONS:
- Answer questions directly and concisely
- Use emojis sparingly (💼 🎓 🤖 📊 only when appropriate)
- If asked about something not in the profile, politely say you don't have that information
- Keep responses under 150 words unless detailed explanation is requested
- Be friendly and professional
- Don't make up information not provided above"""

# =========================
# GREETINGS
# =========================
GREETINGS = {
    r'\bhi\b': f"Hi 👋 I'm Hindav's AI assistant! I can tell you about his work as an ML Engineer (since Sep 2024), skills, projects, and career goals. What would you like to know?",
    r'\bhello\b': "Hello! 😊 I'm here to answer questions about Hindav Deshmukh - an ML Engineer. Ask me about his ADR prediction system, skills, or career journey!",
    r'\bhey\b': "Hey there! 👋 Ask me anything about Hindav - his ML work at Newgen Technomate, healthcare projects, or what he's looking for next!",
    r'\bwho are you\b': f"I'm an AI assistant trained on Hindav Deshmukh's professional profile. I can answer questions about his ML experience ({EXP_DURATION}), projects, and skills!"
}

def check_greeting(q):
    q_lower = q.lower().strip()
    
    if q_lower in ['hi', 'hello', 'hey', 'who are you', 'hi there', 'hello there']:
        for pattern, response in GREETINGS.items():
            if re.search(pattern, q_lower):
                return response
    
    for pattern, response in GREETINGS.items():
        if re.fullmatch(pattern, q_lower):
            return response
    
    return None

# =========================
# OPENROUTER API CALL
# =========================
# =========================
# OPENROUTER API CALL
# =========================
# =========================
# LOCAL FALLBACK AI
# =========================
def generate_local_answer(question: str):
    """
    Generates an answer locally based on keywords if the AI is offline.
    This ensures 'unlimited' availability.
    """
    q = question.lower()
    
    # extensive keyword matching
    if any(k in q for k in ['experience', 'work', 'job', 'company', 'role', 'career']):
        return (
            f"Hindav has {PROFILE['career']['experience_years']} years of experience. "
            f"He is currently a {PROFILE['career']['current_role']} at {PROFILE['career']['current_company']} ({PROFILE['career']['domain']} domain). "
            f"Key responsibilities include {', '.join(PROFILE['experience'][0]['key_project']['responsibilities'][:2])}."
        )
    
    if any(k in q for k in ['project', 'build', 'created', 'system']):
        return (
            f"Hindav's featured project is the '{PROFILE['projects'][0]['name']}', a {PROFILE['projects'][0]['description']}. "
            f"He also worked on: 2. {PROFILE['projects'][1]['name']} ({PROFILE['projects'][1]['technologies'][0]}), "
            f"3. {PROFILE['projects'][2]['name']}."
        )
        
    if any(k in q for k in ['skill', 'stack', 'tech', 'program', 'language']):
        return (
            f"Here is his tech stack:\n"
            f"* **Languages**: {', '.join(PROFILE['skills']['languages'])}\n"
            f"* **ML/AI**: {', '.join(PROFILE['skills']['ml'][:5])}...\n"
            f"* **Backend**: {', '.join(PROFILE['skills']['backend'])}\n"
            f"* **Tools**: {', '.join(PROFILE['skills']['tools'])}"
        )
        
    if any(k in q for k in ['education', 'college', 'degree', 'study']):
        return (
            f"He completed his {PROFILE['education']['degree']} at {PROFILE['education']['college']} "
            f"in {PROFILE['education']['graduation_year']} with a CGPA of {PROFILE['education']['cgpa']}."
        )
        
    if any(k in q for k in ['contact', 'email', 'reach', 'social']):
        return "You can reach Hindav via his LinkedIn profile or the contact form on this website."

    # Generic fallback that summarizes everything
    return (
        f"Hindav Deshmukh is an ML Engineer with {PROFILE['career']['experience_years']} years of experience in Healthcare Analytics. "
        f"He is skilled in Python, Random Forest, and Deep Learning. "
        f"Currently, he works at {PROFILE['career']['current_company']} building predictive systems."
    )

async def ask_openrouter(question: str, use_thinking_model: bool = False):
    """
    Call OpenRouter API with aggressive fallback to local generation
    """
    # Primary and fallback models
    models_to_try = [
        "google/gemini-2.0-flash-exp:free",
        "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
    ]
    
    import asyncio

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Hindav Profile Q&A"
    }

    system_prompt = create_system_prompt()
    
    # Try online models first
    async with httpx.AsyncClient(timeout=15.0) as client:
        for model in models_to_try:
            # Single attempt per model to fail fast and rotate
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            try:
                response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
                
                # If rate limited, skip immediately to next model
                if response.status_code == 429:
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    answer = data["choices"][0]["message"]["content"].strip()
                    return answer, 0.95
                        
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue
    
    # =====================================================
    # FAILOVER TO UNLIMITED LOCAL MODE
    # =====================================================
    # If all API calls fail (Rate limits, down, etc.), we generate
    # the answer locally. This guarantees 100% uptime / "unlimited" usage.
    print("All APIs failed or busy. Switching to Local Mode.")
    local_answer = generate_local_answer(question)
    return local_answer, 0.8  # slightly lower confidence to indicate local fallback

# =========================
# QUICK RESPONSE PATTERNS
# =========================
QUICK_RESPONSES = {
    r'\b(what|tell).*(name)\b': f"His name is {PROFILE['identity']['name']}.",
    r'\b(how old|age)\b': f"Hindav is {AGE} years old.",
    r'\b(where|location|based|live)\b': f"He's based in {PROFILE['identity']['location']}.",
}

def check_quick_response(q):
    q_lower = q.lower()
    for pattern, response in QUICK_RESPONSES.items():
        if re.search(pattern, q_lower):
            return response
    return None

# =========================
# REQUEST MODEL
# =========================
class QuestionRequest(BaseModel):
    question: str

# =========================
# MAIN HANDLER
# =========================
async def handle_question(question: str):
    ql = question.lower().strip()
    
    # Check for greetings first
    greeting = check_greeting(question)
    if greeting:
        return greeting, 1.0
    
    # Check for quick responses
    quick = check_quick_response(question)
    if quick:
        return quick, 1.0
    
    # Try AI, failover to Local Generator
    answer, confidence = await ask_openrouter(question)
    return answer, confidence

# =========================
# API ENDPOINTS
# =========================
@app.post("/ask")
async def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    answer, confidence = await handle_question(request.question)
    return {"answer": answer, "confidence": round(confidence, 2)}

@app.post("/api/ask")
async def ask_api(request: QuestionRequest):
    answer, confidence = await handle_question(request.question)
    return {
        "answer": answer,
        "confidence": round(confidence, 2),
        "success": True
    }

@app.post("/proxy-chat")
async def proxy_chat(request: QuestionRequest):
    answer, confidence = await handle_question(request.question)
    return {
        "answer": answer,
        "success": True,
        "confidence": round(confidence, 2)
    }

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model": SELECTED_MODEL,
        "provider": "OpenRouter"
    }

# =========================
# MODEL INFO ENDPOINT
# =========================
@app.get("/model-info")
async def model_info():
    return {
        "current_model": "Hybrid (Cloud + Local Fallback)",
        "status": "Unlimited",
        "cost": "Free"
    }

# =========================
# STATIC FILES (Root Mount)
# =========================
app.mount("/", StaticFiles(directory="static", html=True), name="static")