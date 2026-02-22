# Hindav Deshmukh | Machine Learning Engineer Portfolio

A premium, modern, and high-performance portfolio website showcasing Hindav Deshmukh's expertise in **Machine Learning Engineering**, **Data Science**, and **Full-Stack Development**.

� **Live Demo:** [hindav.vercel.app](https://hindav.vercel.app)

---

## 🌟 Highlights

- **AI-Powered Assistant**: An integrated chatbot powered by a hybrid architecture (OpenRouter + Local Fallback) to answer professional queries in real-time.
- **Project Showcase**: A curated gallery of projects ranging from Healthcare Analytics and Computer Vision to GenAI & RAG systems.
- **Modern Tech Stack**: Built with FastAPI for a robust backend and optimized Vanilla CSS/JS for a sleek, responsive frontend experience.
- **Dynamic Content**: Auto-calculating experience and real-time model status via API endpoints.

---

## 🛠️ Technology Stack

### **Backend & AI**
- **Framework**: FastAPI (Python 3.10+)
- **AI Models**: Google Gemini 2.0 Flash, Llama 3.2 (via OpenRouter)
- **Logic**: Hybrid logic with local NLP fallback for 100% uptime.
- **Deployment**: Vercel (Serverless Functions)

### **Frontend**
- **Core**: HTML5, Vanilla JavaScript (ES6+)
- **Styling**: Component-based CSS3 (Custom Variables, Flexbox, Grid)
- **Features**: Light/Dark Mode, smooth animations, responsive grids.
- **Icons & Fonts**: Font Awesome, Google Fonts (Outfit/Inter)

---

## 📂 Key Projects

1.  **Predictive ADR Management System (Professional)**
    *   *System to predict adverse drug reactions using ensemble algorithms (92% accuracy).*
    *   **Tech**: Python, Random Forest, Gradient Boosting, Scikit-learn, Flask.
2.  **StoxAi: AI Stock Prediction System**
    *   *Hybrid stock forecasting using LSTM and GPT-4 based sentiment analysis (98% accuracy).*
    *   **Tech**: TensorFlow, FastAPI, GPT-4, Streamlit, Plotly.
3.  **AskMyDocs**
    *   *AI-powered document chatbot (RAG) for conversational interaction with uploaded files.*
    *   **Tech**: LangChain, LLM, Streamlit, RAG.
4.  **Movie Character Recognition**
    *   *Real-time character identification using Computer Vision and Deep Learning.*
    *   **Tech**: OpenCV, CNN, Flask, Deep Learning.
5.  **Mumbai Housing Prices 2025**
    *   *Geospatial analysis on 20K+ listings for price prediction and urban study.*
    *   **Tech**: Pandas, Scikit-learn, Geospatial Analysis.
6.  **Digital Toolbox**
    *   *Web suite for image processing, high-accuracy OCR (EasyOCR), and PDF management.*
    *   **Tech**: EasyOCR, PyMuPDF, OpenCV, Streamlit.

---

## 🤖 Intelligent Profile Assistant

The portfolio features a custom-built API that powers the on-site AI assistant. 
- **Endpoint**: `/ask` (POST)
- **Hybrid Logic**: If cloud APIs (OpenRouter) are hit by rate limits or latency, a **local NLP fallback** takes over to ensure 100% uptime.
- **Context-Aware**: Trained on Hindav's professional experience, skills, and background.

```bash
# Example Q&A call
curl -X POST "https://hindav.vercel.app/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Hindav experience in healthcare?"}'
```

---

## 🚀 Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hindav/Portfolio.git
   cd Portfolio
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file:
   ```env
   OPENROUTER_API_KEY=your_key_here
   ```

4. **Run the application:**
   ```bash
   python app.py
   # or
   uvicorn app:app --reload
   ```

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for more details.

---

## 🤝 Connect
- **LinkedIn**: [linkedin.com/in/hindav](https://www.linkedin.com/in/hindav)
- **GitHub**: [github.com/hindav](https://github.com/hindav)
- **Email**: [Direct Contact via Portfolio](https://hindav.vercel.app#contact)

