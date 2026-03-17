# 🔍 PROOF AI — Multimodal Real-Time Trust Layer
### Gemini Live Agent Challenge 2026

![PROOF AI](https://img.shields.io/badge/PROOF_AI-Multimodal_Trust_Layer-c0392b?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Powered_by-Gemini_Live-4285F4?style=for-the-badge&logo=google)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Every AI lies sometimes. PROOF AI catches it — in real-time, across every modality.**

---

## 🌐 Live Site
**[proof-ai.github.io](https://ilikekulfi3000-crypto.github.io/proof-ai.com)**  
*(Replace with your actual GitHub Pages URL after deployment)*

---

## 🎯 What is PROOF AI?

PROOF AI is a **real-time verification layer** that rides alongside any AI system as a trust co-pilot. As AI generates responses — text, images, documents, audio, or video — PROOF AI simultaneously runs:

1. **🔍 Hallucination Detector** — Claims cross-referenced against Google Search grounding
2. **🧠 Contextual Drift Monitor** — Catches AI self-contradictions across sessions  
3. **📊 Neural Confidence Scorer** — Assigns a live PROOF Score (0–100) with grade A–F

---

## 📦 Repository Structure

```
proof-ai/
├── index.html              # Production website (GitHub Pages)
├── backend.py              # FastAPI + Gemini verification engine
├── requirements.txt        # Python dependencies
├── demo.py                 # CLI demo script
├── README.md
└── .github/
    └── workflows/
        └── deploy.yml      # Auto-deploy to GitHub Pages
```

---

## ⚡ Quick Start

### Website (no backend needed)
The `index.html` is a full static site — just open it in any browser!

### Full Stack (with AI verification)
```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/proof-ai.git
cd proof-ai

# 2. Install
pip install -r requirements.txt

# 3. Set API Key
export GEMINI_API_KEY=your_gemini_api_key_here

# 4. Run
python backend.py

# 5. Open
open http://localhost:8000/app
```

---

## 🏗️ Architecture

```
AI Output (any modality)
         │
         ▼
   [PROOF AI Engine]
    ├── Claim Extractor   ← Gemini Live (function calling)
    ├── Fact Verifier     ← Gemini + Google Search Grounding
    ├── Drift Monitor     ← Semantic coherence analysis
    └── Score Calculator  ← Accuracy(60%) + Coherence(25%) + Confidence(15%)
         │
         ▼
   PROOF Score (0–100) + Trust Certificate
```

---

## 📊 PROOF Score Formula

```
PROOF = (Accuracy × 0.60) + (Coherence × 0.25) + (Confidence × 0.15)
```

| Grade | Score | Meaning |
|-------|-------|---------|
| A | 90–100 | Highly Trustworthy |
| B | 75–89 | Generally Reliable |
| C | 60–74 | Requires Caution |
| D | 40–59 | Significant Issues |
| F | 0–39 | Unreliable — Do Not Trust |

---

## 🎯 Built For

**[Gemini Live Agent Challenge](https://devpost.com/submit-to/28633-gemini-live-agent-challenge)**  
Submission Date: March 16, 2026

### How Gemini Live is Used:
- **Streaming claim extraction** via Gemini Live's real-time API
- **Google Search grounding** for fact verification (not another LLM!)
- **Multimodal input** — text, image, audio, document, video
- **Function calling** to structure the verification pipeline

---

## 📜 License
MIT — Build on it, deploy it, make AI trustworthy.
