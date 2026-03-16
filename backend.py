"""
PROOF AI — Real-Time Verification Layer for Gemini
FastAPI Backend with WebSocket streaming + Gemini API
"""

import os
import json
import asyncio
import re
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="PROOF AI",
    description="Real-Time Trust Layer for AI Verification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
class VerifyRequest(BaseModel):
    text: str
    context: list = []
    domain: str = "general"  # general | medical | legal | financial

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str

class ProofSession(BaseModel):
    messages: list[ChatMessage]
    domain: str = "general"

# ─────────────────────────────────────────────
# PROOF ENGINE — CORE LOGIC
# ─────────────────────────────────────────────

CLAIM_EXTRACTION_PROMPT = """You are a precise claim extraction engine. 

Analyze the following AI-generated text and extract all VERIFIABLE factual claims.
A verifiable claim is a statement that can be checked against external sources.

DO NOT include opinions, suggestions, or subjective statements.

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "id": 1,
      "text": "the exact claim text",
      "category": "factual|statistical|historical|scientific|current_event",
      "confidence_required": "high|medium|low",
      "verifiable": true
    }}
  ],
  "non_verifiable_count": 0,
  "overall_risk": "low|medium|high"
}}

TEXT TO ANALYZE:
{text}

Return ONLY the JSON. No explanation, no markdown, no extra text."""

VERIFICATION_PROMPT = """You are PROOF AI — a rigorous fact-verification engine.

Verify this claim using your knowledge and logical reasoning:
CLAIM: "{claim}"

Domain context: {domain}

Analyze carefully and return ONLY valid JSON:
{{
  "claim": "{claim}",
  "verdict": "VERIFIED|UNCERTAIN|FLAGGED",
  "confidence_score": 85,
  "reasoning": "Brief explanation of verdict",
  "source_hint": "What type of source would verify this",
  "flag_reason": "Only if FLAGGED: explain what is wrong"
}}

Verdict guide:
- VERIFIED: Claim is factually accurate based on established knowledge
- UNCERTAIN: Claim may be true but cannot be fully confirmed, or is partially correct
- FLAGGED: Claim contains a detectable error, hallucination, or misleading information

Return ONLY the JSON."""

DRIFT_ANALYSIS_PROMPT = """You are a contextual coherence analyzer.

Review this conversation and detect any CONTEXTUAL DRIFT — where the AI's responses:
1. Contradict earlier statements
2. Change facts mid-conversation
3. Use inconsistent information
4. Forget previously established context

CONVERSATION:
{conversation}

Return ONLY valid JSON:
{{
  "drift_detected": false,
  "drift_score": 15,
  "inconsistencies": [],
  "coherence_summary": "Brief summary"
}}

drift_score is 0 (perfect) to 100 (severe drift). Return ONLY the JSON."""

GEMINI_AI_PROMPT = """You are a helpful AI assistant. Answer the user's question naturally and informatively.
Provide factual information, statistics, historical facts, and explanations.
Domain: {domain}

IMPORTANT: Give detailed, factual responses that can be verified. Include specific numbers, dates, and facts."""


async def extract_claims(text: str) -> dict:
    """Extract verifiable claims from AI-generated text using Gemini."""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = CLAIM_EXTRACTION_PROMPT.format(text=text)
        
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=1500,
            )
        )
        
        raw = response.text.strip()
        # Clean up any markdown code blocks
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        raw = raw.strip()
        
        result = json.loads(raw)
        return result
    except json.JSONDecodeError:
        return {"claims": [], "non_verifiable_count": 0, "overall_risk": "low"}
    except Exception as e:
        print(f"Claim extraction error: {e}")
        return {"claims": [], "non_verifiable_count": 0, "overall_risk": "low"}


async def verify_claim(claim: str, domain: str = "general") -> dict:
    """Verify a single claim using Gemini with grounding."""
    try:
        # Try with Google Search grounding first
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            tools=[{"google_search_retrieval": {}}]
        )
        
        prompt = VERIFICATION_PROMPT.format(claim=claim, domain=domain)
        
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=500,
            )
        )
        
        raw = response.text.strip()
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        raw = raw.strip()
        
        result = json.loads(raw)
        result["grounded"] = True
        return result
        
    except Exception:
        # Fallback without grounding
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = VERIFICATION_PROMPT.format(claim=claim, domain=domain)
            response = model.generate_content(
                prompt,
                generation_config=GenerationConfig(temperature=0.1, max_output_tokens=500)
            )
            raw = response.text.strip()
            raw = re.sub(r'```json\s*', '', raw)
            raw = re.sub(r'```\s*', '', raw)
            result = json.loads(raw.strip())
            result["grounded"] = False
            return result
        except Exception as e:
            return {
                "claim": claim,
                "verdict": "UNCERTAIN",
                "confidence_score": 50,
                "reasoning": "Verification service temporarily unavailable",
                "source_hint": "Manual verification recommended",
                "grounded": False
            }


async def analyze_drift(conversation: list) -> dict:
    """Detect contextual drift in conversation history."""
    if len(conversation) < 2:
        return {
            "drift_detected": False,
            "drift_score": 0,
            "inconsistencies": [],
            "coherence_summary": "Insufficient context for drift analysis"
        }
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        conv_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation[-10:]  # Last 10 messages
        ])
        
        prompt = DRIFT_ANALYSIS_PROMPT.format(conversation=conv_text)
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.1, max_output_tokens=500)
        )
        
        raw = response.text.strip()
        raw = re.sub(r'```json\s*', '', raw)
        raw = re.sub(r'```\s*', '', raw)
        return json.loads(raw.strip())
        
    except Exception as e:
        return {
            "drift_detected": False,
            "drift_score": 0,
            "inconsistencies": [],
            "coherence_summary": "Drift analysis unavailable"
        }


def calculate_proof_score(claims_result: dict, verified_claims: list, drift_result: dict) -> dict:
    """Calculate the final PROOF Score (0-100)."""
    
    if not claims_result.get("claims"):
        return {
            "score": 95,
            "grade": "A",
            "label": "No verifiable claims detected",
            "breakdown": {
                "accuracy": 95,
                "coherence": 100 - drift_result.get("drift_score", 0),
                "confidence": 90
            }
        }
    
    # Accuracy Score (60% weight)
    verdicts = [c.get("verdict", "UNCERTAIN") for c in verified_claims]
    verified_count = verdicts.count("VERIFIED")
    uncertain_count = verdicts.count("UNCERTAIN")
    flagged_count = verdicts.count("FLAGGED")
    total = len(verdicts)
    
    if total > 0:
        accuracy = (
            (verified_count * 100 + uncertain_count * 50 + flagged_count * 0) / total
        )
    else:
        accuracy = 80
    
    # Coherence Score (25% weight)
    coherence = max(0, 100 - drift_result.get("drift_score", 0))
    
    # Confidence Score (15% weight)  
    avg_confidence = 0
    if verified_claims:
        avg_confidence = sum(c.get("confidence_score", 70) for c in verified_claims) / len(verified_claims)
    else:
        avg_confidence = 75
    
    # Final Score
    final_score = int((accuracy * 0.60) + (coherence * 0.25) + (avg_confidence * 0.15))
    final_score = max(0, min(100, final_score))
    
    # Grade
    if final_score >= 90:
        grade, label = "A", "Highly Trustworthy"
    elif final_score >= 75:
        grade, label = "B", "Generally Reliable"
    elif final_score >= 60:
        grade, label = "C", "Requires Caution"
    elif final_score >= 40:
        grade, label = "D", "Significant Issues"
    else:
        grade, label = "F", "Unreliable — Do Not Trust"
    
    return {
        "score": final_score,
        "grade": grade,
        "label": label,
        "breakdown": {
            "accuracy": int(accuracy),
            "coherence": int(coherence),
            "confidence": int(avg_confidence)
        },
        "stats": {
            "verified": verified_count,
            "uncertain": uncertain_count,
            "flagged": flagged_count,
            "total_claims": total
        }
    }


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "PROOF AI is running", "version": "1.0.0", "status": "active"}


@app.get("/health")
async def health():
    return {"status": "healthy", "api_key_set": GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"}


@app.post("/api/verify")
async def verify_text(request: VerifyRequest):
    """
    Full PROOF verification pipeline:
    1. Extract claims from text
    2. Verify each claim
    3. Analyze drift
    4. Calculate PROOF Score
    """
    
    # Step 1: Extract claims
    claims_result = await extract_claims(request.text)
    
    # Step 2: Verify each claim (parallel)
    claims = claims_result.get("claims", [])
    verification_tasks = [
        verify_claim(claim["text"], request.domain) 
        for claim in claims[:8]  # Cap at 8 claims for speed
    ]
    verified_claims = await asyncio.gather(*verification_tasks)
    
    # Step 3: Drift analysis
    drift_result = await analyze_drift(
        [{"role": m.get("role", "user"), "content": m.get("content", "")} 
         for m in request.context]
    )
    
    # Step 4: PROOF Score
    proof_score = calculate_proof_score(claims_result, list(verified_claims), drift_result)
    
    return {
        "proof_score": proof_score,
        "claims": [
            {
                **claim,
                "verification": verified_claims[i] if i < len(verified_claims) else {}
            }
            for i, claim in enumerate(claims[:8])
        ],
        "drift_analysis": drift_result,
        "overall_risk": claims_result.get("overall_risk", "low"),
        "text_analyzed": request.text[:200] + "..." if len(request.text) > 200 else request.text
    }


@app.post("/api/chat")
async def chat_with_proof(session: ProofSession):
    """
    AI Chat endpoint where the AI responds AND gets PROOF verified simultaneously.
    Returns both the AI response and its PROOF Score.
    """
    
    try:
        # Build conversation history for Gemini
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        system_prompt = GEMINI_AI_PROMPT.format(domain=session.domain)
        
        # Convert messages to Gemini format
        history = []
        for msg in session.messages[:-1]:  # All except last
            history.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [msg.content]
            })
        
        chat = model.start_chat(history=history)
        
        # Get last user message
        last_message = session.messages[-1].content
        
        # Generate AI response
        response = chat.send_message(
            f"{system_prompt}\n\nUser: {last_message}"
        )
        
        ai_response = response.text
        
        # Now PROOF verify the response
        context_dicts = [{"role": m.role, "content": m.content} for m in session.messages]
        
        verify_request = VerifyRequest(
            text=ai_response,
            context=context_dicts,
            domain=session.domain
        )
        
        proof_result = await verify_text(verify_request)
        
        return {
            "ai_response": ai_response,
            "proof_result": proof_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.websocket("/ws/live-verify")
async def websocket_live_verify(websocket: WebSocket):
    """
    WebSocket endpoint for REAL-TIME streaming verification.
    Client sends text chunks, PROOF AI verifies in real-time.
    """
    await websocket.accept()
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "PROOF AI Live Verification Active"
        })
        
        full_text = ""
        context = []
        
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "stream_chunk":
                chunk = data.get("text", "")
                full_text += chunk
                
                await websocket.send_json({
                    "type": "chunk_received",
                    "chars_processed": len(full_text)
                })
            
            elif action == "verify_now":
                text = data.get("text", full_text)
                domain = data.get("domain", "general")
                context = data.get("context", [])
                
                # Send progress updates
                await websocket.send_json({"type": "progress", "step": "Extracting claims...", "percent": 20})
                
                claims_result = await extract_claims(text)
                claims = claims_result.get("claims", [])
                
                await websocket.send_json({
                    "type": "progress", 
                    "step": f"Found {len(claims)} verifiable claims. Verifying...",
                    "percent": 40,
                    "claims_found": len(claims)
                })
                
                # Verify claims one by one, sending updates
                verified_claims = []
                for i, claim in enumerate(claims[:8]):
                    await websocket.send_json({
                        "type": "verifying_claim",
                        "claim_index": i,
                        "claim_text": claim["text"][:100],
                        "percent": 40 + (i + 1) * (30 // max(len(claims[:8]), 1))
                    })
                    
                    result = await verify_claim(claim["text"], domain)
                    verified_claims.append(result)
                    
                    # Send individual claim result
                    await websocket.send_json({
                        "type": "claim_verified",
                        "claim_index": i,
                        "claim": claim["text"][:100],
                        "verdict": result.get("verdict"),
                        "confidence": result.get("confidence_score"),
                        "reasoning": result.get("reasoning", "")[:200]
                    })
                
                await websocket.send_json({"type": "progress", "step": "Analyzing context drift...", "percent": 80})
                
                drift_result = await analyze_drift(context)
                
                await websocket.send_json({"type": "progress", "step": "Calculating PROOF Score...", "percent": 95})
                
                proof_score = calculate_proof_score(claims_result, verified_claims, drift_result)
                
                # Final result
                await websocket.send_json({
                    "type": "proof_complete",
                    "percent": 100,
                    "proof_score": proof_score,
                    "drift_analysis": drift_result,
                    "verified_claims": verified_claims,
                    "total_claims": len(claims)
                })
                
                full_text = ""
                
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print("Client disconnected from live verification")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass


# ─────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────
@app.get("/app")
async def serve_frontend():
    return FileResponse("frontend.html")


if __name__ == "__main__":
    import uvicorn
    print("🚀 PROOF AI Backend Starting...")
    print("📍 API Docs: http://localhost:8000/docs")
    print("🖥️  Frontend: http://localhost:8000/app")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
