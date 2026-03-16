"""
PROOF AI — Quick CLI Demo
Run this to test without the frontend UI.
"""

import os
import json
import asyncio
import sys

# Add the proof_ai directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set your API key here or via environment variable
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_KEY_HERE")

import google.generativeai as genai
import re

genai.configure(api_key=API_KEY)

CLAIM_PROMPT = """Extract verifiable factual claims from this text. Return JSON only:
{{"claims": [{{"text": "claim here", "category": "factual"}}]}}

TEXT: {text}"""

VERIFY_PROMPT = """Verify this claim. Return JSON only:
{{"verdict": "VERIFIED|UNCERTAIN|FLAGGED", "confidence_score": 85, "reasoning": "brief reason"}}

CLAIM: {claim}"""


async def demo_verify(text: str):
    print(f"\n{'='*60}")
    print(f"📝 TEXT: {text[:100]}...")
    print(f"{'='*60}")
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Extract claims
    print("\n🔍 Extracting claims...")
    resp = model.generate_content(CLAIM_PROMPT.format(text=text))
    raw = re.sub(r'```json|```', '', resp.text).strip()
    
    try:
        data = json.loads(raw)
        claims = data.get("claims", [])
        print(f"   Found {len(claims)} verifiable claims")
    except:
        claims = []
        print("   No structured claims found")
    
    # Verify each claim
    verified = []
    for i, claim in enumerate(claims[:5]):
        print(f"\n⚡ Verifying claim {i+1}: '{claim['text'][:60]}...'")
        try:
            vresp = model.generate_content(VERIFY_PROMPT.format(claim=claim['text']))
            vraw = re.sub(r'```json|```', '', vresp.text).strip()
            result = json.loads(vraw)
            verdict = result.get('verdict', 'UNCERTAIN')
            score = result.get('confidence_score', 50)
            
            icon = {'VERIFIED': '✅', 'UNCERTAIN': '⚠️', 'FLAGGED': '❌'}.get(verdict, '⚠️')
            print(f"   {icon} {verdict} ({score}%) — {result.get('reasoning', '')[:80]}")
            verified.append(result)
        except Exception as e:
            print(f"   ⚠️  Verification error: {e}")
    
    # Calculate PROOF Score
    verdicts = [v.get('verdict') for v in verified]
    v_count = verdicts.count('VERIFIED')
    u_count = verdicts.count('UNCERTAIN')
    f_count = verdicts.count('FLAGGED')
    total = len(verdicts)
    
    if total > 0:
        accuracy = (v_count * 100 + u_count * 50) / total
    else:
        accuracy = 75
    
    proof_score = int(accuracy * 0.85)
    
    grade = 'A' if proof_score >= 90 else 'B' if proof_score >= 75 else 'C' if proof_score >= 60 else 'D' if proof_score >= 40 else 'F'
    
    print(f"\n{'='*60}")
    print(f"🎯 PROOF SCORE: {proof_score}/100  (Grade: {grade})")
    print(f"   ✅ Verified: {v_count}  ⚠️  Uncertain: {u_count}  ❌ Flagged: {f_count}")
    print(f"{'='*60}\n")
    
    return proof_score


if __name__ == "__main__":
    # Test cases — these demo hallucination detection
    test_texts = [
        "The speed of light is exactly 299,792,458 meters per second. Albert Einstein published his theory of special relativity in 1905.",
        "The Great Wall of China is visible from space with the naked eye. It was built in 221 BC by Emperor Qin Shi Huang.",
        "Python was created by Guido van Rossum in 1991. The latest version is Python 4.0, released in 2024.",
    ]
    
    if API_KEY == "YOUR_KEY_HERE":
        print("❌ Please set your GEMINI_API_KEY environment variable!")
        print("   export GEMINI_API_KEY=your_key_here")
        sys.exit(1)
    
    print("🚀 PROOF AI — CLI Demo")
    print("Testing hallucination detection on 3 sample texts...\n")
    
    for text in test_texts:
        asyncio.run(demo_verify(text))
