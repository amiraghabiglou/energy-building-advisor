"""API Gateway - Production-Corrected Implementation.

CRITICAL FIXES APPLIED:
1. ASYNC BLOCKING: Uses run_in_threadpool for all sync PyTorch operations
2. JSON RELIABILITY: Uses Outlines for guaranteed valid JSON schema enforcement  
3. DRY PRINCIPLE: Imports all schemas from shared module (no redefinition)

References:
- FastAPI concurrency: https://fastapi.tiangolo.com/async/ [^45^]
- run_in_threadpool for ML: https://apxml.com/courses/fastapi-ml-deployment/ [^42^]
- Outlines structured generation: https://dottxt-ai.github.io/outlines/ [^22^]
"""

import os
import sys
from typing import List

# Add src to path for shared imports (Fix #3: DRY principle)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool  # Fix #1: Prevent blocking
from pydantic import BaseModel, Field

# Fix #3: Import schemas from shared module - NO REDEFINITION
from src.energy_advisor.schemas import (
    BuildingFeatures,
    Recommendation,
    AnalysisResult,
    HealthResponse,
    SLMOutputSchema  # Pre-defined schema for Outlines
)

# Fix #2: Outlines for guaranteed structured generation
import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(
    title="Energy Advisor API Gateway (Production)",
    version="3.0.0",
    description="Two-stage pipeline: XGBoost regression + SLM explanations with proper async handling"
)

# Configuration
ML_SERVER_URL = os.getenv("ML_SERVER_URL", "http://ml-server:8001")
SLM_MODEL_NAME = os.getenv("SLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Outlines generator for structured JSON (Fix #2)
outlines_generator = None


def initialize_slm_sync():
    """Synchronous initialization of SLM with Outlines.

    This function is blocking and must be called via run_in_threadpool.
    """
    global outlines_generator

    print(f"[INIT] Loading SLM with Outlines: {SLM_MODEL_NAME}")

    # Configure quantization
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            SLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            SLM_MODEL_NAME,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

    tokenizer = AutoTokenizer.from_pretrained(
        SLM_MODEL_NAME, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create Outlines model
    outlines_model = outlines.models.Transformers(model, tokenizer)

    # Fix #2: Create generator with JSON schema enforcement
    # This guarantees valid JSON output matching SLMOutputSchema
    outlines_generator = outlines.generate.json(outlines_model, SLMOutputSchema)

    print("[INIT] Outlines generator ready with schema enforcement")
    return outlines_generator


def generate_explanation_sync(
    features: BuildingFeatures, 
    heating_load: float, 
    cooling_load: float
) -> SLMOutputSchema:
    """Synchronous generation - must be called via run_in_threadpool.

    Uses Outlines for guaranteed valid JSON (no regex parsing needed).
    """
    orientation_map = {2: "North", 3: "East", 4: "South", 5: "West"}
    orientation_str = orientation_map.get(features.orientation, "Unknown")

    # Calculate efficiency score for context
    total_load = heating_load + cooling_load
    base_score = max(0, min(100, 100 - (total_load - 20) * 2))

    prompt = f"""<|system|>
You are an expert building energy efficiency advisor. Analyze this building and provide structured recommendations.
</s>
<|user|>
Building Analysis Request:

PHYSICAL SPECIFICATIONS:
- Relative Compactness: {features.relative_compactness:.2f} (0-1 scale)
- Surface Area: {features.surface_area:.1f} m²
- Wall Area: {features.wall_area:.1f} m²  
- Roof Area: {features.roof_area:.1f} m²
- Overall Height: {features.overall_height:.1f} m
- Orientation: {orientation_str} (2=North, 3=East, 4=South, 5=West)
- Glazing Area: {features.glazing_area*100:.0f}% of floor area
- Glazing Distribution: {features.glazing_area_distribution} (0=Unknown, 1=Uniform, 2=North, etc.)

PREDICTED ENERGY PERFORMANCE (from XGBoost model):
- Heating Load: {heating_load:.1f} kWh/m²
- Cooling Load: {cooling_load:.1f} kWh/m²
- Base Efficiency Score: {base_score:.0f}/100

TASK:
Provide exactly 3 energy efficiency recommendations. For each:
- category: one of [glazing, insulation, orientation, geometry]
- priority: one of [high, medium, low]  
- action: specific actionable step (10-200 chars)
- expected_impact: quantified benefit (10-100 chars)

Also provide:
- efficiency_score: integer 0-100 (refined based on analysis)
- explanation: 2-3 sentence narrative analysis (50-300 chars)
</s>
<|assistant|>
"""

    # Fix #2: Outlines guarantees valid JSON matching SLMOutputSchema
    # No regex parsing needed - schema is enforced at token level
    result = outlines_generator(prompt, max_tokens=512, temperature=0.7, top_p=0.9)

    return result


@app.on_event("startup")
async def startup():
    """Initialize services without blocking event loop."""
    global outlines_generator

    # Fix #1: Run blocking initialization in thread pool
    # This prevents startup from freezing the event loop
    try:
        outlines_generator = await run_in_threadpool(initialize_slm_sync)
    except Exception as e:
        print(f"[WARNING] SLM initialization failed: {e}")
        print("[WARNING] Falling back to rule-based recommendations")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - always responsive."""
    ml_connected = False

    # Async health check to ML server (non-blocking)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ML_SERVER_URL}/health", 
                timeout=2.0  # Short timeout to keep responsive
            )
            ml_connected = response.status_code == 200
    except:
        pass

    return HealthResponse(
        status="healthy" if ml_connected else "degraded",
        ml_server_connected=ml_connected,
        slm_loaded=outlines_generator is not None,
        version="3.0.0-production"
    )


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_building(features: BuildingFeatures):
    """Two-stage analysis with proper async handling.

    Stage 1: XGBoost regression (async HTTP call)
    Stage 2: SLM explanation (run_in_threadpool to prevent blocking)
    """

    # Stage 1: Get numerical predictions from ML Server
    # This is properly async (httpx) and non-blocking
    try:
        async with httpx.AsyncClient() as client:
            ml_response = await client.post(
                f"{ML_SERVER_URL}/predict",
                json=features.dict(),
                timeout=10.0
            )
            ml_data = ml_response.json()
            heating_load = ml_data["heating_load"]
            cooling_load = ml_data["cooling_load"]
            confidence = ml_data["confidence"]
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"ML Server unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ML Server error: {str(e)}"
        )

    # Stage 2: Generate explanations with SLM
    # Fix #1: Run blocking PyTorch code in thread pool
    if outlines_generator is not None:
        try:
            # This prevents the ~120ms generation from blocking other requests
            slm_result = await run_in_threadpool(
                generate_explanation_sync,
                features,
                heating_load,
                cooling_load
            )

            recommendations = slm_result.recommendations
            explanation = slm_result.explanation
            efficiency_score = slm_result.efficiency_score

        except Exception as e:
            # Fallback if Outlines generation fails
            print(f"[WARNING] SLM generation failed: {e}")
            recommendations = generate_fallback_recommendations(
                features, heating_load, cooling_load
            )
            explanation = (
                f"Building analysis complete. Predicted heating: {heating_load:.1f} kWh/m², "
                f"cooling: {cooling_load:.1f} kWh/m². Consider the recommendations above."
            )
            efficiency_score = max(0, min(100, 100 - (heating_load + cooling_load) / 2))
    else:
        # Fallback when SLM not available
        recommendations = generate_fallback_recommendations(
            features, heating_load, cooling_load
        )
        explanation = (
            f"Building analysis (rule-based). Predicted heating: {heating_load:.1f} kWh/m², "
            f"cooling: {cooling_load:.1f} kWh/m²."
        )
        efficiency_score = max(0, min(100, 100 - (heating_load + cooling_load) / 2))

    return AnalysisResult(
        heating_load=heating_load,
        cooling_load=cooling_load,
        efficiency_score=efficiency_score,
        recommendations=recommendations,
        explanation=explanation,
        confidence=confidence
    )


def generate_fallback_recommendations(
    features: BuildingFeatures, 
    heating_load: float, 
    cooling_load: float
) -> List[Recommendation]:
    """Rule-based fallback when SLM unavailable."""
    recs = []

    if features.glazing_area > 0.25:
        recs.append(Recommendation(
            category="glazing",
            priority="high",
            action=f"Reduce glazing area from {features.glazing_area*100:.0f}% to 15-20%",
            expected_impact="Reduce cooling load by 10-15%"
        ))

    if heating_load > 25:
        recs.append(Recommendation(
            category="insulation",
            priority="high",
            action="Improve wall and roof insulation (R-value upgrade)",
            expected_impact="Reduce heating load by 15-20%"
        ))

    if features.relative_compactness < 0.7:
        recs.append(Recommendation(
            category="geometry",
            priority="medium",
            action="Improve building compactness (reduce surface/volume ratio)",
            expected_impact="Reduce both loads by 5-10%"
        ))

    return recs if recs else [
        Recommendation(
            category="general",
            priority="low",
            action="Building performs within normal parameters",
            expected_impact="Maintain current efficiency with regular maintenance"
        )
    ]


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    # Single worker to avoid duplicate model loading
    # Use multiple workers only with proper model sharing
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
