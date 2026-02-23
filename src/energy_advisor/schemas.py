"""Shared Pydantic schemas - Single Source of Truth.

All services import from this module to maintain DRY principle.
No schema redefinition allowed in service files.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BuildingFeatures(BaseModel):
    """Building physical features - Shared between ML Server and API Gateway."""
    relative_compactness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Surface area / volume^(2/3)",
        example=0.98
    )
    surface_area: float = Field(
        ..., 
        ge=300.0, 
        le=800.0, 
        description="Total surface area in m²",
        example=514.5
    )
    wall_area: float = Field(
        ..., 
        ge=200.0, 
        le=420.0, 
        description="Wall area in m²",
        example=294.0
    )
    roof_area: float = Field(
        ..., 
        ge=100.0, 
        le=250.0, 
        description="Roof area in m²",
        example=110.25
    )
    overall_height: float = Field(
        ..., 
        ge=3.0, 
        le=10.0, 
        description="Overall height in meters",
        example=7.0
    )
    orientation: int = Field(
        ..., 
        ge=2, 
        le=5, 
        description="Orientation (2=North, 3=East, 4=South, 5=West)",
        example=4
    )
    glazing_area: float = Field(
        ..., 
        ge=0.0, 
        le=0.5, 
        description="Glazing area as ratio of floor area",
        example=0.1
    )
    glazing_area_distribution: int = Field(
        ..., 
        ge=0, 
        le=5, 
        description="Glazing distribution (0=Unknown, 1=Uniform, 2=North, etc.)",
        example=1
    )


class Recommendation(BaseModel):
    """Energy efficiency recommendation - Used by API Gateway and SLM output."""
    category: str = Field(
        ...,
        pattern="^(glazing|insulation|orientation|geometry|general)$",
        description="Recommendation category"
    )
    priority: str = Field(
        ...,
        pattern="^(high|medium|low)$",
        description="Priority level"
    )
    action: str = Field(
        ...,
        min_length=10,
        max_length=200,
        description="Specific action to take"
    )
    expected_impact: str = Field(
        ...,
        min_length=10,
        max_length=100,
        description="Expected quantitative impact"
    )


class SLMOutputSchema(BaseModel):
    """Schema for guaranteed JSON output from SLM via Outlines.

    This schema is passed to Outlines to enforce valid JSON generation.
    """
    recommendations: List[Recommendation] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="List of energy efficiency recommendations"
    )
    explanation: str = Field(
        ...,
        min_length=20,
        max_length=500,
        description="Natural language analysis of building performance"
    )
    efficiency_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall energy efficiency score (0-100, higher is better)"
    )


class EnergyPrediction(BaseModel):
    """Numerical predictions from XGBoost ML Server."""
    heating_load: float = Field(..., description="Predicted heating load in kWh/m²")
    cooling_load: float = Field(..., description="Predicted cooling load in kWh/m²")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    model_version: str = Field(default="xgboost-v1.0", description="Model version")


class AnalysisResult(BaseModel):
    """Final combined result from API Gateway (Stage 1 + Stage 2)."""
    heating_load: float
    cooling_load: float
    efficiency_score: float
    recommendations: List[Recommendation]
    explanation: str
    confidence: float
    model_version: str = Field(default="v3.0-production", description="Pipeline version")


class HealthResponse(BaseModel):
    """Health check response - Used by all services."""
    status: str = Field(..., description="Service status")
    ml_server_connected: bool = Field(..., description="ML server connectivity")
    slm_loaded: bool = Field(..., description="SLM model loaded status")
    version: str = Field(..., description="Service version")
