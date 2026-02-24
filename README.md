# Energy-Efficient Building Design Advisor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/amiraghabiglou/energy-building-advisor/actions/workflows/ci.yml/badge.svg)](https://github.com/amiraghabiglou/energy-building-advisor/actions)

A production-ready microservice architecture that provides energy efficiency analysis and natural language recommendations for building designs. 

This project demonstrates a novel **two-stage hybrid ML pipeline**, combining the raw predictive accuracy of Gradient Boosted Trees with the reasoning and formatting capabilities of a Small Language Model (SLM).

## Concept & Architecture

Applying SLMs directly to tabular regression tasks often leads to catastrophic overfitting or severe hallucination. To solve this, the pipeline is strictly decoupled:

1. **Stage 1 (Numerical Regression):** An isolated XGBoost microservice trained on the [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/energy+efficiency). It interprets 8 architectural features (compactness, surface area, glazing, etc.) to precisely predict heating and cooling loads.
2. **Stage 2 (SLM Reasoning):** An API Gateway orchestrates the results from Stage 1 into a dynamic prompt for **TinyLlama (1.1B)**. Using the **Outlines** library, the SLM's output is strictly constrained to a predefined JSON schema at the token level, guaranteeing 100% valid, structured recommendations without parsing errors.

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client / UI   │────▶│   API Gateway    │────▶│   ML Server     │
│   Request       │     │   (FastAPI)      │     │   (XGBoost)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               │ Orchestrates predictions & builds prompt
                               ▼
                        ┌──────────────────┐
                        │   SLM Service    │
                        │   (TinyLlama)    │
                        │   via Outlines   │
                        └──────────────────┘
                               │ Guaranteed JSON Schema
                               ▼
                        ┌──────────────────┐
                        │ Structured Output│
                        │ & Explanations   │
                        └──────────────────┘
```

📁 Repository Structure
```text
.
├── src/
│   └── energy_advisor/
│       └── schemas.py              # Single source of truth for all Pydantic schemas (DRY principle)
│
├── services/
│   ├── api/                        # FastAPI gateway orchestrating the SLM and thread-pool execution
│   └── ml_server/                  # Isolated, lightweight XGBoost inference server
│
├── scripts/
│   └── train_model.py              # Offline training pipeline (ensures container immutability)
│
├── docker/                         # Context-aware Dockerfiles for each microservice
│
├── models/                         # Serialized model artifacts (.pkl generated offline)
│
└── docker-compose.yml
```
# Deployment & Execution

This repository enforces container immutability. Models are not trained at runtime; you must generate the serialized model artifacts before building the Docker images.

## 1. Generate Model Artifacts (Required)

Install the data science dependencies locally and execute the offline training script to generate the `.pkl` artifact.

```bash
python scripts/train_model.py
```
Verify that models/xgboost_model.pkl has been created successfully.

## 2.Build and Launch Containers
Once the model artifact is staged, launch the isolated microservices:

```bash
docker-compose up --build
```
Services Available
API Gateway: http://localhost:8000
API Documentation (Swagger): http://localhost:8000/docs
ML Server (Internal): http://localhost:8001
API Usage Example
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{
           "relative_compactness": 0.98,
           "surface_area": 514.5,
           "wall_area": 294.0,
           "roof_area": 110.25,
           "overall_height": 7.0,
           "orientation": 4,
           "glazing_area": 0.1,
           "glazing_area_distribution": 1
         }'
```
