FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install ML Server dependencies
COPY services/ml_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared schemas and the ML server code
COPY src/energy_advisor/schemas.py ./src/energy_advisor/schemas.py
COPY services/ml_server/predictor.py ./services/ml_server/predictor.py

# Copy the pre-trained model artifact (Enforcing Immutability)
COPY models/xgboost_model.pkl ./models/xgboost_model.pkl

EXPOSE 8001

CMD ["uvicorn", "services.ml_server.predictor:app", "--host", "0.0.0.0", "--port", "8001"]