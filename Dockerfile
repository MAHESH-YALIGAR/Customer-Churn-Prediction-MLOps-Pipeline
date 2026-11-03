
# 1. Use Python base image
FROM python:3.13-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements file
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application source code
COPY prediction/ ./prediction/

# 6. Copy fitted preprocessor
# Copy trained model
COPY artifacts/model_pusher/model.pkl ./artifacts/model_pusher/model.pkl

# Copy fitted preprocessor
COPY artifacts/model_pusher/preprocessor.pkl ./artifacts/model_pusher/preprocessor.pkl
# 7. Expose FastAPI port
EXPOSE 8000

# 8. Start FastAPI application
CMD ["uvicorn", "prediction.app:app", "--host", "0.0.0.0", "--port", "8000"]
