# Base image
FROM python:3.11-slim

# Define working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Run the API
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
