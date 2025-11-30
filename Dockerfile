FROM python:3.13.1-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy  necessary files
COPY Homework/Modules/predict.py .
COPY model_C=1.0.bin .
COPY serve_prod.py .

EXPOSE 9696

# Start your production server
CMD ["python", "serve_prod.py"]
