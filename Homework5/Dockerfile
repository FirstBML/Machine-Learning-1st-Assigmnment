FROM python:3.12-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies listed in pyproject.toml manually
RUN pip install --no-cache-dir fastapi uvicorn pydantic scikit-learn requests ipykernel

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
