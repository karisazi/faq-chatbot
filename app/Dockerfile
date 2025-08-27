# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (no cache for smaller image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code after dependencies
COPY . .

# Expose port used by Streamlit
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
