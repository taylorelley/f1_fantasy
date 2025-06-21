# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    sassc \
    && rm -rf /var/lib/apt/lists/*

# Provide a `sass` command using sassc
RUN ln -s /usr/bin/sassc /usr/local/bin/sass

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY f1_optimizer.py .
COPY templates/ templates/
COPY static/ static/
COPY default_data/ default_data/

# Compile SCSS to CSS during build
RUN sass static/scss/style.scss static/style.css

# Create necessary directories
RUN mkdir -p uploads results

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
