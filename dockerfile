# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PORT=8080

# Copy local code to the container
WORKDIR /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the app using Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 dashboard:server