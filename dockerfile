# Use an official Python runtime as a parent image
FROM python:3.14-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Define environment variables (optional, can be overridden at runtime)
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run smart_stream.py when the container launches
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 dashboard:server