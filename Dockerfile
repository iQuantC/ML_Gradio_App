# Use official Python base image
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Gradio's default port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app.py"] 
