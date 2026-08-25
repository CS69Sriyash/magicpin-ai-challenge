# 1. Base Image: Use a lightweight, secure Python image
FROM python:3.11-slim

# 2. Environment Tuning
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# 3. Dependencies
# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt .
# Install dependencies without caching to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# 4. App Code
# Copy the application files into the working directory
COPY *.py ./

# 5. Network
# Expose port 8080 to the outside world
EXPOSE 8080

# 6. Execution
# Command to run the application using uvicorn
CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8080"]
