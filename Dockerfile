# Base Stage Uses an official NVIDIA CUDA runtime image compatible with PyTorch (cu121).
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base
 
# Set the working directory inside the container.
WORKDIR /app

# Prevent Python from writing .pyc files and ensure output is unbuffered.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies, including Python 3.11, pip, and build tools.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# This is the final image for the application.
FROM base

# Copy the requirements file first to leverage Docker's build cache.
COPY requirements.txt .

# Install all Python dependencies directly into the system's Python site-packages.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the rest of the application source code into the working directory.
COPY . .

# Expose the port the application will run on.
EXPOSE 8000

# Define the command to start the Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

