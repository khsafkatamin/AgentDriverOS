FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

USER root
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Link Python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files into the container
COPY . .

# Default command
CMD ["bash"]
