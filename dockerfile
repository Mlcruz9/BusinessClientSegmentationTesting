# Use an official Python runtime as a parent image
FROM python:3.10.2-slim

# Set non-interactive installation to avoid stuck builds
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    tar \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Download and compile SQLite from source to get the latest version
RUN curl -O https://www.sqlite.org/2022/sqlite-autoconf-3370200.tar.gz \
    && tar xvfz sqlite-autoconf-3370200.tar.gz \
    && cd sqlite-autoconf-3370200 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && make clean \
    && cd .. \
    && rm -rf sqlite-autoconf-3370200* \
    && ldconfig

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install ollama using their provided script
RUN curl -fsSL https://ollama.com/install.sh | sh

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME World

# Command to run ollama and streamlit
CMD ollama serve & sleep 10 && ollama run phi3 && streamlit run main.py
