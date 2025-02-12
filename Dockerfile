FROM python:3.11

# Install required system dependencies for building C code
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    cmake g++ libzmq3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN useradd --create-home --shell /bin/bash appuser

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Copy the entire project
COPY --chown=appuser:appuser . .

# Run make and install
RUN make -C python_fbas/constellation/brute-force-search install PREFIX=/home/appuser/.local/bin

# Ensure the user-installed binaries are accessible
ENV PATH="/home/appuser/.local/bin:${PATH}"

RUN pip install --no-cache-dir --user .

# Expose Jupyter Notebook port
EXPOSE 8888

# Default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/app/python_fbas/constellation/notebooks/"]
