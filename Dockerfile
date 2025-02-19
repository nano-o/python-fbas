FROM python:3.11

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

# Copy only dependency files first (helps with caching)
COPY --chown=appuser:appuser pyproject.toml ./

# Install Python dependencies before copying the entire project
RUN pip install --no-cache-dir --user .

# Copy the entire project (triggers rebuild only if source changes)
COPY --chown=appuser:appuser . .

RUN pip install --no-cache-dir --user --no-deps .

# Ensure the user-installed binaries are accessible
ENV PATH="/home/appuser/.local/bin:${PATH}"

ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]
CMD ["bash"]

