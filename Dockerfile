FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libzmq3-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN useradd --create-home --shell /bin/bash appuser

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Copy the entire project
COPY --chown=appuser:appuser . .

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
# Ensure the user-installed binaries are accessible
ENV PATH="/home/appuser/.local/bin:${PATH}"

RUN pip install --no-cache-dir --user .[qbf]

CMD ["python-fbas", "--help"]
