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

# Copy only dependency metadata first for better layer caching
COPY --chown=appuser:appuser pyproject.toml /app/pyproject.toml

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
# Ensure the user-installed binaries are accessible
ENV PATH="/home/appuser/.local/bin:${PATH}"

RUN pip install --no-cache-dir --user pip-tools
RUN pip-compile --extra qbf --output-file /tmp/requirements.txt pyproject.toml
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# Copy the entire project last
COPY --chown=appuser:appuser . .
RUN pip install --no-cache-dir --user --no-deps .

ENTRYPOINT ["python-fbas"]
CMD ["--help"]
