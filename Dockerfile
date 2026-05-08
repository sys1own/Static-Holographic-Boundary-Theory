# Use the Python 3.11 slim image for a lightweight, rigid environment
FROM python:3.11-slim as base

# 1. Install System-Level Dependencies
# We include curl and ca-certificates for Tectonic, and build-essential for any C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libfontconfig1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Tectonic (Self-Contained LaTeX Engine)
# This removes the need for a 5GB TeXLive installation while allowing PDF generation
RUN curl --proto '=https' --tlsv1.2 -sSf https://drop-sh.fullyjustified.net | sh \
    && mv tectonic /usr/local/bin/

# 3. Set up the working directory
WORKDIR /app

# 4. Handle Python Dependencies
# Using a requirements file; consider using a lockfile (poetry.lock/uv.lock) for Tier 1 rigidity
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Universal Source Code
COPY . .

# 6. Environment Configuration
# Set PYTHONPATH for internal shbt modules and ensure Tectonic is in the path
ENV PYTHONPATH=/app/src
ENV PATH="/usr/local/bin:${PATH}"
ENV SHBT_CONTAINER=true

# 7. Pre-Build/Cache LaTeX dependencies (Optional but recommended)
# This runs a dummy build to ensure the LaTeX packages are baked into the image layers
# RUN tectonic src/shbt/manuscript/main.tex --setup

# Default command: Secure the Metric Lock and generate the Manuscript
# This allows the container to act as a "Black Box" that outputs a verified universe
CMD ["sh", "-c", "python src/shbt/main.py --sector universal && python scripts/build_manuscript.py"]
