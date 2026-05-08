# Use the Python 3.11 slim image for a lightweight, rigid environment
FROM python:3.11-slim as base

# 1. Install System-Level Dependencies
# build-essential: Required for high-precision math C-extensions
# libfontconfig1: Required by Tectonic/XeTeX for font management
# graphviz: Required for generating Noether Bridge diagrams
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libfontconfig1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Tectonic (Self-Contained LaTeX Engine)
# This allows PDF generation without the 5GB overhead of a full TeXLive distribution
RUN curl --proto '=https' --tlsv1.2 -sSf https://drop-sh.fullyjustified.net | sh \
    && mv tectonic /usr/local/bin/

# 3. Set the working directory
WORKDIR /app

# 4. Install Python Dependencies
# We use requirements.txt, but the Justfile will enforce the requirements.lock for rigidity
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Project Structure
COPY . .

# 6. Environment Configuration
# PYTHONPATH: Ensures the 'shbt' package is discoverable
# SHBT_CONTAINERIZED_LATEX: Critical flag required by test_manuscript_containerization.py
ENV PYTHONPATH=/app/src
ENV PATH="/usr/local/bin:${PATH}"
ENV SHBT_CONTAINERIZED_LATEX=1
ENV SHBT_CONTAINER=true

# 7. Pre-warm Tectonic (Optional)
# This ensures standard LaTeX packages are cached in the image layer
# RUN tectonic --version

# Default command: Execute the Universal Audit and generate the Manuscript
# This sequence ensures the physics is verified before the report is written
CMD ["sh", "-c", "python src/shbt/main.py --sector universal && python scripts/build_manuscript.py"]
