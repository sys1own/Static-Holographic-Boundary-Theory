# Use the Python 3.11 slim image for a lightweight, rigid environment
# Fixed: 'as' casing to match 'FROM' to satisfy Docker warnings
FROM python:3.11-slim AS base

# 1. Install System-Level Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libfontconfig1 \
    graphviz \
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Tectonic (Direct Binary Download)
# We fetch the specific x86_64-unknown-linux-musl binary to ensure zero-leakage rigidity.
# Using version 0.15.0 to match the requirements.lock.
RUN wget https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \
    && tar -xzf tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \
    && mv tectonic /usr/local/bin/ \
    && rm tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz

# 3. Set the working directory
WORKDIR /app

# 4. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Project Structure
COPY . .

# 6. Environment Configuration
# SHBT_CONTAINERIZED_LATEX=1 is required for test_manuscript_containerization.py
ENV PYTHONPATH=/app/src
ENV PATH="/usr/local/bin:${PATH}"
ENV SHBT_CONTAINERIZED_LATEX=1
ENV SHBT_CONTAINER=true

# 7. Pre-warm Tectonic
# This ensures the executable is functional and pathing is correct
RUN tectonic --version

# Default command: Execute the Universal Audit and generate the Manuscript
CMD ["sh", "-c", "python src/shbt/main.py --sector universal && python scripts/build_manuscript.py"]
