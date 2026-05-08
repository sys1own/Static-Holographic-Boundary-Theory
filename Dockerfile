FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libfontconfig1 \
    graphviz \
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Tectonic 0.15.0
RUN wget https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \
    && tar -xzf tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \
    && mv tectonic /usr/local/bin/ \
    && rm tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz

WORKDIR /app

# Upgrade pip before installing requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Replace the previous ENV block
ENV SHBT_INTERNAL_LATEX_BACKEND=tectonic
ENV SHBT_CONTAINERIZED_LATEX=1

RUN tectonic --version

CMD ["sh", "-c", "python src/shbt/main.py --sector universal && python scripts/build_manuscript.py"]
