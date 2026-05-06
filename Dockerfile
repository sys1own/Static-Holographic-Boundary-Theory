# Use the Python 3.11 slim image for a lightweight, rigid environment
FROM python:3.11-slim

# Set the working directory within the container
WORKDIR /app

# Copy requirements first to leverage Docker's cache layers
COPY requirements.txt .

# Install the specific SHBT dependency stack
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure into the container[cite: 11]
COPY . .

# Set PYTHONPATH so internal shbt modules are discoverable[cite: 11]
ENV PYTHONPATH=/app/src

# Default command: Execute the high-level derivation audit
CMD ["python", "scripts/derive_universe.py"]
