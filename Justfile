# Static Holographic Boundary Theory - Command Registry
# Usage: 'just <recipe>'

# Run the complete zero-parameter derivation for all sectors
audit:
    python -m shbt.main --sector universal --zero-parameter

# Generate the formal PDF manuscript using the containerized engine
manuscript:
    python scripts/build_manuscript.py

# Verify the cryptographic hashes of all physics configuration profiles
verify-integrity:
    python scripts/verify_config_integrity.py

# Check for dependency drift against the requirements.lock
verify-lock:
    python scripts/verify_dependency_lock.py

# Run the full suite of 309+ logical and physical tests
test:
    pytest tests/
