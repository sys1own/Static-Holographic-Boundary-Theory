# Static Holographic Boundary Theory - Command Registry

# Run the complete zero-parameter derivation
audit:
    python scripts/verify_dependency_lock.py
    python -m shbt.main --sector universal --zero-parameter

# Generate the formal PDF manuscript
manuscript:
    python scripts/build_manuscript.py

# Verify the cryptographic hashes of physics profiles
verify-integrity:
    python scripts/verify_config_integrity.py

# Check for dependency drift against the lockfile
verify-lock:
    python scripts/verify_dependency_lock.py

# Run the full test suite
test:
    python scripts/verify_dependency_lock.py
    pytest tests/
