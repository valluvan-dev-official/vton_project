"""
ml_inference/__init__.py (top-level, build-context root)

NOTE: This file is at the ROOT of the ml_inference/ build context directory
(i.e. vton_project-main/ml_inference/__init__.py).  It is NOT the package
__init__.py.

The actual Python package __init__.py lives at:
  ml_inference/ml_inference/__init__.py

This file exists only so that local development imports
`import ml_inference` from the project root still work via the
editable install (`pip install -e .`), which installs the INNER
ml_inference/ directory (discovered by find_packages() in setup.py).

Do NOT add code here — it is not part of the installed package.
"""
