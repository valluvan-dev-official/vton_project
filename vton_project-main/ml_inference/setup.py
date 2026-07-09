"""
setup.py — Makes ml_inference pip-installable inside the SageMaker container.

The SageMaker PyTorch inference toolkit imports SAGEMAKER_PROGRAM (inference.py)
as a top-level module, so inference.py lives at /opt/ml/code/inference.py and
imports the package via absolute import:

    from ml_inference.predictor import VTONPredictor

For that absolute import to resolve, the ml_inference package must be installed
into the Python environment.  The Dockerfile does:

    COPY . /tmp/ml_inference_src/
    RUN pip install --no-cache-dir -e /tmp/ml_inference_src/

which installs this package (find_packages discovers ml_inference/).
"""
from setuptools import setup, find_packages

setup(
    name="ml_inference",
    version="1.0.0",
    description="DCI-VTON SageMaker inference package",
    packages=find_packages(
        exclude=["scripts", "scripts.*", "*.egg-info"]
    ),
    python_requires=">=3.9",
)
