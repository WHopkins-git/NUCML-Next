"""
NUCML-Next Setup Script
========================
Next-Generation Nuclear Data Evaluation Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nucml-next",
    version="1.1.0-alpha",
    author="NUCML-Next Team",
    author_email="nucml@example.com",
    description="Next-Generation Nuclear Data Evaluation with Physics-Informed Deep Learning (Production-Ready with EXFOR Data)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WHopkins-git/NUCML-Next",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "pyarrow>=12.0.0",
        "fastparquet>=2023.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "openmc": [
            "openmc>=0.13.0",
        ],
    },
    keywords="nuclear physics machine-learning deep-learning gnn transformer cross-sections",
    project_urls={
        "Documentation": "https://github.com/WHopkins-git/NUCML-Next/wiki",
        "Source": "https://github.com/WHopkins-git/NUCML-Next",
        "Issues": "https://github.com/WHopkins-git/NUCML-Next/issues",
    },
)
