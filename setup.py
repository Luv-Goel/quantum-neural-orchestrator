from setuptools import setup, find_packages

setup(
    name="quantum-neural-orchestrator",
    version="0.1.0",
    description="Advanced Quantum-Inspired Neural Architecture for Multi-Dimensional Data Processing and Autonomous System Orchestration",
    author="Luv-Goel",
    author_email="luv@example.com",
    url="https://github.com/Luv-Goel/quantum-neural-orchestrator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tensorflow>=2.12.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "flake8>=6.0.0",
        ],
        "quantum": [
            "qiskit>=0.44.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="quantum neural-network machine-learning orchestration ai",
    project_urls={
        "Documentation": "https://github.com/Luv-Goel/quantum-neural-orchestrator#readme",
        "Source": "https://github.com/Luv-Goel/quantum-neural-orchestrator",
        "Tracker": "https://github.com/Luv-Goel/quantum-neural-orchestrator/issues",
    },
)