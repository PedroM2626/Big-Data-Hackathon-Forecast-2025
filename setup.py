from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="big-data-hackathon-forecast-2025",
    version="0.1.0",
    author="Pedro Morato, Pietra Paz, Alisson Guarniêr",
    author_email="team@example.com",
    description="Previsão de Vendas Semanais por PDV/SKU para o Big Data Hackathon 2025",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/big-data-hackathon-forecast-2025",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "python-dateutil>=2.8.2",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "xgboost>=1.5.0",
        "jupyter>=1.0.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.12b0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.940",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "forecast-train=src.models.train:main",
            "forecast-eval=src.models.evaluate:main",
            "forecast-pipeline=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
