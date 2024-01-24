from setuptools import setup, find_packages

setup(
    name="neural_analysis",
    version="1.0.1",
    author="Daniel Deng",
    author_email="hdeng3@caltech.edu",
    description="Collection of neural analysis tools",
    url="https://github.com/ddeng00/neural_analysis.git",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "statsmodels",
        "tqdm",
        "scipy",
        "scikit-learn",
        "matplotlib",
    ],
)
