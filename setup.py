from setuptools import setup, find_packages

setup(
    name="EvolutionSimulation",  # Change this if needed
    version="0.1",
    packages=find_packages(),  # Auto-detects 'Evolution' module
    install_requires=[
        "numpy",  
        "matplotlib"  # Add any other dependencies
    ],
    python_requires=">=3.6",
    author="Giyushino",
    description="A simulation of evolutionary processes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/giyushino/EvolutionSimulation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

