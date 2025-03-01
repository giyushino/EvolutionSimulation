from setuptools import setup, find_packages

setup(
    name="my_module",  # Replace with your package name
    version="0.1",
    packages=find_packages(),  # Automatically finds submodules in your repo
    install_requires=[  # List dependencies here
        "torch",
    ],
    python_requires=">=3.9",  # Set minimum Python version
    author="giyushino",
    author_email="allanzhang440@gmail.com",
    description="Testing evolutionary strategies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/giyushino/EvolutionSimulation",  # Replace with your repo URL
    classifiers=[  # Metadata for PyPI
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

