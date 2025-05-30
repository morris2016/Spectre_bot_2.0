from setuptools import setup, find_packages
import os
import logging

logger = logging.getLogger(__name__)

# Function to read requirements from requirements.txt
def parse_requirements(filename="requirements.txt"):
    """Load requirements from a pip requirements file."""
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        logger.warning(f"Warning: {filename} not found. Proceeding without defined requirements.")
        return []

setup(
    name="quantum-spectre",
    version="1.0.0",
    packages=find_packages(), # Automatically find all packages
    install_requires=parse_requirements(), # Read from requirements.txt
    author="QuantumSpectre Team",
    author_email="info@quantumspectre.ai", # Corrected email format
    description="QuantumSpectre Elite Trading System",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url="https://github.com/your-username/quantum-spectre", # Example URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License", # Assuming MIT based on README
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9", # Ensure Python version compatibility
    # entry_points={ # Optional: if you have command-line scripts
    # 'console_scripts': [
    # 'spectre-bot=main:main', # Assuming a main.py with a main() function
    # ],
    # },
    # include_package_data=True, # If you have non-code files inside your packages
    # package_data={ # Example:
    # '': ['*.pem', '*.key'], # Include all .pem and .key files
    # 'your_package_name': ['data/*.csv'],
    # },
)
