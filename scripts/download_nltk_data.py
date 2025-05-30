#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
NLTK Data Downloader

This script downloads required NLTK data packages to avoid runtime download errors.
Run this script during setup to ensure all required NLTK data is available locally.
"""

import os
import sys
import nltk
import ssl

def download_nltk_data():
    """Download required NLTK data packages."""
    print("Downloading NLTK data packages...")
    
    # Create a directory for NLTK data if it doesn't exist
    nltk_data_dir = os.path.expanduser("~/.nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Required NLTK packages
    required_packages = [
        'vader_lexicon',  # For sentiment analysis
        'punkt',          # For sentence tokenization
        'stopwords',      # For stopword filtering
        'wordnet',        # For lemmatization
        'averaged_perceptron_tagger'  # For part-of-speech tagging
    ]
    
    # Handle SSL certificate issues that might occur
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download each package
    for package in required_packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, download_dir=nltk_data_dir, quiet=False)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
            
            # Fallback: Try to download to the default location
            try:
                print(f"Trying fallback download for {package}...")
                nltk.download(package, quiet=False)
                print(f"Fallback download successful for {package}")
            except Exception as e2:
                print(f"Fallback download failed for {package}: {str(e2)}")
    
    print("\nNLTK data download complete.")
    print(f"Data directory: {nltk_data_dir}")
    print("If you continue to experience NLTK data loading errors, manually download the data from:")
    print("https://www.nltk.org/data.html")

if __name__ == "__main__":
    download_nltk_data()
