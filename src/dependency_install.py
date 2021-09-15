"""
install dependencies needed for models
elasticsearch, newspaper3k, numpy, matplotlib, scipy, pandas, spacy, spacytextblob
"""

import subprocess
import sys

PACKAGES1 = ["elasticsearch", "newspaper3k", "numpy", "matplotlib", "scipy", "pandas"]

def install1(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_spacy():
    # subprocess.check_call([sys.executable, "pip", "install", "-U", "pip", "setuptools", "wheel"])
    subprocess.check_call([sys.executable, "pip", "install", "-U", "spacy"])
    subprocess.check_call([sys.executable, "python3", "spacy", "download", "en_core_web_sm"])

def install_spacy_textblob():
    subprocess.check_call([sys.executable, "pip", "install", "spacytextblob"])
    subprocess.check_call([sys.executable, "python3", "-m", "textblob.download_corpora"])
    subprocess.check_call([sys.executable, "python3", "-m", "spacy", "download", "en_core_web_sm"])

def main():
    install1(PACKAGES1)
    # install_spacy()
    install_spacy_textblob()


if __name__ == "__main__":
    main()
