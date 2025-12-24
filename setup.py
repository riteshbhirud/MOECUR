from setuptools import setup, find_packages

setup(
    name="relaxed-cur",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "relaxed-cur=main:main",
        ],
    },
)
