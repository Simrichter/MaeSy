from setuptools import setup, find_packages

setup(
    name="maesy",
    version="0.1.0",
    description="Vision Transformer framework for object detection in robot soccer",
    author="Simon Ian Richter",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1", #1.13.1+cu117",
        "torchvision", #>=0.14.1+cu117",
        "numpy",
        "pillow>=9.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "requests>=2.27.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.5.0",
        "pycocotools>=2.0.4",
        "scipy>=1.7.0",
        "scikit-learn",
        "onnxscript",
        "timm",
        "wandb",
        "faiss-cpu",
        "cv-bridge",
        "pyyaml",
        "scikit-image"
    ],
    entry_points={
        "console_scripts": [
            "maesy = maesy.command_line:main",
        ]
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
