import pathlib

from setuptools import find_packages, setup

# Read the contents of your README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="gibbonbot",
    version="0.1.0",
    description="Reinforcement Learning framework for training a 4-limb robotic ape",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gibbonBot-RL",
    author="Your Name",
    author_email="your.email@example.com",
    # Package structure
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # Include data files
    package_data={
        "gibbonbot": ["data/*.xml"],
    },
    # Entry points for command line scripts
    entry_points={
        "console_scripts": [
            "gibbonbot-train=scripts.train:main",
        ],
    },
    # Dependencies
    install_requires=[
        "stable-baselines3==2.8.0",
        "mujoco==3.8.0",
        "mujoco-mjx==3.8.0",
        "gymnasium==1.2.3",
        "jax==0.10.0",
        "jaxlib==0.10.0",
        "flax==0.12.7",
        "optax==0.2.8",
        "numpy==2.4.4",
        "pandas==3.0.2",
        "matplotlib==3.10.9",
        "torch==2.11.0",
        "opencv-python",
    ],
    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.0",
            "flake8>=6.0",
        ],
        "gpu": [
            "jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        ],
    },
    python_requires=">=3.8",
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="reinforcement-learning robotics mujoco simulation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gibbonBot-RL/issues",
        "Source": "https://github.com/yourusername/gibbonBot-RL",
    },
)
