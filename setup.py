from setuptools import find_packages, setup

setup(
    name="saerl",
    version="0.0.1",
    description="Multi-Objective RL with SAE",
    author="Ryan Kortvelesy",
    author_email="rk627@cam.ac.uk",
    packages=find_packages(),
    install_requires=["torch", "torchrl", "moviepy", "wandb"],
)
