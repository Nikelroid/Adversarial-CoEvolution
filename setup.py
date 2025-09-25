from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Adversarial-CoEvoloution",
    version= "0.1",
    author="VLAvengers",
    packages=find_packages(),
    install_requires = requirements,
)