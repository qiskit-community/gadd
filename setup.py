"""Set up the gadd package."""

import os
import setuptools

# Package dependencies
REQUIREMENTS = ["qiskit>=1.0", "qiskit-ibm-runtime>=0.23.0", "rustworkx>=0.6.0", "matplotlib"]

# Handle version.
VERSION_PATH = os.path.join(os.path.dirname(__file__), "gadd", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()


setuptools.setup(
    name="qiskit-gadd",
    version=VERSION,
    description="Qiskit-based implementation for dynamical decoupling.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Qiskit-Community/gadd",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum api dynamical decoupling",
    packages=setuptools.find_packages(exclude=["test*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
    project_urls={
        "Paper": "https://arxiv.org/abs/2403.02294",
        "Source Code": "https://github.com/Qiskit-Community/gadd",
    },
)
