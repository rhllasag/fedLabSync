from glob import glob
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
from setuptools import setup, find_packages
setup(
    name="fedLabSync",
    version="0.0.1",
    author="Raul Homero Llasag Rosero",
    author_email="raul.hllasag@dei.uc.pt",
    url = 'https://github.com/rhllasag',
    description="Package used to run Hybrid Federated Learning based on Label Synchornization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "multiprocess==0.70.15",
    ],
    packages=find_packages(),
    package_dir={'fedLabSync':
                 'fedLabSync'},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
       'console_scripts': [
           'fd=fedLabSync.bin.fd:entry_func',
       ],
    },
    data_files=[
        ('haramps', glob('fedLabSync/bin/defaults/utime/*.yaml')),
        ('dataset_haramps', glob('fedLabSync/bin/defaults/utime/dataset_configurations/*.yaml')),
    ],
    python_requires='>=3.7',
)