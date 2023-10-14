import setuptools
from glob import glob
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="fedLabSync",
    version="0.0.1",
    author="Raul Homero Llasag Rosero",
    author_email="rosero@dei.uc.pt",
    url = 'https://github.com/rhllasag',
    description="Package used to run Hybrid Federated Learning based on Label Synchornization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['fedLabSync'],
    install_requires=[
        "multiprocess==0.70.15",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    data_files=[
        ('weights', glob('models/*.h5')),
        ('yaml', glob('models/*.yaml')),
        ('json', glob('models/*.json')),
    ],
    include_package_data = True,
    python_requires='>=3.8',
)