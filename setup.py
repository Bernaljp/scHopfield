"""Setup script for scHopfield package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='scHopfield',
    version='0.1.0',
    author='scHopfield Contributors',
    author_email='',
    description='Single-cell Hopfield network analysis for gene regulatory networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Bernaljp/scHopfield',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
            'nbsphinx>=0.8',
            'ipykernel>=6.0',
        ],
        'optional': [
            'seaborn>=0.11.0',
            'python-igraph>=0.9.0',
            'dynamo-release>=1.0.0',
        ],
        'all': [
            'seaborn>=0.11.0',
            'python-igraph>=0.9.0',
            'dynamo-release>=1.0.0',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords='single-cell bioinformatics hopfield-network gene-regulatory-networks RNA-seq energy-landscape',
    project_urls={
        'Bug Reports': 'https://github.com/Bernaljp/scHopfield/issues',
        'Source': 'https://github.com/Bernaljp/scHopfield',
        'Documentation': 'https://schopfield.readthedocs.io',
    },
)
