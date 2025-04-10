#!/usr/bin/env python
"""The setup script."""
from setuptools import setup, find_packages

setup(
    author="LEA - Uni Paderborn",
    author_email='upblea@mail.upb.de',
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Environment :: MacOS X'
    ],
    description="dimensional-performance-factor",
    install_requires=['pandas~=2.2.1',
                      'numpy~=1.26.0',
                      'matplotlib~=3.8.0',
                      'scipy~=1.12.0'
                      ],
    license="GNU General Public License v3",

    # long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='dimensional-performance-factor',
    name='dimensional-performance-factor',
    packages=find_packages(include=['dimensional-performance-factor', 'dimensional-performance-factor.*']),
    url='https://github.com/upb-lea/dimensional-performance-factor',
    project_urls={
        "Source Code": "https://github.com/upb-lea/dimensional-performance-factor",
    },
    version='0.0.0',
)
