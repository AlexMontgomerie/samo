import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="samo",
    version="0.2.4",
    author="Alex Montgomerie, Zhewen Yu",
    author_email="am9215@ic.ac.uk",
    description="Streaming Architecture Mapping Optimiser",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexMontgomerie/samo",
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "networkx>=2.5",
        "tqdm>=4.62.3",
        "numpy>=1.19.2",
        "tabulate>=0.8.9",
    ]
)
