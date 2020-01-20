import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nbep7",
    version="0.5",
    author="Graham Markall",
    author_email="gmarkall@nvidia.com",
    description="NBEP 7: External Memory Manager Plugin Interface prototyping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gmarkall/nbep-7",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
