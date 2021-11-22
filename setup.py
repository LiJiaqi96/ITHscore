from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ITHscore",
    version="0.2.3",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.17.0",
        "matplotlib>=3.1.3",
        "scikit-learn>=0.23.2",
        "scipy>=1.5.2",
        "six>=1.15.0",
        "pydicom>=1.4.1",
        "SimpleITK>=1.2.4",
        "pyradiomics>=3.0"
    ],
    author="Jackie Li",
    author_email="li-jq18@mails.tsinghua.edu.cn",
    description="package for calculating ITHscore from CT image",
    license="MIT",
    url="https://github.com/pypa/sampleproject",
    long_description=long_description,
    long_description_content_type="text/markdown"
)
