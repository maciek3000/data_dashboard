from setuptools import setup, find_packages
import pathlib


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "readme.md").read_text(encoding="utf-8")

setup(
    name="data_dashboard",
    version="0.1.0",
    description="Dashboard to explore the data and to create baseline Machine Learning model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maciek3000/data_dashboard",
    author="Maciej Dowgird",
    author_email="dowgird.maciej@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    package_dir={"data_dashboard": "data_dashboard"},
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.2.3",
        "numpy>=1.19.5",
        "scipy>=1.6.1",
        "beautifulsoup4>=4.9.3",
        "scikit-learn>=0.24.1",
        "seaborn>=0.11.1",
        "bokeh>=2.3.0",
        "Jinja2>=2.11.3",
        "xgboost>=1.3.3",
        "lightgbm>=3.2.0"
    ],
    package_data={
        "data_dashboard": ["static/*", "templates/*", "examples/*"]
    },
    project_urls={
        "Github": "https://github.com/maciek3000/data_dashboard",
    },
)
