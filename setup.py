__version__ = "1.0.0"

from setuptools import setup, find_packages


setup(
    name="terra_ai_datasets",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "pydantic>=1.8.2",
        "pandas>=1.3.5",
        "opencv-python>=4.6.0.66",
        "librosa>=0.8.1",
        "pillow>=7.1.2",
        "tqdm>=4.64.1",
        "scikit-learn>=1.0.2",
        "tensorflow>=2.0",
        "joblib>=1.1.0",
        "pymorphy2>=0.9.1",
        "gensim>=3.6.0",
        "gensim>=3.6.0",
    ],
)
