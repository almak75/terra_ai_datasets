__version__ = "1.1.1"

from setuptools import setup, find_packages

DESCRIPTION = "Framework to create a dataset to train a neural network."
LONG_DESCRIPTION = "terra_ai_datasets is a framework to create " \
                   "a dataset to train a neural network model based on a Keras."

setup(
    name="terra_ai_datasets",
    version=__version__,
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
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
