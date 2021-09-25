from setuptools import setup


def readme():
    with open("README.rst") as readme_file:
        return readme_file.read()


configuration = {
    "name": "pynndescent",
    "version": "0.5.5",
    "description": "Nearest Neighbor Descent",
    "long_description": readme(),
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    "keywords": "nearest neighbor, knn, ANN",
    "url": "http://github.com/lmcinnes/pynndescent",
    "author": "Leland McInnes",
    "author_email": "leland.mcinnes@gmail.com",
    "maintainer": "Leland McInnes",
    "maintainer_email": "leland.mcinnes@gmail.com",
    "license": "BSD",
    "packages": ["pynndescent"],
    "install_requires": [
        "scikit-learn >= 0.18",
        "scipy >= 1.0",
        "numba >= 0.51.2",
        "llvmlite >= 0.30",
        "joblib >= 0.11",
    ],
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "nose.collector",
    "tests_require": ["nose"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)
