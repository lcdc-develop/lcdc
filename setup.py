from setuptools import find_packages, setup

setup(
    name='lcdc',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    license='MIT',
    install_requires=["scikit-learn", 
                      "numpy",
                      "matplotlib", 
                      "datasets",
                      "tqdm",
                      "strenum"],
)