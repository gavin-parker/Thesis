from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = [
'numpy==1.14.1',
'opencv-contrib-python==3.4.0.12',
'scikit-image==0.13.1',
'scipy==1.0.0',
'subprocess32==3.2.7']

setup(  name = 'trainer',
        version='0.1',
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(),
        include_package_data=True,
        description='Classifier test')