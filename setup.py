from setuptools import setup, find_packages

_classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License (GPL)',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Operating System :: OS Independent']

setup(
    name='cwave',
    version = "1.0",
    packages = find_packages(),
    description='cwave',
    long_description=open('README.rst').read(),
    url='',
    download_url='',
    license='GPLv3',
    author='Davide Albanese',
    author_email='davide.albanese@gmail.com',
    maintainer='Davide Albanese',
    maintainer_email='davide.albanese@fmach.it',
    install_requires=[
        'numpy>=1.7.0',
        'scipy>=0.13',
        'six'],
    classifiers = _classifiers
)
