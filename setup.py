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
    version = "1.1.dev",
    packages = find_packages(),
    description='cwave',
    long_description=open('README.rst').read(),
    url='cwave.readthedocs.io',
    download_url='https://github.com/compbio-fmach/cwave/releases',
    license='GPLv3',
    author='Davide Albanese',
    author_email='davide.albanese@gmail.com',
    maintainer='Davide Albanese',
    maintainer_email='davide.albanese@gmail.com',
    install_requires=[
        'numpy>=1.7.0',
        'scipy>=0.13',
        'six'],
    classifiers = _classifiers
)
