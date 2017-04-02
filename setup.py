import os
from setuptools import setup, find_packages


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except:
        return 'Please see: https://github.com/TariqAHassan/BioVida.'


setup(
    name='biovida',
    version='0.1',
    author='Tariq A. Hassan',
    author_email='laterallattice@gmail.com',
    description=('Automated BioMedical Information Curation for Machine Learning Applications.'),
    long_description=read('README.md'),
    license='BSD',
    keywords='machine-learning, biomedical-informatics, data-science, bioinformatics, imaging-informatics',
    url='https://github.com/TariqAHassan/BioVida.git',
    packages=find_packages(exclude=("data.*",
                                    "data",
                                    "tests.*",
                                    "tests")
                           ),
    package_data={'biovida': ['images/_resources/*.h5', 'images/_resources/*.p'],},
    data_files=[('', ['LICENSE.md'])],
    install_requires=['bs4', 'h5py', 'keras', 'lxml', 'numpy', 'pandas', 'Pillow',
                      'pydicom', 'requests', 'scikit-image', 'scipy', 'tensorflow', 'theano', 'tqdm'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Natural Language :: English',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'License :: OSI Approved :: BSD License'
    ],
    include_package_data=True
)
