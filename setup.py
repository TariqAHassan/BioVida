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
    long_description=read('docs/README.md'),
    license='BSD',
    keywords='medicine, biological sciences, machine learning, data science',
    url='https://github.com/TariqAHassan/BioVida.git',
    packages=find_packages(),
    data_files=[('', ['LICENSE.md'])],
    install_requires=['inflect', 'pandas', 'numpy', 'requests', 'tqdm'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Natural Language :: English',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'License :: OSI Approved :: BSD License'
    ],
    include_package_data=True
)