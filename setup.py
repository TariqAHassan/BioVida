import os
from setuptools import setup, find_packages


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except:
        return 'Please see: https://github.com/TariqAHassan/BioVida.'


def required_dependencies():
    """
    Adds TensorFlow to dependencies if Theano cannot be imported.
    """
    dependencies = ['bs4', 'h5py', 'keras', 'lxml', 'numpy', 'pandas', 'Pillow',
                    'pydicom', 'requests', 'scikit-image', 'scipy', 'tqdm']

    try:
        import theano
    except ImportError:
        dependencies += ['tensorflow']

    return dependencies


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
    packages=find_packages(),
    package_data={'biovida': ['images/resources/*.h5', 'images/resources/*.p'],},
    data_files=[('', ['LICENSE.md'])],
    install_requires=required_dependencies(),
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Natural Language :: English',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'License :: OSI Approved :: BSD License'
    ],
    include_package_data=True
)
