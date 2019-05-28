import os
import setuptools

setuptools.setup(
    name='STAR-GCN',
    version="0.1.dev0",
    author="Jiani Zhang, Xingjian Shi",
    author_email="jnzhang@cse.cuhk.edu.hk, xshiab@connect.ust.hk",
    packages=setuptools.find_packages(),
    description='GluonGraph for recommender systems',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='https://github.com/jennyzhang0215/STAR-GCN',
    install_requires=['numpy', 'scipy', 'matplotlib', 'six', 'pyyaml', 'sklearn', 'pandas', 'mxnet'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)