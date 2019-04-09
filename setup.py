import setuptools
__author__ = 'sunil.patel <innoplexus.com>'


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="extensible_classification_framework",
    version="0.3",
    author="Sunil Patel",
    author_email="sunil.patel@innoplexus.com",
    description="Extensible Classsification Framework is an engineering effort to make a well defined ensemble engine for the text classifcation task. This notebook is an usage guide for the first relese of Extensible framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
      'myapp': ['extensible_classification_framework/src/utils/resources/ontology_for_tokenizer.list'],
   },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.4',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Text Editors :: Text Processing',
    ],
    install_requires=[
        'nltk==3.2.4',
        'chakin==0.0.6',
        'torch==0.4.1',
        'tqdm==4.31.1',
        'torchtext==0.3.1',
        'matplotlib==2.2.2',
        'pandas==0.24.1',
        'gensim==3.7.1',
        'numpy==1.16.1',
        'scikit_learn==0.20.0',
    ],

)