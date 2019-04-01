import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="extensible_classification_framework",
    version="0.0.1",
    author="Sunil Patel",
    author_email="sunil.patel@innoplexus.com",
    description="A package to test various model on the text classification task",
    long_description="""
    This is an AutoML approach specifically designed for text classification problem. It has 3 modules
     1) A convolution modul
     2) A Attention based LSTM classifier
     3) A Feed Forward Network 

     These 3 module can be run with various combinations to test best model that gives maximum efficiency.
    """,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
     package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.list']
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Natural Language :: English",
        
    ],
)