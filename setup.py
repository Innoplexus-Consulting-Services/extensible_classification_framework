import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="extensible_classification_framework",
    version="0.3",
    author="Sunil Patel",
    author_email="sunil.patel@innoplexus.com",
    description="Extensible Classsification framework is an engineering effort to make a well defined ensemble engine for the text classifcation task. This notebook is an usage guide for the first relese of Extensible framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
      'myapp': ['extensible_classification_framework/src/utils/resources/ontology_for_tokenizer.list'],
   },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
