from setuptools import setup, find_packages

setup(
    name='web-table-classification',
    version='1.0.0',
    install_requires=['pyjnius'],
    packages=find_packages(),
    url='',
    license='',
    author='',
    author_email='',
    description='',
#    package_data={'web_table_classification': ['runtime_testing/resources/dwtc-extension-1.0-jar-with-dependencies.jar']},
    package_data={"": ["*.jar"]},

    include_package_data = True
    #data_files=[('resources', ['runtime_testing/resources/dwtc-extension-1.0-jar-with-dependencies.jar'])]
)
