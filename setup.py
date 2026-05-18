from setuptools import setup

try:
    with open("README.md", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""

setup(
    name='TheHawkesPackage',
    version="0.0.1",
    packages=['TheHawkesPackage', 'TheHawkesPackage.spatio_temporal'],
    url='',
    license='MIT',
    author='Jens Fischer',
    author_email='jefischer@posteo.de',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=["numpy", "scipy"],
)
