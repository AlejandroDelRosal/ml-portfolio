from setuptools import setup, find_packages

setup(
    name="palindromospy",
    version="0.1.0",
    author="Manuel Alejandro Del Rosal Saucedo",
    author_email="mars.delrosal@gmail.com", # Puedes usar el de tu portafolio
    description="A library for symmetry analysis and palindrome generation",
    long_description=open("/home/cavlex98/python/Palindromos/README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AlejandroDelRosal/Portafolio/tree/main/Palindromos",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[], # No tenemos dependencias externas
)