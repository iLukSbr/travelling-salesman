from setuptools import setup
import os

def generate_epydoc():
    os.system("epydoc --config epydoc.config")

setup(
    name="seu_projeto",
    version="1.0.0",
    packages=["algorithms", "utils"],
    install_requires=[
        "matplotlib",
        "numpy",
        "psutil",
        "numba",
        "cupy",
    ],
    cmdclass={
        'build_docs': generate_epydoc
    },
)

# Para gerar a documentação, você pode executar:
# python setup.py build_docs