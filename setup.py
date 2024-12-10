from setuptools import setup, find_packages

setup(
    name='rfid_system_pso_optimization',
    version='1.0.0',
    packages=find_packages(),
    author_email='huynhvikhang6a13@gmail.com',
    url='https://github.com/netprtony/RFID-System-PSO',  
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'colorama'
    ],
    entry_points={
        'console_scripts': [
            'rfid-system=rfid_system.main:main',
        ],
    },
    python_requires='>=3.12.3',
)