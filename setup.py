from setuptools import setup

setup(
    name='gpu_monitor',
    version='0.1',
    py_modules=['main'],
    install_requires=[
        'pynvml==11.5.3',
        'requests==2.31.0',
        'pyyaml==6.0.1',
        'prometheus_client==0.17.0',
        'tabulate==0.9.0',
        'matplotlib==3.7.1',
    ],
    entry_points={
        'console_scripts': [
            'gpu_monitor=main:main',
        ],
    },
)
