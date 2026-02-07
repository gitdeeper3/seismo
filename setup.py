from setuptools import setup, find_packages

setup(
    name="seismo-framework",
    version="1.0.0",
    packages=find_packages(),
    package_data={
        'seismo_framework': [
            'config/*.yaml',
            'config/*.yml',
            'config/*.json',
            'config/regions/*.yaml',
            'config/templates/*.yaml',
            'data/samples/*.csv',
            'data/samples/*.json',
        ],
    },
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'python-dateutil>=2.8.0',
        'pytz>=2021.3',
        'tzlocal>=4.0',
        'pyyaml>=6.0',
    ],
    entry_points={
        'console_scripts': [
            'seismo-monitor=seismo_framework.cli.monitor:main',
            'seismo-analyze=seismo_framework.cli.analyze:main',
            'seismo-report=seismo_framework.cli.report:main',
        ],
    },
)
