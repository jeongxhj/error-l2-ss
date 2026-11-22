# Metadata lives here, not in pyproject's [project]: the ROS humble images pin setuptools 58,
# which predates PEP 621 and would build a nameless package from pyproject alone.
from setuptools import find_packages, setup

setup(
    name='forza_config',
    version='0.1.0',
    description='Pydantic schemas and path-based loaders for the ForzaETH car configs: car model, '
                'Pacejka tyre, steering LUT, trailing. No ROS.',
    license='MIT',
    python_requires='>=3.8',
    packages=find_packages(include=['forza_config*']),
    install_requires=['pydantic>=2,<3', 'pyyaml>=5.3', 'numpy'],
    extras_require={'test': ['pytest']},
)
