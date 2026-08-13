from setuptools import setup

package_name = 'proxemics_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Septimiu',
    maintainer_email='septi@example.com',
    description='Synthetic proxemics-circle publisher for home dev without a camera',
    license='MIT',
    entry_points={
        'console_scripts': [
            'fake_publisher = proxemics_sim.fake_circle_publisher:main',
        ],
    },
)
