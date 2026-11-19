from setuptools import setup

package_name = 'py_episode'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/robot_reset.yaml']),
        ('share/' + package_name + '/config', ['config/T_camera2end.yaml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='neil',
    maintainer_email='ken1999429@gmail.com',
    description='Python interface nodes for the robot arm',
    license='BSD',
    entry_points={
        'console_scripts': [
            'robot_interface_full = py_episode.robot_interface_node_full:main',
            'moveit_api_demo = py_episode.moveit_api_demo:main',
            'main = py_episode.main:main'
        ],
    },
)