"""Bring up Nav2 with the proxemics costmap layer.

Wraps nav2_bringup's navigation_launch (controller, planner, recoveries,
BT navigator + lifecycle) and points it at our params. Localization/SLAM
is NOT started here — your existing SLAM must already be publishing
map->odom before you launch this.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_dir = get_package_share_directory('proxemics_bringup')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(
                bringup_dir, 'config', 'nav2_params.yaml'),
            description='Full path to the Nav2 parameters file'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    nav2_bringup_dir, 'launch', 'navigation_launch.py')),
            launch_arguments={
                'params_file': params_file,
                'use_sim_time': use_sim_time,
            }.items()),
    ])
