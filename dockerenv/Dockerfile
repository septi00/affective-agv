FROM ros:galactic

# Nav2 + build tooling for the proxemics layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-galactic-navigation2 \
    ros-galactic-nav2-bringup \
    ros-galactic-nav2-costmap-2d \
    ros-galactic-nav2-core \
    python3-colcon-common-extensions \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Auto-source ROS on every shell.
RUN echo 'source /opt/ros/galactic/setup.bash' >> /root/.bashrc && \
    echo '[ -f /dis_ws/install/setup.bash ] && source /dis_ws/install/setup.bash' \
    >> /root/.bashrc

WORKDIR /dis_ws
