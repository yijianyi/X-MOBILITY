# X-Mobility ROS Melodic 详细部署步骤

本文针对 **Ubuntu 18.04 x86_64 工控机/PC + NVIDIA 独立显卡 + ROS Melodic**。

## 1. 部署结构与版本

```text
相机 / 底盘 / TF / RViz
         │ ROS Melodic 消息
         ▼
x_mobility_navigator（宿主机，Python 2.7）
         │ 本机 Unix socket：float32 数组
         ▼
TensorRT 推理服务（Docker，Python 3.10 + TensorRT 8.6.1.6）
```

Melodic 的官方目标平台是 Ubuntu 18.04，默认 Python 2.7，参见
[ROS REP 3](https://github.com/ros-infrastructure/rep/blob/master/rep-0003.rst#melodic-morenia-may-2018---may-2023)。
因此，ROS 节点使用系统 Python 2.7 以及 apt 提供的 `rospy`、`cv_bridge`、TF，
GPU 推理独立运行，宿主机无需将 Melodic 的 C++ Python 扩展重新编译为 Python 3。
不要修改 `/usr/bin/python` 的指向，也不要在 Melodic Python 环境安装根目录训练依赖。

| 组件 | 本文使用的环境 |
| --- | --- |
| 宿主机 | Ubuntu 18.04，amd64 |
| ROS | Melodic，Python 2.7 |
| 推理基础镜像 | `nvcr.io/nvidia/tensorrt:23.08-py3` |
| 镜像内系统 / Python | Ubuntu 22.04 / Python 3.10 |
| 镜像内 CUDA / TensorRT | CUDA 12.2.1 / TensorRT 8.6.1.6 |
| 宿主机 NVIDIA 驱动 | 本文按 535 或更新的兼容驱动准备 |
| NVIDIA Container Toolkit | 下文固定 1.14.6，用于旧版 Ubuntu 18.04 |

镜像版本和驱动要求来自
[NVIDIA TensorRT 容器发布说明](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/index.html#tensorrt-release-23-08)。
GPU 也需要在该镜像支持范围内。此镜像不是针对更新 GPU 架构的通用方案；
若显卡需要 TensorRT 10 或更新版本，需要另行迁移推理 API。
Melodic 已结束维护，本文使用其既有发行包。

## 2. 检查宿主机和目录

以下安装命令在部署机器执行，不是在本仓库当前的开发机器执行。
先准备一个没有激活 Conda、Noetic 或 ROS 2 的终端：

```bash
lsb_release -a
uname -m
/usr/bin/python2.7 --version
nvidia-smi
```

预期为 Ubuntu 18.04、`x86_64`、Python 2.7；`nvidia-smi` 应能显示显卡和驱动。
驱动不满足要求时，先按显卡型号安装 NVIDIA 官方兼容驱动并重启，再确认
`nvidia-smi` 正常。宿主机不需要安装与容器相同版本的 CUDA toolkit。

设置仓库、模型和通信目录。将第一行替换为部署机上的真实仓库路径：

```bash
export X_MOBILITY_ROOT=/absolute/path/to/X-MOBILITY
export X_MOBILITY_MODELS="$HOME/x_mobility_models"
export X_MOBILITY_IPC=/tmp/x-mobility
mkdir -p "$X_MOBILITY_MODELS" "$X_MOBILITY_IPC"
chmod 700 "$X_MOBILITY_IPC"
```

新终端需要重新设置这三个变量，或将这些 `export` 行保存到个人环境脚本。
容器与 ROS 节点应使用同一宿主机 UID/GID，以访问权限为 `0600` 的 socket。

## 3. 安装 ROS Melodic

已经安装且 `rosversion -d` 为 `melodic` 时，跳过添加源和安装基础 ROS，
仍需确认本节末尾的运行依赖已安装。

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
printf '%s\n' 'deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu bionic main' \
  | sudo tee /etc/apt/sources.list.d/ros-melodic.list
sudo apt-get update
sudo apt-get install -y ros-melodic-ros-base ros-melodic-rviz \
  ros-melodic-cv-bridge ros-melodic-geometry-msgs ros-melodic-nav-msgs \
  ros-melodic-sensor-msgs ros-melodic-tf2-ros ros-melodic-tf2-geometry-msgs \
  ros-melodic-tf python-numpy python-setuptools build-essential
source /opt/ros/melodic/setup.bash
rosversion -d
/usr/bin/python2.7 -c 'import rospy, cv_bridge, tf2_ros, tf2_geometry_msgs, numpy; print("ROS imports OK")'
```

参考 [Melodic 官方安装说明](http://wiki.ros.org/melodic/Installation/Ubuntu)。
若旧发行版仓库出现 404 或密钥错误，先按官方存档修复仓库；不要混入其他
Ubuntu 版本的 ROS deb 包。`cv_bridge` 和 TF 的导入检查必须在 Python 2.7 下通过。

## 4. 安装 Docker 和 NVIDIA Container Toolkit

已具备可用 GPU Docker 环境时，直接运行本节最后的验证命令。
新环境可先安装 Ubuntu 提供的 Docker，确认版本不低于 19.03：

```bash
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo docker version
```

以下是 NVIDIA Container Toolkit 的仓库配置。
1.14.6 的平台列表包含 Ubuntu 18.04，参见
[平台说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.6/supported-platforms.html)
和[安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.6/install-guide.html)。

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
apt-cache madison nvidia-container-toolkit
sudo apt-get install -y \
  nvidia-container-toolkit=1.14.6-1 nvidia-container-toolkit-base=1.14.6-1 \
  libnvidia-container-tools=1.14.6-1 libnvidia-container1=1.14.6-1
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker run --rm --gpus all nvcr.io/nvidia/tensorrt:23.08-py3 nvidia-smi
```

预期容器能显示同一张 GPU。若软件源已不提供该旧版 toolkit，应从 NVIDIA
官方存档准备与 Ubuntu 18.04 兼容的包，再完成相同的 GPU 容器检查。
不要直接以“宿主机 `nvidia-smi` 正常”代替容器检查。

本文 Docker 命令均显式使用 `sudo`，不要求修改用户组。

## 5. 构建 Melodic catkin 包

使用新的专用工作空间，避免复用上一次 Noetic/Python 3 构建产生的缓存。

```bash
source /opt/ros/melodic/setup.bash
mkdir -p ~/x_mobility_melodic_ws/src
ln -s "$X_MOBILITY_ROOT/ros_deployment/x_mobility_navigator" \
  ~/x_mobility_melodic_ws/src/x_mobility_navigator
cd ~/x_mobility_melodic_ws
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python2.7
source devel/setup.bash
rospack find x_mobility_navigator
/usr/bin/python2.7 -c 'from x_mobility_navigator.inference_client import InferenceClient; print("Melodic client OK")'
```

`ln -s` 已存在时不必重复执行。CMake 固定使用 Python 2.7；不要传 Python 3。
需要安装空间时执行 `catkin_make install`，随后使用 `install/setup.bash`。

## 6. 构建 Python 3 推理镜像

从仓库根目录构建，使用专门的部署 Dockerfile：

```bash
cd "$X_MOBILITY_ROOT"
sudo docker build -f ros_deployment/Dockerfile.inference \
  -t x-mobility-melodic-inference:trt8 .
sudo docker run --rm --gpus all --entrypoint python3 \
  x-mobility-melodic-inference:trt8 \
  -c 'import tensorrt, pycuda.driver, numpy; print(tensorrt.__version__, numpy.__version__)'
```

镜像内固定 NumPy 1.24.4、PyCUDA 2024.1.1；TensorRT 使用基础镜像提供的版本。
不要在此镜像安装根目录 `requirements.txt`，也不要再安装最新的 TensorRT。
容器不需要 ROS、ROS master 地址或 `--network=host`；它仅通过共享目录中的
Unix socket 与宿主机通信。根目录原有 Dockerfile 继续用于训练和 ONNX 导出。

## 7. 导出 ONNX，并在目标 GPU 构建 engine

先在原来的训练环境准备 checkpoint：

```bash
cd /path/to/X-MOBILITY
python3 onnx_conversion.py -p /path/to/checkpoint.ckpt -o /path/to/x_mobility.onnx
```

导出前检查 `onnx_conversion.py` 中输入 `image` 的尺寸。默认是
`(1, 1, 3, 1200, 1920)`，必须与实际相机的高度和宽度一致。
例如相机输出 640×480，则改为 `(1, 1, 3, 480, 640)` 后重新导出。
不要启用 `--enable-semantic`、`--enable-rgb`、`--enable-depth`。

将 ONNX 复制到部署机的 `$X_MOBILITY_MODELS/x_mobility.onnx`。
然后在**部署 GPU 和同一个推理镜像**中生成 engine：

```bash
sudo docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$X_MOBILITY_MODELS:/models" --entrypoint python3 \
  x-mobility-melodic-inference:trt8 \
  /opt/x-mobility/trt_conversion.py \
  -o /models/x_mobility.onnx -t /models/x_mobility.engine
ls -lh "$X_MOBILITY_MODELS/x_mobility.engine"
```

如 ONNX 解析失败，保留解析器输出，先处理不支持的算子或导出配置；
不要使用其他 TensorRT 大版本生成的 engine 替代。
更换 GPU、TensorRT 版本或图像尺寸后应重新构建 engine。

服务启动时会检查：输入名称为 `image`、`route`、`speed`、`action_input`、
`history_input`、`sample_input`；输出名称为 `action_output`、`path_output`、
`history_output`、`sample_output`，元素数量分别为 6、10、1024、512。
输入输出必须是固定形状 float32，绑定顺序由 engine 查询，GPU 缓冲区重复使用。

## 8. 启动推理服务，独立验证 Python 2 → Python 3 通信

终端 A：

```bash
sudo docker run --rm --gpus all --name x-mobility-inference \
  --user "$(id -u):$(id -g)" \
  -v "$X_MOBILITY_MODELS:/models:ro" \
  -v "$X_MOBILITY_IPC:/tmp/x-mobility" \
  x-mobility-melodic-inference:trt8 \
  --engine /models/x_mobility.engine --socket /tmp/x-mobility/inference.sock
```

出现 `Inference ready on /tmp/x-mobility/inference.sock` 表示 engine 已加载。
在另一个终端检查共享 socket：

```bash
ls -l /tmp/x-mobility/inference.sock
source /opt/ros/melodic/setup.bash
source ~/x_mobility_melodic_ws/devel/setup.bash
rosrun x_mobility_navigator check_inference --width 1920 --height 1200
```

宽高必须使用导出 ONNX 时的实际值。该检查由 Python 2.7 客户端发送测试输入，
通过 Python 3 服务执行真正 GPU 推理，并验证四个输出；不会发布 ROS 速度指令，
也不需要先启动底盘。预期最后出现 `Inference check passed`。

这一步未通过前，先不要进入导航联调。检查日志可使用：

```bash
sudo docker logs x-mobility-inference
```

## 9. 启动 ROS master、相机、底盘和 TF

终端 B：

```bash
source /opt/ros/melodic/setup.bash
roscore
```

终端 C：启动机器人现有的相机、底盘、里程计及 TF launch 文件。
本仓库不包含特定硬件驱动，使用该机器已有的驱动配置。

默认接口如下，可通过最后一列的 launch 参数改名：

| 方向 | 默认话题 | 消息类型 | 参数 |
| --- | --- | --- | --- |
| 订阅 | `/front_stereo_camera/left/image_raw` | `sensor_msgs/Image` | `image_topic` |
| 订阅 | `/chassis/odom` | `nav_msgs/Odometry` | `odom_topic` |
| 订阅 | `/move_base_simple/goal` | `geometry_msgs/PoseStamped` | `goal_topic` |
| 订阅 | `/route` | `nav_msgs/Path` | `route_topic` |
| 发布 | `/cmd_vel` | `geometry_msgs/Twist` | `cmd_vel_topic` |
| 发布 | `/x_mobility_path` | `nav_msgs/Path` | `path_topic` |

例如实际接口是 `/camera/color/image_raw` 和 `/odom`，检查：

```bash
rostopic list
rostopic type /camera/color/image_raw
rostopic hz /camera/color/image_raw
rostopic echo -n 1 /camera/color/image_raw/encoding
rostopic echo -n 1 /odom/twist/twist
rosrun tf tf_echo odom base_link
```

图像需为原始 `sensor_msgs/Image`，不是 `CompressedImage`，尺寸要与 engine 一致。
节点会转为 RGB、CHW、float32 并归一化到 `[0, 1]`，支持图像行尾填充。
速度读取 `Odometry.twist.twist.linear.x`，单位为 m/s。
TF 必须能将目标/路径的 `header.frame_id` 转到机器人坐标系 `base_link`。

多机 ROS 时，在相关 ROS 终端配置 `ROS_MASTER_URI` 和本机可被对端访问的
`ROS_IP`/`ROS_HOSTNAME`；GPU 推理容器仍与导航节点部署在同一台工控机上。

## 10. 启动导航与 RViz

终端 D，真实机器人使用墙上时钟：

```bash
source /opt/ros/melodic/setup.bash
source ~/x_mobility_melodic_ws/devel/setup.bash
roslaunch x_mobility_navigator x_mobility_navigator.launch \
  image_topic:=/camera/color/image_raw odom_topic:=/odom \
  cmd_vel_topic:=/cmd_vel robot_frame:=base_link \
  inference_socket:=/tmp/x-mobility/inference.sock \
  use_sim_time:=false launch_rviz:=true
```

将话题名称替换为第 9 步实际存在的接口。图像和里程计驱动无需修改源代码。
节点启动后等待图像、里程计和目标；默认推理频率为 5 Hz。

在 RViz 中：

1. 将 **Fixed Frame** 设为 `odom`，或机器人实际使用的世界坐标系。
2. 确认 Camera、TF 显示正常。
3. 选择 **2D Nav Goal**，在目标位置按下鼠标并拖动设置朝向。
4. 确认 `/cmd_vel` 有输出，`/x_mobility_path` 显示输入导航路线。

也可以从终端发送目标并检查：

```bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  '{header: {frame_id: odom}, pose: {position: {x: 3.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
rostopic echo /cmd_vel
rostopic hz /x_mobility_path
```

`/x_mobility_path` 当前显示送入模型的路线，而非模型解码的 `path_output`。
无地图模式 `is_mapless:=true` 会从机器人到目标构建直线路线。

### 使用外部全局路径

在同一个启动命令中添加：

```text
is_mapless:=false route_topic:=/move_base/GlobalPlanner/plan
```

将 `route_topic` 替换为规划器实际发布的 `nav_msgs/Path`。
这种模式使用外部路径，RViz 目标不会直接生成路线。

### 无界面或仿真

- 无界面：设置 `launch_rviz:=false`。
- Melodic Gazebo 或 rosbag 仿真：设置 `use_sim_time:=true`，确保存在 `/clock`。
- 推理频率：例如 `inference_rate:=5.0`。
- 推理连接/单次读写超时：`inference_timeout:=5.0` 秒。
- 使用不同机器人坐标系：例如 `robot_frame:=base_footprint`，并保证 TF 连通。

原先 Noetic 文档中的 Isaac Sim 4.2 配置不能直接作为 Melodic 已验证配置使用；
本文以原生 Melodic 驱动或 Gazebo 接入为准。

## 11. 停止、重启与故障排查

停止 ROS 节点：在终端 D 按 Ctrl+C。停止 GPU 服务：

```bash
sudo docker stop x-mobility-inference
```

服务正常停止会移除 socket。异常退出留下的旧 socket 会在下次启动时检测并清理；
已有活动服务或同名普通文件时不会覆盖。通信目录权限为 700，socket 为 600。

推理连接中断、超时或响应无效时，ROS 节点发布零速度，在后续推理周期尝试重连；
重启服务不需要修改 ROS 参数。底盘应沿用已有的速度指令超时机制。

| 现象 | 检查与处理 |
| --- | --- |
| Python 报 `cv_bridge` / `_tf2` 导入错误 | 确认 ROS 节点使用 `/usr/bin/python2.7`；重新开终端，只 source Melodic 和该工作空间 |
| CMake 拒绝 Python 3 | 在专用 Melodic 工作空间使用 `-DPYTHON_EXECUTABLE=/usr/bin/python2.7` |
| 容器看不到 GPU | 检查宿主驱动、toolkit runtime 和 `--gpus all`；重跑第 4 步容器验证 |
| engine 无法反序列化 | 在相同镜像和目标 GPU 上重新执行第 7 步；查看 GPU 和 TensorRT 兼容性 |
| `Input size does not match engine: image` | 对照相机宽高重新导出 ONNX、构建 engine；不要只修改 launch |
| `six inputs and four outputs` | 重新导出默认模型，关闭 semantic/RGB/depth 输出标志 |
| `Inference service unavailable` | 确认服务已就绪、挂载了同一 IPC 目录、UID/GID 相同、socket 路径一致 |
| socket `Permission denied` | 检查目录和 socket 所有者；按第 8 步用宿主机当前 UID/GID 运行容器 |
| 一直等待输入 | 检查图像/里程计话题、目标或外部路径；确认 TF 连通 |
| TF 无法转换 | 检查目标/路径 frame、`robot_frame`、发布 TF 的驱动和时间来源 |
| 仿真中节点不继续运行 | 检查 `/use_sim_time` 与 `/clock`；真实机器人保持 `use_sim_time:=false` |
| 有速度输出但底盘不动 | 检查底盘实际订阅话题、速度复用器、急停状态和已有控制器是否争用 |

## 12. 验证范围与部署验收

开发机可以使用 Python 3 和 NumPy 执行逻辑测试，ROS/GPU 接口通过测试替身替代：

```bash
cd "$X_MOBILITY_ROOT"
python3 -m unittest discover -s ros_deployment/tests -v
```

允许本机 socket 监听的环境可以额外运行真实客户端/服务集成测试：

```bash
XMOBILITY_RUN_SOCKET_SERVER_TEST=1 \
  python3 -m unittest discover -s ros_deployment/tests -v
```

这条测试使用模拟推理结果；真正 Python 2.7 ↔ Python 3 ↔ GPU 的验收是第 8 步的
`rosrun x_mobility_navigator check_inference`。最终应依次确认：

1. Melodic Python 2.7 依赖导入通过。
2. catkin 包构建通过。
3. GPU 容器和 engine 构建通过。
4. `check_inference` 的四个输出形状及数值通过。
5. 相机、里程计、TF 与时间来源一致。
6. 设置目标后，RViz、路线和底盘速度输出符合预期。

当前开发环境缺少 Melodic/Python 2.7、Docker 和 GPU 推理运行时，且 socket 操作受
沙箱限制；开发机检查不能代替以上目标机验收。
