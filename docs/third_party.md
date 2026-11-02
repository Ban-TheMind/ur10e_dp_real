# 来源与许可

`diffusion_policy/` 是原项目所需的Transformer RGB策略、扩散模型、视觉编码和归一化依赖闭包。上游 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) 使用MIT，原LICENSE保留在根目录。逐文件来源指纹见 `core_source_manifest.json`；本任务不重写其核心算法。

夹爪SDK来自 [robotiq/grippers](https://github.com/robotiq/grippers)，提交 `9017b5707cb28a8ea21fd7c4e84a50f7c56aaa5d`，BSD-3-Clause。其源码和LICENSE与UPSTREAM保留。libserialport0.1.2源码包及LGPL说明也一并保存。桥接main.cpp/operations.hpp来自既有UR10e工程，此项目移除ROS构建依赖，并增加非阻塞command操作，让采集期间持续读取实际夹爪反馈。

键盘采集交互参考 [scy-v/lerobot_ur5e_keyteleop](https://github.com/scy-v/lerobot_ur5e_keyteleop) 与LeRobot，但没有复制许可不明确的该仓库实现，也不依赖LeRobot的数据格式。本项目按现有DP动作约定实现独立采集器；DP RealEnv的软件时间、分源记录思路作为工程参考。

作者epoch250权重来自作者托管的Lift-PH仿真实验，并列满分而非唯一最佳或真机权重；默认本地epoch3050来自已有Lift仿真实验。两者身份、大小和SHA256见 `weights/manifest.json`。没有把已有作者或贡献者署名替换为本项目署名。
