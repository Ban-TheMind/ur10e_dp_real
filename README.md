# UR10e DP Real

面向 UR10e、Robotiq 2F-85 和双 RealSense 的 Lift 实验项目：键盘采集真实示教、检查数据、从已有 RGB Diffusion Policy 权重微调，以及运行策略并记录真实执行。代码与模型可以独立于原仿真项目运行。

首版保留原始 **Transformer RGB DP**：双84×84图像、末端位置/四元数、Panda兼容双指观测，输出10维绝对动作（位置3＋rotation6d 6＋夹爪1）。本地epoch3050为默认权重；作者epoch250为参考，两者都是仿真Lift-PH权重。权重见 [Release](https://github.com/Ban-TheMind/ur10e_dp_real/releases/tag/weights-v1) 与 [清单](weights/manifest.json)。

## 安装

在仓库根目录运行。采集电脑与训练电脑可以分别安装，无需 ROS，也不会改动旧项目环境。

```bash
# 机器人旁电脑：Python3.8以上，已有图形桌面/X11
python3 -m venv .venv_robot
.venv_robot/bin/python -m pip install -e '.[hardware]'
cmake -S gripper_sdk -B gripper_sdk/build -DBUILD_TESTING=ON
cmake --build gripper_sdk/build -j4

# 训练/推理电脑：推荐 Python3.10，先按本机CUDA安装匹配的torch/torchvision
python3.10 -m venv .venv_train
.venv_train/bin/python -m pip install -e '.[train]'
```

同一台GPU电脑直接连接机器人时，在同一独立环境安装 `.[hardware,train]`。首版推理与设备读控在同一台电脑运行，不包含跨电脑网络推理服务。Foxy原环境无需改变；新程序通过ur_rtde直接控制，不能同时运行另一条发运动命令的ROS控制链路。

夹爪桥接用官方SDK与固定版本libserialport源码离线构建；需要C/C++17编译器、make和CMake≥3.16。构建机没有CMake时可在独立环境 `pip install cmake==3.27.9`，用该环境的cmake。训练依赖参考已验证的Python3.10、torch2.5.1、torchvision0.20.1、diffusers0.11.1、robomimic0.2.0；完整本地版本见 [验证记录](docs/validation.md)。

## 权重

两份原始检查点作为 `weights-v1` Release 附件保存，不进入普通Git历史。已登录目标私有仓库的GitHub CLI可下载：

```bash
python tools/fetch_weights.py
# 只下载公开作者参考权重，不需要访问私有Release：
python tools/fetch_weights.py --author-only
```

也可在Release网页下载至 `weights/`。脚本检查SHA256；原始文件含模型、EMA及原优化器状态，微调只选择模型权重初始化新训练，不沿用原优化器。只加载本项目或作者可信来源的pickle权重。

## 配置现场

```bash
cp configs/site.example.json configs/site.local.json
ur-dp cameras
ur-dp observe --site configs/site.local.json
```

填写机器人IP、夹爪串口和两台相机序列号。固定 `agentview_image` 为外部相机、`robot0_eye_in_hand_image` 为腕部相机，不要交换。相机原始RGB按640×480保存，训练时统一缩放84×84。速度、转速、工作范围及超时都在配置中，可以按实际布置调整。

`policy_frame.base_T_policy` 描述策略坐标系在UR基座中的位姿；`tcp_T_policy_tool` 描述策略使用的末端参考点相对当前UR TCP的位姿。示例均为单位矩阵，表示暂按同一坐标/同一工具点解释，不表示已经对齐了仿真和现场。它们是动作/状态坐标适配，不是相机手眼外参。见 [数据与坐标约定](docs/data_contract.md)。无需填写“已标定”或“已实测”开关。

`observe` 只读取设备，不创建运动控制接口。需要夹爪激活时单独运行 `ur-dp gripper-activate --site configs/site.local.json`；正常采集和执行不重复激活、不自动回零。

## 键盘采集

```bash
ur-dp collect --site configs/site.local.json --output data/lift
ur-dp check data/lift
```

聚焦双相机窗口：W/S沿基座X正/负，A/D沿Y正/负，Q/E沿Z正/负；O张爪，P闭爪。默认固定朝向，R切换旋转模式，I/K、J/L、U/H分别调整基座系三个旋转轴。松键停止位移目标，窗口失去焦点不继续解释运动按键。

C开始一条演示，Enter保存并标为成功，Backspace保存并标为失败；失败原始数据保留但默认不用于监督训练。Esc结束程序并停止机械臂，尚未结束的演示保留为不完整记录。保存图像在后台进行，结束一条演示不阻塞下一条的机器人控制。

首轮保持相机固定，完成接近—下降—闭爪—抬升；不将回到初态的过程录进同一条成功演示。先录几条检查画面和轨迹。图像、真实状态、目标指令分别保留；读数和命令不混写。

## 微调

```bash
ur-dp finetune --checkpoint weights/local_lift_3050.ckpt \
  --data data/lift --output outputs/lift_ft --settings configs/finetune.json

# 同一数据、同一配置恢复，包含优化器、学习率、EMA、采样与随机数状态
ur-dp finetune --checkpoint weights/local_lift_3050.ckpt \
  --data data/lift --output outputs/lift_ft --settings configs/finetune.json \
  --resume outputs/lift_ft/update_001500.pt
```

默认动作网络LR1e-5、视觉LR1e-6，全量微调，100更新warmup后保持固定LR，总上限5000次参数更新；保存0/500/1500/3000/5000节点。这些是起始配置，不是收敛经验定值。可修改 `max_updates`、`batch_size`、学习率、冻结编码器和cosine等设置；精确恢复时配置必须保持一致。

只用完整、成功的demonstration；按完整episode划分80%训练/20%验证，再生成序列窗口。保留源权重归一化规则。日志同时记录真实参数更新次数、等效epoch、训练损失和EMA离线验证损失。小集拟合可将 `val_fraction` 设为0；正式适配应保留独立演示和新初态。若数据坐标、频率或内容改变，重新开始一个适配run。

## 直接部署与微调后评测

```bash
ur-dp run --site configs/site.local.json --checkpoint weights/local_lift_3050.ckpt \
  --output outputs/zero_shot --duration 30 --device cuda

ur-dp run --site configs/site.local.json --checkpoint outputs/lift_ft/update_003000.pt \
  --output outputs/finetuned --duration 30 --device cuda

ur-dp summarize outputs/finetuned
```

`run` 直接运行策略；Esc停止，Enter结束并记录成功，Backspace结束并记录失败。建议现场事先统一成功定义，例如物块离桌≥5cm并保持2秒，由操作者观察物块后标记，不能把空夹爪上升当作抓取成功。超时或Esc未标记的回合不计入成功率分母。

推理在后台进行，控制循环持续工作；每批动作预测完成后按20Hz执行，再重新观测规划，保留实际推理延迟。默认沿用源模型100去噪步；可以通过 `--inference-steps` 做明确的推理速度调整。固定朝向模式也作用于策略执行，原始策略动作与实际下发动作都保存。

程序没有点云避障。现有MoveIt点云验证属于另一条规划链路，不会自动约束这里的RTDE指令。当前配置使用速度/工作区限制、设备错误退出和正常停止。控制参数在 [现场配置](configs/site.example.json) 中集中管理。

## 检查与复现

```bash
python -m unittest discover -s tests -v
ur-dp model-check --checkpoint weights/local_lift_3050.ckpt --device cpu
ctest --test-dir gripper_sdk/build --output-on-failure
python tools/smoke_train.py
```

最后一项使用明确标注的合成数据，验证真实权重的梯度更新和断点恢复，不是抓取效果实验。[验证记录](docs/validation.md)说明已执行检查；源码来源和许可证见 [第三方说明](docs/third_party.md)。
