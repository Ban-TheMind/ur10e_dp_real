# 真实成功集：微调与从零训练

项目根目录为 `/home/zanglab/projects/ur10e_dp_real`，训练环境为 `.venv_train`。
该环境独立安装依赖，不继承系统包；Python解释器来自保留的`dp_preexp`，不能删除后者。
已验证版本清单保存在`outputs/finetune_preflight_20261107/requirements_train.lock.txt`。

## 数据

`data/lift_20261106`与`data/lift_20261107`保留原始采集记录。
`data/lift_success`仅链接23条完整成功轨迹；9条失败及1条不完整记录不参与训练。
`data/lift_success/split.json`固定18条训练、5条验证，分别4767与1184个20Hz窗口。
两组共用相同划分，验证轨迹始终不进入训练；不宣称另有独立离线测试集。

## 两组入口

以下命令必须在项目根目录运行。正式运行通过桌面终端执行并显示进度，日志同时保存在输出目录。

```bash
.venv_train/bin/python -u -m ur10e_dp_real.cli finetune \
  --checkpoint weights/local_lift_3050.ckpt \
  --data data/lift_success --output outputs/lift_finetune_20261107_run01 \
  --settings configs/lift_finetune.json

.venv_train/bin/python -u -m ur10e_dp_real.cli train \
  --policy-config configs/policy_rgb_transformer.json \
  --data data/lift_success --output outputs/lift_scratch_20261107_run01 \
  --settings configs/lift_scratch.json
```

微调使用源EMA、重新建立优化器/调度器/EMA并保留源归一化，5000更新。
从零训练仅用架构配置，视觉骨干与动作网络均随机初始化，不读取预训练参数。
归一化只拟合真实训练集，旋转与已归一化的夹爪动作保留几何表示。
batch32时每epoch149更新，500epoch共74500更新；每5epoch验证和保存。
两组初始化、归一化、预算均有差异，属于实用方案对照，不是仅改变初始化的消融。

保留阶段权重、`best_val.pt`及`best_val.json`、滚动`latest.pt`。
离线最佳依据完整验证集EMA去噪损失，不等于真机成功率最佳。
恢复时给对应命令增加`--resume <检查点>`，保持设置与数据不变。
GPU恢复核验模型/EMA误差在1e-7数值容差内，采样与Torch/CUDA随机数状态一致，不承诺跨GPU逐位相同。

## 可见终端与自动衔接

`tools/run_training_terminal.py finetune`与`scratch`各在一个桌面终端中运行。
第一终端执行微调，第二终端显示等待。自动检查每30分钟核对完成证据，触发第一终端运行
`tools/analyze_real_training.py`完整验证集评测并输出CSV、PNG、PDF。
评测成功后第一终端发布通过标记，第二终端开始500epoch从零训练；共用GPU锁，串行运行。
从零训练结束后按同样流程评测与导出图表。控制状态、日志及运行协议见
`outputs/real_lift_20261107/README.md`。自动化ID为`ur10e-dp`。

离线测试不会操作机器人。真机闭环测试应另外统一初态与成功定义。
保持电脑和Codex应用运行，保证定时检查能够触发阶段衔接。
