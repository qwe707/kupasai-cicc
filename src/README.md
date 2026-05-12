# 源码目录说明

- `app/` 演示界面（建议 React / Vue / Streamlit 三选一）
- `agents/` MLLM 复核智能体（问答 / 冲突 / 合规初筛）
- `pipelines/` 标注数据集
- `services/` 后端服务（FastAPI / 系统服务封装）
- `evaluation/` 评估脚本与离线评测集
- `yolo-service/` YOLO 模块（训练、检测、评估、导出、检测分流）
- `yolo-design/` 设计文档
  - `1标注训练/` 数据集与训练流程
  - `2迁移部署相关/` 本机 → 5060Ti 迁移方案
  - `3接口相关/` API 重构设计
  - `4yolo-agent交互方案/` YOLO 与 Agent 对接方案
  - `拼接更新20260511/` 裁剪拼接 + Agent 评分完整设计（源码快照 + API 文档 + 流程图 + 方案迭代纪录）
- `yolo-service/` YOLO 模块(训练、检测、评估、导出、检测分流)

