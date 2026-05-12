# CICC 2026 · 建交赛道参赛项目

本仓库用于参加 **CICC 2026 场景开放与行业应用挑战赛 —— 建交赛道**（城市建设与交通）。
比赛官网：<https://www.kupasai.com/cicc2026/scence>

> 飞书协作空间与本仓库一一对应：飞书放过程产物（讨论 / 决策 / 任务 / 多维表格），GitHub 放可追溯资产（代码 / 文档 / 提交包）。

## 1. 当前选定方向

- **赛道**：建交（城市建设与交通）
- **赛题**：⚠️ 待 `docs/00-overview/topic-selection.md` 内最终勾选
- **报名类别**：企业 / 高校 / 团队（待确认）
- **目标**：进决赛 + 拿名次（一/二/三等奖均争取）

## 2. 仓库结构

```
docs/                  方案、提交、研究文档
  00-overview/         比赛信息、选题、团队
  01-solution/         问题定义、技术方案、评估方案
  02-submission/       初赛 / 决赛 / 路演脚本
  03-product/          截图、架构图、演示故事板
  04-research/         调研笔记、对标分析、参考资料
src/                   代码
  app/                 前端 / 演示界面
  agents/              智能体（如标书智能体、问答智能体）
  pipelines/           数据处理 / 模型推理流水线
  services/            后端服务、API
  evaluation/          指标、评估脚本
data/                  数据
  sample/              示例数据
  processed/           预处理后数据（.gitignore 大文件）
  prompts/             Prompt 模板
notebooks/             探索 / 实验
scripts/               一次性脚本（数据下载、清洗等）
tests/                 单测
assets/                PPT、视频、海报
deliverables/          打包后的提交物
  registration/        报名材料
  initial-round/       初赛提交
  final-round/         决赛提交
.github/               Issue / PR 模板、CI
```

## 3. 关键里程碑

| 阶段 | 目标 | 主交付物 |
| --- | --- | --- |
| M0 选题 | 锁定赛题 | `topic-selection.md` |
| M1 报名 | 完成报名 | `deliverables/registration/` |
| M2 初赛 | 提交初赛包 | `deliverables/initial-round/` |
| M3 原型 | 跑通最小闭环 | `src/`、`screenshots/` |
| M4 决赛 | 上海现场路演 | `deliverables/final-round/` + 视频 + PPT |

> 比赛各节点的具体日期请以官方通知为准（前端 bundle 中未直接暴露）。

## 4. 提交物对照官方要求

### 初赛
- [ ] 项目方案说明书（项目背景、场景需求、技术方案、核心功能、创新点、价值分析）
- [ ] 技术方案展示材料（系统架构图、功能流程图、原型界面、核心实现路径）
- [ ] （可选）原型 / 功能演示（Demo 视频、截图、演示程序）

### 决赛
- [ ] 完整解决方案报告（总体架构、核心技术、数据与模型、部署方案、应用价值）
- [ ] 系统原型或应用产品（智能体应用、数据处理工具、行业 AI 应用系统）
- [ ] 3—5 分钟产品演示视频
- [ ] 项目路演 PPT（背景与问题、技术与创新、效果与价值、推广与落地）
- [ ] （可选）开源代码仓库 / API 文档 / 技术说明文档

## 5. 飞书协作空间

详见 `docs/00-overview/feishu-workspace.md`。

## 6. Git 协作规范

分支管理、Pull Request、rebase 与 `main` 保护规则见 `docs/00-overview/git-workflow.md`。

## 7. 合规与原创承诺

参赛项目须为团队原创，严禁抄袭与侵权；所涉及数据须遵守数据安全与合规要求。
