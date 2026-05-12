# 5.9 数据到账后的数据拆分、Baseline 与 YOLO-MLLM 对接方案

> 背景：官方数据集预计 5.9 才能拿到。5.9 前不应等待数据，而应先完成环境、baseline、接口和脚本骨架；5.9 后再按真实数据分布校准 schema、拆分数据集并迭代训练。

---

## 1. 总体判断

当前路线分成两条并行线：

```text
5.9 前：不依赖官方数据的工程线
  环境搭建 → YOLO 预训练推理 → MLLM 调用 → baseline 串联 → Prompt/JSON 固化 → 脚本骨架

5.9 后：依赖官方数据的数据线
  manifest → schema 探针 → 去重 → 分桶 → holdout 隔离 → 分层抽样 → 预标/校对 → YOLO 训练
```

核心原则：

1. **baseline 先跑通**：先证明“图像输入 → 检测 → 大模型判断 → 告警 JSON”链路可用。
2. **数据到手先探查**：不要直接标 10000 张，先确认类别、场景、质量和 schema 是否匹配。
3. **正式方案走双轨 C**：baseline 是第 0 版；最终架构是“YOLO/规则快轨 + MLLM 慢轨 + 仲裁器”。

---

## 2. 5.9 前：无官方数据阶段该做什么

这一阶段的目标不是训练最终模型，而是把所有不依赖数据的事情提前完成。

| 模块 | 要做什么 | 验收标准 |
| --- | --- | --- |
| Python / CUDA / PyTorch | 环境可运行，GPU 可见 | `torch.cuda.is_available()` 为 true；能跑一次 YOLO 推理 |
| YOLO baseline | 用预训练 YOLOv8/v11 跑样例图 | 输出 bbox、class、confidence |
| MLLM baseline | 本地 Qwen2.5-VL / 官方 QWEN3-VL / 豆包 Vision Pro 至少接通一个 | 输入图片 + 文本后输出固定 JSON |
| 串联 baseline | YOLO 输出转成 prompt 传给 MLLM | 单图输出告警 JSON |
| Prompt 与 JSON schema | 固定违规判定字段 | 输出能被 Pydantic/jsonschema 校验 |
| 数据脚本骨架 | manifest、去重、分桶、抽样、预标导出 | 空数据或样例数据能跑通流程 |
| Demo 骨架 | Streamlit / Gradio 简单页面 | 上传图片后展示 bbox + JSON |

5.9 前不建议做：

| 不建议 | 原因 |
| --- | --- |
| 自训 YOLO | 没有官方数据，训练结果没有代表性 |
| 定死 12 类 schema | 官方数据类别和分布未验证，容易返工 |
| 复杂多 Agent | baseline 还没跑通时过度设计 |
| 全量工程化部署 | 初赛先要能讲清、能演示、可验证 |

---

## 3. 5.9 数据到账当天：先做数据体检

拿到官方 10000 张数据后，第一天不直接标注，先做 5 件事。

### 3.1 原始数据只读归档

```text
data/
  raw/
    official/
      2026-05-09/
        images/              # 官方原图，只读保存
        source_manifest.*     # 如果官方自带索引，原样保存
```

要求：

- 不手动改名，不覆盖原图。
- 所有派生数据放到 `data/interim/`、`data/labels/`、`data/splits/`。
- 原始数据默认不进 GitHub，只记录处理脚本和文档。

### 3.2 生成 manifest

`manifest.csv` 至少包含：

| 字段 | 说明 |
| --- | --- |
| image_id | 内部唯一 ID |
| path | 原图路径 |
| width / height | 图像尺寸 |
| file_md5 | 文件级去重 |
| phash | 近似去重 |
| source | official/self/public/synthetic |
| bucket_id | 后续 CLIP 分桶结果 |
| split | holdout/train/val/internal_test |
| label_status | unlabeled/prelabeled/verified/gold |

### 3.3 100 张目视探针

随机抽 100 张，三人分担看，记录：

| 图ID | 场景 | 出现目标 | 新发现类别 | 是否疑似违规 | 数据质量 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| xxx | 施工围挡 | person, fence | helmet | 否 | 清晰 | 光照正常 |

目的：

- 验证 schema 候选类是否覆盖真实数据。
- 发现长尾类，如安全帽、反光衣、警示牌、吊臂等。
- 判断是否有保护区边界或需要人工配置红线。

### 3.4 MLLM caption + CLIP 统计

用 MLLM 给全量图片生成一句话 caption，再用文本频次和 CLIP 零样本分类估算分布。

输出：

```text
类别候选统计：
person: 约 N 张
vehicle: 约 N 张
construction_vehicle: 约 N 张
fence/barrier: 约 N 张
crane_arm: 约 N 张
obstacle/pile: 约 N 张
helmet/vest/sign: 约 N 张
```

决策：

- 数量极少且与赛题弱相关的类：暂时合并或弃用。
- 数量少但赛题强相关的类：保留，后续重点抽样或合成补充。
- 高频但弱区分类：作为上下文类，不一定作为 YOLO 主训练类。

### 3.5 冻结 schema v1

第一版 schema 不追求完美，但要能指导标注和训练。

建议初赛只保留 6-8 个核心可检目标：

```yaml
classes:
  - person
  - vehicle
  - construction_vehicle
  - crane_arm
  - fence_or_barrier
  - obstacle_or_pile
  - warning_sign
  - other_risk_object

attributes:
  violation_type: [越界, 吊装违规, 占道, 堆物, 无]
  in_protection_zone: bool
  visibility: [clear, occluded, blurry]
```

---

## 4. 数据集拆分方案

### 4.1 拆分顺序

正确顺序：

```text
raw 原始数据
  ↓
manifest + md5/phash
  ↓
pHash 去重（避免 train/test 泄漏）
  ↓
CLIP 场景分桶
  ↓
抽 gold holdout（人工精标，永不训练）
  ↓
剩余样本按桶分层拆 train/val/internal_test
  ↓
train 部分进入预标/校对/主动学习
```

不要先拆分再去重，否则相似帧可能同时出现在训练集和测试集，评估分数会虚高。

### 4.2 推荐目录

```text
data/
  raw/
    official/2026-05-09/          # 官方原始数据，只读
  interim/
    manifest.csv
    dedup_log.json
    buckets.csv
  splits/
    gold_holdout.txt              # 100-300 张，最终评测
    train.txt
    val.txt
    internal_test.txt
    active_learning_pool.txt
    pseudo_label_pool.txt
  labels/
    prelabeled/                   # MLLM/GroundingDINO 预标
    verified/                     # 人工校对
    gold/                         # holdout 精标
  reports/
    data_probe_report.md
    split_report.md
```

### 4.3 gold holdout

用途：最终评测和方案说明书硬证据。

建议：

- 数量：**100-300 张**，如果时间紧至少 100 张。
- 来源：去重和分桶之后抽取，覆盖主要场景桶和违规类型。
- 标签：必须人工精标，不能用伪标签。
- 纪律：永远不参与训练、预标回流、主动学习选样。

### 4.4 train / val / internal_test

从剩余样本中按场景桶和违规类别分层拆分：

| Split | 比例 | 用途 |
| --- | --- | --- |
| train | 70% | 训练 YOLO / 伪标签迭代 |
| val | 15% | 调参、早停、模型选择 |
| internal_test | 15% | 阶段性内部评估，不能频繁看 |

分层维度：

1. 场景桶：地铁口、施工围挡、吊装、堆物、夜间/雨天、其他。
2. 违规类型：越界、吊装违规、占道、堆物、无。
3. 质量：清晰、遮挡、模糊、夜间。

### 4.5 预标与校对样本池

初始不需要校对全部 10000 张：

```text
去重后样本
  ↓
按桶分层抽样约 2000-3000 张
  ↓
MLLM + GroundingDINO 预标
  ↓
人工校对
  ↓
先精标 200-300 张种子集训练 YOLO v1
  ↓
Active Learning 每轮选 300-500 张不确定样本继续校对
```

---

## 5. Baseline 方案

### 5.1 Baseline 是什么

Baseline 是最小可运行基准版，不是最终方案。

```text
图片
  ↓
预训练 YOLOv8/v11
  ↓
检测结果转 JSON / 文本
  ↓
MLLM 判断是否违规
  ↓
输出告警 JSON
```

Baseline 的价值：

1. 验证环境：YOLO、MLLM、API、GPU 是否能跑。
2. 验证接口：YOLO 输出能否稳定传给 MLLM。
3. 提供对照：后续双轨方案 C 要证明比 baseline 更快、更准、更稳。

### 5.2 Baseline 输入输出

YOLO 输出：

```json
{
  "image_id": "demo_001",
  "detections": [
    {"class": "person", "bbox": [120, 80, 200, 300], "conf": 0.91},
    {"class": "truck", "bbox": [330, 180, 600, 420], "conf": 0.87}
  ]
}
```

MLLM 输出：

```json
{
  "is_violation": true,
  "violation_type": "疑似越界",
  "risk_level": "中",
  "confidence": 0.72,
  "reason": "图中人员和车辆靠近疑似保护区围挡，存在违规进入风险。",
  "evidence": ["person", "truck", "fence"]
}
```

### 5.3 Baseline 验收标准

| 项目 | 标准 |
| --- | --- |
| 单图链路 | 输入 1 张图能输出 JSON |
| 批量链路 | 输入 100 张样例图能生成 jsonl |
| 稳定性 | MLLM 输出能通过 schema 校验 |
| 可解释 | JSON 中有 reason 和 evidence |
| 可对比 | 记录耗时、进入 MLLM 次数、失败样本 |

---

## 6. YOLO 与大模型对接方案

### 6.1 v0：串联 baseline

最简单版本：

```text
image → yolo_detect(image) → render_prompt(detections) → mllm_judge(image, prompt) → json
```

Prompt 模板：

```text
你是城市轨道交通保护区安全管控助手。
请结合图像和 YOLO 检测结果判断是否存在违规风险。

YOLO 检测结果：
{detections_json}

输出严格 JSON：
{
  "is_violation": bool,
  "violation_type": "越界|吊装违规|占道|堆物|无",
  "risk_level": "高|中|低",
  "confidence": 0-1,
  "reason": "不超过100字",
  "evidence": ["目标类别或区域"]
}
```

### 6.2 v1：加入几何关系

YOLO 不只把 bbox 给大模型，还先计算对象关系：

```text
person P1 inside protection_zone
truck V1 distance_to_zone = 2.3m
crane_arm C1 overlap_with_zone = 0.18
```

再交给 MLLM 判断。这样可以避免让大模型凭肉眼猜距离。

### 6.3 v2：升级成双轨方案 C

正式方案：

```text
图片
  ↓
YOLO 前置检测
  ↓
几何引擎 + 规则引擎
  ├─ 高置信规则命中 → 快轨直接告警
  ├─ 明确无风险 → 直接放行
  └─ 低置信 / 复杂场景 → 慢轨 MLLM 复核
        ↓
     仲裁器融合规则结果和 MLLM 结果
        ↓
     告警 JSON / 人工复核 / 样本回流
```

### 6.4 仲裁逻辑

| 情况 | 处理 |
| --- | --- |
| YOLO 高置信 + 规则明确命中 | 快轨直接告警，不调用 MLLM |
| YOLO 无目标且无异常 | 放行 |
| YOLO 中低置信 / 遮挡 / 边界不清 | 进入 MLLM 慢轨 |
| 规则与 MLLM 一致 | 输出高置信结果 |
| 规则与 MLLM 冲突 | 输出中风险 + 标记人工复核 |

### 6.5 Baseline 与方案 C 的区别

| 维度 | Baseline | 方案 C |
| --- | --- | --- |
| 目标 | 先跑通 | 正式比赛架构 |
| 链路 | YOLO → MLLM 串联 | 快轨 + 慢轨 + 仲裁 |
| MLLM 调用 | 每张图都可能调用 | 只处理复杂/不确定样本 |
| 规则引擎 | 无或很弱 | 核心模块 |
| 性能 | 慢 | 快 |
| 可解释 | 主要靠 MLLM 文本 | 规则证据 + MLLM 解释 |
| 用途 | 对照组 | 主方案 |

---

## 7. 需要沉淀的结果文件

```text
outputs/
  baseline/
    baseline_predictions.jsonl
    baseline_latency.csv
    baseline_error_cases/

reports/
  data_probe_report.md
  split_report.md
  baseline_report.md
  yolo_mllm_integration_report.md
```

其中 `baseline_report.md` 至少包含：

1. 用了什么 YOLO 权重。
2. 用了什么 MLLM。
3. 测试了多少张图。
4. 平均耗时、P95 耗时。
5. JSON 解析成功率。
6. 典型成功样例和失败样例。

---

## 8. 当前结论

5.9 前的重点不是训练，而是把工程底座搭好：

```text
环境能跑 → baseline 能串 → 输出格式固定 → 数据脚本准备好 → 5.9 后快速吃数据
```

5.9 后的重点不是全量人工标注，而是：

```text
先体检数据 → 冻结 schema → 去重分桶 → 隔离 holdout → 分层抽样 → 预标校对 → 主动学习
```

最终对外讲法：

> 我们先用 baseline 证明 YOLO 与多模态大模型能完成端到端判图，再通过双轨方案 C 将简单样本交给 YOLO/规则快轨、复杂样本交给 MLLM 慢轨，实现性能、准确率和可解释性的平衡。
