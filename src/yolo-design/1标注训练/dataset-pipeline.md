# 10000 张数据集处理 SOP

> 适用范围：CICC2026 建交赛道，10000 张原始图像数据集的端到端处理方案
> 目标：把 10000 张原图，通过"少标 + 智能标 + 自动标"的组合拳，**实际人工标注量压缩到 ~900 张**，节省 80% 工时

---

## 0. 核心原则

1. **80% 标注价值来自 20% 样本** —— 不追求"标完所有"，追求"标到关键的"
2. **Schema 迭代收敛** —— 不是定一次，是 5 步迭代到稳定
3. **MLLM 当自动标注员** —— 人工只做"二选一/微调"，速度 ×5-10
4. **Active Learning 主动学习** —— 让模型告诉你"哪张最该标"
5. **数据 vs 代码解耦** —— 数据流程独立推进，不阻塞架构开发

---

## 1. 完整流程总览

```
① Schema 设计 (1天，迭代收敛)
        ↓
② 全量去重 (半天，10000 → ~6000)
        ↓
③ CLIP 分桶 (半天，按场景聚类)
        ↓
④ 分层抽样 (1小时，6000 → ~2200)
        ↓
⑤ MLLM + GroundingDINO 自动预标 (1天)
        ↓
⑥ 人工校对 (3天，三人分担)
        ↓
⑦ 精标 200 张种子集 (半天)
        ↓
⑧ 训 YOLO v1 (半天)
        ↓
⑨ Active Learning 选 300 张 → 校对 → 训 v2
        ↓
⑩ 再 300 张 → 训 v3 (收敛)
        ↓
⑪ 用 v3 自动伪标剩余 → 训 v4
```

**总人工成本**：~30 工时（三人各 10 小时），相比全量标注 167 小时节省 **82%**

---

## 2. 阶段一：Schema 设计（迭代收敛 5 步法）

### 核心思想：Schema 不是定一次，是迭代到稳定

```
v0 (拍脑袋猜) → 探针1 → v1 (修正主类) → 探针2 → v2 (发现长尾)
   → 探针3 → v3 (冻结)
```

### Step 1: v0 Schema（30 分钟）
基于赛题 + 常识写草稿，**故意写宽**，把不确定的事明确列出来。

```yaml
# schema_v0.yaml
classes:
  candidates:
    - person, car, truck, construction_vehicle
    - crane_arm, fence, obstacle, traffic_cone
    - other_machinery
attributes:
  - in_protection_zone: bool
  - violation_type: [越界, 占道, 吊装违规, 堆物]
unsure:
  - 工人 vs 路人 怎么区分？
  - 保护区边界图里有没有标注？
  - 夜间/雨天数据多不多？
```

### Step 2: 100 张目视探针（1 小时，三人分担）
完全随机抽 100 张，每人看 33 张，记录：

| 图ID | 一句话描述 | 出现的类 | 新发现的类 | 场景 | 质量 |
|------|------------|----------|-----------|------|------|
| 001  | 地铁口工人搬护栏 | person, fence | helmet | 地铁口 | 清晰 |

### Step 3: MLLM 全量打 caption（2 小时机器跑，0 人工）

```python
def generate_caption(image):
    return mllm.chat(
        image=image,
        prompt="""用一句中文话描述这张图，包含：
        1. 场景类型 2. 主要目标 3. 是否有违规迹象。50字内"""
    )
```

拿到 10000 条文字描述后，**用文本工具分析**：
```python
words = []
for cap in captions:
    words.extend(jieba.cut(cap))
print(Counter(words).most_common(50))
# [('工人', 3200), ('围挡', 2800), ('吊车', 180), ...]
```

### Step 4: CLIP 零样本分布验证（1 小时）

```python
candidate_classes = ["person", "car", "truck", "crane_arm", "fence", "obstacle", "helmet"]
text_features = clip.encode_text(candidate_classes)
img_features = clip.encode_images(all_images)
sims = img_features @ text_features.T

for i, cls in enumerate(candidate_classes):
    n = (sims[:, i] > 0.25).sum()
    print(f"{cls}: {n} 张")
```

**决策点**：
- 数量 < 100 的类 → 砍掉或合并
- 新发现且数量 > 500 的类 → 加入 schema
- 数量 200-500 的稀缺类 → 保留但需重点采样 / 合成补充

### Step 5: v3 冻结 + 50 张精标验证

```yaml
# schema_v3.yaml
version: 3.0
classes:
  - id: 0, name: person,                估计数量: 9200
  - id: 1, name: car,                   估计数量: 3500
  - id: 2, name: construction_vehicle,  估计数量: 1400
  - id: 3, name: crane_arm,             估计数量: 150  # 长尾
  - id: 4, name: fence,                 估计数量: 4500
  - id: 5, name: obstacle,              估计数量: 1200
  - id: 6, name: helmet,                估计数量: 2800  # 新增
弃用类: [traffic_cone (<50), sign (相关性弱)]
attributes:
  in_protection_zone: bool
  violation_type: [越界, 占道, 吊装违规, 堆物, 无]
```

**验证**：三人按 v3 各自标同 50 张 → 一致率 > 90% 才能进入下阶段，< 80% 回去改 schema

---

## 3. 阶段二：去重瘦身

### 三种去重技术效果对比

| 技术 | 检出范围 | 典型命中率 | 推荐 |
|------|---------|-----------|------|
| MD5/SHA | 完全相同文件 | 1-5% | 必做但收益小 |
| **pHash/dHash** | 像素级近似 | 视频抽帧 60-85% / 多源采集 15-25% | **ROI 最高** |
| CLIP 语义相似 | 场景级相似 | 10-50% | 谨慎用，会误删多样性 |

### 实操步骤

```python
# Step 1: 探针（30 分钟先看效果再决定）
python dedup_probe.py --sample 1000

# Step 2: pHash 去重（保守阈值 5）
import imagehash
hashes = {}
for img_path in all_images:
    h = imagehash.phash(Image.open(img_path), hash_size=8)
    if any(abs(h - existing) < 5 for existing in hashes):
        continue
    hashes[h] = img_path
```

### pHash 距离阈值选择

| 距离 | 含义 | 推荐场景 |
|------|------|---------|
| 1-5 | 极相似（连续视频帧） | 安全保守起步 |
| 5-10 | 相似（同场景不同时刻） | 激进点用 |
| > 15 | 不同 | - |

### 4 个常见踩坑

| 坑 | 对策 |
|---|------|
| 把"违规对照组"删了（人正常 vs 越界两帧很像） | 阈值取严 ≤5；视频强制每 N 秒留 1 帧 |
| CLIP 把不同违规类型聚一起 | CLIP 只用来分桶，不用来砍数量 |
| 不同摄像头同场景被当重复 | 保留摄像头 ID，跨摄像头不去重 |
| 破坏数据集元信息 | 建 dedup_log.json 记录"删了哪张、相似哪张" |

### 预期收益

| 数据来源 | 重复率 | 去重后剩余 |
|----------|--------|-----------|
| 视频抽帧 | 70-90% | 1000-3000 |
| 多源采集（最常见） | 20-40% | 6000-8000 |
| 网络爬取 | 30-50% | 5000-7000 |
| 已精选 | 5-15% | 8500-9500 |

---

## 4. 阶段三：CLIP 分桶（按场景聚类）

### 关键：分桶必须在预标之前，不是之后

**为什么**：
- 预标完才发现数据严重不均衡 → 已浪费 MLLM 算力 + 校对工时
- 分桶在前，能决定"标什么"，避免冗余

### 实操

```python
features = clip_model.encode_images(images)  # 6000×512
clusters = KMeans(n_clusters=20).fit(features)

# 输出每个簇的样本数 + 代表图
for cid in range(20):
    samples = [imgs[i] for i in range(len(imgs)) if clusters.labels_[i] == cid]
    print(f"簇{cid}: {len(samples)} 张, 代表: {samples[:3]}")
```

**典型输出**：
```
桶1: 地铁口正常巡检 (4000)
桶2: 施工围挡 (800)
桶3: 吊装作业 (200)        ← 长尾！必须重点采样
桶4: 堆物占道 (500)
桶5: 夜间/雨天 (300)
桶6: 异常其他 (200)
```

### 分桶 ≠ 分类

⚠️ 不是"把所有 person 放一起" —— 那是分类，不是分桶
✅ 分桶 = "按场景聚类"，一个桶里可能同时有 person+car+crane

---

## 5. 阶段四：分层抽样

### 按桶配额抽样（保多样性 + 砍冗余）

```python
quotas = {
    "桶1_地铁口": 600,    # 4000张里抽600，够用
    "桶2_施工": 500,      # 800张里抽500
    "桶3_吊装": 200,      # 全用，稀缺
    "桶4_堆物": 400,
    "桶5_夜间": 300,      # 全用
    "桶6_其他": 200,      # 全用
}
# 合计: 2200 张进入预标
```

**收益**：
- ✅ 稀缺类一张不浪费
- ✅ 高频类不过度冗余
- ✅ MLLM 预标算力省一半
- ✅ 训出来的模型自然均衡

---

## 6. 阶段五：MLLM 自动预标

### 6.1 主力：MLLM 直出 YOLO 格式

```python
def auto_prelabel(image_path):
    response = mllm.chat(
        image=image_path,
        prompt="""请检测图中所有目标，输出YOLO格式：
        类别: person, car, truck, crane_arm, fence, obstacle
        格式: class_id cx cy w h (归一化坐标)
        每行一个目标。"""
    )
    return parse_yolo_format(response)
```

### 6.2 进阶：GroundingDINO + SAM 精修 bbox

MLLM 输出的 bbox 偏移 10-20 像素，用 GroundingDINO 精修：

```python
# Step 1: MLLM 列出"图里有什么"
classes = mllm.list_classes(image)  # ["person", "crane_arm"]

# Step 2: GroundingDINO 精确定位
boxes = grounding_dino(image, text=" . ".join(classes))

# Step 3: SAM 精确 mask（如需要）
masks = sam.segment(image, boxes)
```

### 预标质量预期

| 场景 | 准确率 |
|------|--------|
| 简单场景（person/car） | 85-95% |
| 复杂场景（crane_arm/fence） | 60-75% |
| **整体可用率** | **~80%** |

---

## 7. 阶段六：人工校对（二选一模式）

### 不让标注员画框，只让看 → 接受/拒绝/微调

```
[展示原图 + MLLM 预标的 bbox]
  ✅ 全部正确  (~80% 场景)
  ✏️ 部分调整  (~15% 场景)
  ❌ 全部重标  (~5% 场景)
```

### 速度对比

| 方式 | 每张耗时 | 2200 张总耗时 |
|------|---------|--------------|
| 从零画框 | 60s | 37 小时 |
| MLLM 预标 + 人工校对 | **8-12s** | **6-7 小时** |
| MLLM 预标 + 接受/拒绝 | **3-5s** | **3-4 小时** |

**三人分担 → 每人 2-3 小时搞定**。

### 工具推荐

- **CVAT**（首选）：原生支持"导入预标 → 校对模式"，数据本地可控，适合敏感场景
- Label Studio：插件丰富，能集成 MLLM 在线预标
- Roboflow：自带 Smart Labeling，付费但极快

---

## 8. 阶段七-十：训练 + Active Learning 迭代

### 关键洞察：Active Learning 选 1000 张 ≈ 随机选 5000 张

### 完整迭代流程

```
W2 Day1:  人工精标 200 张种子集（每人 1 小时）
          ↓
          训 YOLO v1（mAP 假设 0.55）
          ↓
W2 Day3:  v1 在剩余 8000 张未标注上推理
          挑出"不确定样本"：
            - 置信度 0.4-0.7（模型纠结）
            - dropout 多次推理方差大的
            - 与已标注集 CLIP 距离最大的
          挑 300 张 → 校对（每人 1 小时）
          ↓
          训 YOLO v2（mAP 0.72）
          ↓
W3 Day1:  再选 300 张 → 校对 → 训 v3（mAP 0.80+）✅ 收敛
          ↓
W3 Day3:  用 v3 自动伪标剩余 → 高置信样本进训练集
          训 YOLO v4（mAP 0.85+）
```

### Active Learning 选样代码（10 行）

```python
def select_to_label(model, unlabeled_images, k=300):
    scores = []
    for img in unlabeled_images:
        preds = model.predict(img)
        # 不确定性 = 1 - max_confidence
        uncertainty = 1 - max([p.conf for p in preds], default=0)
        scores.append((img, uncertainty))
    return sorted(scores, key=lambda x: -x[1])[:k]
```

### 自训练 / 半监督（最后一波收割）

```python
# v3 模型推理剩余 2000 张
for img in unlabeled:
    preds = v3.predict(img)
    for box in preds:
        if box.conf > 0.85:        # 高置信 → 直接当真标签
            save_label(img, box)
        elif box.conf > 0.5:       # 中置信 → 进 active learning 池
            queue_for_review(img, box)
        # 低置信 → 丢弃
```

---

## 9. 时间投入总览

| 阶段 | 耗时 | 谁做 |
|------|------|------|
| Schema 5 步迭代 | 1 天 | 三人协作 |
| 去重 | 半天 | 1 人 |
| 分桶 + 分层抽样 | 半天 | 1 人 |
| MLLM 全量预标 | 1 天（机器） | 0 人工 |
| 人工校对 2200 张 | 6-7 小时 | 三人各 2 小时 |
| 精标 200 张种子 | 3 小时 | 三人各 1 小时 |
| 训练 v1 | 半天 | 1 人 |
| AL 选 300 → 校对 → 训 v2 | 1 天 | - |
| AL 选 300 → 校对 → 训 v3 | 1 天 | - |
| 自训练 → 训 v4 | 半天 | 1 人 |
| **合计** | **~7 工作日** | **每人纯标注 ≈ 10 小时** |

**对比全人工标 10000 张**：
- 全标：167 小时纯标注 + ~14 天周期
- 本方案：30 小时纯标注 + 7 天周期
- **节省 82% 工时 + 50% 周期**

---

## 10. 评测集隔离（关键纪律）

### 4 个高压线

| 雷区 | 后果 | 对策 |
|------|------|------|
| 评测集混入训练 | 看起来分数高，实际模型烂 | 评测集**必须从一开始就隔离** |
| 评测集用伪标签 | 自欺欺人 | **评测集 100 张必须人工精标** |
| 类别极度不均衡 | 少数类永远学不会 | AL 选样按类别配额 |
| 全部依赖 MLLM 预标 | 系统性偏差进训练集 | 至少 20% 样本人工精标 |

### 评测集准备

- 数量：**100 张**（精标，不入训练）
- 来源：从去重后的 6000 张里**最早**抽出，物理隔离到 `data/test_holdout/`
- 平衡：每个类至少 10-20 张
- 多样性：覆盖所有场景桶

---

## 11. 推荐技术栈速查

| 环节 | 工具 | 备注 |
|------|------|------|
| 去重 | imagehash + scikit-learn | Python 库 |
| 分桶 | CLIP ViT-L/14 + KMeans | 或 DINOv2 |
| MLLM 预标 | Qwen2.5-VL-7B (vllm) | 主力 |
| MLLM 备选 | 豆包 1.5 Vision Pro | 云端辅助 |
| bbox 精修 | GroundingDINO 1.6 | 零样本检测 |
| 像素级 mask | SAM 2.1 | 仅在需要时用 |
| 标注校对 | **CVAT**（推荐） | 数据本地可控 |
| Active Learning | 自写 50 行 | 不用框架 |
| 训练 | Ultralytics YOLOv11 | YOLOv11s 主力 |
| 数据增强 | Albumentations | 训练时无脑用 |
| 合成稀有场景 | SD3.5 + ControlNet | crane_arm 不足时补 |

---

## 12. 创新点提炼（写进说明书第 5 章）

本流程可提炼至少 3 个创新点：

1. **MLLM 驱动的弱监督数据闭环** —— 利用大模型零样本能力生成伪标签，配合 Active Learning 选择策略，将人工标注成本降低 82%

2. **基于 MLLM + CLIP 的零样本数据探查** —— Schema 与数据迭代收敛方法，避免传统"先想清楚再开干"的盲目性

3. **分层抽样 + 主动学习的高效标注范式** —— 用 1000 张达到传统 5000 张的训练效果

---

## 13. 关键文档与代码位置（待开发）

```
src/
├── data/
│   ├── dedup.py              # pHash + CLIP 去重
│   ├── bucket.py             # CLIP 聚类分桶
│   ├── sampler.py            # 分层抽样
│   ├── auto_label.py         # MLLM + GroundingDINO 预标
│   └── active_learner.py     # Active Learning 选样
├── train/
│   └── yolo_iter.py          # 迭代训练脚本
└── eval/
    └── holdout_eval.py       # 评测集独立测评

docs/01-solution/
├── dataset-pipeline.md       # 本文档
├── solution-architecture.md  # 整体架构
└── evaluation-plan.md        # 评测方案

data/
├── raw/                      # 10000 原图（gitignore）
├── deduped/                  # 去重后（gitignore）
├── prelabeled/               # MLLM 预标（gitignore）
├── verified/                 # 人工校对（gitignore）
├── train/ val/               # 训练集（gitignore）
└── test_holdout/             # 评测集 100 张（gitignore）
```

---

## 14. 三个反共识建议

1. **别全标 10000 张** —— 实际只需 ~900-1000 张精标 + 自动伪标的剩余样本
2. **别先分类再去重** —— 顺序倒了，去重在前是无监督的
3. **别把分桶放最后** —— 分桶必须在预标之前，决定"标什么"

---

**最后一句**：这套流程**本身就是创新点**，写进说明书第 5 章就是核心竞争力。10000 张数据不再是负担，而是用最少人工撬动最大模型效果的杠杆。
