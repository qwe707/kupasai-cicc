# YOLO 与 MLLM 在线协同推理方案

> 适用范围：CICC2026 建交赛道，城市轨道交通保护区判图智能体的在线推理链路。
> 本文只描述小模型和大模型在运行时如何交互，不包含数据集处理、标注、YOLO 训练、Active Learning 或伪标签流程。

---

## 1. 目标约束

核心目标：

- 提高综合检出率。
- 提高最终判定准确率。
- 在 `100 张图 <= 60s` 的约束下尽量压缩总耗时。

基本原则：

- YOLO 负责全量快速检测。
- MLLM 只处理必须复核或值得复核的样本。
- YOLO 空检不等于安全，必须经过轻量异常筛查。
- 在线链路按比赛指标优化，不以业务风险等级作为主要分流依据。

推荐总体策略：

```text
YOLO 全量检测
+ 复核优先级分流
+ 空检异常兜底
+ MLLM 精查
+ 固定比例质检
```

---

## 2. 总体流程

```text
输入 100 张图像
  ↓
Step 1. 图像预处理
  ↓
Step 2. YOLO batch 全量推理
  ↓
Step 3. 按 YOLO 输出分两路
        A. 有检测框
        B. 空检测
  ↓
Step 4A. 有框样本复核优先级判断
Step 4B. 空检样本异常筛查
  ↓
Step 5. 构造 MLLM evidence packet
  ↓
Step 6. MLLM 复核
  ↓
Step 7. 结果融合与最终判定
  ↓
Step 8. 输出检测结果 / 告警 / 质检样本
```

---

## 3. Step 1：图像预处理

对每张图生成基础信息：

```text
image_id
原图尺寸
图像质量分数：清晰度、亮度、遮挡、雨雾/夜间
是否存在保护区 ROI / 红线掩码
可选：场景来源，例如无人机、固定摄像头、巡检车
```

图像质量分数用于估计模型失误概率：

```text
模糊、夜间、强反光、小目标、远景
→ YOLO 和 MLLM 都更容易出错
→ 提高复核优先级
```

---

## 4. Step 2：YOLO 全量 Batch 推理

YOLO 对输入图像全量推理。

推荐策略：

```text
模型：优先 YOLOv11m / YOLOv11l
输入尺寸：960 或 1280，精度优先
conf 阈值：偏低，例如 0.15 ~ 0.30
NMS IoU：正常偏高，避免过早删框
推理方式：batch 推理，不逐张串行处理
```

YOLO 输出示例：

```json
{
  "image_id": "img_001",
  "detections": [
    {
      "box_id": 0,
      "class": "construction_vehicle",
      "conf": 0.62,
      "bbox_xyxy": [120, 330, 480, 720]
    }
  ]
}
```

YOLO 的在线职责不是直接最终判定违规，而是：

```text
尽量把疑似目标捞出来。
宁可多报，不要漏报。
```

---

## 5. Step 3：按 YOLO 输出分流

YOLO 输出后分为两路：

```text
A. 有检测框样本
B. 空检测样本
```

关键纪律：

```text
空检测 != 安全
空检测只表示 YOLO 没找到目标
```

因此，空检测样本不能直接放行，必须进入轻量异常筛查。

---

## 6. Step 4A：有框样本复核优先级判断

在线推理中不使用“高风险 / 低风险”作为主要分流名称，而使用：

```text
review_priority：复核优先级
```

复核优先级由以下因素共同决定：

```text
1. YOLO 置信度
2. bbox 与 ROI / 红线的空间关系
3. 类别是否属于比赛关注类
4. 图像质量
5. 目标尺寸与遮挡情况
6. 多目标复杂度
7. 历史弱类表现
```

建议分为四档：

```text
P0：必须复核
P1：优先复核
P2：放行 + 抽检
P3：直接放行
```

### 6.1 P0：必须复核

满足任一条件即进入 P0：

```text
YOLO conf < 0.40
bbox 与保护区 ROI 有交集
bbox 距离 ROI 边界很近
图像质量差：模糊、夜间、雨雾、强遮挡
检测到小目标或严重遮挡目标
同图多目标聚集，场景复杂
YOLO 类别不稳定，多个类别置信度接近
```

动作：

```text
进入 MLLM 精查
```

### 6.2 P1：优先复核

满足任一条件可进入 P1：

```text
YOLO conf 在 0.40 ~ 0.60
目标类别属于比赛重点关注类
目标尺寸过小，或者只检测到局部
当前类别是历史漏检 / 误检高发类
```

动作：

```text
时间预算充足 → MLLM 精查
时间预算紧张 → 进入抽检 / 延迟复核队列
```

### 6.3 P2：放行 + 抽检

同时满足以下条件可进入 P2：

```text
YOLO conf >= 0.60
目标清楚
bbox 远离 ROI
图像质量正常
场景复杂度低
没有异常筛查信号
```

动作：

```text
自动放行
进入固定比例抽检池，例如 5% ~ 10%
```

### 6.4 P3：直接放行

同时满足以下条件可进入 P3：

```text
YOLO conf >= 0.80
目标类别稳定
bbox 明显远离 ROI
图像质量好
无异常信号
```

动作：

```text
直接放行
只做极低比例质量抽检，例如 1% ~ 3%
```

---

## 7. Step 4B：空检样本异常筛查

YOLO 空检后，不能直接放行。需要运行独立于 YOLO 的轻量兜底信号。

可选信号：

```text
1. CLIP 风险语义相似度
2. 场景分类器 suspicious score
3. 与标准状态图的变化检测
4. 图像质量检测
5. 连续帧 / 同一路线历史一致性
```

最低配建议：

```text
CLIP 风险语义相似度 + 固定比例抽检
```

空检样本分为三档：

```text
E0：空检异常
E1：空检不确定
E2：空检正常
```

### 7.1 E0：空检异常

满足任一条件进入 E0：

```text
CLIP 对“施工车辆、吊车、堆土、人员侵入、障碍物”等风险 prompt 相似度高
场景分类器 suspicious score 高
当前图和标准状态图差异明显
图像质量差，YOLO 可能漏检
连续多帧突然从有目标变成无目标
```

动作：

```text
进入 MLLM 精查
必要时触发 GroundingDINO 兜底定位
```

### 7.2 E1：空检不确定

满足以下情况可进入 E1：

```text
异常分数中等
图像质量一般
场景桶属于历史易漏检场景
```

动作：

```text
进入抽检池
时间预算充足时送 MLLM
```

### 7.3 E2：空检正常

同时满足以下条件可进入 E2：

```text
CLIP / 场景分类器均判断正常
图像质量好
与历史正常图差异小
无异常信号
```

动作：

```text
放行
极低比例抽检
```

---

## 8. Step 5：构造 MLLM Evidence Packet

进入 MLLM 的样本不要只给一句话，也不要只给 crop。应构造结构化证据包。

每个样本包含：

```text
1. 原图
2. bbox crop
3. expanded crop：bbox 周围扩大 10% ~ 30%
4. bbox 坐标
5. YOLO 类别和置信度
6. ROI / 红线信息
7. 触发复核原因
8. 可选：检索到的规则摘要
```

示例：

```json
{
  "image_id": "img_001",
  "review_reason": [
    "low_confidence",
    "roi_overlap"
  ],
  "detections": [
    {
      "box_id": 0,
      "class": "construction_vehicle",
      "conf": 0.37,
      "bbox_xyxy": [120, 330, 480, 720],
      "roi_overlap": 0.24,
      "crop": "crop_001_0.jpg",
      "expanded_crop": "crop_001_0_expanded.jpg"
    }
  ],
  "image_quality": {
    "blur": "medium",
    "brightness": "normal"
  }
}
```

---

## 9. Step 6：MLLM 精查

MLLM 的任务不是重新做全量目标检测，而是回答：

```text
YOLO 提供的候选目标是否真实存在？
是否与保护区 ROI / 红线相关？
是否构成疑似违规？
有没有明显漏掉的关键目标？
最终是否需要告警？
```

输出必须固定 JSON，避免长文本生成拖慢速度并降低稳定性。

有框样本输出示例：

```json
{
  "image_id": "img_001",
  "is_violation": true,
  "violation_type": "construction_vehicle_intrusion",
  "confidence": "high",
  "evidence_box_ids": [0],
  "missed_objects": [],
  "need_human_review": false,
  "reason": "检测框内疑似工程车辆，且与保护区区域有交集。"
}
```

空检异常样本输出示例：

```json
{
  "image_id": "img_023",
  "is_violation": false,
  "has_suspicious_object": true,
  "suggested_objects": [
    {
      "class": "crane_or_boom",
      "rough_location": "upper-right"
    }
  ],
  "need_grounding_dino": true,
  "need_human_review": true
}
```

---

## 10. Step 7：结果融合

融合规则应简单明确：

```text
1. YOLO 高置信 + MLLM 确认
   → 输出违规 / 告警

2. YOLO 检出 + MLLM 否定
   → 作为误检剔除

3. YOLO 低置信 + MLLM 确认
   → 保留，提升为有效检测

4. YOLO 空检 + 异常筛查高 + MLLM 发现可疑目标
   → 进入 GroundingDINO 或人工复核

5. YOLO 与 MLLM 冲突，且影响最终判断
   → 标记人工复核，不直接放行
```

最终输出示例：

```json
{
  "image_id": "img_001",
  "final_decision": "violation",
  "violation_type": "construction_vehicle_intrusion",
  "level": "suspected",
  "evidence": [
    {
      "box_id": 0,
      "bbox_xyxy": [120, 330, 480, 720],
      "source": "YOLO+MLLM"
    }
  ],
  "need_human_review": false
}
```

---

## 11. Step 8：时间预算控制

`100 张 <= 60s` 约束下，不能让 MLLM 调用无限扩张。建议使用动态预算。

建议预算：

```text
YOLO batch 推理：5 ~ 15s
预处理 / crop / ROI 计算：2 ~ 5s
MLLM 复核预算：35 ~ 45s
结果融合：1 ~ 3s
```

MLLM 复核样本数建议控制：

```text
默认最多 20 ~ 40 张 / 100 张
超过预算时，按 P0 > E0 > P1 > E1 > P2 抽检排序
```

优先级队列：

```text
第一优先：低置信 + ROI 相关
第二优先：空检异常高
第三优先：图像质量差
第四优先：中置信关键目标
第五优先：低优先级抽检
```

---

## 12. 最终一句话

```text
YOLO 对 100 张图全量 batch 检测；有框样本根据置信度、ROI 关系、图像质量和目标复杂度计算复核优先级；空检样本不直接放行，而是经过 CLIP / 场景分类 / 变化检测等轻量异常筛查；高优先级样本构造原图 + crop + bbox + ROI + 触发原因的 evidence packet 送入 MLLM 精查；MLLM 只负责复核、补充语义判断和发现疑似漏检，最终通过规则融合输出告警、放行或人工复核结果。
```
