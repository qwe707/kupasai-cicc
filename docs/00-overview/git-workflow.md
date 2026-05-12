# Git 分支协作规范

> 适用仓库：`https://github.com/qwe707/kupasai-cicc`
> 目标：队员各自拉分支开发，提交 Pull Request，由负责人统一 review 和 merge 到 `main`。

---

## 1. 分支模型

| 分支 | 用途 | 规则 |
| --- | --- | --- |
| `main` | 稳定主线，只放确认后的代码和文档 | 不直接 push；通过 Pull Request 合并 |
| `feature/<name>` | 新功能 / 新模块 | 例如 `feature/yolo-baseline` |
| `docs/<name>` | 文档修改 | 例如 `docs/dataset-sop` |
| `fix/<name>` | bug 修复 | 例如 `fix/json-parser` |
| `experiment/<name>` | 探索性实验 | 不保证进入 `main` |

建议分支名只用英文、小写、短横线，避免空格和中文。

---

## 2. 队员日常流程

### 2.1 第一次拉仓库

```powershell
git clone https://github.com/qwe707/kupasai-cicc.git
cd kupasai-cicc
```

### 2.2 新建个人分支

```powershell
git checkout main
git pull origin main
git checkout -b feature/yolo-baseline
```

### 2.3 提交修改

```powershell
git status
git add .
git commit -m "feat: add yolo baseline"
git push -u origin feature/yolo-baseline
```

### 2.4 合并前 rebase 主线

在开 Pull Request 前，先把自己的分支基于最新 `main` 重新整理：

```powershell
git fetch origin
git checkout feature/yolo-baseline
git rebase origin/main
git push --force-with-lease
```

注意：

- rebase 的是自己的功能分支，不是 `main`。
- rebase 后需要 `--force-with-lease`，不要用 `--force`。
- 如果发生冲突，先解决冲突，再执行 `git rebase --continue`。

---

## 3. Pull Request 规则

1. 每个任务开一个 PR，不要把无关修改混在一起。
2. PR 标题写清楚做了什么，例如：`feat: add yolo baseline pipeline`。
3. PR 描述至少包含：
   - 改了什么
   - 怎么验证
   - 是否影响数据 / 模型 / 提交物
4. 负责人 review 后再 merge。
5. 合并方式优先用 **Squash and merge** 或 **Rebase and merge**，保持 `main` 历史清晰。

---

## 4. main 分支保护建议

在 GitHub 页面设置：

```text
Settings → Rules → Rulesets
```

点击：

```text
New ruleset → New branch ruleset
```

基础配置：

| 项 | 值 |
| --- | --- |
| Ruleset name | `protect-main` |
| Enforcement status | `Active` |
| Target branches | Include by pattern: `main` |

建议开启：

| 设置 | 建议 |
| --- | --- |
| Require a pull request before merging | 开启 |
| Require approvals | 至少 1 人 |
| Block force pushes | 开启 |
| Restrict deletions | 开启 |
| Require linear history | 建议开启 |
| Require status checks | 有 CI 后再开启 |

这样可以避免队员误 push 到 `main` 或误删主分支。

如果 GitHub 页面是旧版，也可以走：

```text
Settings → Branches → Add branch protection rule
```

`Branch name pattern` 填 `main`，然后开启同样的 PR、approval、linear history、block force pushes 和 deletion protection。

---

## 5. 添加协作者

仓库负责人在 GitHub 页面设置：

```text
Settings → Collaborators and teams → Add people
```

建议权限：

| 角色 | 权限 | 说明 |
| --- | --- | --- |
| 普通队员 | `Write` | 可以 push 自己的分支、开 PR |
| 只查看资料的人 | `Read` | 只能查看和 clone |
| 仓库负责人 | `Admin` | 管理设置、协作者、分支保护 |

不建议给普通队员 `Admin` 权限，避免误改仓库规则或删除分支保护。

邀请后，队友需要在 GitHub 通知或邮件里接受邀请，之后才能 push 分支。

---

## 6. 仓库初始化后检查清单

| 检查项 | 目标状态 |
| --- | --- |
| `main` 分支保护 | 已开启，不能直接 push |
| 协作者 | 队员已加入，普通队员为 `Write` 权限 |
| PR 合并方式 | 建议保留 `Squash and merge` / `Rebase and merge` |
| Force push | 对 `main` 禁止 |
| Branch deletion | 对 `main` 禁止 |
| Secrets / 数据 / 权重 | 不进 GitHub |

---

## 7. 不允许提交的内容

以下内容不进 GitHub：

| 类型 | 原因 |
| --- | --- |
| 原始数据、标注数据、监控截图 | 涉及隐私和数据安全 |
| 模型权重 `.pt/.onnx/.safetensors` | 文件大，且可能涉及训练资产 |
| `.env`、token、密钥 | 安全风险 |
| 飞书临时文件、提交包 zip、视频 | 大文件和过程产物 |

当前 `.gitignore` 已默认屏蔽这些内容。若确实需要共享大文件，优先放飞书或后续配置 Git LFS。

---

## 8. 推荐提交信息

| 类型 | 示例 |
| --- | --- |
| 新功能 | `feat: add yolo baseline pipeline` |
| 修复 | `fix: handle invalid mllm json output` |
| 文档 | `docs: update dataset split plan` |
| 实验 | `exp: try groundingdino prelabeling` |
| 配置 | `chore: update gitignore for model outputs` |

提交信息建议用英文，简短说明“做了什么”。

---

## 9. 负责人合并流程

负责人合并前检查：

```powershell
git fetch origin
git checkout main
git pull origin main
```

在 GitHub 页面 review PR：

1. 看文件是否只改了相关范围。
2. 看是否误提交数据、模型权重、密钥。
3. 看是否有验证说明。
4. 通过后 merge。

本地同步：

```powershell
git checkout main
git pull origin main
```

---

## 10. 冲突处理原则

1. 谁的分支产生冲突，谁先解决。
2. 不确定时不要强行覆盖别人文件。
3. 文档冲突优先保留双方内容再整理。
4. 代码冲突解决后至少跑一次对应脚本或测试。

