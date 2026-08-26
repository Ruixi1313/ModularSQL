# ModularSQL Pipeline Steps Reference

DeepEye-SQL 的 7 个 stage + 我们的 Pattern 1 post-processing。

## DeepEye-SQL Pipeline (Steps 1-7)

执行入口：`script/run_pipeline.sh` 或 `script/run_stages_3_to_7.sh`（跳过 1-2）

### Step 1: Dataset Preprocessing
- **脚本**: `runner/preprocess_dataset.py`
- **做什么**: 读 BIRD `data/bird/dev/dev.json` + `database_description/*.csv` → 构建 `dataset/bird/dev.snapshot`
- **LLM 调用**: 0（纯本地处理）
- **耗时**: <30 秒
- **输出**: `workspace_*/dataset/bird/dev.snapshot` (每个 question 的完整 `database_schema` + 元数据)

### Step 2: Vector Database
- **脚本**: `runner/create_vector_db_parallel.py`
- **做什么**: 用 sentence-transformers (`all-MiniLM-L6-v2`) 对所有 DB 列值做 embedding，建本地向量索引
- **LLM 调用**: 0（本地 CPU embedding）
- **耗时**: 5-10 分钟（11 个 DB）
- **输出**: `workspace_*/vector_database/bird/dev/` (per-column FAISS-style 索引)
- **特点**: 跨 run 可复用（cell values 不变）

### Step 3: Value Retrieval
- **脚本**: `runner/run_value_retrieval.py`
- **做什么**: 用 LLM 从 question/evidence 抽 keywords → 在 vector DB 检索相关 cell values
- **LLM 调用**: 每题 1-2 次（keyword extraction）
- **耗时**: 99 题 ~3 min, 1534 题 ~15-30 min
- **输出**: `workspace_*/value_retrieval/bird/dev.snapshot` (+ retrieved values per question)

### Step 4: Schema Linking ⭐ 重头戏 1
- **脚本**: `runner/run_schema_linking.py`
- **做什么**: **3 个 linker 并行**判断哪些表/列跟 question 相关：
  - **Direct linker**: 让 LLM 直接列出相关表/列
  - **Reversed linker**: 让 LLM 先写一个候选 SQL，再反推涉及的列
  - **Value-based linker**: 用 retrieved values + DB column matching
- **LLM 调用**: 每题 ~8-12 次
- **耗时**: 99 题 ~48 min, 1534 题 **~6-7 hours**（最慢 stage 之一）
- **输出**: `workspace_*/schema_linking/bird/dev.snapshot` (+ `final_linked_tables_and_columns`)

### Step 5: SQL Generation ⭐ 重头戏 2
- **脚本**: `runner/run_sql_generation.py`
- **做什么**: **N-version Programming** — 3 个独立生成器各产 4 个候选，共 12 个候选 SQL：
  - **DC (Divide & Conquer) generator**: 把复杂 question 拆子问题
  - **Skeleton generator**: 先写 SQL 骨架再填值
  - **ICL generator**: few-shot in-context learning (BIRD train 检索的示例)
- **LLM 调用**: 每题 ~12-20 次（最重）
- **耗时**: 99 题 ~70 min, 1534 题 **~10-15 hours**（最慢 stage）
- **输出**: `workspace_*/sql_generation/bird/dev.snapshot` (+ `sql_candidates: [12 strings]`)

### Step 6: SQL Revision
- **脚本**: `runner/run_sql_revision.py`
- **做什么**: "**Unit Testing Tool-Chain**" — 对 12 个候选跑 deterministic checkers：
  - **Syntax checker** (sqlglot)
  - **Logic checker** (NULL handling, aggregation patterns)
  - **Quality checker** (使用 ORDER BY+LIMIT 替代 min/max 子查询等)
  - 检测出问题 → 触发 LLM 修复
- **LLM 调用**: 每题 ~3-5 次（仅修复时触发）
- **耗时**: 99 题 ~3 min, 1534 题 ~30-60 min
- **输出**: `workspace_*/sql_revision/bird/dev.snapshot` (+ `sql_candidates_after_revision`)

### Step 7: SQL Selection
- **脚本**: `runner/run_sql_selection.py`
- **做什么**: 从修复后候选 SQL 选最终答案：
  - 先按执行结果分组（结果相同的合并）
  - 然后 **tournament selection**（pairwise comparison）选 top-1
  - **Shortcut**: 如果某个候选在 ≥60% 的小组里出现，直接选它
- **LLM 调用**: 每题 ~3-5 次（仅 tournament 时触发）
- **耗时**: 99 题 ~1 min, 1534 题 ~5-15 min（最快 stage）
- **输出**: `workspace_*/sql_selection/bird/dev.snapshot` (+ `final_selected_sql` ← 这就是终极答案)

## ModularSQL 改进 (Post-processing)

### Pattern 1: DISTINCT-aware Semantic Verifier
- **脚本**: `src/improvements/apply_pattern1.py`
- **做什么**: 对 `final_selected_sql` 应用规则：
  1. SQL 有 JOIN
  2. 无 aggregate/GROUP BY
  3. 无 DISTINCT
  4. 执行结果有重复行
  5. NULL-density < 5%（防止误伤数据自然重复的列）
  - 满足全部 → inject DISTINCT 重新执行
- **LLM 调用**: **0**
- **耗时**: <30 秒（纯 Python + SQLite）
- **dev99 验证**: +4 pp (71.72% → 75.76%)
- **输出**: `results/<config>_pattern1.csv`

### Pattern 2: Profile-Augmented Schema (废弃)
- **脚本**: `src/improvements/inject_profile_metadata.py`
- **做什么**: 把 column 的 profile hints (PK / Common Prefix / Contains NULLs) 注入 BIRD 的 `column.description`
- **dev99 验证**: -5pp（slim 版）/ -4pp（verbose 版）→ **报负面发现**
- **状态**: 不进 final stack，作为 paper 的 contradicting 实验

## 完整流程

```
[BIRD dev.json + dev_databases/]
        ↓
Step 1: Preprocess          (本地, 0 LLM)
        ↓
Step 2: Vector DB           (本地 embedding, 0 LLM)
        ↓
Step 3: Value Retrieval     (LLM × 1-2/题)
        ↓
Step 4: Schema Linking      (LLM × 8-12/题, 3 个并行 linker)
        ↓
Step 5: SQL Generation      (LLM × 12-20/题, 3 个生成器 × 4 候选 = 12 SQL/题)
        ↓
Step 6: SQL Revision        (LLM × 3-5/题, 修复触发型)
        ↓
Step 7: SQL Selection       (LLM × 3-5/题, tournament + shortcut)
        ↓
[final_selected_sql per question]
        ↓
Pattern 1: DISTINCT Verifier  (本地规则, 0 LLM)  ← 我们的改进
        ↓
[final accuracy on BIRD-Dev]
```

## 时间预算（1534 全量, parallel=16）

| Stage | 已知 99-sample | 推算 1534 | 实际全量 (running) |
|---|---|---|---|
| 1. Preprocess | <30s | <30s | ~10s ✓ |
| 2. Vector DB | 5min | 10min | ~10min ✓ |
| 3. Value Retrieval | 3min | 30min | ~30min ✓ |
| 4. Schema Linking | 48min | 7h | ~7h (06:25 AM 完成 ✓) |
| 5. SQL Generation | 70min | **12-15h** | 跑中 ~10% |
| 6. Revision | 3min | 60min | TBD |
| 7. Selection | 1min | 15min | TBD |
| Pattern 1 post-proc | <1min | <1min | TBD |
| **Total** | 2h | **~24h** | ~22-24h |

## 成本预算

- 99-sample 1 次完整跑 ≈ $0.83
- 1534 全量 1 次 ≈ **$13-25**
- Pattern 1 post-proc 0 美元
- Pattern 2（如果跑全量）+ $13 一次（**已决定不跑**）
