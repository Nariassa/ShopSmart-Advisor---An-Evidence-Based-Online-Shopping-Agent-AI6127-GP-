# 证据约束生成与购买信心评分模块：Generation Testing 交付

本轮日期为 2026-10-15。使用 20 个已有问题，分别生成 Qwen2.5 7B 和 Llama 3.1 8B 的英文回答，计划 40 个问题—模型结果。

本轮测试条件是 annotation_candidate_replay_development：791 条已标注候选仅提供问题、片段 ID 和展示顺序，程序从 chunks.parquet 回填原文，并逐条核对 candidate_text_sha256。全部候选保留，不按 relevance_label、confidence 或 reason 过滤或重排，这些字段不会传给模型。

没有获得 检索与排序评估模块 的正式检索排名，不能把展示顺序当作 BM25/dense/hybrid 排名。本轮是固定候选池上的生成测试，不是端到端检索测试，也不是独立正式测试集。问题文件虽命名为 formal_queries_v1.jsonl，但这些问题已被用于项目开发，因此交付中明确标为 development。

## 执行配置

- 两模型使用完全相同的候选证据、顺序、v1 选择规则及固定 PCS 时间。
- 选择输出只包含 fact_id 和 product_name，由程序核对商品名称并提取原文。
- 每条选中事实单独生成一条英文改写，不支持多证据归纳；改写未通过 NLI 则保留原文。
- 本地 NLI：cross-encoder/nli-deberta-v3-small，CPU，阈值 0.9。
- temperature=0，seed=42，num_ctx=8192，num_predict=2048。
- max_products=3，max_facts=18，max_claims=8，max_context_chars=8000。
- keep_alive=2m 减少重复加载；切换模型和完成该模型全部问题后释放该模型。运行前已有的 Ollama 服务不由测试关闭。

完整配置、软件版本、模型权重 digest、输入哈希和代码快照随包保留。耗时包含首题模型冷加载及后续常驻推理，不用于声称系统级吞吐量。

## 一条命令执行或断点续跑

先启动 Ollama，在项目根目录运行：

```powershell
& ../.venv/Scripts/python.exe -m src.generation.testing
```

默认目录为 outputs/member_c_generation_testing_revised_final_2026-10-15/。每题保存独立检查点，不会把失败输出伪装为成功回答，也不会静默删除失败题。再次执行会保留已有结果；输入、参数、代码或模型版本不同则拒绝混合结果。

重新完整运行到新目录：

```powershell
& ../.venv/Scripts/python.exe -m src.generation.testing --output-dir outputs/member_c_generation_testing_rerun
```

仅准备候选原文和冻结配置：

```powershell
& ../.venv/Scripts/python.exe -m src.generation.testing --prepare-only
```

## 系统集成、交互与综合评估模块 接收的文件

| 文件 | 用途 |
| --- | --- |
| summary.md / summary.json | 运行结果、引用核验、自动检查线索及限制 |
| qwen2.5_7b_answers.md / llama3.1_8b_answers.md | 两模型逐题英文回答 |
| results_all.jsonl / 两模型 results.jsonl | 最终回答、PCS、引用、实际输入事实、全部调用及失败原因 |
| runs/ | 每个问题—模型的原始检查点 |
| evidence_map.json / queries.jsonl | 固定原始候选和问题 |
| ragas_inputs.jsonl | 实际给选择模型的事实上下文和最终回答，无虚构 reference |
| blinded_review_packets.jsonl / blinded_review_answers.md | 隐去模型名称的回答、结论和原始来源，随机顺序 |
| model_key_for_E.csv | 盲评 ID 与模型对应关系，正式评分前不提供给标注者 |
| human_review_template.csv | 回答级评分表，分数留空 |
| claim_support_review_template.csv | 逐结论支持性评分表，标签留空 |
| automatic_review_flags.csv | 自动标记需复核的问题 |
| failures_and_abstentions.jsonl | 失败、非模型输出及无证据拒答记录 |
| paired_input_checks.json | 两模型每题是否同证据、同选择 Prompt、同 PCS |
| manifest.json / runtime_versions.json / model_digests.json | 固定配置、软件与模型版本 |
| input_provenance.json / code_snapshot/ | 候选原文校验及代码版本追溯 |

## 检查范围

引用检查只验证产品、原文和字符位置。方面覆盖检查使用输入方面标签，正反证据保留检查使用未校准的情感标签，这些都只是复核线索，不是人工回答质量分数。

程序分别报告模型选择通过数、选择拒绝数、英文改写通过数、NLI 拒绝数、规则补入证据数和最终引用数。模型支持率不等于引用核验率，最终引用核验率也不等于事实准确率。

PCS 暂未校准；缺少原始评论 ID、时间、购买验证和有用票的情况仍需披露。方面分析与购买决策解释模块 的正式极化和覆盖接口未在本批使用。新版启用保守标题类型规则，候选是否满足商品功能、兼容性等具体需求仍需人工检查。

## 2026-10-15 四项修复重测

旧版基线保留在 outputs/member_c_generation_testing_2026-10-15/。新版从 --baseline-dir 复用其冻结的问题、791 条原始候选及 PCS 时间，不再次回填或重排标注候选。配置 generation_revision=bounded_type_scope_display_v3_attribution_aware_nli 标识修复版本，不能仅根据 prompt_version=v1 将其视作历史原版 Prompt。

1. Prompt、Ollama 传输 Schema 和运行时检查均按 max_claims 构建；本批两阶段上限均为 8。模型少选允许，超限仍失败，不截断伪装成功。
2. 提取目标商品类型、原始需求、显式条件和排除短语，并传入选择 Prompt。商品标题类型过滤在 PCS 排序、上下文构建和规则补入之前执行。保留 excluded_products 日志；不把功能证据缺失解释为已满足要求。配件本身是目标时保留配件类型，不一律排除所有配件。
3. 评论改写以 One reviewer reports that 开头。独立守卫检查个人范围、显式否定、数字及单位、条件、不确定性和时间限制。通过守卫后，NLI 比较原文与归属引导语之后的结论内容，阈值仍为 0.9；最终展示完整归属句。守卫与 NLI 任一拒绝均保留原文。规则不保证完整语义等价，可能误拒绝同义改写。
4. 强化完整商品名称复制，只归一化 Unicode NFC、HTML 实体和空白，不删除型号或后缀。显示层清理 HTML；结构化结论、引用原文和字符位置不改动。原文评论展示增加 One reviewer wrote 归属。

outputs/member_c_generation_testing_revised_2026-10-15/ 是主动停止的早期诊断批次：归属引导语最初整体送入 NLI，导致过量误拒绝。结果及代码快照保留并标为 DIAGNOSTIC_ONLY，不作为最终重测。

在两个模型完成后运行 scripts/compare_generation_revision.py 生成逐题前后对照。原始候选保持相同，但类型过滤会改变实际 Prompt 事实和商品集合；这是联合修复的流程回归，不是固定最终事实的 Prompt 消融。相同商品的 PCS 应保持一致，新增/移除商品不能强行按原位比较 PCS。

rewrite_challenge.json 保存 15 个开发者编写困难样例的守卫加真实本地 NLI 结果；这些诊断样例不构成独立测试准确率。55 项自动测试通过，包括 Schema 传递、配件过滤/规则补入、个人范围与条件检查、归属与 NLI 内容分离，以及清洗后引用位置不变。

## 正式检索证据接入

检索与排序评估模块 提供 query_id → 检索证据映射后，可通过现有 src.generation.batch 的 --queries、--evidence-map 和 --nli-model 重跑，两个模型分别运行并保持参数一致。必须将该条件与本轮候选池回放分开保存。

系统集成、交互与综合评估模块按评估规则进行人评、IAA、适用的 RAGAS 指标计算。没有参考答案时，需参考答案的指标不能计算；可先使用不需要参考答案的指标。不得把本轮自动检查或 NLI 分数当作人工评分。
