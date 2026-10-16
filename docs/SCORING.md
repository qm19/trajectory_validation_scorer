# 数据结构与评分规范

## 输入与三阶段

UTF-8 JSONL，一行一个含 messages 数组的对象。支持 user / assistant / tool / system；system 和隐藏 reasoning 不作为执行证据。原消息索引从 0 开始并保留。

1. A 只读全部 user 消息，合并最终有效要求，对交付物逐一考虑 C1–C9；不根据已执行的检查反推义务。
2. B 按段提取 create / modify / check / failure / fix / user_correction，记录方法、对象、结果、消息编号与局限。
3. C 对每项义务评分。正分只能引用对应义务的 check / failure 事件。程序核对逐项覆盖并重算总分。

无缓存、无重试时，单条调用数为证据段数 K + 2。

| 类别 | 检查目标 |
|---|---|
| C1 | 生成后的文件存在、位置、格式/可打开性 |
| C2 | 内容完整与已取得来源一致 |
| C3 | 章节、顺序、字段、页数、数量与格式 |
| C4 | 公式、计算、日期、筛选、分类与规则 |
| C5 | 渲染后实际查看的视觉呈现 |
| C6 | 程序、交互和公式工作簿的运行 |
| C7 | 引用、公开来源、URL、日期与证据 |
| C8 | 外部动作执行后的状态 |
| C9 | 多交付物关键事实与版本一致 |

## 等级与指标

0 无检查；1 发起但缺结果；2 方法、范围或判读不足；3 检查合适、有结果并判读；4 进一步覆盖主要失效方式、边界或互补方法。重复同一检查不加分。

`action_score = 100 × Σscore / (4 × 义务数)`。

- attempt_rate：score≥1 的义务比例。
- observed_check_rate：score≥2 的义务比例。
- adequate_check_rate：score≥3 的义务比例。
- final_version_check_rate：final_version_checked=true 的义务比例。
- observed_outcome：pass / fail / inconclusive / not_observed，不影响分数。

无适用义务时主分与各率为 null；缺失类别为 N/A，不补 0。总体均分按轨迹等权，默认各率与类别分按义务等权。

## 最终对象

字段路径 `meta_info.unique_info.verifi_action_score`，schema_version=1.0。成功时保留原完整评分对象，包括 final_deliverables、requirement_updates、obligations、evidence_events、obligation_scores、category_scores 和汇总，另加：

- status：success / failed / pending。
- trace_sha256：归一化 messages 的 SHA-256，不使用 meta_info 标签。
- scoring_fingerprint：评分模型与配置身份。
- normalization：chunk_chars / max_tool_chars，用于离线重查哈希。
- source_file / source_line_number / source_group：来源、物理行号及输入父目录。
- attempts_total：当前断点身份下累计记录的轨迹尝试数。
- updated_at：UTC ISO 时间。
- llm_call_stats：已记录响应的字符量、服务端 usage 和 HTTP 尝试数，包含当前断点记录内的失败重试；缓存不新增调用。

failed / pending 主分为 null，不伪造 0 分。硬崩溃前未写入断点、或未取得服务端 usage 的请求，不能据此准确计费。最终证据对象保留供追溯，清理的是阶段文件。

结构与引用合法不证明语义打分无误。来源实际对照、C5 视觉判读、图片移除、工具截断和 4 分边界需要人工校准。
