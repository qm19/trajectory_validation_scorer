# Agent 轨迹校验动作评分

从 agent 轨迹的 `messages` 中评价实际检查动作是否充分，支持并发打标、元数据写回、断点恢复、失败重试、tqdm 实时进度及离线 HTML 分析。

本项目衡量**可观察的校验行为**。检查发现交付失败仍可得充分动作分，最终交付正确率和修复是否成功不直接参与动作分。

## 安装与模型配置

Python **3.10+**，运行依赖为 tqdm 和 filelock。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Linux/macOS 激活使用 `source .venv/bin/activate`。在源码目录直接运行，也可用 `python -m pip install -r requirements.txt`。

将 `.env.example` 复制为 `.env` 并填入模型地址、密钥和模型名；已有 `.env` 时保留并编辑。系统环境变量优先于 `.env`。

```dotenv
LLM_URL=http://YOUR_HOST:PORT/v1/chat/completions
LLM_API_KEY=YOUR_API_KEY
LLM_MODEL_NAME=YOUR_MODEL_NAME
LLM_TIMEOUT_SECONDS=180
LLM_MAX_RETRIES=3
LLM_MAX_OUTPUT_TOKENS=32768
LLM_JSON_MODE=false
LLM_NO_PROXY=7.156.135.202
```

接口兼容 Chat Completions，URL 可为完整地址或 /v1 基址。按既有服务要求，发送 `extra_body.enable_thinking=false`、`extra_body.google.thinking_config.include_thoughts=true` 和 `thinking_level="minimal"`。LLM_NO_PROXY 追加到 no_proxy / NO_PROXY 并保留现有排除项。实际密钥不进入安装包、断点或报告。

## 快速开始

### 编辑配置直接运行

修改根目录 [settings.py](settings.py) 中的 CONFIG：

```python
from trajectory_scorer.config import RunConfig
CONFIG = RunConfig(
    inputs=[r"data/input"],
    pattern="*/sample_100.jsonl",
    output_dir="outputs/batch1",
    env_file=".env",
    workers=10,
    max_attempts=3,
    json_retries=2,
    chunk_chars=100000,
    max_tool_chars=16000,
    in_place=False,
    cleanup_on_success=True,
    generate_report=True,
)
```

```powershell
python run.py
```

配置文件内的相对路径以配置文件目录为基准。inputs 支持多个文件或目录。目录默认只匹配直接子文件 `*.jsonl`；使用 `pattern` 明确指定递归范围，避免混入旧评分文件或输入副本。

### 命令行与 Python API

```powershell
python -m trajectory_scorer annotate --input data/input.jsonl --output-dir outputs/batch1 --workers 10
python -m trajectory_scorer annotate --config settings.py --workers 5 --max-attempts 4
```

多个文件重复传 `--input`；目录可加 `--pattern "*/sample_100.jsonl"`。安装后也可使用 `trajectory-validation-scorer` 命令。更多参数见 `python -m trajectory_scorer annotate --help`。

```python
from trajectory_scorer.batch import run_batch
from trajectory_scorer.config import RunConfig
summary = run_batch(RunConfig(inputs=["data/input.jsonl"], output_dir="outputs/batch1"))
assert summary["success"] == summary["total"]
```

## 最终结果写到哪里

默认保留输入，生成保留完整记录结构的 `*.annotated.jsonl` 副本，只更新以下叶字段：

```text
meta_info.unique_info.verifi_action_score
```

该字段是**完整评分对象**，不是单个数字。读取主分使用 `.verifi_action_score.action_score`。

```json
{
  "schema_version": "1.0",
  "status": "success",
  "action_score": 75.0,
  "adequate_check_rate": 1.0,
  "final_version_check_rate": 0.75,
  "trace_sha256": "...",
  "scoring_fingerprint": "...",
  "source_file": "...",
  "source_line_number": 1,
  "source_group": "model_a",
  "attempts_total": 1
}
```

以上为字段节选。真实对象还含交付物、义务、检查证据、逐义务分、类别分、版本及调用统计。原有 messages、tools 和其他元数据均保留；缺失的 meta_info / unique_info 会创建，已存在但不是对象时预检报错，避免覆盖原始内容。

| 状态 | 含义 | 主分 | 统计 |
|---|---|---|---|
| success | 已完成且通过结构、引用与计算校验 | 0–100；无义务时 null | 有义务时计分 |
| failed | 本次尝试已用尽，含 error | null | 不计为 0 分 |
| pending | 限额或中断后未完成 | null | 不计为 0 分 |

典型输出：

```text
outputs/batch1/
├── model_a/sample_100.annotated.jsonl
├── model_b/sample_100.annotated.jsonl
├── analysis.html
├── run_summary.json
└── .annotation.lock
```

多文件输出保留输入公共父目录下的相对目录，避免同名文件冲突。原始空行保留，物理行号不漂移；轨迹内消息编号从 0 开始。

`in_place=True` / `--in-place` 可原子替换原文件，默认首次写回前生成 `.bak` 原始备份，不覆盖已有备份。备份不是阶段结果，不自动删除。`.annotation.lock` 是不含打标数据的控制锁文件。

## 进度、重试与统计

进度按完成轨迹更新，包含成功、失败、复用和重试计数，不会等待前一行完成才更新。tqdm 写 stderr，结束汇总写 stdout，打印总数、成功/失败/待处理、均分、中位数、充分检查率、复验率和来源分组统计。

三层重试：

1. `max_attempts=3`：每条轨迹每次程序运行最多尝试 3 次，**包含首次**；复用已成功的阶段缓存。
2. `json_retries=2`：每个阶段最多额外修复 2 次不合规模型 JSON/引用响应。
3. `.env` 中 `LLM_MAX_RETRIES=3` 或 `http_max_attempts=3`：每次 HTTP 请求最多尝试 3 次，包含首次。

次数叠加会增加请求量。单条失败不终止其他轨迹。退出码：全部成功 0；仍有失败/待处理 1；配置、文件或系统错误 2；安全中断 130。

## 断点续跑与中间结果清理

重新运行相同命令、使用相同输出目录即可续跑。

- 每条轨迹尝试和结果提交到 SQLite；成功阶段另有缓存。
- Ctrl+C 后停止提交新任务，等待已发出请求结束，再保存部分输出。硬结束进程时仍可恢复已提交的断点，等待受 HTTP 超时约束。
- 已有成功结果从断点、原记录注释或已打标副本复用。即使成功后缓存已清理，再运行也能复用最终注释，避免重复 API 调用。
- 比较归一化 messages SHA-256、评分参数指纹及引用/分数一致性。修改模型、服务、提示词、分段或截断参数使旧结果失效；改变并发和重试数不使其失效。
- `max_new_records=10` 只新增最多 10 条，保留其他记录为 pending；续跑恢复为 0。
- **全部成功、写回读回通过、HTML 报告完成后**，默认统一删除本次 .state 中的 SQLite、WAL、阶段 JSON 和无效响应文件。
- 仍有失败、待处理、中断或交付错误时保留状态；下次运行对失败条目重新提供 max_attempts 预算。

最终评分对象内的义务/证据用于审计，仍然保留；清理的是中间文件，不删除最终对象中的可追溯证据。

## 独立生成 HTML 分析报告

正式 annotate 默认自动生成 analysis.html。它内嵌最终数据，支持分组/状态/关键词筛选、C1–C9 指标切换、分页、逐义务理由和证据查看，可以离线打开。

也可从已打标记录独立生成，**不需要 .env / API key，不调用 LLM**：

```powershell
python -m trajectory_scorer analyze --input outputs/batch1 --pattern "**/*.annotated.jsonl" --report outputs/batch1/analysis.html
```

同时生成 analysis.summary.json。总体均分按轨迹等权，充分检查率、复验率和默认类别分按义务等权。失败、待处理不是 0 分；无适用类别是 N/A。

## 评分体系与限制

A 仅从全部 user 消息实例化 C1–C9 义务；B 分段提取可见事件；C 对每项义务给 0–4 分。程序校验引用、逐义务覆盖、总分与类别分。写入成功不是读回检查，图片渲染不等于查看，检查失败不扣动作分。meta_info 中的质量标签不参与评分。

程序校验不能证明语义判断无误。来源实际对照、C5 视觉判读、图像占位与 4 分边界仍需抽样复核。完整说明见 [评分规范](docs/SCORING.md)。

## 项目结构与文档

```text
trajectory_validation_scorer/
├── pyproject.toml / requirements.txt / .env.example
├── settings.py / run.py
├── trajectory_scorer/
│   ├── config.py       # 配置及评分指纹
│   ├── batch.py        # 并发、重试、写回、清理
│   ├── records.py      # JSONL 与原子写入
│   ├── state.py        # 持久化断点
│   ├── reporting.py    # 统计与离线 HTML
│   ├── core.py         # API、阶段缓存与校验
│   ├── prompts.py      # 版本化提示词
│   └── cli.py / __main__.py
├── docs/ / examples/ / tests/
└── CHANGELOG.md
```

- [配置参数](docs/CONFIGURATION.md)
- [数据与评分规范](docs/SCORING.md)
- [恢复与清理实现](docs/ARCHITECTURE.md)
- [验证说明](docs/VALIDATION.md)

旧 preflight / score / report 命令保留兼容。**旧 score 仍输出独立评分 JSONL；上述新工作流使用 annotate。** .env、历史数据与报告不进入安装包。

## 测试与构建

```powershell
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps --wheel-dir dist
```

测试覆盖数据保留、续跑、失败重试、原地备份、中断、参数失效、清理保护及报告。需要旧私有轨迹的集成测试在数据不存在时跳过，其他测试无需 API。分享目录不包含真实轨迹、真实打标结果、API 密钥、运行缓存或历史报告。
