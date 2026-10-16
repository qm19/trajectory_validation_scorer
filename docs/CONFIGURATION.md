# 配置参数

`settings.py` 定义 `CONFIG = RunConfig(...)` 或 dict。相对路径以配置文件目录为基准；命令行显式参数覆盖配置文件。模型设置优先级：RunConfig 非 None 值 > 系统环境变量 > `.env` > 内置默认。

| 参数 | 默认 | 说明 |
|---|---:|---|
| inputs | 空 | 文件/目录列表 |
| pattern | `*.jsonl` | 目录匹配范围，支持 `**/*.jsonl` |
| output_dir | outputs | 输出及运行控制目录 |
| env_file | .env | analyze 命令不需要 |
| workers | 10 | 全批次同时处理轨迹数 |
| max_attempts | 3 | 每条每次运行的总尝试数，含首次 |
| retry_delay_seconds | 2 | 轨迹重试退避起点 |
| retry_max_delay_seconds | 20 | 退避上限 |
| json_retries | 2 | 每阶段额外 JSON 修复次数 |
| chunk_chars | 100000 | 轨迹段字符阈值，不等于 token 上限 |
| max_tool_chars | 16000 | 工具内容前后保留字符数；0 不截断 |
| max_new_records | 0 | 0 全量；正数限制本次新增处理条数 |
| in_place | false | true 原子替换输入；false 生成注释副本 |
| backup_on_in_place | true | 首次原地写回前保留 .bak，不覆盖已有备份 |
| cleanup_on_success | true | 全部成功及报告完成后删除中间结果 |
| generate_report | true | 自动生成脚本 HTML |
| report_path | null | 默认 output_dir/analysis.html |
| progress | true | tqdm 进度 |
| print_summary | true | 控制台总体及分组统计 |
| request_timeout_seconds | null | 覆盖 LLM_TIMEOUT_SECONDS |
| http_max_attempts | null | 覆盖 LLM_MAX_RETRIES，包含首次 |
| max_output_tokens | null | 覆盖 LLM_MAX_OUTPUT_TOKENS |
| direct_connection | null | 覆盖 LLM_DIRECT_CONNECTION |
| no_proxy | null | 覆盖 LLM_NO_PROXY |

目录默认不递归。五组模型子目录可使用 `pattern="*/sample_100.jsonl"`。副本模式从扫描中排除 output_dir，避免输出递归成为输入；原地模式建议将 output_dir 设在独立位置。

评分指纹包含模型服务地址的哈希、模型、提示词版本、请求 profile、分段、截断、最大输出与 JSON 模式。凭据不参与指纹，修改并发和重试不使已成功结果失效。

硬关闭进程不能撤销已发出的 API 请求。安全中断等待请求结束后写回；HTTP 超时和重试会影响等待时间。各层重试叠加，不能把 max_attempts 当作整条轨迹的 HTTP 请求总上限。
