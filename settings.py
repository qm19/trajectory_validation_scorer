"""Edit this CONFIG and run `python run.py`. Relative paths use this file's folder."""
from trajectory_scorer.config import RunConfig

CONFIG = RunConfig(
    inputs=["examples"], # Replace this format-only example with your data.
    pattern="*.jsonl",                     # Use */sample_100.jsonl for model subfolders.
    output_dir="outputs/current",
    env_file=".env",
    workers=10,
    max_attempts=3,                         # Per-trajectory attempts including the first.
    retry_delay_seconds=2,
    retry_max_delay_seconds=20,
    json_retries=2,                         # Extra JSON repair calls per stage.
    chunk_chars=100000,
    max_tool_chars=16000,                   # 0 = no tool-content truncation.
    max_new_records=0,                     # 0 = all; positive = staged run.
    in_place=False,                        # True replaces each original atomically.
    backup_on_in_place=True,               # Original .bak kept when in_place=True.
    cleanup_on_success=True,
    generate_report=True,
    report_path=None,                      # Default: output_dir/analysis.html.
    progress=True,
    print_summary=True,
    request_timeout_seconds=None,          # None reads LLM_TIMEOUT_SECONDS from .env.
    http_max_attempts=None,                # None reads LLM_MAX_RETRIES (total attempts).
    max_output_tokens=None,
    direct_connection=None,
    no_proxy=None,                         # .env/default includes 7.156.135.202.
)
