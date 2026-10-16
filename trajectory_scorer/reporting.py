"""Deterministic summaries and a self-contained HTML report; never calls an LLM."""
from __future__ import annotations

import html
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .core import ScoringError, compute_category_scores, normalize_messages, trace_digest, validate_events
from .records import atomic_lines, get_score, read_lines, validate_success

LABELS = {"C1":"交付与可访问性","C2":"内容完整与来源一致","C3":"结构与格式约束","C4":"计算与规则执行",
          "C5":"视觉呈现","C6":"功能与运行","C7":"引用与证据","C8":"外部动作状态","C9":"跨交付物一致性"}


def summarize(items):
    good = [r["score"] for r in items if r["status"] == "success"]
    values = [r["action_score"] for r in good if r["action_score"] is not None]
    scores = [s for r in good for s in r["obligation_scores"]]
    n = len(scores)
    categories = []
    for category in LABELS:
        cs, applicable = [], 0
        for r in good:
            ids = {o["id"] for o in r["obligations"] if o["category"] == category}
            applicable += bool(ids)
            cs.extend(s for s in r["obligation_scores"] if s["obligation_id"] in ids)
        categories.append({"category":category,"count":len(cs),"trajectories":applicable,
            "score":25*statistics.mean(s["score"] for s in cs) if cs else None,
            "adequate":sum(s["score"]>=3 for s in cs)/len(cs) if cs else None,
            "final":sum(bool(s["final_version_checked"]) for s in cs)/len(cs) if cs else None})
    return {"total":len(items),"success":len(good),"failed":sum(r["status"]=="failed" for r in items),
        "pending":sum(r["status"]=="pending" for r in items),"scored_count":len(values),
        "mean":round(statistics.mean(values),2) if values else None,
        "median":statistics.median(values) if values else None,"min":min(values) if values else None,
        "max":max(values) if values else None,"zero_scores":sum(v==0 for v in values),
        "obligations":n,"adequate_check_rate":sum(s["score"]>=3 for s in scores)/n if n else None,
        "final_version_check_rate":sum(bool(s["final_version_checked"]) for s in scores)/n if n else None,
        "grade_counts":[sum(s["score"]==i for s in scores) for i in range(5)],"categories":categories,
        "recorded_completion_calls":sum(r.get("llm_call_stats",{}).get("completion_calls",0) for r in good),
        "recorded_http_attempts":sum(r.get("llm_call_stats",{}).get("http_attempts",0) or 0 for r in good),
        "recorded_total_tokens":sum(r.get("llm_call_stats",{}).get("total_tokens") or 0 for r in good)}


def collect(paths):
    items = []
    for path in paths:
        for line, _, row in read_lines(Path(path)):
            if row is None:
                continue
            value = get_score(row)
            status = value.get("status", "pending") if isinstance(value, dict) else "pending"
            if status not in {"success", "failed", "pending"}:
                raise ScoringError(f"不支持的打标状态: {path}:{line}")
            if status == "success":
                cap = value.get("normalization", {}).get("max_tool_chars", 16000)
                messages = normalize_messages(row, cap)
                validate_success(value, trace_digest(messages))
                validate_events({"events":value["evidence_events"]},{"obligations":value["obligations"]},
                                {m["message_index"] for m in messages})
            group = (value or {}).get("source_group", Path(path).parent.name) if isinstance(value, dict) else Path(path).parent.name
            title = " / ".join(d["description"] for d in value.get("final_deliverables",[])) if isinstance(value,dict) else ""
            items.append({"file":str(Path(path).resolve()),"line":line,"group":group,"status":status,
                          "title":title or f"原始第 {line} 条", "score":value if isinstance(value,dict) else {}})
    if not items:
        raise ScoringError("没有可分析的轨迹")
    stats = summarize(items)
    stats["groups"] = [{"group":g, **summarize([r for r in items if r["group"]==g])} for g in sorted({r["group"] for r in items})]
    stats["input_files"] = [str(Path(p).resolve()) for p in paths]
    stats["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats["usage_scope"] = "已记录的响应与HTTP尝试；崩溃前未记入断点的调用、未收到服务端usage的请求不能据此推算完整费用。"
    return stats, items


def print_summary(stats):
    def pct(v): return "N/A" if v is None else f"{v*100:.2f}%"
    print("\n校验动作打标汇总")
    print(f"总计 {stats['total']} | 成功 {stats['success']} | 失败 {stats['failed']} | 待处理 {stats['pending']}")
    print(f"平均动作分 {stats['mean']} | 中位数 {stats['median']} | 义务 {stats['obligations']}")
    print(f"充分检查率 {pct(stats['adequate_check_rate'])} | 最终版本复验率 {pct(stats['final_version_check_rate'])}")
    if "reused_this_run" in stats:
        print(f"本次复用 {stats['reused_this_run']} | 新成功 {stats['new_successes_this_run']} | 新增模型调用 {stats['new_completion_calls_this_run']}")
    print("来源分组 | 成功/总数 | 失败 | 平均动作分 | 充分检查率 | 最终复验率")
    for g in stats["groups"]:
        print(f"{g['group']} | {g['success']}/{g['total']} | {g['failed']} | {g['mean']} | {pct(g['adequate_check_rate'])} | {pct(g['final_version_check_rate'])}")
    for path in stats.get("output_files",[]): print("已打标数据:", path)
    if stats.get("report"): print("HTML 报告:", stats["report"])
    if stats.get("summary_file"): print("统计 JSON:", stats["summary_file"])
    if "state_cleaned" in stats: print("临时阶段缓存/断点:", "已清理" if stats["state_cleaned"] else "已保留，可续跑")


CSS = """
:root{--ink:#17383d;--muted:#667d80;--line:#dce6e2;--teal:#087d72}*{box-sizing:border-box}html{scroll-padding-top:20px}body{margin:0;background:#f2f5f4;color:var(--ink);font:14px/1.75 'Segoe UI','Microsoft YaHei',sans-serif}header{padding:22px max(24px,calc((100% - 1200px)/2));background:#17383d;color:white;font-weight:700}main{max-width:1200px;margin:auto;padding:32px 24px}h1{font-size:34px;line-height:1.3}h2{font-size:23px}h3{font-size:17px}p{margin:10px 0 16px}.muted,small{color:var(--muted)}.card{background:white;border:1px solid var(--line);padding:24px;border-radius:13px;margin:20px 0;min-width:0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:20px}.metric{border-left:3px solid var(--teal);padding-left:15px}.metric b{display:block;font-size:28px}.metric span{font-size:12px;color:var(--muted)}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#f4f8f6;color:var(--muted);font-size:12px}td small{display:block}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end}.field{flex:1;min-width:140px}.field label{display:block;font-size:12px;color:var(--muted)}input,select,button{font:inherit;border:1px solid var(--line);border-radius:7px;padding:9px 12px;background:white;color:var(--ink)}input,select{width:100%}button{cursor:pointer}button:hover{background:#e8f4ee}button:disabled{opacity:.4;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #e2b355;outline-offset:2px}.link{color:var(--teal);border:0;padding:0;background:transparent;text-align:left;font-weight:600}.heat{display:inline-block;border-radius:5px;padding:5px 10px;min-width:55px;text-align:center}.note{padding:15px 18px;background:#fff8e8;border-left:3px solid #d7a145;border-radius:0 8px 8px 0;margin:16px 0}.pill{font-size:12px;border-radius:20px;padding:4px 9px;background:#e8f4ee;display:inline-block}.pager,.top{display:flex;justify-content:space-between;gap:16px;align-items:center}.pager{margin-top:16px}.stack{display:flex;height:24px;border-radius:5px;overflow:hidden;margin:12px 0;background:#edf1ef}.stack span{height:100%;display:flex;align-items:center;justify-content:center;color:white;font-size:11px}.obligation{padding:18px;border:1px solid var(--line);border-radius:10px;margin:14px 0}.event{border-left:3px solid var(--teal);padding:10px 15px;margin:12px 0;background:#f4f8f6}details{border:1px solid var(--line);padding:12px 15px;border-radius:8px;margin:12px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f5;border:1px solid var(--line);border-radius:7px;padding:15px;font:12px/1.65 Consolas,'Microsoft YaHei',monospace;max-height:360px;overflow:auto}.empty{text-align:center;padding:30px;color:var(--muted)}footer{text-align:center;font-size:12px;color:var(--muted);padding:25px;overflow-wrap:anywhere}.task{min-width:220px;max-width:330px}.source{font-size:12px;color:var(--muted);overflow-wrap:anywhere}@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.top{align-items:flex-start;flex-direction:column}h1{font-size:28px}main{padding:20px 14px}.card{padding:18px}.toolbar{align-items:stretch}.field{min-width:100%}}@media print{header,.toolbar,.pager,button{display:none}body{background:white}main{max-width:none}.card{break-inside:avoid}}
"""

JS = r"""
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let rows=[],page=0;const size=15,P=x=>x==null?'N/A':(x*100).toFixed(1)+'%';
function heat(){let metric=$('metric').value;$('heat').innerHTML=D.stats.groups.map(g=>`<tr><td>${E(g.group)}</td>${g.categories.map(c=>{let v=c[metric],x=metric==='score'?v:v==null?null:v*100;return `<td><span class="heat" style="background:${x==null?'#edf1ef':`hsl(${x*1.3} 35% 90%)`}">${x==null?'N/A':x.toFixed(1)}</span><small>${c.count} 项 / ${c.trajectories} 条${c.trajectories>0&&c.trajectories<10?' · 小样本':''}</small></td>`}).join('')}</tr>`).join('')}
function filter(){let group=$('group').value,status=$('status').value,q=$('search').value.toLowerCase().trim();rows=D.items.filter(r=>(!group||r.group===group)&&(!status||r.status===status)&&(!q||`${r.group} ${r.line} ${r.title} ${(r.score.obligations||[]).map(o=>o.target).join(' ')}`.toLowerCase().includes(q)));page=0;table()}
function table(){let start=page*size;$('body').innerHTML=rows.slice(start,start+size).map(r=>`<tr><td>${E(r.group)}</td><td>${r.line}</td><td class="task"><button class="link" data-id="${r.id}">${E(r.title.slice(0,130))}</button></td><td>${r.status==='success'?'成功':r.status==='failed'?'失败':'待处理'}</td><td>${r.status==='success'?(r.score.action_score==null?'N/A':r.score.action_score.toFixed(1)):'—'}</td><td>${r.status==='success'?P(r.score.adequate_check_rate):'—'}</td><td>${r.status==='success'?P(r.score.final_version_check_rate):'—'}</td></tr>`).join('')||'<tr><td colspan="7" class="empty">没有符合条件的记录。</td></tr>';$('count').textContent=rows.length?`${start+1}–${Math.min(start+size,rows.length)} / ${rows.length} 条`:'0 条';$('prev').disabled=page===0;$('next').disabled=start+size>=rows.length}
function open(id){let r=D.items.find(x=>x.id===Number(id));if(!r)return;$('detail').hidden=false;$('title').textContent=`${r.group} · 原始第 ${r.score.source_line_number||r.line} 条`;$('source').textContent=`已打标文件：${r.file}；哈希：${r.score.trace_sha256||'未评分'}`;if(r.status!=='success')$('content').innerHTML=`<div class="note">${r.status==='failed'?'失败，可续跑重试':'尚未完成打标'}</div><pre>${E(r.score.error||'')}</pre>`;else $('content').innerHTML=`<p>动作分 ${r.score.action_score==null?'N/A':r.score.action_score} / 100；充分检查率 ${P(r.score.adequate_check_rate)}；最终复验率 ${P(r.score.final_version_check_rate)}</p>${r.score.obligations.map(o=>{let s=r.score.obligation_scores.find(s=>s.obligation_id===o.id),events=r.score.evidence_events.filter(e=>e.obligation_ids.includes(o.id)&&['check','failure'].includes(e.type));return `<article class="obligation"><span class="pill">${E(o.category)} / ${E(o.id)} · ${s.score}/4</span><h3>${E(o.target)}</h3><p><b>充分检查要求：</b>${E(o.sufficient_check)}</p><p><b>评分理由：</b>${E(s.reason)}</p><p>观察结果 ${E(s.observed_outcome)}；最终复验 ${s.final_version_checked?'是':'否'}</p><p class="source">证据消息：${s.supporting_message_indices.join(', ')||'无'}</p>${s.evidence_limitations?`<div class="note">${E(s.evidence_limitations)}</div>`:''}<details><summary>关联检查事件 ${events.length} 项</summary>${events.map(e=>`<div class="event"><b>消息 ${e.message_indices.join(', ')} · ${E(e.type)}</b><p>${E(e.method)}</p><p>${E(e.observed_result)}</p>${e.limitations?`<div class="note">${E(e.limitations)}</div>`:''}</div>`).join('')||'没有关联检查事件'}</details></article>`}).join('')}<details><summary>轨迹证据局限与复核标记</summary><pre>${E(JSON.stringify({trajectory_evidence_limitations:r.score.trajectory_evidence_limitations,obligation_review_flags:r.score.obligation_review_flags},null,2))}</pre></details>`;$('detail').scrollIntoView({behavior:'smooth',block:'start'})}
['group','status'].forEach(x=>$(x).addEventListener('change',filter));$('search').addEventListener('input',filter);$('metric').addEventListener('change',heat);$('prev').addEventListener('click',()=>{page--;table()});$('next').addEventListener('click',()=>{page++;table()});$('body').addEventListener('click',e=>{let b=e.target.closest('[data-id]');if(b)open(b.dataset.id)});$('close').addEventListener('click',()=>{$('detail').hidden=true});heat();filter();
"""


def render_report(items, stats, path: Path):
    esc = lambda v: html.escape(str(v))
    pct = lambda v: "N/A" if v is None else f"{v*100:.2f}%"
    group_rows = "".join(f'<tr><td>{esc(g["group"])}</td><td>{g["success"]}/{g["total"]}</td><td>{g["failed"]}</td><td>{g["pending"]}</td><td>{g["mean"] if g["mean"] is not None else "N/A"}</td><td>{g["median"] if g["median"] is not None else "N/A"}</td><td>{pct(g["adequate_check_rate"])}</td><td>{pct(g["final_version_check_rate"])}</td><td>{g["zero_scores"]}</td></tr>' for g in stats["groups"])
    colors = ["#bd6d65", "#bb944e", "#c2b04b", "#3c9b8a", "#173e49"]
    bars = ""
    for g in stats["groups"]:
        segments = "".join(f'<span style="width:{100*n/max(1,g["obligations"])}%;background:{colors[i]}" title="{i} 分：{n} 项">{str(i)+"分" if n/max(1,g["obligations"])>.09 else ""}</span>' for i,n in enumerate(g["grade_counts"]))
        bars += f'<p><b>{esc(g["group"])}</b> · {g["obligations"]} 项义务</p><div class="stack">{segments}</div>'
    options = "".join(f'<option value="{esc(g["group"])}">{esc(g["group"])}</option>' for g in stats["groups"])
    category_heads = "".join(f'<th title="{label}">{c}</th>' for c,label in LABELS.items())
    for i, item in enumerate(items):
        item = item.copy()
        item["id"] = i
        items[i] = item
    payload = json.dumps({"stats":stats,"items":items},ensure_ascii=False,separators=(",",":")).replace("<","\\u003c")
    body = f"""<header>AGENT / 轨迹校验动作分析</header><main><h1>校验行为与检查完成度</h1><p class="muted">从已打标 JSONL 中直接统计，无需额外 LLM 调用。结果字段：meta_info.unique_info.verifi_action_score。</p>
    <section class="card"><div class="grid"><div class="metric"><span>成功 / 全部</span><b>{stats['success']} / {stats['total']}</b></div><div class="metric"><span>失败 / 待处理</span><b>{stats['failed']} / {stats['pending']}</b></div><div class="metric"><span>平均动作分</span><b>{stats['mean'] if stats['mean'] is not None else 'N/A'}</b></div><div class="metric"><span>充分检查率 / 最终复验率</span><b style="font-size:22px">{pct(stats['adequate_check_rate'])} / {pct(stats['final_version_check_rate'])}</b></div></div><div class="note">失败和待处理记录不计为 0 分；成功但未观察到检查动作的记录才可能为 0 分。当前有 {stats['zero_scores']} 条有效 0 分结果。</div></section>
    <section class="card"><h2>来源分组统计</h2><div class="table-wrap"><table><thead><tr><th>分组</th><th>成功 / 总数</th><th>失败</th><th>待处理</th><th>平均分</th><th>中位数</th><th>充分检查率</th><th>最终复验率</th><th>0 分</th></tr></thead><tbody>{group_rows}</tbody></table></div><p class="muted">分组来自评分时的输入父目录，不保证任务可配对或难度相同。模型/批次均分只能描述当前样本，不能自动视为能力排名。</p></section>
    <section class="card"><h2>义务动作等级分布</h2><p class="muted">0 未检查 · 1 发起但无有效结果 · 2 范围或判读不足 · 3 合适且明确判读 · 4 覆盖主要边界或互补方法。</p>{bars}</section>
    <section class="card"><div class="top"><h2>C1–C9 类别热力表</h2><div class="field" style="max-width:220px"><label for="metric">指标</label><select id="metric"><option value="score">动作分 / 100</option><option value="adequate">充分检查率 %</option><option value="final">最终复验率 %</option></select></div></div><div class="table-wrap"><table><thead><tr><th>分组</th>{category_heads}</tr></thead><tbody id="heat"></tbody></table></div><p class="muted">N/A 表示无适用义务；少于 10 条适用轨迹时标记小样本。C1 交付、C2 内容、C3 结构、C4 规则、C5 视觉、C6 功能、C7 引用、C8 外部状态、C9 一致性。</p></section>
    <section class="card"><h2>逐轨迹结果与证据</h2><div class="toolbar"><div class="field"><label for="group">分组</label><select id="group"><option value="">全部</option>{options}</select></div><div class="field"><label for="status">状态</label><select id="status"><option value="">全部</option><option value="success">成功</option><option value="failed">失败</option><option value="pending">待处理</option></select></div><div class="field" style="flex:2"><label for="search">搜索交付物、义务或原始行号</label><input id="search" type="search" placeholder="输入关键词"></div></div><div class="table-wrap"><table><thead><tr><th>分组</th><th>文件行号</th><th>交付物 / 详情</th><th>状态</th><th>动作分</th><th>充分检查率</th><th>最终复验率</th></tr></thead><tbody id="body"></tbody></table></div><div class="pager"><span id="count" aria-live="polite"></span><div><button id="prev">上一页</button> <button id="next">下一页</button></div></div></section>
    <section class="card" id="detail" hidden><div class="top"><h2 id="title"></h2><button id="close">关闭详情</button></div><p class="source" id="source"></p><div id="content"></div></section>
    <section class="card"><h2>口径与限制</h2><p>主分 = 100 × 各义务分之和 / (4 × 义务数)。平均分为成功且有适用义务的轨迹等权平均；充分检查率为 score≥3 的义务占比；最终版本复验率为 final_version_checked=true 的义务占比。后两者按义务等权，不是每条轨迹比例的平均。</p><p>检查发现失败仍可得 3 或 4 分；修复成功不直接加动作分。最终版本复验也可能范围不足，所以复验率可以高于充分检查率。无适用义务时各率和主分为 null，不补 0。</p><p>本报告复核了成功记录与轨迹哈希、引用和汇总的对应关系；义务粒度、图像占位、工具截断与 LLM 语义判断仍需人工校准。原消息编号从 0 开始。</p></section></main><footer>{esc(stats['generated_at'])} · 独立离线 HTML · 数据内嵌</footer>"""
    document = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>轨迹校验动作分析报告</title><style>{CSS}</style></head><body>{body}<script id="data" type="application/json">{payload}</script><script>{JS}</script></body></html>'
    atomic_lines(Path(path), [document])
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise ScoringError("HTML 报告未成功写出")
