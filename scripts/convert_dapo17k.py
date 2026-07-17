#!/usr/bin/env python3
"""DAPO-Math-17k → vime prompt-data jsonl(math 管线验证用)。

vime 需要 {prompt, label, metadata}:
  - prompt : 喂 agent 的题面(**不含答案**)
  - label  : ground-truth 答案(整数),judge 侧比对用;**agent 拿不到**(vime 只把 input-key=prompt 发给 agent)
  - metadata.answer : 同 label,冗余存一份便于 staging

用法:
  python3 convert_dapo17k.py <dapo_raw.jsonl|parquet-导出的 jsonl> <out.jsonl>

DAPO 原始 schema 兼容多种:直接 {question, answer};或 verl 风格
{prompt:[{role,content}], reward_model:{ground_truth}}。下面都 handle。
"""
from __future__ import annotations

import json
import sys


def _extract_prompt(rec: dict) -> str | None:
    # 直接字段
    for k in ("question", "prompt", "problem", "query"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # verl 风格:prompt = [{"role":"user","content":...}]
    p = rec.get("prompt")
    if isinstance(p, list):
        for m in reversed(p):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return c.strip()
    return None


def _extract_answer(rec: dict) -> str | None:
    for k in ("answer", "ground_truth", "solution", "final_answer"):
        v = rec.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    rm = rec.get("reward_model")
    if isinstance(rm, dict):
        v = rm.get("ground_truth")
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    n_ok = n_skip = 0
    with open(src) as fin, open(dst, "w") as fout:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            prompt = _extract_prompt(rec)
            answer = _extract_answer(rec)
            if prompt is None or answer is None:
                n_skip += 1
                continue
            fout.write(json.dumps({
                "prompt": prompt,                               # agent 可见,不含答案
                "label": answer,                                # judge 侧,agent 不可见
                "metadata": {"source": "dapo17k", "answer": answer},
            }, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[convert_dapo17k] wrote {n_ok} rows, skipped {n_skip} (missing prompt/answer) → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
