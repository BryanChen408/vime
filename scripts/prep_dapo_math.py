#!/usr/bin/env python3
"""DAPO-17k → operator_samples 格式:保留 prompt(消息列表,vime Dataset 要 list),
补 metadata.op_name(operator_samples 硬需)+ metadata.answer(judge-only,给 math_judge)。

用法: python3 prep_dapo_math.py <dapo_raw.jsonl> <out.jsonl>
"""
from __future__ import annotations
import json, sys, re


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__); return 2
    src, dst = sys.argv[1], sys.argv[2]
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for i, ln in enumerate(fin):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            prompt = r.get("prompt")            # 保持原样(消息列表)
            label = r.get("label") or r.get("answer")
            if prompt is None or label is None:
                continue
            md = dict(r.get("metadata") or {})
            # op_name 必须是安全文件名(无 / \),operator_samples 用它做 task 标识
            md["op_name"] = re.sub(r"[^0-9A-Za-z_]", "_", f"math_{i:06d}")
            md["answer"] = str(label).strip()   # judge-only,不进 agent prompt
            fout.write(json.dumps({
                "prompt": prompt,
                "label": str(label).strip(),
                "metadata": md,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[prep_dapo_math] wrote {n} rows (prompt=list, +metadata.op_name/answer) → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
