"""Compara o modelo treinado com o modelo base no conjunto de teste.

Uso: python evaluate.py [--adapter out/descritor/adapter] [-n 100] [--show 5]

Para cada exemplo gera a descrição com e sem o adaptador e confere o formato:
JSON válido com os cinco campos, purpose dentro do limite e calls citando só
funções da lista recebida. Grava as saídas em <adapter>/../eval.jsonl.
"""

import argparse
import json
import pathlib
import re
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

FIELDS = ["purpose", "calls", "effects", "errors", "concurrency"]
CALLEE = re.compile(r"^- (\S+) \[\w+\]", re.M)


def check(text, user_input):
    """Devolve (descrição, motivo da rejeição)."""
    try:
        d = json.loads(text)
    except ValueError:
        return None, "JSON inválido"
    if not isinstance(d, dict) or sorted(d) != sorted(FIELDS):
        return None, "campos errados"
    if not isinstance(d["purpose"], str) or not d["purpose"].strip():
        return None, "purpose vazio"
    if len(d["purpose"].split()) > 25:
        return None, "purpose longo"
    if not isinstance(d["calls"], list) or len(d["calls"]) > 5:
        return None, "calls demais"
    allowed = set(CALLEE.findall(user_input))
    for c in d["calls"]:
        if not isinstance(c, dict) or c.get("fn") not in allowed:
            return None, "calls cita função fora da lista"
    if not all(isinstance(d[k], list) for k in ("effects", "errors")) or not isinstance(d["concurrency"], str):
        return None, "tipos errados"
    return d, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="out/descritor/adapter")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    adapter = pathlib.Path(args.adapter)
    base = json.loads((adapter / "gograph.json").read_text())["base"]
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        ),
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    # Espalha a amostra pelos repositórios de teste em vez de pegar os primeiros.
    rows = rows[:: max(1, len(rows) // args.n)][: args.n]

    def generate(msgs):
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=400, do_sample=False)
        return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    stats = {"treinado": {}, "base": {}}
    secs = {"treinado": 0.0, "base": 0.0}
    with open(adapter.parent / "eval.jsonl", "w", encoding="utf-8") as log:
        for i, r in enumerate(rows):
            user = r["messages"][1]["content"]
            rec = {"name": r["name"], "professor": json.loads(r["messages"][2]["content"])}
            for label in ("treinado", "base"):
                start = time.time()
                if label == "base":
                    with model.disable_adapter():
                        text = generate(r["messages"])
                else:
                    text = generate(r["messages"])
                secs[label] += time.time() - start
                d, why = check(text, user)
                stats[label][why or "ok"] = stats[label].get(why or "ok", 0) + 1
                rec[label] = d if d else text
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i < args.show:
                print(f"\n### {r['name']}")
                for k in ("professor", "treinado", "base"):
                    print(f"{k:9s} {json.dumps(rec[k], ensure_ascii=False)[:600]}")

    print()
    for label in ("treinado", "base"):
        ok = stats[label].get("ok", 0)
        rest = {k: v for k, v in stats[label].items() if k != "ok"}
        print(f"{label:9s} válidas {ok}/{len(rows)}  {secs[label] / len(rows):.1f}s por descrição  {rest}")


if __name__ == "__main__":
    main()
