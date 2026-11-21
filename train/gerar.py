"""Gera respostas do modelo para as tarefas do medidor, sem passar pelo Ollama.

Uso: python gerar.py --tasks tarefas.jsonl --out respostas.jsonl [--adapter out/x/adapter]

As tarefas vêm de "gograph bench -dump" (campos hash, system, prompt) e as
respostas vão para "gograph bench -answers" (campos hash, answer). Sem
--adapter mede o modelo base.

Com --fix-from <resultado.jsonl> faz a segunda tentativa: para cada função
que não compilou, pede a correção com o erro do compilador (campos fix_system
e fix_prompt que o medidor gravou); as que já compilavam mantêm a resposta.
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_NEW_TOKENS = 1200  # mesmo limite do medidor no Ollama
MAX_PROMPT_TOKENS = 7500  # comporta --budget=24000 chars (~5150 tokens médios)


def load_tasks(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def fix_tasks(path):
    """Tarefas da segunda tentativa e as respostas que já compilavam."""
    tasks, kept = [], {}
    for r in load_tasks(path):
        if r.get("built") or not r.get("fix_prompt"):
            kept[r["hash"]] = (r.get("answer", ""), r.get("elapsed_ms", 0))
        else:
            tasks.append({"hash": r["hash"], "system": r["fix_system"], "prompt": r["fix_prompt"], "prev_ms": r.get("elapsed_ms", 0)})
    return tasks, kept


@torch.no_grad()
def generate(model, tok, tasks, batch=8, max_new_tokens=MAX_NEW_TOKENS, progress=None):
    """Devolve {hash: (resposta, elapsed_ms)}. Geração gulosa, como o medidor (temperatura 0)."""
    import time
    was_training = model.training
    model.eval()
    side, tok.padding_side = tok.padding_side, "left"  # geração em lote alinha pelo fim
    prompts = []
    for t in tasks:
        msgs = [{"role": "system", "content": t["system"]}, {"role": "user", "content": t["prompt"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    # Lotes de tamanho parecido desperdiçam menos preenchimento.
    order = sorted(range(len(tasks)), key=lambda i: len(prompts[i]))
    answers = {}
    try:
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            t0 = time.time()
            enc = tok(
                [prompts[i] for i in idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_TOKENS,
                add_special_tokens=False,
            ).to(model.device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
            )
            batch_ms = int((time.time() - t0) * 1000 / max(len(idx), 1))
            for row, i in zip(out, idx):
                text = tok.decode(row[enc["input_ids"].shape[1] :], skip_special_tokens=True)
                prev_ms = tasks[i].get("prev_ms", 0)
                answers[tasks[i]["hash"]] = (text.strip(), prev_ms + batch_ms)
            if progress:
                progress(min(start + batch, len(order)), len(order))
    finally:
        tok.padding_side = side
        if was_training:
            model.train()
    return answers


def load_model(adapter="", cpu=False):
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if cpu:
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, device_map={"": "cpu"})
    else:
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


def write_answers(path, answers):
    with open(path, "w", encoding="utf-8") as f:
        for h, val in answers.items():
            if isinstance(val, tuple):
                a, ms = val
            else:
                a, ms = val, 0
            f.write(json.dumps({"hash": h, "answer": a, "elapsed_ms": ms}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="", help="JSONL de gograph bench -dump")
    ap.add_argument("--fix-from", default="", help="JSONL de resultado do medidor: gera a segunda tentativa")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="", help="adaptador LoRA; vazio = modelo base")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--limit", type=int, default=0, help="só as N primeiras tarefas (ensaio)")
    ap.add_argument("--cpu", action="store_true", help="roda na CPU (só para ensaio, é lento)")
    args = ap.parse_args()

    kept = {}
    if args.fix_from:
        tasks, kept = fix_tasks(args.fix_from)
    elif args.tasks:
        tasks = load_tasks(args.tasks)
    else:
        raise SystemExit("informe --tasks ou --fix-from")
    if args.limit:
        tasks = tasks[: args.limit]

    answers = dict(kept)
    if tasks:
        model, tok = load_model(args.adapter, args.cpu)
        done = generate(
            model,
            tok,
            tasks,
            batch=args.batch,
            max_new_tokens=args.max_new_tokens,
            progress=lambda a, b: print(f"  geradas {a}/{b}", flush=True),
        )
        answers.update(done)
    write_answers(args.out, answers)
    print(f"{len(tasks)} respostas geradas, {len(kept)} mantidas -> {args.out}")


if __name__ == "__main__":
    main()
