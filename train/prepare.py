"""Filtra corpus/dataset.jsonl e separa treino e teste por repositório.

Uso: python prepare.py [--dataset ../corpus/dataset.jsonl] [--out data]

Grava data/train.jsonl e data/test.jsonl no formato de chat
({"messages": [system, user, assistant]}) e imprime o que cada filtro tirou.
"""

import argparse
import collections
import hashlib
import json
import pathlib
import re

# Prompt curto do modelo treinado. As regras completas (describe.SystemPrompt)
# só o modelo professor precisa ler: o modelo treinado aprende o formato pelos
# exemplos, e um prompt curto corta quase metade dos tokens de cada exemplo.
# O mesmo texto precisa ser usado na inferência (Modelfile do Ollama).
SYSTEM = (
    "Você descreve funções Go para um índice de código. Responda com um objeto "
    "JSON com os campos purpose, calls, effects, errors e concurrency, em "
    "português, falando só do que a função faz localmente."
)

MAX_INPUT_CHARS = 30000  # 1,25 × o orçamento de 24000 chars
LEAK_JACCARD = 0.6  # purpose quase igual ao purpose de uma função chamada
TRIVIAL_LINES = 7
TRIVIAL_SHARE = 0.05  # fração máxima de funções triviais no treino
TEST_SHARE = 0.05
REPLAY_SHARE = 0.3  # descrições na mistura, em proporção aos exemplos de escrita

STOP = set(
    "a o e de da do das dos em no na nos nas um uma para por com que se ao à "
    "os as ou é como sem sua seu".split()
)
# Itens de errors que só dizem que não há erro (mesma regra de describe.negation).
NEGATION = re.compile(r"(?i)^(não (devolve|retorna|há|gera|produz)|nenhum|sem) erro")
CALLEE = re.compile(r"^- \S+ \[\w+\]: (.+)$", re.M)


def words(text):
    return {w for w in re.findall(r"\w+", text.lower()) if w not in STOP and len(w) > 2}


def leak(purpose, user_input):
    p = words(purpose)
    best = 0.0
    for m in CALLEE.finditer(user_input):
        c = words(m.group(1))
        if p and c:
            best = max(best, len(p & c) / len(p | c))
    return best


def code_lines(user_input):
    code = user_input.split("```go\n", 1)[1].split("\n```", 1)[0]
    return code.count("\n") + 1


def stable(key):
    return hashlib.sha256(key.encode()).hexdigest()


def load(path, test_repos):
    """Lê um JSONL de exemplos (escrita, alteração, correção) no formato de chat."""
    seen, train, test = set(), [], []
    too_long = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["hash"], r.get("kind", ""))
        if key in seen:
            continue
        if len(r["input"]) > MAX_INPUT_CHARS:
            too_long += 1
            continue
        seen.add(key)
        row = {
            "source": r["source"],
            "name": r["name"],
            "hash": r["hash"],
            "kind": r.get("kind", ""),
            "messages": [
                {"role": "system", "content": r["system"]},
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": r["output"]},
            ],
        }
        (test if r["source"] in test_repos else train).append(row)
    print(f"carregado   {pathlib.Path(path).name:16s}  treino={len(train):6d}  teste={len(test):5d}  descartados(>{MAX_INPUT_CHARS} chars)={too_long:5d}")
    return train, test


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            r = {k: v for k, v in r.items() if k != "kind"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_hashes(path):
    """Funções cujo corpo só usa, do próprio projeto, o que a entrada mostra.

    Vem de "gograph corpus audit". Um exemplo de escrita fora desta lista
    ensina o modelo a adivinhar nomes e campos que ele não vê.
    """
    keep = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["missing"] == 0:
            keep.add(r["hash"])
    return keep


def replay(rows, n, salt):
    """Escolhe n exemplos de uma tarefa já treinada, sempre os mesmos."""
    return sorted(rows, key=lambda r: stable(salt + r["hash"]))[:n]


def shuffled(rows, salt):
    return sorted(rows, key=lambda r: stable(salt + r["hash"] + r["messages"][0]["content"] + r["messages"][1]["content"]))


def mix(args, out_dir, describe_rows, test_repos):
    """Grava as misturas de treino das rodadas seguintes à do descritor.

    train_mix.jsonl: escrita pela assinatura mais uma parte das descrições.
    train_dev.jsonl: os exemplos do professor (escrita por pedido, alteração e
    correção) mais uma parte das duas tarefas anteriores. As tarefas antigas
    entram de novo para o modelo não desaprendê-las enquanto aprende as novas.
    """
    path = pathlib.Path(args.implement)
    if not path.exists():
        return
    describe_train = [r for r in describe_rows if r["source"] not in test_repos]
    train, test = load(path, test_repos)
    old = replay(describe_train, int(len(train) * REPLAY_SHARE), "replay")
    # Mesma ordem da primeira versão deste arquivo: um treino interrompido
    # precisa encontrar os mesmos exemplos ao continuar.
    mixed = sorted(train + old, key=lambda r: stable("mix" + r["hash"] + r["messages"][0]["content"]))
    write(out_dir / "train_mix.jsonl", mixed)
    write(out_dir / "test_implement.jsonl", test)
    print(f"mistura     {len(mixed):6d}  ({len(train)} de escrita + {len(old)} de descrição)")
    print(f"teste escr. {len(test):6d}")

    path = pathlib.Path(args.teach)
    if not path.exists():
        return
    dev, dev_test = load(path, test_repos)
    n = int(len(dev) * REPLAY_SHARE)
    old = replay(train, n, "dev-escrita") + replay(describe_train, n, "dev-descricao")
    write(out_dir / "train_dev.jsonl", shuffled(dev + old, "dev"))
    write(out_dir / "test_dev.jsonl", dev_test)
    kinds = collections.Counter(r["messages"][0]["content"].split(".")[0] for r in dev)
    print(f"desenv.     {len(dev) + len(old):6d}  ({len(dev)} do professor + {len(old)} das tarefas anteriores)")
    for k, v in kinds.most_common():
        print(f"            {v:6d}  {k}")
    print(f"teste des.  {len(dev_test):6d}")

    # Rodada limpa: parte do adaptador de descrições, que nunca viu escrita,
    # então a escrita entra inteira. Os exemplos de escrita (pela assinatura e
    # por pedido) só entram se a auditoria disser que a resposta usa apenas o
    # que a entrada mostra. Alteração e correção ficam todas: nelas a entrada
    # já traz a própria função.
    path = pathlib.Path(args.audit)
    if not path.exists():
        return
    keep = clean_hashes(path)
    write_clean = [r for r in train if r["hash"] in keep]
    dev_clean = [r for r in dev if r["kind"] in ("edit", "fix") or r["hash"] in keep]
    n_write = len(write_clean) + sum(1 for r in dev_clean if r["kind"] == "create")
    old = replay(describe_train, int(n_write * REPLAY_SHARE), "limpo-descricao")
    write(out_dir / "train_limpo.jsonl", shuffled(write_clean + dev_clean + old, "limpo"))
    kinds = collections.Counter(r["kind"] or "assinatura" for r in write_clean + dev_clean)
    print(f"limpo       {len(write_clean) + len(dev_clean) + len(old):6d}  (auditoria aprovou {len(keep)} funções)")
    print(f"            {kinds['assinatura']:6d}  escrita pela assinatura (de {len(train)})")
    print(f"            {kinds['create']:6d}  escrita por pedido (de {sum(1 for r in dev if r['kind'] == 'create')})")
    print(f"            {kinds['edit']:6d}  alteração")
    print(f"            {kinds['fix']:6d}  correção")
    print(f"            {len(old):6d}  descrição (repetição)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../corpus/dataset.jsonl")
    ap.add_argument("--implement", default="../corpus/implement.jsonl")
    ap.add_argument("--teach", default="../corpus/teach.jsonl")
    ap.add_argument("--audit", default="../corpus/audit.jsonl")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    dropped = collections.Counter()
    seen = set()
    rows, trivial = [], []
    for line in open(args.dataset, encoding="utf-8"):
        try:
            r = json.loads(line)
            o = r["output"]
            if isinstance(o, str):
                o = json.loads(o)
            # Ordem fixa dos campos: é a ordem que o modelo vai aprender a gerar.
            out = {
                "purpose": o["purpose"].strip(),
                "calls": [{"fn": c["fn"], "why": c["why"].strip()} for c in o["calls"]],
                "effects": list(o["effects"]),
                "errors": [e for e in o["errors"] if not NEGATION.match(e.strip())],
                "concurrency": o["concurrency"].strip(),
            }
        except (ValueError, KeyError, TypeError, AttributeError):
            dropped["JSON inválido"] += 1
            continue
        if r["hash"] in seen:  # mesmo código em mais de um repositório
            dropped["código repetido"] += 1
            continue
        seen.add(r["hash"])
        if len(r["input"]) > MAX_INPUT_CHARS:
            dropped["entrada longa demais"] += 1
            continue
        if leak(out["purpose"], r["input"]) >= LEAK_JACCARD:
            dropped["purpose copia a função chamada"] += 1
            continue
        row = {
            "source": r["source"],
            "name": r["name"],
            "hash": r["hash"],
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["input"]},
                {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
            ],
        }
        empty = not (out["calls"] or out["effects"] or out["errors"] or out["concurrency"])
        if empty and code_lines(r["input"]) <= TRIVIAL_LINES:
            trivial.append(row)
        else:
            rows.append(row)

    # Triviais (getters e afins): mantém só uma fração, escolhida pelo hash.
    keep = int(len(rows) * TRIVIAL_SHARE / (1 - TRIVIAL_SHARE))
    trivial.sort(key=lambda r: r["hash"])
    dropped["triviais em excesso"] += max(0, len(trivial) - keep)
    rows += trivial[:keep]

    # Teste por repositório: nenhum repositório aparece nos dois lados.
    per_repo = collections.Counter(r["source"] for r in rows)
    test_repos, n_test = set(), 0
    for repo in sorted(per_repo, key=stable):
        if n_test >= len(rows) * TEST_SHARE:
            break
        test_repos.add(repo)
        n_test += per_repo[repo]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: r["hash"])
    counts = collections.Counter()
    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as tr, open(
        out_dir / "test.jsonl", "w", encoding="utf-8"
    ) as te:
        for r in rows:
            side = "test" if r["source"] in test_repos else "train"
            counts[side] += 1
            (te if side == "test" else tr).write(json.dumps(r, ensure_ascii=False) + "\n")

    mix(args, out_dir, rows, test_repos)

    for reason, n in dropped.most_common():
        print(f"descartado  {n:6d}  {reason}")
    print(f"treino      {counts['train']:6d}  em {len(per_repo) - len(test_repos)} repositórios")
    print(f"teste       {counts['test']:6d}  em {len(test_repos)} repositórios: {', '.join(sorted(test_repos))}")


if __name__ == "__main__":
    main()
