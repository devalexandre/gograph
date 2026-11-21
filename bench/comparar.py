"""Compara medições do gograph bench lado a lado, com segmentação por missing e McNemar pareado.

Uso:
  python3 bench/comparar.py [--dir bench] [--audit-dir bench] base-ctx [base-oracle base-ctx-fix ...]

Para cada função registra:
  - compilou (sim/não)
  - testes passaram (sim/não/sem teste)
  - missing (0 ou >0, segundo bench/audit-<repo>.jsonl em ctx=real, --budget=24000)
  - tempo de inferência (elapsed_ms)

Todas as tabelas mostram:
  - Total (324 funções)
  - Subconjunto missing == 0
  - Subconjunto missing > 0
  - Tempo médio de inferência por função

Quando 2 ou mais modelos são passados, calcula a comparação pareada entre o
primeiro modelo (A / baseline) e cada modelo seguinte (B) — ou pares adjacentes:
  - b = funções que funcionam em A e quebraram em B
  - c = funções que estavam quebradas em A e funcionam em B
  - McNemar com correção de continuidade: chi2 = (|b - c| - 1)^2 / (b + c)
  - p-valor correspondente (1 grau de liberdade, survival function da chi2(1): erfc(sqrt(chi2 / 2)))
  - calculado separadamente para compilação e para testes (apenas funções que TÊM teste nos dois modelos, mesmo denominador)
  - reportado para Total, missing == 0 e missing > 0.
"""

import argparse
import collections
import glob
import json
import math
import os
import sys


def load_audit(audit_dir, repos):
    """Devolve {(repo, hash): missing_count} e {(repo, name): missing_count}."""
    by_hash = {}
    by_name = {}
    for repo in repos:
        path = os.path.join(audit_dir, f"audit-{repo}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            m = int(r.get("missing", 0))
            by_hash[(repo, r["hash"])] = m
            by_name[(repo, r["name"])] = m
    return by_hash, by_name


def load_model(bench_dir, prefix, repos, audit_by_hash, audit_by_name):
    """Devolve {(repo, hash): rec} com campos padronizados."""
    records = {}
    for path in sorted(glob.glob(os.path.join(bench_dir, f"{prefix}-*.jsonl"))):
        repo = os.path.basename(path)[len(prefix) + 1 : -len(".jsonl")]
        if repo not in repos:
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            h = r.get("hash", "")
            name = r.get("name", "")
            missing = audit_by_hash.get((repo, h), audit_by_name.get((repo, name), 0))
            built = bool(r.get("built"))
            covered = bool(r.get("covered"))
            passed = bool(r.get("passed")) if covered else False
            fixed = bool(r.get("fixed"))
            attempt = r.get("attempt", 1 if fixed else (0 if built else -1))
            elapsed_ms = int(r.get("elapsed_ms", 0))
            err_msg = str(r.get("error", ""))
            infra_error = bool(r.get("infra_error")) or ("ollama: 500" in err_msg) or err_msg.startswith("infra:")
            records[(repo, h)] = {
                "repo": repo,
                "name": name,
                "hash": h,
                "built": built,
                "test_status": ("sim" if passed else "não") if covered else "sem teste",
                "covered": covered,
                "passed": passed,
                "fixed": fixed,
                "attempt": attempt,
                "missing": missing,
                "missing_cat": "missing==0" if missing == 0 else "missing>0",
                "elapsed_ms": elapsed_ms,
                "infra_error": infra_error,
            }
    return records


def summarize(records, subset_filter=None):
    rows = [r for r in records.values() if not r.get("infra_error") and (subset_filter is None or subset_filter(r))]
    n = len(rows)
    built = sum(1 for r in rows if r["built"])
    covered = sum(1 for r in rows if r["covered"])
    passed = sum(1 for r in rows if r["passed"])
    fixed = sum(1 for r in rows if r["fixed"])
    ms_list = [r["elapsed_ms"] for r in rows if r["elapsed_ms"] > 0]
    avg_ms = (sum(ms_list) / len(ms_list)) if ms_list else 0.0
    return n, built, covered, passed, fixed, avg_ms


def fmt_cell(stats, width=40):
    n, built, covered, passed, fixed, avg_ms = stats
    if n == 0:
        return f"{'-':>{width}}"
    b_pct = 100.0 * built / n
    t_pct = 100.0 * passed / covered if covered > 0 else 0.0
    text = f"{built}/{n} ({b_pct:4.1f}%) comp | {passed}/{covered} ({t_pct:4.1f}%) test"
    if fixed:
        text += f" [{fixed} fix]"
    if avg_ms > 0:
        text += f" {avg_ms/1000:.2f}s"
    return f"{text:>{width}}"


def mcnemar_cc(b, c):
    """McNemar com correção de continuidade (1 grau de liberdade):
    chi2 = (|b - c| - 1)^2 / (b + c)
    p-valor = P(Chi2_1 >= chi2) = erfc(sqrt(chi2 / 2)).
    """
    if b + c == 0:
        return 0.0, 1.0
    diff = abs(b - c)
    if diff <= 1:
        return 0.0, 1.0
    chi2 = ((diff - 1.0) ** 2) / float(b + c)
    p_val = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p_val


def paired_comparison(rec_a, rec_b, subset_filter=None):
    all_common = sorted(set(rec_a.keys()) & set(rec_b.keys()))
    if subset_filter is not None:
        all_common = [k for k in all_common if subset_filter(rec_a[k])]
    common_keys = [k for k in all_common if not rec_a[k].get("infra_error") and not rec_b[k].get("infra_error")]
    excluded_infra = len(all_common) - len(common_keys)

    # 1. Compilação (todas as funções em common_keys)
    n_comp = len(common_keys)
    a_comp = sum(1 for k in common_keys if rec_a[k]["built"])
    b_comp = sum(1 for k in common_keys if rec_b[k]["built"])
    both_comp = sum(1 for k in common_keys if rec_a[k]["built"] and rec_b[k]["built"])
    b_only_a = sum(1 for k in common_keys if rec_a[k]["built"] and not rec_b[k]["built"])  # b: funcionava em A, quebrou em B
    c_only_b = sum(1 for k in common_keys if not rec_a[k]["built"] and rec_b[k]["built"])  # c: quebrada em A, funciona em B
    chi2_comp, p_comp = mcnemar_cc(b_only_a, c_only_b)

    # 2. Testes: apenas funções que TÊM teste nos dois modelos (mesmo denominador)
    test_keys = [k for k in common_keys if rec_a[k]["covered"] and rec_b[k]["covered"]]
    n_test = len(test_keys)
    a_test = sum(1 for k in test_keys if rec_a[k]["passed"])
    b_test = sum(1 for k in test_keys if rec_b[k]["passed"])
    both_test = sum(1 for k in test_keys if rec_a[k]["passed"] and rec_b[k]["passed"])
    tb_only_a = sum(1 for k in test_keys if rec_a[k]["passed"] and not rec_b[k]["passed"])
    tc_only_b = sum(1 for k in test_keys if not rec_a[k]["passed"] and rec_b[k]["passed"])
    chi2_test, p_test = mcnemar_cc(tb_only_a, tc_only_b)

    return {
        "excluded_infra": excluded_infra,
        "comp": {
            "n": n_comp,
            "a": a_comp,
            "b_tot": b_comp,
            "both": both_comp,
            "b": b_only_a,
            "c": c_only_b,
            "delta": c_only_b - b_only_a,
            "chi2": chi2_comp,
            "p": p_comp,
        },
        "test": {
            "n": n_test,
            "a": a_test,
            "b_tot": b_test,
            "both": both_test,
            "b": tb_only_a,
            "c": tc_only_b,
            "delta": tc_only_b - tb_only_a,
            "chi2": chi2_test,
            "p": p_test,
        },
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=here, help="pasta com os arquivos <prefixo>-<repo>.jsonl")
    ap.add_argument("--audit-dir", default=here, help="pasta com os arquivos audit-<repo>.jsonl")
    ap.add_argument("--pair-base", default="", help="compara todos contra este prefixo (padrão: primeiro prefixo da lista)")
    ap.add_argument("prefixes", nargs="*", default=["base-ctx"])
    args = ap.parse_args()

    repos_dir = os.path.join(here, "..", "corpus", "src")
    if os.path.exists(repos_dir):
        repos = set(os.listdir(repos_dir))
    else:
        repos = {
            "FiloSottile__mkcert",
            "charmbracelet__bubbletea",
            "docker__compose",
            "gorilla__mux",
            "nektos__act",
            "qax-os__excelize",
        }

    audit_by_hash, audit_by_name = load_audit(args.audit_dir, repos)
    prefixes = args.prefixes
    models = {p: load_model(args.dir, p, repos, audit_by_hash, audit_by_name) for p in prefixes}
    active_repos = sorted({r["repo"] for m in models.values() for r in m.values()})
    if active_repos:
        repos = set(active_repos)

    WIDTH = 44
    subsets = [
        ("TOTAL (todas as funções)", None),
        ("SUBCONJUNTO missing == 0", lambda r: r["missing"] == 0),
        ("SUBCONJUNTO missing > 0", lambda r: r["missing"] > 0),
    ]

    for title, sfilter in subsets:
        print(f"\n=== {title} ===")
        print(f"{'repositório':26}" + "".join(f"{p:>{WIDTH}}" for p in prefixes))
        print("-" * (26 + WIDTH * len(prefixes)))
        for repo in sorted(repos):
            row_cells = []
            for p in prefixes:
                repo_Filter = (lambda r, rp=repo: r["repo"] == rp and (sfilter is None or sfilter(r)))
                st = summarize(models[p], repo_Filter)
                row_cells.append(fmt_cell(st, WIDTH))
            print(f"{repo:26}" + "".join(row_cells))
        print("-" * (26 + WIDTH * len(prefixes)))
        tot_cells = []
        for p in prefixes:
            st = summarize(models[p], sfilter)
            tot_cells.append(fmt_cell(st, WIDTH))
        print(f"{'TOTAL':26}" + "".join(tot_cells))

    print("\n=== RESUMO SEGMENTADO POR MODELO ===")
    print(f"{'modelo':18} | {'segmento':14} | {'compilação':20} | {'testes (cobertas)':20} | {'tempo médio/fn':14}")
    print("-" * 98)
    for p in prefixes:
        for seg_name, sfilter in [("Total", None), ("missing == 0", lambda r: r["missing"] == 0), ("missing > 0", lambda r: r["missing"] > 0)]:
            n, built, cov, pas, fix, avg_ms = summarize(models[p], sfilter)
            b_str = f"{built:3d}/{n:<3d} ({100*built/max(n,1):5.1f}%)"
            t_str = f"{pas:3d}/{cov:<3d} ({100*pas/max(cov,1):5.1f}%)"
            ms_str = f"{avg_ms/1000:.2f}s ({avg_ms:.0f}ms)" if avg_ms > 0 else "-"
            print(f"{p:18} | {seg_name:14} | {b_str:20} | {t_str:20} | {ms_str:14}")

    if len(prefixes) >= 2:
        base_p = args.pair_base or prefixes[0]
        if base_p in models:
            print(f"\n=== COMPARAÇÃO PAREADA (McNemar com correção de continuidade) vs '{base_p}' ===")
            for cand_p in prefixes:
                if cand_p == base_p:
                    continue
                print(f"\n--- Par: A = {base_p}  vs  B = {cand_p} ---")
                print(f"{'Segmento':14} | {'Métrica':10} | {'N':4} | {'A':4} | {'B':4} | {'b (A+B-)':8} | {'c (A-B+)':8} | {'Delta(c-b)':10} | {'chi2':7} | {'p-valor':8}")
                print("-" * 102)
                for seg_name, sfilter in [("Total", None), ("missing == 0", lambda r: r["missing"] == 0), ("missing > 0", lambda r: r["missing"] > 0)]:
                    res = paired_comparison(models[base_p], models[cand_p], sfilter)
                    if seg_name == "Total" and res["excluded_infra"] > 0:
                        print(f"[excluídos por erro de infra: {res['excluded_infra']}]")
                    for m_key, m_label in [("comp", "Compilação"), ("test", "Testes")]:
                        m = res[m_key]
                        print(
                            f"{seg_name:14} | {m_label:10} | {m['n']:4d} | {m['a']:4d} | {m['b_tot']:4d} | "
                            f"{m['b']:8d} | {m['c']:8d} | {m['delta']:+10d} | {m['chi2']:7.3f} | {m['p']:8.4f}"
                        )


if __name__ == "__main__":
    main()
