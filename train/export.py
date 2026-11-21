"""Junta o adaptador LoRA ao modelo base e registra o resultado no Ollama.

Uso: python export.py [--adapter out/descritor/adapter] [--name gograph-descritor]

Roda na CPU (não disputa a GPU). Grava o modelo completo em <adapter>/../merged
converte para GGUF (Q8) com o llama.cpp e cria o modelo no Ollama.
"""

import argparse
import json
import pathlib
import subprocess

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare import SYSTEM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="out/descritor/adapter")
    ap.add_argument("--name", default="gograph-descritor")
    ap.add_argument("--quantize", default="q8_0", help="f16, q8_0 (tipos do conversor do llama.cpp)")
    ap.add_argument("--llama-cpp", default="llama.cpp", help="pasta do clone do llama.cpp")
    ap.add_argument("--no-ollama", action="store_true", help="só grava o modelo juntado")
    args = ap.parse_args()

    adapter = pathlib.Path(args.adapter)
    base = json.loads((adapter / "gograph.json").read_text())["base"]
    merged = adapter.parent / "merged"

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(base).save_pretrained(merged)
    print(f"modelo juntado em {merged}")

    if args.no_ollama:
        return

    # O Ollama só quantiza safetensors para o motor MLX; para rodar em GPU
    # NVIDIA o caminho é GGUF, gerado pelo conversor do llama.cpp.
    convert = pathlib.Path(args.llama_cpp) / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit(
            f"conversor não encontrado em {convert}; rode antes:\n"
            "  git clone --depth 1 https://github.com/ggml-org/llama.cpp"
        )
    gguf = merged.parent / f"{args.name}-{args.quantize}.gguf"
    subprocess.run(
        # O conversor precisa do gguf-py do próprio clone; --with o instala só
        # para esta execução, sem mexer no ambiente de treino.
        ["uv", "run", "--with", str(convert.parent / "gguf-py"), "--with", "sentencepiece", "--with", "protobuf",
         "python", str(convert), str(merged), "--outtype", args.quantize, "--outfile", str(gguf)],
        check=True,
    )

    # temperature 0: a descrição precisa ser a mesma a cada execução.
    modelfile = merged.parent / "Modelfile"
    modelfile.write_text(
        f'FROM ./{gguf.name}\nSYSTEM """{SYSTEM}"""\nPARAMETER temperature 0\nPARAMETER num_ctx 4096\n'
    )
    subprocess.run(["ollama", "create", args.name, "-f", "Modelfile"], cwd=merged.parent, check=True)
    print(f"pronto: ollama run {args.name}")

if __name__ == "__main__":
    main()
