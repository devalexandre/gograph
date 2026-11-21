"""QLoRA do descritor de funções Go, dimensionado para uma GPU de 6 GB.

Uso: python train.py [--model Qwen/Qwen2.5-Coder-1.5B-Instruct] [--out out/descritor]

Interromper com Ctrl+C é seguro: rodar de novo com --resume continua do último
checkpoint. O adaptador final fica em <out>/adapter.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class EscritaCallback(TrainerCallback):
    """Mede a tarefa de escrita a cada checkpoint e guarda o melhor adaptador.

    O eval_loss diz pouco sobre escrever código que compila: um modelo pode
    baixar a perda e passar a compilar menos. A cada checkpoint este callback
    gera as funções de validação com o modelo como está, manda o compilador
    conferir (comando --eval-cmd) e grava o resultado em <out>/escrita.jsonl.
    O adaptador com mais funções compilando fica em <out>/melhor.
    """

    def __init__(self, tok, tasks, cmd, out, batch, patience=0):
        self.tok, self.tasks, self.cmd, self.out, self.batch = tok, tasks, cmd, pathlib.Path(out), batch
        self.patience, self.stale = patience, 0
        self.best = -1
        self.log = self.out / "escrita.jsonl"

    def measure(self, model, step):
        from gerar import generate, write_answers

        answers = (self.out / f"validacao-{step}.jsonl").resolve()
        write_answers(answers, generate(model, self.tok, self.tasks, batch=self.batch))
        done = subprocess.run(self.cmd.format(answers=answers), shell=True, capture_output=True, text=True)
        counts = re.findall(r"compilam:\s*(\d+) de (\d+)", done.stdout)
        if done.returncode != 0 or not counts:
            print(f"escrita: a medição falhou no passo {step}:\n{done.stdout[-600:]}{done.stderr[-600:]}", flush=True)
            return None
        built = sum(int(a) for a, _ in counts)
        total = sum(int(b) for _, b in counts)
        best = built > self.best
        self.stale = 0 if best else self.stale + 1
        if best:
            self.best = built
            model.save_pretrained(self.out / "melhor")
            self.tok.save_pretrained(self.out / "melhor")
            (self.out / "melhor" / "gograph.json").write_text(json.dumps({"step": step, "built": built, "total": total}))
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "built": built, "total": total, "best": best}) + "\n")
        print(f"escrita: passo {step}: {built} de {total} compilam{' (melhor até aqui)' if best else ''}", flush=True)
        return built

    def on_save(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero and model is not None:
            self.measure(model, state.global_step)
            if self.patience and self.stale >= self.patience:
                print(f"escrita: {self.stale} medições seguidas sem melhora; parando o treino", flush=True)
                control.should_training_stop = True


class AnswerLossTrainer(Trainer):
    """Calcula os logits só nas posições da resposta.

    O vocabulário tem ~150 mil entradas: projetar a sequência inteira custa
    centenas de MB por exemplo, e o prompt (maior parte dos tokens) nem entra
    na perda. Projetar só a resposta é o que faz o treino caber em 6 GB.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = model.get_base_model()
        hidden = base.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).last_hidden_state
        labels = inputs["labels"][:, 1:]
        keep = labels != -100
        logits = base.lm_head(hidden[:, :-1][keep])
        loss = torch.nn.functional.cross_entropy(logits.float(), labels[keep])
        return (loss, {"logits": None}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out/descritor")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-tokens", type=int, default=1536, help="exemplos maiores são descartados")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16, help="passos acumulados por atualização")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0, help="usa só N exemplos (teste rápido)")
    ap.add_argument("--max-steps", type=int, default=-1, help="para depois de N atualizações (teste rápido)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--train-file", default="train.jsonl", help="arquivo de treino dentro de --data")
    ap.add_argument("--init-adapter", default="", help="continua o treino a partir deste adaptador")
    ap.add_argument("--save-steps", type=int, default=200, help="intervalo entre checkpoints")
    ap.add_argument("--eval-tasks", default="", help="tarefas de validação (gograph bench -dump); mede a escrita a cada checkpoint")
    ap.add_argument("--eval-cmd", default="", help="comando que confere as respostas; {answers} é o arquivo gerado")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=0, help="para depois de N medições de escrita seguidas sem melhora (0 = não para)")
    ap.add_argument("--cpu", action="store_true", help="roda na CPU: só para ensaiar o fluxo, é lento demais para treinar")
    ap.add_argument(
        "--big-gpu",
        action="store_true",
        help="GPU com 16 GB ou mais: modelo sem compressão em 4 bits e sem economia de memória (mais rápido)",
    )
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(ex):
        msgs = ex["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(msgs[-1]["content"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        # A perda conta só sobre a resposta; o prompt entra como contexto.
        return {"input_ids": p + a, "labels": [-100] * len(p) + a}

    data = load_dataset(
        "json",
        data_files={"train": f"{args.data}/{args.train_file}", "test": f"{args.data}/test.jsonl"},
    )
    if args.max_samples:
        data["train"] = data["train"].select(range(min(args.max_samples, len(data["train"]))))
    data["test"] = data["test"].select(range(min(150, len(data["test"]))))
    data = data.map(encode, remove_columns=data["train"].column_names)
    before = len(data["train"])
    data = data.filter(lambda ex: len(ex["input_ids"]) <= args.max_tokens)
    print(f"treino: {len(data['train'])} exemplos ({before - len(data['train'])} acima de {args.max_tokens} tokens)")

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id)
        labels = torch.full((len(batch), width), -100)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            labels[i, :n] = torch.tensor(b["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    if args.cpu:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, device_map={"": "cpu"})
    elif args.big_gpu:
        # Com memória de sobra o modelo entra inteiro, em bf16: é o mesmo
        # formato em que o adaptador é juntado na exportação.
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.rank,
                lora_alpha=2 * args.rank,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
    model.print_trainable_parameters()
    model.config.use_cache = False

    callbacks = []
    if args.eval_tasks:
        if not args.eval_cmd:
            raise SystemExit("--eval-tasks precisa de --eval-cmd")
        tasks = [json.loads(line) for line in open(args.eval_tasks, encoding="utf-8") if line.strip()]
        pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
        callbacks.append(EscritaCallback(tok, tasks, args.eval_cmd, args.out, args.eval_batch, args.patience))
        print(f"validação de escrita: {len(tasks)} funções a cada {args.save_steps} atualizações")

    trainer = AnswerLossTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=collate,
        callbacks=callbacks,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=0.03,  # fração do total de passos
            fp16=not args.big_gpu and not args.cpu,
            bf16=args.big_gpu and not args.cpu,
            use_cpu=args.cpu,
            optim="adamw_torch" if args.big_gpu or args.cpu else "paged_adamw_8bit",
                        logging_steps=10,
            eval_strategy="steps",
            eval_steps=400,
            save_steps=args.save_steps,
            save_total_limit=3,
            report_to="none",
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    if torch.cuda.is_available():
        print(f"pico de memória da GPU: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    # O fim do treino quase nunca cai num checkpoint: mede o modelo final também.
    for cb in callbacks:
        if trainer.state.global_step % args.save_steps != 0:
            cb.measure(model, trainer.state.global_step)
        print(f"escrita: melhor resultado na validação: {cb.best} funções compilando, adaptador em {cb.out / 'melhor'}")

    adapter = pathlib.Path(args.out) / "adapter"
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    (adapter / "gograph.json").write_text(json.dumps({"base": args.model}, indent=2))
    print(f"adaptador salvo em {adapter}")


if __name__ == "__main__":
    main()
