"""Produce "golden" reference data from HuggingFace transformers.

The Rust engine is checked against these files tensor-by-tensor, so every
name saved here is a hook point the engine must also emit:

    embed.out
    layer.{i}.attn_norm.out
    layer.{i}.attn.{q,k,v}          projections, before RoPE   [T, heads*head_dim]
    layer.{i}.attn.{q,k}_rope       after RoPE                  [T, heads*head_dim]
    layer.{i}.attn.probs            softmax(QK^T/sqrt(d))       [H, T, S]
    layer.{i}.attn.out              heads merged, before o_proj [T, hidden]
    layer.{i}.attn.o_proj
    layer.{i}.resid.post_attn       residual stream after attention
    layer.{i}.mlp_norm.out
    layer.{i}.mlp.{gate,up,act,hidden,down}   act = silu(gate), hidden = act*up
    layer.{i}.resid.post_mlp        residual stream leaving the layer
    final_norm.out
    logits                                                      [T, vocab]

Usage:  .venv/bin/python reference/dump_golden.py
"""

import json
import os

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "smollm2-135m")
OUT = os.path.join(ROOT, "golden")

PROMPT = "The capital of France is Paris. The capital of Germany is"
GEN_PROMPT = "Once upon a time, in a small village,"
GEN_TOKENS = 32

TOKENIZER_CASES = [
    "Hello world",
    "The capital of France is Paris. The capital of Germany is",
    "  leading spaces and trailing   ",
    "numbers: 12345 and 3.14159, year 2026!",
    "tabs\tand\nnewlines\n\n  indented",
    "contractions: don't, I'm, we've, they'll, she'd, it's",
    "UPPER lower MiXeD",
    "unicode: café naïve 日本語 Ελληνικά emoji 🤖🚀",
    "code: fn main() { println!(\"hi\"); } // x+=1;",
    "<|im_start|>user\nhi<|im_end|>\n<|endoftext|>",
    "fractions ½ and arabic digits ٣٤",
    "",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))

    # ---- tokenizer cases -------------------------------------------------
    cases = [{"text": t, "ids": tok.encode(t).ids} for t in TOKENIZER_CASES]
    with open(os.path.join(OUT, "tokenizer_cases.json"), "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)

    # ---- full activation trace for PROMPT -------------------------------
    ids = tok.encode(PROMPT).ids
    input_ids = torch.tensor([ids])
    T = len(ids)

    acts = {}

    def save(name, t):
        if isinstance(t, tuple):
            t = t[0]
        acts[name] = t.detach()[0].contiguous().clone()  # drop batch dim

    def out_hook(name):
        return lambda mod, inp, out: save(name, out)

    def in_hook(name):
        return lambda mod, inp: save(name, inp[0])

    m = model.model
    hs = [m.embed_tokens.register_forward_hook(out_hook("embed.out"))]
    for i, layer in enumerate(m.layers):
        p = f"layer.{i}."
        a, mlp = layer.self_attn, layer.mlp
        hs += [
            layer.input_layernorm.register_forward_hook(out_hook(p + "attn_norm.out")),
            a.q_proj.register_forward_hook(out_hook(p + "attn.q")),
            a.k_proj.register_forward_hook(out_hook(p + "attn.k")),
            a.v_proj.register_forward_hook(out_hook(p + "attn.v")),
            a.o_proj.register_forward_pre_hook(in_hook(p + "attn.out")),
            a.o_proj.register_forward_hook(out_hook(p + "attn.o_proj")),
            layer.post_attention_layernorm.register_forward_pre_hook(in_hook(p + "resid.post_attn")),
            layer.post_attention_layernorm.register_forward_hook(out_hook(p + "mlp_norm.out")),
            mlp.gate_proj.register_forward_hook(out_hook(p + "mlp.gate")),
            mlp.up_proj.register_forward_hook(out_hook(p + "mlp.up")),
            mlp.act_fn.register_forward_hook(out_hook(p + "mlp.act")),
            mlp.down_proj.register_forward_pre_hook(in_hook(p + "mlp.hidden")),
            mlp.down_proj.register_forward_hook(out_hook(p + "mlp.down")),
            layer.register_forward_hook(out_hook(p + "resid.post_mlp")),
        ]
    hs.append(m.norm.register_forward_hook(out_hook("final_norm.out")))

    with torch.no_grad():
        out = model(input_ids, output_attentions=True, use_cache=False)
    for h in hs:
        h.remove()

    acts["logits"] = out.logits[0].contiguous()
    cfg = model.config
    H, KVH = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // H
    pos = torch.arange(T)[None]
    for i in range(cfg.num_hidden_layers):
        p = f"layer.{i}."
        acts[p + "attn.probs"] = out.attentions[i][0].contiguous()
        # RoPE isn't a module, so recompute it exactly the way HF does.
        q = acts[p + "attn.q"].view(1, T, H, D).transpose(1, 2)
        k = acts[p + "attn.k"].view(1, T, KVH, D).transpose(1, 2)
        cos, sin = m.rotary_emb(q, pos)
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        acts[p + "attn.q_rope"] = qr[0].transpose(0, 1).reshape(T, H * D).contiguous()
        acts[p + "attn.k_rope"] = kr[0].transpose(0, 1).reshape(T, KVH * D).contiguous()

    save_file(acts, os.path.join(OUT, "trace.safetensors"))

    # ---- greedy generation ----------------------------------------------
    gen_ids = tok.encode(GEN_PROMPT).ids
    with torch.no_grad():
        g = model.generate(
            torch.tensor([gen_ids]), max_new_tokens=GEN_TOKENS, do_sample=False
        )
    new = g[0, len(gen_ids):].tolist()

    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(
            {
                "prompt": PROMPT,
                "prompt_ids": ids,
                "gen_prompt": GEN_PROMPT,
                "gen_prompt_ids": gen_ids,
                "gen_ids": new,
                "gen_text": tok.decode(new),
            },
            f,
            ensure_ascii=False,
            indent=1,
        )

    # ---- chat template + greedy reply with the instruct model ----------
    from transformers import AutoTokenizer

    inst_dir = os.path.join(ROOT, "models", "smollm2-135m-instruct")
    inst_tok = AutoTokenizer.from_pretrained(inst_dir)
    inst = AutoModelForCausalLM.from_pretrained(inst_dir, dtype=torch.float32)
    chat = []
    for q in ["What is the capital of France?", "Write a haiku about rust."]:
        msgs = [{"role": "user", "content": q}]
        text = inst_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        chat_ids = inst_tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            g = inst.generate(torch.tensor([chat_ids]), max_new_tokens=48, do_sample=False)
        reply = g[0, len(chat_ids):].tolist()
        if reply and reply[-1] == inst.config.eos_token_id:
            reply = reply[:-1]
        chat.append({"user": q, "text": text, "ids": chat_ids, "reply_ids": reply,
                     "reply": inst_tok.decode(reply)})
        print("chat:", q, "->", repr(inst_tok.decode(reply)))
    with open(os.path.join(OUT, "chat.json"), "w") as f:
        json.dump(chat, f, ensure_ascii=False, indent=1)

    print(f"prompt tokens: {T}, saved {len(acts)} tensors to golden/trace.safetensors")
    print("greedy continuation:", repr(tok.decode(new)))


if __name__ == "__main__":
    main()
