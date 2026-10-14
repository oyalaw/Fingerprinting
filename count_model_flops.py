#!/usr/bin/env python3
"""
Count parameters and training FLOPs per sample for each model variant, using
torch.utils.flop_counter (built into PyTorch >= 2.1; you have 2.5.1).

Usage (run from your Fingerprinting repo root, so your model code imports):
    python3 count_model_flops.py run_metadata.csv \
        --factory ai_fingerprint.workloads.base:build_workload \
        --runs-root collected_experiments --out model_flops.csv

--factory is called as factory(cfg) with the run's parsed config.yaml. The returned object
(e.g. a Workload) is searched for the nn.Module. Input shape is found by trying plausible
batches until the model accepts one; the shape used is written to the input_shape column.
"""
import argparse, csv, importlib, os
import torch
from torch.utils.flop_counter import FlopCounterMode


def _find_module(obj, depth=0):
    """Return the torch nn.Module held by obj (obj itself, or an attribute like .model)."""
    if isinstance(obj, torch.nn.Module):
        return obj
    if depth > 2 or obj is None:
        return None
    for name in ("model", "net", "network", "module", "_model", "_net", "torch_model"):
        m = getattr(obj, name, None)
        if isinstance(m, torch.nn.Module):
            return m
    for name, val in vars(obj).items() if hasattr(obj, "__dict__") else []:
        if isinstance(val, torch.nn.Module):
            return val
    for name, val in vars(obj).items() if hasattr(obj, "__dict__") else []:
        if not name.startswith("__") and hasattr(val, "__dict__") and not isinstance(val, type):
            m = _find_module(val, depth + 1)
            if m is not None:
                return m
    return None


def build(factory, cfg):
    """Call your factory (e.g. build_workload(config)) and dig out the nn.Module.
    If the workload builds its model lazily, common setup methods are tried."""
    obj = factory(cfg)
    m = _find_module(obj)
    if m is None:
        for meth in ("setup", "prepare", "build", "build_model", "load_model", "initialize", "init_model"):
            f = getattr(obj, meth, None)
            if callable(f):
                try:
                    r = f()
                    m = _find_module(r) or _find_module(obj)
                except TypeError:
                    continue
                if m is not None:
                    break
    if m is None:
        attrs = [k for k in vars(obj)] if hasattr(obj, "__dict__") else dir(obj)
        raise RuntimeError(f"No nn.Module found on {type(obj).__name__}; attributes: {attrs}")
    return m


def candidate_inputs(cfg, batch=1):
    """Plausible input batches, most likely first. The first one the model accepts is used,
    and its shape is recorded in the output so you can sanity-check it."""
    ai = cfg.get("ai", {}) or {}
    fam, app = ai.get("family", ""), ai.get("application", "")
    seq = int(ai.get("sequence_length") or 128)
    dim = int(ai.get("input_dim") or 9)
    vocab = int(ai.get("vocab_size") or 10000)
    s = int(ai.get("input_size") or 224)
    tokens = lambda: torch.randint(0, vocab, (batch, seq))
    seqfeat = lambda: torch.randn(batch, seq, dim)
    img3 = lambda n=s: torch.randn(batch, 3, n, n)
    img1 = lambda n=28: torch.randn(batch, 1, n, n)
    flat = lambda n: torch.randn(batch, n)
    if fam == "transformer" or "text" in app or "language" in app:
        order = [tokens, seqfeat, img3]
    elif fam == "rnn":
        order = [seqfeat, tokens, lambda: torch.randn(batch, 128, dim)]
    elif fam == "cnn":
        order = [img3, lambda: img3(32), img1]
    else:  # autoencoders and anything else
        order = [img1, lambda: img3(32), img3, lambda: img1(s), lambda: flat(784), lambda: flat(dim),
                 lambda: flat(3 * 32 * 32), seqfeat]
    return order


def make_input_for(model, cfg):
    model.eval()
    errs = []
    for mk in candidate_inputs(cfg):
        x = mk()
        try:
            with torch.no_grad():
                model(x)
            return x
        except Exception as e:
            errs.append(f"{tuple(x.shape)}:{type(e).__name__}")
    raise RuntimeError("No candidate input worked: " + "; ".join(errs))


def first_output(y):
    if isinstance(y, torch.Tensor):
        return y
    if isinstance(y, (tuple, list)):
        return first_output(y[0])
    if isinstance(y, dict):
        return first_output(next(iter(y.values())))
    if hasattr(y, "logits"):
        return y.logits
    raise TypeError(f"Unrecognised model output type {type(y)}")


def lstm_flops(model, x):
    """Analytic forward FLOPs of nn.LSTM layers. FlopCounterMode has no formula for the fused
    aten.lstm op (nn.GRU decomposes into counted mm/addmm on CPU, nn.LSTM does not), so without
    this an LSTM model's count is only its Linear head. Per step, layer and direction:
    4 gates x 2*h*(in + h) multiply-add FLOPs; backward is taken as 2x forward."""
    seen = []
    def hook(m, inp, out):
        s = inp[0]
        if hasattr(s, "batch_sizes"):  # PackedSequence
            steps = int(s.batch_sizes.sum())
        else:
            steps = s.shape[0] * s.shape[1]  # batch * time, independent of batch_first
        h, dirs = m.hidden_size, 2 if m.bidirectional else 1
        f = 0
        for layer in range(m.num_layers):
            inp_dim = m.input_size if layer == 0 else h * dirs
            f += dirs * 4 * 2 * h * (inp_dim + h)
        seen.append(f * steps)
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.LSTM)]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hs:
            h.remove()
    return sum(seen)


def count(model, x):
    model.train()
    with FlopCounterMode(display=False) as fc:
        y = first_output(model(x))
    fwd = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc2:
        y = first_output(model(x))
        y.float().sum().backward()
    fwd_bwd = fc2.get_total_flops()
    ops = {str(k): v for k, v in fc2.get_flop_counts().get("Global", {}).items()}
    extra = lstm_flops(model, x)
    if extra:
        fwd += extra
        fwd_bwd += 3 * extra
        ops["analytic.lstm"] = 3 * extra
    return fwd, fwd_bwd, ops


def layer_histogram(model):
    hist = {}
    for m in model.modules():
        if len(list(m.children())) == 0:
            n = type(m).__name__
            hist[n] = hist.get(n, 0) + 1
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata_csv")
    ap.add_argument("--factory", required=True, help="module.path:function")
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", default="model_flops.csv")
    a = ap.parse_args()

    import sys, yaml
    sys.path.insert(0, os.getcwd())  # so your repo's packages import when run from the repo root
    mod, fn = a.factory.split(":")
    factory = getattr(importlib.import_module(mod), fn)

    rows, seen = [], {}
    with open(a.metadata_csv) as f:
        for r in csv.DictReader(f):
            key = (r["family"], r["architecture"], r["variant"], r["dataset"])
            if not r.get("family"):
                continue  # unlabeled / incomplete run
            if key in seen:  # one count per distinct model; reuse for repeats
                rows.append({**seen[key], "experiment_id": r["experiment_id"]})
                continue
            run = os.path.join(a.runs_root, r["experiment_id"])
            cfg_path = next((p for p in [os.path.join(run, "server", "config.yaml"),
                                         os.path.join(run, "client_1", "config_effective.yaml")]
                             if os.path.exists(p)), None)
            out = {"experiment_id": r["experiment_id"], "family": key[0], "architecture": key[1],
                   "variant": key[2], "dataset": key[3]}
            try:
                cfg = yaml.safe_load(open(cfg_path))
                torch.manual_seed(0)
                model = build(factory, cfg)
                params = sum(p.numel() for p in model.parameters())
                buffers = sum(b.numel() for b in model.buffers())
                x = make_input_for(model, cfg)
                fwd, fwd_bwd, ops = count(model, x)
                hist = layer_histogram(model)
                out.update({
                    "params": params, "buffers": buffers,
                    "input_shape": "x".join(map(str, x.shape)), "input_dtype": str(x.dtype),
                    "fwd_flops_per_sample": fwd, "train_flops_per_sample": fwd_bwd,
                    "flops_per_param": round(fwd_bwd / max(params, 1), 3),
                    "leaf_module_count": sum(hist.values()),
                    "layer_types": "|".join(f"{k}:{v}" for k, v in sorted(hist.items())),
                    "flop_ops": "|".join(f"{k}:{v}" for k, v in sorted(ops.items())),
                })
            except Exception as e:
                out["error"] = f"{type(e).__name__}: {e}"
            seen[key] = out
            rows.append(out)
            print(out["experiment_id"][-8:], key[2], out.get("train_flops_per_sample", out.get("error")))

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
