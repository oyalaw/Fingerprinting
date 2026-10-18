#!/usr/bin/env python3
"""
Plan a data-collection campaign: repeated, randomized FL runs with ready-to-use role configs.

Every planned run gets one config per role (server, client_1..3) derived from a reference run
in collected_experiments, so device-specific settings (device label, OS, network interface,
hosts, ports, collector) are kept exactly as on the testbed, while the model, dataset, rounds,
seeds, and experiment number are set by the plan.

What the plan enforces
  * repeats      each (variant, dataset) is run --repeats times, with a different data-partition
                 seed per run (weights are already initialized randomly on every run)
  * order        runs are grouped in blocks; each block contains every (variant, dataset) once,
                 shuffled, so families and models are interleaved over days instead of run in
                 sequence
  * datasets     --cross-family-datasets adds a shared dataset across families (RNN on AG News
                 like the transformers; CNN and autoencoder on CIFAR-10), so that family is not
                 confounded with the dataset
  * rounds       --rounds for all models, --rounds-large for models whose payload exceeds
                 --large-mb (large uploads over Wi-Fi make 100 rounds last more than a day);
                 the minimum is FULL_SCALE_MIN_ROUNDS (50) in ai_fingerprint/config.py
  * held out     --holdout-per-family models are drawn (seeded) before any data is collected
                 and written to plan.json, so the test split is fixed in advance
  * numbering    experiment numbers continue after the highest expN already collected for each
                 branch, so existing_output_policy=error never fires on a planned run

Outputs (in --out-dir)
  plan.json       settings, held-out models, reference run
  schedule.csv    one row per run in execution order
  runs/NNN_<variant>_<dataset>_expK/{server,client_1,client_2,client_3}.yaml
  RUNBOOK.md      the commands to start each run on each machine

Usage
  python3 plan_campaign.py --out-dir campaigns/c2 --repeats 3 --cross-family-datasets \\
      --transport tcp
"""
import argparse, copy, csv, glob, json, os, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_fingerprint import registry
from ai_fingerprint.config import FULL_SCALE_MIN_ROUNDS
from ai_fingerprint.dataset_catalog import get_dataset_spec
from ai_fingerprint.experiment_layout import apply_hierarchical_layout, branch_directory

ROLES = ("server", "client_1", "client_2", "client_3")
DATASET_KEYS = ("input_size", "sequence_length", "input_dim", "vocab_size", "max_text_length")
# shared datasets added by --cross-family-datasets: family -> (application, dataset)
CROSS_FAMILY = {"rnn": ("text_classification", "ag_news"),
                "cnn": ("image_classification", "cifar10"),
                "autoencoder": ("anomaly_detection", "cifar10")}
SAME_NET = {"beta_vae": "vae_conv"}   # identical network; planned once


def load_yaml(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_dirs(root):
    """Collected runs directly under root or one folder down (e.g. collected_experiments/old)."""
    found = glob.glob(os.path.join(root, "run_*")) + glob.glob(os.path.join(root, "*", "run_*"))
    return sorted(p for p in found if os.path.isdir(p))


def collected_server_configs(root):
    out = []
    for p in (os.path.join(r, "server", "config.yaml") for r in run_dirs(root)):
        if not os.path.exists(p):
            continue
        try:
            out.append((p, load_yaml(p)))
        except Exception:
            pass
    return out


def default_variants(y_csv):
    """Variants of the existing dataset, plus the registry variants of any architecture that has
    only one variant there (e.g. the added DistilBERT depths), so every architecture has >= 2."""
    import pandas as pd
    Y = pd.read_csv(y_csv, usecols=["architecture", "variant"]).drop_duplicates()
    Y["variant"] = Y["variant"].replace(SAME_NET)
    seen = set(Y["variant"])
    for arch, g in Y.groupby("architecture"):
        if g["variant"].nunique() < 2:
            seen.update(registry.variants_for_architecture(arch))
    return sorted(v for v in seen if v in registry.VARIANTS and v not in SAME_NET)


def profile_for(variant, dataset, application, servers):
    """ai/data/execution block for (variant, dataset): taken from a collected run of the same
    variant (or architecture, or family), with dataset-specific fields from any run on that dataset."""
    spec = registry.VARIANTS[variant]
    fam, arch = spec.family, spec.architecture
    def pick(pred):
        for _, c in servers:
            if pred(c["ai"]):
                return c
        return None
    base = (pick(lambda a: a["variant"] == variant and a["dataset"] == dataset)
            or pick(lambda a: a["variant"] == variant)
            or pick(lambda a: a["architecture"] == arch)
            or pick(lambda a: a["family"] == fam))
    if base is None:
        raise SystemExit(f"no collected run to derive a profile for {variant}")
    ai = copy.deepcopy(base["ai"])
    ai.update({"family": fam, "architecture": arch, "variant": variant,
               "application": application, "dataset": dataset})
    same_ds = pick(lambda a: a["dataset"] == dataset and a["application"] == application)
    if same_ds is not None and ai.get("dataset") != base["ai"]["dataset"]:
        for k in DATASET_KEYS:
            if k in same_ds["ai"]:
                ai[k] = same_ds["ai"][k]
    if ai["dataset"] != base["ai"]["dataset"] or ai["application"] != base["ai"]["application"]:
        n = get_dataset_spec(dataset).num_classes
        if n:
            ai["num_classes"] = n
    return ai, copy.deepcopy(base["execution"])


def used_experiment_numbers(servers):
    used = defaultdict(set)
    for _, c in servers:
        loc = str(c.get("experiment", {}).get("storage_locator") or "").replace("\\", "/")
        if "/exp" in loc:
            branch, exp = loc.rsplit("/", 1)
            if exp[3:].isdigit():
                used[branch].add(int(exp[3:]))
    return used


def round_minutes(collected):
    """Median minutes per FL round for each collected variant (server round_metrics timestamps)."""
    out = {}
    for p in (os.path.join(r, "client_1", "round_metrics.csv") for r in run_dirs(collected)):
        if not os.path.exists(p):
            continue
        try:
            import pandas as pd
            r = pd.read_csv(p, usecols=["variant", "timestamp_start_utc", "timestamp_end_utc"])
            d = (pd.to_datetime(r["timestamp_end_utc"], format="ISO8601")
                 - pd.to_datetime(r["timestamp_start_utc"], format="ISO8601")).dt.total_seconds()
            out.setdefault(r["variant"].iloc[0], []).append(d.median() / 60)
        except Exception:
            pass
    return {k: float(np.median(v)) for k, v in out.items()}


def payload_mb(variant, table):
    return table.get(SAME_NET.get(variant, variant))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--collected", default="collected_experiments")
    ap.add_argument("--reference-run", default="run_20260907t220204z_e367861f",
                    help="collected run whose role configs supply the device-specific settings")
    ap.add_argument("--y", default="fingerprinting_dataset/fingerprinting_Y_ground_truth.csv")
    ap.add_argument("--variants", nargs="*", help="default: all variants of the current dataset's architectures")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cross-family-datasets", action="store_true")
    ap.add_argument("--rounds", type=int, default=FULL_SCALE_MIN_ROUNDS)
    ap.add_argument("--rounds-large", type=int, default=FULL_SCALE_MIN_ROUNDS)
    ap.add_argument("--large-mb", type=float, default=50.0)
    ap.add_argument("--payload-table", default="fingerprinting_results/variant_per_model.csv")
    ap.add_argument("--transport", choices=("tcp", "tls"), default="tls",
                    help="tls (default) uses the testbed certificates in tls/ from `main.py tls-provision`")
    ap.add_argument("--fl-framework", default="custom_fedavg_tcp", choices=registry.fl_frameworks(),
                    help="federated.fl_framework for every run (only implemented ones are accepted)")
    ap.add_argument("--flare-kits", default="flare_kits/aifp/prod_00",
                    help="with --fl-framework nvidia_flare: provisioned prod_* folder, same path on every machine")
    ap.add_argument("--holdout-per-family", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--results-root", default="experiments/results")
    ap.add_argument("--device", nargs="*", default=[], metavar="ROLE=LABEL",
                    help="override a role's device label from the reference run, e.g. server=nvidia_dgx_spark")
    ap.add_argument("--require-clock-sync", action="store_true",
                    help="roles refuse to start if the system clock is not synchronized")
    ap.add_argument("--hosts", nargs="*", default=[], metavar="ROLE=HOSTNAME",
                    help="expected hostname per role, e.g. client_1=jetson-nano; roles refuse to start elsewhere")
    a = ap.parse_args()

    if min(a.rounds, a.rounds_large) < FULL_SCALE_MIN_ROUNDS:
        sys.exit(f"full_scale runs need at least {FULL_SCALE_MIN_ROUNDS} rounds "
                 "(FULL_SCALE_MIN_ROUNDS in ai_fingerprint/config.py)")
    hosts = dict(h.split("=", 1) for h in a.hosts)
    for item in a.device:
        role, label = item.split("=", 1)
        if role not in ROLES or label not in registry.DEVICES:
            sys.exit(f"--device {item}: role must be one of {ROLES} and label one of the registry devices")
    if not registry.fl_framework_implemented(a.fl_framework):
        sys.exit(f"--fl-framework {a.fl_framework} is listed but not implemented yet")
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; choose a new --out-dir")
    servers = collected_server_configs(a.collected)
    ref = next((Path(r) for r in run_dirs(a.collected) if os.path.basename(r) == a.reference_run), None)
    if ref is None:
        sys.exit(f"reference run {a.reference_run} not found under {a.collected}")
    templates = {r: load_yaml(ref / r / "config.yaml") for r in ROLES}
    for item in a.device:
        role, label = item.split("=", 1)
        templates[role]["device"]["label"] = label

    payload = {}
    if os.path.exists(a.payload_table):
        for row in csv.DictReader(open(a.payload_table)):
            payload[row["model"]] = float(row["payload_MB"])

    variants = a.variants or default_variants(a.y)
    variants = [v for v in variants if v not in a.exclude]
    cells = []
    for v in variants:
        spec = registry.VARIANTS[v]
        own = profile_for(v, *_own_dataset(v, servers), servers)
        cells.append((v, own[0]["application"], own[0]["dataset"]))
        if a.cross_family_datasets and spec.family in CROSS_FAMILY:
            app, ds = CROSS_FAMILY[spec.family]
            if ds != own[0]["dataset"] and ds in spec.datasets_by_application.get(app, []):
                cells.append((v, app, ds))

    # held-out models, fixed before collection
    by_family = defaultdict(list)
    for v in variants:
        by_family[registry.VARIANTS[v].family].append(v)
    holdout = {f: sorted(rng.sample(vs, min(a.holdout_per_family, len(vs)))) for f, vs in sorted(by_family.items())}

    # blocks: every cell once per block, shuffled within the block
    order = []
    for rep in range(a.repeats):
        block = cells[:]
        rng.shuffle(block)
        order += [(rep, c) for c in block]

    minutes = round_minutes(a.collected)
    used = used_experiment_numbers(servers)
    next_num = {}
    rows = []
    (out / "runs").mkdir(parents=True)
    for i, (rep, (v, app, ds)) in enumerate(order, 1):
        ai, execution = profile_for(v, ds, app, servers)
        mb = payload_mb(v, payload)
        rounds = a.rounds_large if (mb is not None and mb > a.large_mb) else a.rounds
        part_seed = a.seed * 1000 + i
        probe = copy.deepcopy(templates["server"]); probe["ai"] = ai
        branch = str(branch_directory(a.results_root, probe)).replace("\\", "/")
        rel = branch.split(a.results_root.rstrip("/") + "/", 1)[-1]
        if rel not in next_num:
            next_num[rel] = max(used.get(rel, {0})) + 1
        exp = f"exp{next_num[rel]}"
        next_num[rel] += 1
        run_dir = out / "runs" / f"{i:03d}_{v}_{ds}_{exp}"
        run_dir.mkdir()
        for role in ROLES:
            c = copy.deepcopy(templates[role])
            c["ai"] = copy.deepcopy(ai)
            c["execution"] = copy.deepcopy(execution)
            c["execution"]["seed"] = part_seed
            part = c["data"].setdefault("partition", {})
            part.update({"seed": part_seed, "client_count": 3})
            fed = c["federated"]
            fed.update({"rounds": rounds, "expected_clients": 3, "fl_framework": a.fl_framework,
                        "policy_applied": False, "policy_id": None})
            if a.fl_framework == "nvidia_flare":
                fed["flare"] = {"kits_dir": a.flare_kits}
            c["transport"]["kind"] = a.transport
            if a.transport == "tls":
                c["transport"].update({"certfile": "tls/server.pem", "keyfile": "tls/server.key",
                                       "cafile": "tls/ca.pem", "verify_peer": True,
                                       "server_hostname": "ai-fingerprint-server",
                                       "auto_generate_self_signed": True})
            c["experiment"]["run_id"] = None          # the server generates it at start-up
            c["experiment"]["existing_output_policy"] = "error"
            apply_hierarchical_layout(c, root=a.results_root, experiment_id=exp,
                                      role_token="server" if role == "server" else role)
            c["experiment"]["run_id"] = None
            c["campaign"] = {"order": i, "repeat": rep + 1, "planned_device": c["device"]["label"],
                             "plan": str(out / "plan.json"), "require_clock_sync": a.require_clock_sync}
            if role in hosts:
                c["campaign"]["expected_host"] = hosts[role]
            with open(run_dir / f"{role}.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(c, f, sort_keys=False)
        rows.append({"order": i, "repeat": rep + 1, "fl_framework": a.fl_framework, "family": ai["family"], "architecture": ai["architecture"],
                     "variant": v, "application": app, "dataset": ds, "rounds": rounds,
                     "partition_seed": part_seed, "experiment": exp, "branch": rel,
                     "payload_mb": "" if mb is None else round(mb, 2),
                     "est_hours": "" if v not in minutes else round(minutes[v] * rounds / 60, 1),
                     "held_out": v in holdout.get(ai["family"], []),
                     "config_dir": str(run_dir), "status": "planned"})

    with open(out / "schedule.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump({"seed": a.seed, "repeats": a.repeats, "fl_framework": a.fl_framework, "rounds": a.rounds, "rounds_large": a.rounds_large,
               "large_mb": a.large_mb, "transport": a.transport,
               "cross_family_datasets": a.cross_family_datasets, "reference_run": a.reference_run,
               "devices": {r: templates[r]["device"] for r in ROLES},
               "variants": variants, "held_out": holdout, "runs": len(rows)},
              open(out / "plan.json", "w"), indent=2)
    write_runbook(out, rows, templates)
    hours = sum(r["est_hours"] for r in rows if r["est_hours"] != "")
    unknown = sum(1 for r in rows if r["est_hours"] == "")
    print(f"planned {len(rows)} runs ({len(cells)} variant-dataset cells x {a.repeats} repeats) in {out}")
    print(f"held-out models: {json.dumps(holdout)}")
    print(f"estimated duration: {hours:.0f} h ({hours / 24:.1f} days) for runs with a collected reference"
          f"{f', {unknown} runs without an estimate' if unknown else ''}; see {out}/schedule.csv and {out}/RUNBOOK.md")


def _own_dataset(variant, servers):
    """(dataset, application) the variant was collected with; else the architecture's or family's."""
    spec = registry.VARIANTS[variant]
    for key in ("variant", "architecture", "family"):
        target = getattr(spec, key) if key != "variant" else variant
        for _, c in servers:
            if c["ai"].get(key) == target:
                return c["ai"]["dataset"], c["ai"]["application"]
    sys.exit(f"no collected run for {variant}")


def write_runbook(out, rows, templates):
    dev = {r: templates[r]["device"]["label"] for r in ROLES}
    lines = ["# Campaign runbook", "",
             "Run the rows of `schedule.csv` in order. For each run, copy its folder to every machine",
             "(or share it), then start the server, then the proxy (which waits for the server's run",
             "ID with --wait-for-run, so the same proxy config serves every run), then the clients.", "",
             "```bash",
             "python3 main.py run --config <run>/server.yaml          # " + dev["server"] + "  (start first)",
             "python3 main.py proxy --config <proxy config> --wait-for-run   # proxy host; picks up the run ID",
             "python3 main.py run --config <run>/client_1.yaml        # " + dev["client_1"],
             "python3 main.py run --config <run>/client_2.yaml        # " + dev["client_2"],
             "python3 main.py run --config <run>/client_3.yaml        # " + dev["client_3"],
             "```", "",
             "Set `capture.extract_after: false` in the proxy config, so the proxy only captures and",
             "finishes within minutes of the run ending. After the run is collected, copy the proxy's",
             "pcap chunks to the DGX, extract them there in parallel, and check the run before",
             "starting the next one:", "",
             "```bash",
             "rsync -av --progress <user>@<proxy-host>:<repo>/experiments/staging/<run_id>/proxy/pcap/ \\",
             "      /data/pcaps/<run_id>/",
             "python3 extract_pcaps_parallel.py collected_experiments/<run_id> --pcap-dir /data/pcaps/<run_id>",
             "python3 check_run.py collected_experiments/<run_id> --plan " + str(out / "schedule.csv"),
             "```", "",
             "Once the check passes, delete the run's pcap chunks on the proxy laptop to free its disk.", "",
             "When the campaign is collected, build the dataset (X, Y, the single-client compute",
             "features and the experiment ratios) in one step:", "",
             "```bash",
             "python3 prepare_fingerprinting_dataset.py",
             "```", "",
             "Re-run immediately if the check reports FAIL (give the repeat a new experiment number,",
             "because the failed run's output folders already exist). Each role refuses to start on",
             "the wrong kind of machine (Jetson / Linux / Windows), on a host other than",
             "`campaign.expected_host` when set, or with an unsynchronized clock when",
             "`campaign.require_clock_sync` is true; `AIFP_SKIP_DEVICE_CHECK=1` overrides the check.", "",
             "| order | variant | dataset | rounds | experiment | config folder |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['order']} | {r['variant']} | {r['dataset']} | {r['rounds']} | {r['experiment']} | "
              f"`{r['config_dir']}` |" for r in rows]
    (out / "RUNBOOK.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
