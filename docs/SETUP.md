# Testbed setup: what to install on each machine

| Machine | Role | Address | Python for the experiment |
|---|---|---|---|
| DGX Spark | FL server, result collector, time server, analysis | 10.42.0.195 | 3.12 (existing `fingerprinting` conda env) |
| Jetson Orin Nano | client_1 | 10.42.0.210 | 3.10 (JetPack system Python) |
| Dell desktop (Ubuntu) | client_2 | 10.42.0.47 | **3.12 environment** (not the system 3.14) |
| Dell laptop (Windows 11) | client_3 | 10.42.0.145 | **3.12 environment** (not 3.14) |
| Proxy laptop (Ubuntu) | hotspot + capture | 10.42.0.1 | system Python 3.12 |

Why 3.12 on the desktop and Windows laptop: Flower 1.13.1 needs grpcio <= 1.64.3, which has no
packages for Python 3.13/3.14, and one Python minor version across the PCs keeps runs comparable.
The Jetson stays on its JetPack Python (3.10) because NVIDIA's CUDA PyTorch wheels target it.

Requirement files (in `requirements/`):

| File | Contents | Install on |
|---|---|---|
| `core.txt` | numpy, PyYAML, Pillow, psutil, scikit-learn | every machine |
| `fl.txt` | core + torch, torchvision, transformers, datasets | DGX, all clients |
| `frameworks.txt` | flwr 1.13.1, nvflare 2.6.3 | DGX and clients, if Flower/FLARE are used |
| `proxy.txt` | core | proxy laptop |
| `analysis.txt` | fl + pandas, matplotlib, pytest | DGX |

Install PyTorch first with the platform command below, then the requirement files, so pip does
not replace the CUDA build with a CPU one.

---

## 1. Every machine

```bash
git pull                                  # branch round-feature-analysis
python3 main.py identity                  # once: role, client ID, device, OS (menus)
```

Copy `tls/ca.pem` from the DGX into `tls/` in the repository (the DGX keeps all of `tls/`).

## 2. DGX (server)

```bash
conda activate fingerprinting                                 # Python 3.12, torch 2.5.1
pip install -r requirements/analysis.txt -r requirements/frameworks.txt
sudo bash scripts/timesync/dgx_time_server.sh                 # time server for 10.42.0.0/24
python3 result_collector.py                                   # result collector, port 8090
```

Firewall, if enabled: allow 8080, 8081, 8090 (TCP) and 123 (UDP) from 10.42.0.0/24.

## 3. Jetson Orin Nano (client_1)

```bash
sudo apt install -y python3-pip python3-venv
python3 -m venv --system-site-packages ~/aifp && source ~/aifp/bin/activate
# PyTorch for JetPack 6 (CUDA build) from NVIDIA's Jetson index; match your JetPack/L4T release
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
pip install -r requirements/fl.txt -r requirements/frameworks.txt
sudo bash scripts/timesync/linux_client.sh
sudo nvpmodel -m 0 && sudo apt-mark hold 'nvidia-l4t-*'      # fixed power mode; protect L4T
```

If the previous environment is wanted instead, it is in the backup on the DGX
(`/home/dgx-1/jetson_orin_nano_backup_20261015/home_anik/miniconda3`).
Check: `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints True.

## 4. Dell desktop, Ubuntu (client_2)

```bash
conda create -y -n aifp python=3.12 && conda activate aifp     # or: python3.12 -m venv ~/aifp
# PyTorch: pick the command for your GPU/CUDA at https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # example: CUDA 12.4
pip install -r requirements/fl.txt -r requirements/frameworks.txt
sudo bash scripts/timesync/linux_client.sh
sudo snap refresh --hold && sudo systemctl stop unattended-upgrades
```

## 5. Dell laptop, Windows 11 (client_3)

Install Python 3.12 (python.org, tick "Add to PATH") or Miniconda, then in PowerShell:

```powershell
py -3.12 -m venv $HOME\aifp; & $HOME\aifp\Scripts\Activate.ps1
# PyTorch: pick the Windows command at https://pytorch.org/get-started/locally/ (CUDA or CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # example
pip install -r requirements\fl.txt -r requirements\frameworks.txt
```

As Administrator: `powershell -ExecutionPolicy Bypass -File scripts\timesync\windows_client.ps1`.
Also: pause Windows Update, turn off OneDrive sync, AC power, sleep off, Wi-Fi adapter power
saving off (Device Manager > adapter > Power Management).

## 6. Proxy laptop (hotspot + capture)

```bash
sudo apt install -y tshark ethtool
sudo dpkg-reconfigure wireshark-common         # answer Yes: non-superusers may capture
sudo usermod -aG wireshark $USER               # then log out and in again
pip install -r requirements/proxy.txt
sudo bash scripts/install_offload_sudoers.sh wlo1     # lets the proxy disable GRO/GSO/TSO
sudo bash scripts/timesync/linux_client.sh
```

In the proxy config set `capture.extract_after: false`. Keep at least 60 GB free (largest run
at 50 rounds is about 24 GB of pcap with 3 clients).

---

## 7. Check everything (each client, then the server)

```bash
python3 -c "import numpy, yaml, PIL, psutil, sklearn, torch, torchvision, transformers, datasets; print('ok', torch.__version__)"
python3 -c "import flwr, nvflare; print('flwr', flwr.__version__, 'nvflare', nvflare.__version__)"   # if used
python3 main.py identity --show
timedatectl timesync-status | grep -E "Server|Offset"                                  # Linux clients
```

Then run the pilot (5-round smoke test per framework, then the validation run in
`docs/validation_capture.md`) before planning the campaign.
