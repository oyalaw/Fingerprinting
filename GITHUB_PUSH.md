# Push this project to GitHub

Before pushing, confirm that datasets, packet captures, generated experiment
results, model artifacts, and private keys are not staged. The included
`.gitignore` excludes the common large and sensitive outputs.

## Option 1: GitHub CLI

From the project directory:

```bash
git init
git add .
git commit -m "Initial AI fingerprinting experiment framework"
git branch -M main

gh auth login
gh repo create AI-Fingerprinting \
  --private \
  --source=. \
  --remote=origin \
  --push
```

Change `--private` to `--public` only if you intend the research code to be
public.

## Option 2: Existing repository on GitHub

Create an empty repository on GitHub first, then from this project directory:

```bash
git init
git add .
git commit -m "Initial AI fingerprinting experiment framework"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/AI-Fingerprinting.git
git push -u origin main
```

If `origin` already exists, inspect it:

```bash
git remote -v
```

and update it if necessary:

```bash
git remote set-url origin git@github.com:YOUR_USERNAME/AI-Fingerprinting.git
git push -u origin main
```

## Routine updates

After editing the code:

```bash
git status
git add .
git commit -m "Describe the experiment update"
git push
```

## Check what will be published

Before every commit:

```bash
git status
git diff --cached
```

Do not commit:

```text
datasets/
captures/
experiments/
*.pcap
*.pcapng
*.pt
*.pth
*.onnx
*.engine
*.tflite
server.key
private credentials
API tokens
```
