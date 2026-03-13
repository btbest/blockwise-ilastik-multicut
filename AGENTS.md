# Agent setup

## Environment

Install micromamba (Linux x86_64):

```bash
# Preferred (if micro.mamba.pm is reachable):
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Fallback via GitHub releases:
mkdir -p ~/bin
curl -L https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 \
     -o ~/bin/micromamba && chmod +x ~/bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba
```

Create and activate the env (from repo root):

```bash
micromamba create -f environment.yml -y
micromamba activate blockwise-mc
```

If `python-elf` (0.7.4) is not on conda-forge; it can be pip-installed from `libs/elf@b58e4c83/`.

## Tests

```bash
pytest tests/
```

## Demo (end-to-end on synthetic data)

```bash
python run_demo.py          # writes output to /tmp/blockwise_mc_demo/
```
