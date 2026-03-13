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

1. Modify environment.yml to add:
```
  - pip:
    - --no-deps ./libs/elf@b58e4c83
```

Overriding `python-elf` with the local copy at `libs/elf@b58e4c83/` 
is necessary because claude always seems to pull an old version 
from conda-forge.

2. Then:

```bash
micromamba create -f environment.yml -y
micromamba activate blockwise-mc
```


## Tests

```bash
pytest tests/
```

## Demo (end-to-end on synthetic data)

```bash
python run_demo.py          # writes output to /tmp/blockwise_mc_demo/
```
