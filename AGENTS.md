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

The `environment.yml` pip section overrides the conda-forge
`python-elf` with the local copy at `libs/elf@b58e4c83/` because 
claude always seems to pull an old version from conda-forge.
The user will pull python-elf from conda-forge and obtain the same version.

## Tests

```bash
pytest tests/
```

## Demo (end-to-end on synthetic data)

```bash
python run_demo.py          # writes output to /tmp/blockwise_mc_demo/
```
