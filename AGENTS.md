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

The `environment.yml` pip section automatically overrides the conda-forge
`python-elf` with the local patched copy at `libs/elf@b58e4c83/`.  The
conda-forge 0.7.4 release has a bug in `blockwise_mc_impl` where the reduced
graph is sized from edge endpoints only, so isolated nodes (e.g. the phantom
node 0 that arises from vigra's 1-indexed labels) trigger an `IndexError`.
The local copy fixes this by sizing from `new_labels.max()` instead.

## Tests

```bash
pytest tests/
```

## Demo (end-to-end on synthetic data)

```bash
python run_demo.py          # writes output to /tmp/blockwise_mc_demo/
```
