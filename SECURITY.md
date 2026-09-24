# Security

Only load datasets and model checkpoints from sources you trust. This project
uses a restricted loader for the DERCo stimulus pickle files and loads PyTorch
checkpoints with `weights_only=True`.

The repository intentionally excludes datasets, pretrained weights, experiment
outputs, server paths, credentials, and batch scripts that delete or rotate old
outputs. Training writes only beneath the local `outputs/` directory unless you
change the code.

Please report a suspected vulnerability through GitHub's private vulnerability
reporting feature rather than opening a public issue with exploit details.
