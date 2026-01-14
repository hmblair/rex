# TODO

## GPU-specific option filtering

Currently `prefer` and `constraint` from `.rex.toml` are only applied when using a GPU partition. This is a workaround - ideally we should detect GPU-specific values (e.g., containing "GPU", "gpu", "cuda") and only filter those, allowing non-GPU preferences/constraints to work on CPU partitions too.
