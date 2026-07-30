# FlagGems Candidate Staging

`third_party/` is a local staging area for external runtime source. Full
FlagGems trees remain ignored so platform snapshots and generated kernels are
not committed with GraphCast.

The tracked [`flaggems-candidate.lock`](flaggems-candidate.lock) pins one
complete candidate from `flagos-common`. That candidate consists of a fixed
official FlagGems baseline plus only the validated operator-file replacements.
Unchanged operators therefore continue to use the official implementation.
The current lock includes optimized `index`, `index_add`/`index_add_`, Ascend
`addmm`, and Ascend LayerNorm forward; the newer LayerNorm backward work is not
part of this pinned candidate yet.

Prepare the ignored local candidate with:

```bash
FLAGOS_COMMON_ROOT=/path/to/flagos-common \
  bash tools/prepare_flaggems_candidate.sh
```

The default output is:

```text
third_party/flaggems-candidate/FlagGems/
```

Select it without changing the Python environment:

```bash
GRAPHCAST_FLAGGEMS_SOURCE_ROOT="$PWD/third_party/flaggems-candidate/FlagGems" \
GRAPHCAST_FLAGOS_MODE=flaggems \
GRAPHCAST_FLAGGEMS_OPS=index,index_add_,addmm,layer_norm \
  bash run_25km_inference.sh <inputs.nc> <target_template.nc> <forcings.nc>
```

The inference report records both the requested source root and the actual
`flag_gems` module path. Use those fields to distinguish an official install
from the candidate. Operator development and review artifacts stay in
`flagos-common`; GraphCast uses only the public FlagGems runtime API.
