# snakebar

A tqdm-like progress bar that fills your terminal with a one-character-thick snake along a random space-filling curve. Based on https://observablehq.com/@esperanc/random-space-filling-curves

## Installation
```bash
pip install snakebar
```

## Usage

```python
from snakebar import snake_tqdm
for i in snake_tqdm(range(100)):
    # your code here
    pass
```

```python
from snakebar import SnakeTQDM
with SnakeTQDM(total=100) as pbar:
    for i in range(100):
        # your code here
        pbar.update(1)
```