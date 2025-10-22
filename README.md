# snakebar

![snakebar](https://github.com/Majoburo/snakebar/blob/main/docs/snaking.gif?raw=true)

A tqdm-like progress bar that fills your terminal with a one-character-thick snake along a random space-filling curve.

Based on [Random Space-Filling Curves](https://observablehq.com/@esperanc/random-space-filling-curves).

## Installation

```bash
pip install snakebar
```

## Usage

### Basic Python usage

Using `snake_bar` as a drop-in replacement for tqdm:

```python
from snakebar import snake_bar

for i in snake_bar(range(100), desc="Processing"):
    # your code here
    pass
```

### Manual progress bar updates

Using `SnakeBAR` for manual control:

```python
from snakebar import SnakeBAR

with SnakeBAR(total=100, desc="Processing") as bar:
    for i in range(100):
        # your code here
        bar.update(1)
```

### Multi-snake usage

Display multiple colored snakes progressing through the same maze:

```python
from snakebar import multi_snake_bar

# Using as an iterator
for i in multi_snake_bar(range(100), 3, desc="3 Snakes"):
    # your code here
    pass
```

Each snake will be displayed in a different color (bright red, bright green, bright yellow, etc.).

### Independent snake advancement (parallel processes)

Track parallel processes that advance at different rates:

```python
from snakebar import MultiSnakeBAR

# Create a multi-snake bar for 3 processes, each with 100 steps
bar = MultiSnakeBAR(total=100, n_snakes=3, desc="3 Parallel Processes")

with bar:
    # Simulate 3 processes running at different speeds
    while True:
        # Update snake 1 (process 1)
        if process1_has_work():
            bar.update_snake(1, 1)  # Advance snake 1 by 1 step

        # Update snake 2 (process 2)
        if process2_has_work():
            bar.update_snake(2, 2)  # Advance snake 2 by 2 steps

        # Update snake 3 (process 3)
        if process3_has_work():
            bar.update_snake(3, 1)  # Advance snake 3 by 1 step

        # Check if all processes are done
        if all_done():
            break
```

The status line will show individual progress for each snake: `S1:45/100 S2:78/100 S3:32/100`

### CLI usage

You can run the demo from the command line:

```bash
# Single snake
python -m snakebar -n 200 --desc "Processing" --sleep 0.01

# Multiple colored snakes
python -m snakebar -n 200 -s 3 --desc "3 Snakes" --sleep 0.01
```

Options:
- `-n`, `--total`: Total number of steps (default 200)
- `-s`, `--snakes`: Number of snakes to display (default 1, uses colors when > 1)
- `--desc`: Description text to show alongside the progress bar
- `--sleep`: Time in seconds to sleep between steps (simulates work)
- `--seed`: Random seed for reproducible snake paths
- `--ch`: Character to use for the snake (default: █)
- `--bg`: Background character (default: space)

## Features

- Single snake progress bar with customizable characters
- Multi-snake mode with colored snakes (each snake gets a different color via ANSI escape codes)
- Independent progress tracking for each snake - perfect for monitoring parallel processes
- Random space-filling curves generated each run (unless a seed is specified)
- tqdm-style status information (progress %, ETA, rate, and individual snake counters)
- Optimized rendering with rate limiting (60 FPS max) and efficient StringIO-based string building

## API Reference

### `SnakeBAR`

Main class for creating a snake progress bar.

**Constructor:**
```python
SnakeBAR(total: int,
         ch: str = '█',
         bg: str = ' ',
         seed: Optional[int] = None,
         pad_x: int = 0,
         pad_y: int = 0,
         desc: str = "")
```

**Methods:**
- `__enter__()` / `__exit__()`: Context manager support
- `update(n=1)`: Advance progress by n steps
- `set_description(desc)`: Update the description text
- `close()`: Clean up and restore cursor

### `snake_bar`

Convenience function for wrapping iterables.

**Usage:**
```python
for item in snake_bar(iterable, **kwargs):
    # work with item
    pass
```

### `MultiSnakeBAR`

Multi-snake progress bar that displays multiple colored snakes in the same maze.

**Constructor:**
```python
MultiSnakeBAR(total: int,
              n_snakes: int,
              ch: str = '█',
              colors: Optional[List[str]] = None,
              bg: str = ' ',
              seed: Optional[int] = None,
              pad_x: int = 0,
              pad_y: int = 0,
              desc: str = "")
```

**Parameters:**
- `total`: Total number of iterations
- `n_snakes`: Number of colored snakes to display
- `ch`: Character to use for all snakes (default: █)
- `colors`: Optional list of ANSI color codes (auto-generated if not provided)
- `bg`: Background character
- `seed`: Random seed for reproducible paths
- `pad_x`, `pad_y`: Padding around the display
- `desc`: Description text

**Methods:**
- `__enter__()` / `__exit__()`: Context manager support
- `update(n=1)`: Advance all snakes together by n steps (for uniform progress)
- `update_snake(snake_idx, n=1)`: Advance a specific snake by n steps (for independent progress tracking)
- `set_description(desc)`: Update the description text
- `close()`: Clean up and restore cursor

**Note:** Each snake maintains its own independent progress counter. Use `update_snake()` to track parallel processes advancing at different rates, or use `update()` to advance all snakes together uniformly.

### `multi_snake_bar`

Convenience function for multi-snake progress with iterables.

**Usage:**
```python
for item in multi_snake_bar(iterable, n_snakes: int, **kwargs):
    # work with item
    pass
```

## License

MIT

## Credits

Original implementation by Majo Bustamante Rosell.
Based on the Observable notebook by Claudio Esperança.
