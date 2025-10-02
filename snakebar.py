# snakebar.py

__version__ = "0.1.0"
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Iterator, Optional
import numpy as np
import sys, time, shutil

Index = int
Edge = Tuple[Index, Index]

def _row(i: int, ncols: int) -> int:
    return i // ncols

def _col(i: int, ncols: int) -> int:
    return i % ncols

@dataclass
class SpanningTree:
    nrows: int
    ncols: int
    edges: List[Edge]
    # connect[i] = dict(left=bool, right=bool, up=bool, down=bool)
    connect: List[Dict[str, bool]]

@dataclass
class Hamiltonian:
    nrows: int
    ncols: int
    path: List[Index]

def grid_spanning_tree(ncols: int, nrows: int, seed: int | None = None) -> SpanningTree:
    """
    Create a random spanning tree over an ncols x nrows grid using DFS with shuffled neighbors.
    Matches the Observable logic (including the connect[] bookkeeping).
    """
    rng = np.random.default_rng(seed)
    N = ncols * nrows
    visited = np.zeros(N, dtype=bool)
    edges: List[Edge] = []

    def neighbors(k: int) -> List[int]:
        i, j = _col(k, ncols), _row(k, ncols)
        ns = []
        if i > 0:          ns.append(k - 1)        # left
        if j > 0:          ns.append(k - ncols)    # up
        if i + 1 < ncols:  ns.append(k + 1)        # right
        if j + 1 < nrows:  ns.append(k + ncols)    # down
        rng.shuffle(ns)
        return ns

    def visit(k: int) -> None:
        visited[k] = True
        for n in neighbors(k):
            if not visited[n]:
                edges.append((k, n))
                visit(n)

    start = int(rng.integers(0, N))
    visit(start)

    connect = [dict(left=False, right=False, up=False, down=False) for _ in range(N)]
    for (a, b) in edges:
        i, j = (a, b) if a <= b else (b, a)
        # same row: horizontal
        if _row(i, ncols) == _row(j, ncols):
            connect[i]["right"] = True
            connect[j]["left"] = True
        else:
            # vertical
            connect[i]["down"] = True
            connect[j]["up"] = True

    return SpanningTree(nrows=nrows, ncols=ncols, edges=edges, connect=connect)

def hamiltonian_from_spanning_tree(st: SpanningTree) -> Hamiltonian:
    """
    Convert the grid spanning tree to a Hamiltonian path on the doubled grid,
    following the same construction as the Observable notebook.
    """
    nrows2, ncols2 = 2 * st.nrows, 2 * st.ncols
    N2 = nrows2 * ncols2
    edges2: List[Edge] = []

    def index2(i: int, dcol: int, drow: int) -> int:
        return (_row(i, st.ncols) * 2 + drow) * ncols2 + (_col(i, st.ncols) * 2 + dcol)

    # Build the doubled-grid edges according to the local connect flags
    for i, cell in enumerate(st.connect):
        left, right, up, down = cell["left"], cell["right"], cell["up"], cell["down"]

        # Right edge(s)
        if right:
            edges2.append((index2(i, 1, 0), index2(i, 2, 0)))
            edges2.append((index2(i, 1, 1), index2(i, 2, 1)))
        else:
            edges2.append((index2(i, 1, 0), index2(i, 1, 1)))

        # Left boundary (if no left connection)
        if not left:
            edges2.append((index2(i, 0, 0), index2(i, 0, 1)))

        # Down edge(s)
        if down:
            edges2.append((index2(i, 0, 1), index2(i, 0, 2)))
            edges2.append((index2(i, 1, 1), index2(i, 1, 2)))
        else:
            edges2.append((index2(i, 0, 1), index2(i, 1, 1)))

        # Up boundary (if no up connection)
        if not up:
            edges2.append((index2(i, 0, 0), index2(i, 1, 0)))

    # Build 2-regular graph adjacency (each vertex has degree 2)
    links: List[List[int]] = [[] for _ in range(N2)]
    for a, b in edges2:
        links[a].append(b)
        links[b].append(a)

    # Walk the cycle to produce a single Hamiltonian path over all doubled-grid nodes
    # In this construction, number of edges equals number of nodes (2-regular, one cycle).
    visited = np.zeros(N2, dtype=bool)
    j = 0
    path: List[int] = []
    for _ in range(len(edges2)):
        path.append(j)
        visited[j] = True
        a, b = links[j]
        j = b if visited[a] else a

    return Hamiltonian(nrows=nrows2, ncols=ncols2, path=path)

def _terminal_size() -> Tuple[int, int]:
    sz = shutil.get_terminal_size(fallback=(80, 24))
    return sz.columns, sz.lines

def _build_interleaved_canvas(nrows: int, ncols: int, bg=" "):
    H, W = 2*nrows - 1, 2*ncols - 1
    return [[bg]*W for _ in range(H)]

def _rc(idx: int, ncols: int):
    return divmod(idx, ncols)

class SnakeBAR:
    """
    tqdm-like progress bar using the Hamiltonian 'snake' as the fill visualization.
    """
    def __init__(self, total: int, ch: str = "█", bg: str = " ", seed: Optional[int] = None,
                 pad_x: int = 0, pad_y: int = 0, desc: str = ""):
        self.total = max(1, int(total))
        self.ch, self.bg = ch, bg
        self.pad_x, self.pad_y = pad_x, pad_y
        self.desc = desc

        cols, lines = _terminal_size()
        W = max(10, cols - 2*pad_x)
        H = max(5,  lines - 2*pad_y)

        # Choose spanning-tree size to fit interleaved canvas
        st_nrows = max(1, (H + 1) // 4)
        st_ncols = max(1, (W + 1) // 4)

        st  = grid_spanning_tree(st_ncols, st_nrows, seed=seed)
        self.ham = hamiltonian_from_spanning_tree(st)

        nrows, ncols = self.ham.nrows, self.ham.ncols
        self.nrows, self.ncols = nrows, ncols
        self.canvas = _build_interleaved_canvas(nrows, ncols, bg=self.bg)

        # Precompute (y, x) draw order: center of first cell, then (edge, next center) pairs
        order = []
        path = self.ham.path
        order.append(("center", _rc(path[0], ncols)))
        for k in range(len(path) - 1):
            a, b = path[k], path[k+1]
            r0, c0 = _rc(a, ncols)
            r1, c1 = _rc(b, ncols)
            # edge midpoint
            if r0 != r1:
                y = (2*r0 + 2*r1)//2
                x = 2*c0
            else:
                y = 2*r0
                x = (2*c0 + 2*c1)//2
            order.append(("edge", (y, x)))
            order.append(("center", (r1, c1)))
        # Convert to canvas coordinates
        self.draw_seq = []
        for kind, val in order:
            if kind == "center":
                r, c = val
                self.draw_seq.append((2*r, 2*c))
            else:
                y, x = val
                self.draw_seq.append((y, x))

        self._drawn_upto = -1  # last index in draw_seq rendered
        self._start_time = None
        self._hidden = False

    # --- Terminal control helpers
    _hide_cursor = "\x1b[?25l"
    _show_cursor = "\x1b[?25h"
    _clear_screen = "\x1b[2J"
    _home = "\x1b[H"

    def __enter__(self):
        sys.stdout.write(self._hide_cursor)
        sys.stdout.flush()
        self._hidden = True
        # Clear and position once
        sys.stdout.write(self._clear_screen + self._home)
        sys.stdout.flush()
        self._start_time = time.perf_counter()
        self._repaint()  # initial empty canvas
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self._hidden:
            sys.stdout.write(self._show_cursor)
            sys.stdout.flush()
            self._hidden = False

    def _render_canvas(self) -> str:
        body = "\n".join("".join(row) for row in self.canvas)
        if self.pad_x or self.pad_y or self.desc:
            lines = body.splitlines()
            if self.desc:
                # prepend a title line above the art
                title = self.desc
                lines = [title] + lines
            if self.pad_x or self.pad_y:
                side = " " * self.pad_x
                lines = ([""] * self.pad_y) + [side + ln + side for ln in lines] + ([""] * self.pad_y)
            body = "\n".join(lines)
        return body

    def _repaint(self):
        sys.stdout.write(self._home)
        sys.stdout.write(self._render_canvas())
        sys.stdout.flush()

    def update(self, n: int = 1):
        """
        Advance progress by n (like tqdm.update). Redraws only as needed.
        """
        # clamp progress
        n = max(0, n)
        done = min(self.total, getattr(self, "_progress", 0) + n)
        self._progress = done

        # Map progress -> how many draw_seq points to reveal
        total_pts = len(self.draw_seq)
        frac = done / self.total
        target_upto = int(frac * (total_pts - 1))

        # Draw newly revealed points
        for k in range(self._drawn_upto + 1, target_upto + 1):
            y, x = self.draw_seq[k]
            if 0 <= y < len(self.canvas) and 0 <= x < len(self.canvas[0]):
                self.canvas[y][x] = self.ch

        if target_upto > self._drawn_upto:
            self._drawn_upto = target_upto
            self._repaint()

    def set_description(self, desc: str):
        self.desc = desc
        self._repaint()

    # Convenience: iterate over an iterable like tqdm
    def wrap(self, iterable: Iterable) -> Iterator:
        count = 0
        self.__enter__()
        try:
            for item in iterable:
                yield item
                count += 1
                self.update(1)
        finally:
            self.close()

# Syntactic sugar function, like tqdm(iterable)
def snake_bar(iterable: Iterable, **kwargs) -> Iterator:
    total = len(iterable) if hasattr(iterable, "__len__") else None
    if total is None:
        raise ValueError("snake_tqdm requires a sized iterable; or use SnakeTQDM(total=...) manually.")
    bar = SnakeBAR(total=total, **kwargs)
    return bar.wrap(iterable)

def main():
    """
    CLI entry point: renders a snake progress bar for a dummy loop,
    or can be used to visualize the bar with custom settings.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="snakebar", description="tqdm-like progress bar that snakes across your terminal")
    parser.add_argument("-n", "--total", type=int, default=200, help="total number of steps")
    parser.add_argument("--desc", type=str, default="Snaking...", help="label printed above the bar")
    parser.add_argument("--seed", type=int, default=None, help="random seed for the path")
    parser.add_argument("--ch", type=str, default="█", help="character used for the snake (default: full block)")
    parser.add_argument("--bg", type=str, default=" ", help="background character")
    parser.add_argument("--sleep", type=float, default=0.01, help="sleep per step (seconds) for the demo")
    parser.add_argument("--fps", type=int, default=60, help="target frames per second for repaint")
    parser.add_argument("--spf", type=int, default=1, help="steps per frame before repaint")
    args = parser.parse_args()

    # Demo: iterate for the given total
    with SnakeBAR(total=args.total, desc=args.desc, seed=args.seed, ch=args.ch, bg=args.bg) as bar:
        for _ in range(args.total):
            time.sleep(max(0.0, args.sleep))
            bar.update(1)

if __name__ == "__main__":
    main()
