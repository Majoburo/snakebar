# random_space_filling_curves.py
# Translation of https://observablehq.com/@esperanc/random-space-filling-curves
# to Python with NumPy. Optional plotting via Matplotlib.

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

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

# ---------- Optional plotting helpers (Matplotlib) ----------

def draw_graph(st: SpanningTree, cell_size: float = 10.0):
    """
    Visualize the spanning tree edges on the base grid.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, st.ncols * cell_size)
    ax.set_ylim(0, st.nrows * cell_size)

    def X(i): return (_col(i, st.ncols) + 0.5) * cell_size
    def Y(i): return (_row(i, st.ncols) + 0.5) * cell_size

    for a, b in st.edges:
        ax.plot([X(a), X(b)], [Y(a), Y(b)], linewidth=1)

    ax.invert_yaxis()  # match canvas-like coordinates
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax

def draw_path(ham: Hamiltonian, cell_size: float = 10.0, thick: float = 0.5):
    """
    Visualize the Hamiltonian path on the doubled grid.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, ham.ncols * cell_size)
    ax.set_ylim(0, ham.nrows * cell_size)

    def X(i): return (_col(i, ham.ncols) + 0.5) * cell_size
    def Y(i): return (_row(i, ham.ncols) + 0.5) * cell_size

    xs = [X(i) for i in ham.path]
    ys = [Y(i) for i in ham.path]
    ax.plot(xs, ys, linewidth=cell_size * thick, solid_joinstyle='round')
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax

# ---------- Example usage ----------

if __name__ == "__main__":
    st = grid_spanning_tree(ncols=10, nrows=10, seed=None)  # set seed for reproducibility if desired
    ham = hamiltonian_from_spanning_tree(st)

    # Optional: draw
    #fig1, ax1 = draw_graph(st, cell_size=10)
    fig2, ax2 = draw_path(ham, cell_size=10, thick=0.3)
    import matplotlib.pyplot as plt
    plt.show()
    def ascii_path_raster(ham, scale=3, thickness=1, char="#", bg=" "):
        """
        Render the Hamiltonian path as a thick continuous ASCII line.

        Parameters
        ----------
        ham : Hamiltonian  (from the code we wrote)
        scale : int        number of character cells per grid step (>=2 looks good)
        thickness : int    half-thickness of the stroke in characters (>=1)
        char : str         draw character (e.g. "#", "*", "@")
        bg : str           background character (usually space)

        Returns
        -------
        str : multiline ASCII art
        """
        nrows, ncols = ham.nrows, ham.ncols

        # Canvas dimensions: one "scale" per grid cell, draw at centers.
        H = nrows * scale
        W = ncols * scale
        canvas = [[bg] * W for _ in range(H)]

        def rc(idx):  # (row, col)
            return divmod(idx, ncols)

        def to_px(r, c):
            # center of cell in pixel coords
            return r * scale + scale // 2, c * scale + scale // 2  # (y, x)

        def draw_segment(y0, x0, y1, x1):
            if x0 == x1:
                y_start, y_end = (y0, y1) if y0 <= y1 else (y1, y0)
                for y in range(y_start, y_end + 1):
                    for dx in range(-thickness, thickness + 1):
                        xx = x0 + dx
                        if 0 <= y < H and 0 <= xx < W:
                            canvas[y][xx] = char
            elif y0 == y1:
                x_start, x_end = (x0, x1) if x0 <= x1 else (x1, x0)
                for x in range(x_start, x_end + 1):
                    for dy in range(-thickness, thickness + 1):
                        yy = y0 + dy
                        if 0 <= yy < H and 0 <= x < W:
                            canvas[yy][x] = char
            else:
                # Path is axis-aligned; this shouldn't happen. Fall back to a simple line.
                steps = max(abs(x1 - x0), abs(y1 - y0))
                for t in range(steps + 1):
                    y = y0 + (y1 - y0) * t // steps
                    x = x0 + (x1 - x0) * t // steps
                    if 0 <= y < H and 0 <= x < W:
                        canvas[y][x] = char

        # Draw the path as connected thick segments
        for k in range(len(ham.path) - 1):
            r0, c0 = rc(ham.path[k])
            r1, c1 = rc(ham.path[k + 1])
            y0, x0 = to_px(r0, c0)
            y1, x1 = to_px(r1, c1)
            draw_segment(y0, x0, y1, x1)

        return "\n".join("".join(row) for row in canvas)
    print(ascii_path_raster(ham, scale=1,1thickness=1, char = "#"))
