# snakebar.py

__version__ = "1.0.0"
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Iterator, Optional, Union
import random, math
import sys, time, shutil
from io import StringIO

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
    rng = random.Random(seed)
    N = ncols * nrows
    visited = [False] * N
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

    start = rng.randrange(N)
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
    visited = [False] * N2
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

# ANSI color codes for multi-snake mode
_COLORS = [
    "\x1b[91m",  # Bright red
    "\x1b[92m",  # Bright green
    "\x1b[93m",  # Bright yellow
    "\x1b[94m",  # Bright blue
    "\x1b[95m",  # Bright magenta
    "\x1b[96m",  # Bright cyan
    "\x1b[31m",  # Red
    "\x1b[32m",  # Green
    "\x1b[33m",  # Yellow
    "\x1b[34m",  # Blue
    "\x1b[35m",  # Magenta
    "\x1b[36m",  # Cyan
]
_RESET_COLOR = "\x1b[0m"

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
        self._last_repaint = 0.0  # timestamp of last repaint for rate limiting
        self._dirty = True  # flag to track if canvas changed since last repaint

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
        self._progress = 0
        self._repaint()  # initial empty canvas
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        # Force final repaint to show accurate final state (bypasses rate limiting)
        self._repaint(force=True)

        if self._hidden:
            sys.stdout.write(self._show_cursor)
            sys.stdout.flush()
            self._hidden = False

    def _format_status(self) -> str:
        """Return a tqdm-like status line with desc, percent, counts, ETA and rate."""
        done = getattr(self, "_progress", 0)
        total = self.total
        frac = done / total if total else 0.0
        pct = int(frac * 100)

        # timing
        start = self._start_time or time.perf_counter()
        elapsed = max(0.0, time.perf_counter() - start)
        rate = (done / elapsed) if elapsed > 0 else 0.0
        remaining = ((total - done) / rate) if rate > 0 else float("inf")

        def fmt_time(t: float) -> str:
            if not math.isfinite(t):
                return "--:--"
            s = int(round(t))
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h:02d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        e_str = fmt_time(elapsed)
        r_str = fmt_time(remaining)
        rate_str = f"{rate:0.2f} it/s"
        desc = self.desc if self.desc else "Snaking"
        return f"{desc} {pct:3d}%|{done}/{total} [{e_str}<{r_str}, {rate_str}]"

    def _render_canvas(self) -> str:
        # Use StringIO for better performance
        io = StringIO()
        for i, row in enumerate(self.canvas):
            io.write("".join(row))
            if i < len(self.canvas) - 1:
                io.write("\n")
        body = io.getvalue()

        if self.pad_x or self.pad_y or self.desc:
            lines = body.splitlines()
            if self.desc:
                # prepend a title line above the art
                title = self._format_status()
                lines = [title] + lines
            if self.pad_x or self.pad_y:
                side = " " * self.pad_x
                lines = ([""] * self.pad_y) + [side + ln + side for ln in lines] + ([""] * self.pad_y)
            body = "\n".join(lines)
        return body

    def _repaint(self, force=False):
        # Skip if nothing changed
        if not self._dirty and not force:
            return

        # Rate limit repaints to max 60 FPS (16.67ms between frames) unless forced
        current_time = time.perf_counter()
        if not force and (current_time - self._last_repaint) < 0.0167:
            return

        self._last_repaint = current_time
        self._dirty = False

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

        # Ensure we reach the end when at 100%
        if done >= self.total:
            target_upto = total_pts - 1
        else:
            frac = done / self.total
            target_upto = int(frac * (total_pts - 1))

        # Draw newly revealed points
        for k in range(self._drawn_upto + 1, target_upto + 1):
            y, x = self.draw_seq[k]
            if 0 <= y < len(self.canvas) and 0 <= x < len(self.canvas[0]):
                self.canvas[y][x] = self.ch

        if target_upto > self._drawn_upto:
            self._drawn_upto = target_upto
            self._dirty = True
            self._repaint()

    def set_description(self, desc: str):
        self.desc = desc
        self._dirty = True
        self._repaint(force=True)

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


class MultiSnakeBAR:
    """
    Multi-snake progress bar - multiple colored snakes in the same maze.
    """
    def __init__(self, total: int, n_snakes: int, ch: str = "█",
                 colors: Optional[List[str]] = None,
                 bg: str = " ", seed: Optional[int] = None,
                 pad_x: int = 0, pad_y: int = 0, desc: str = ""):
        self.total = max(1, int(total))
        self.n_snakes = max(1, n_snakes)
        self.ch, self.bg = ch, bg
        self.pad_x, self.pad_y = pad_x, pad_y
        self.desc = desc

        # Default colors for different snakes
        if colors is None:
            self.colors = [_COLORS[i % len(_COLORS)] for i in range(n_snakes)]
        else:
            self.colors = colors

        cols, lines = _terminal_size()
        W = max(10, cols - 2*pad_x)
        H = max(5,  lines - 2*pad_y)

        # Choose spanning-tree size to fit interleaved canvas
        st_nrows = max(1, (H + 1) // 4)
        st_ncols = max(1, (W + 1) // 4)

        st = grid_spanning_tree(st_ncols, st_nrows, seed=seed)
        self.ham = hamiltonian_from_spanning_tree(st)

        nrows, ncols = self.ham.nrows, self.ham.ncols
        self.nrows, self.ncols = nrows, ncols
        self.canvas = _build_interleaved_canvas(nrows, ncols, bg=self.bg)

        # Initialize color canvas (empty strings mean no color)
        self.canvas_colors = [[""] * len(self.canvas[0]) for _ in range(len(self.canvas))]
        # Initialize snake index tracker (0 = empty, 1-n = snake index)
        self.canvas_snake_idx = [[0] * len(self.canvas[0]) for _ in range(len(self.canvas))]

        # Precompute draw order (same as single snake)
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

        # Create non-overlapping segments - each snake gets its own exclusive path section
        total_pts = len(self.draw_seq)
        segment_size = total_pts // n_snakes

        self.snake_segments = []
        self.segment_boundaries = []

        for i in range(n_snakes):
            # Non-overlapping: each snake starts where the previous one ended
            start_idx = i * segment_size
            start_idx = max(0, min(start_idx, total_pts - 1))

            end_idx = min(start_idx + segment_size, total_pts)
            if i == n_snakes - 1:
                end_idx = total_pts  # Last snake extends to end to cover any remainder

            self.segment_boundaries.append(start_idx)
            self.snake_segments.append(self.draw_seq[start_idx:end_idx])

        self.segment_boundaries.append(total_pts)

        self._drawn_upto = [0] * n_snakes  # one per snake
        self._start_time = None
        self._hidden = False
        self._progress = [0] * n_snakes  # individual progress per snake
        self._last_repaint = 0.0  # timestamp of last repaint for rate limiting
        self._dirty = True  # flag to track if canvas changed since last repaint

    # --- Terminal control helpers
    _hide_cursor = "\x1b[?25l"
    _show_cursor = "\x1b[?25h"
    _clear_screen = "\x1b[2J"
    _home = "\x1b[H"

    def __enter__(self):
        sys.stdout.write(self._hide_cursor)
        sys.stdout.flush()
        self._hidden = True
        sys.stdout.write(self._clear_screen + self._home)
        sys.stdout.flush()
        self._start_time = time.perf_counter()
        self._progress = [0] * self.n_snakes
        self._repaint()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        # Force final repaint to show accurate final state (bypasses rate limiting)
        self._repaint(force=True)

        if self._hidden:
            sys.stdout.write(self._show_cursor)
            sys.stdout.flush()
            self._hidden = False

    def _format_status(self, max_width: Optional[int] = None) -> str:
        """Return a tqdm-like status line with overall progress and individual snake counters."""
        done_total = sum(self._progress)
        total_all = self.total * self.n_snakes
        frac = done_total / total_all if total_all else 0.0
        pct = int(frac * 100)

        start = self._start_time or time.perf_counter()
        elapsed = max(0.0, time.perf_counter() - start)
        rate = (done_total / elapsed) if elapsed > 0 else 0.0
        remaining = ((total_all - done_total) / rate) if rate > 0 else float("inf")

        def fmt_time(t: float) -> str:
            if not math.isfinite(t):
                return "--:--"
            s = int(round(t))
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h:02d}:{m:02d}:{s:02d}"
            return f"{m:02d}:{s:02d}"

        e_str = fmt_time(elapsed)
        r_str = fmt_time(remaining)
        rate_str = f"{rate:0.2f} it/s"
        desc = self.desc if self.desc else f"Multi-snaking ({self.n_snakes} snakes)"

        # Show individual snake progress
        snake_status = " ".join([f"S{i+1}:{self._progress[i]}/{self.total}" for i in range(self.n_snakes)])

        status = f"{desc} {pct:3d}%|{done_total}/{total_all} [{snake_status}] [{e_str}<{r_str}, {rate_str}]"

        # Truncate to max_width if specified to prevent wrapping
        if max_width is not None and len(status) > max_width:
            status = status[:max_width-3] + "..."

        # Pad to consistent width to prevent jitter
        if max_width is not None and len(status) < max_width:
            status = status.ljust(max_width)

        return status

    def _render_canvas(self) -> str:
        # Render canvas with colors using StringIO for better performance
        io = StringIO()
        for row_idx, row in enumerate(self.canvas):
            for col_idx, char in enumerate(row):
                color = self.canvas_colors[row_idx][col_idx]
                if color:
                    io.write(f"{color}{char}{_RESET_COLOR}")
                else:
                    io.write(char)
            if row_idx < len(self.canvas) - 1:
                io.write("\n")
        body = io.getvalue()

        if self.pad_x or self.pad_y or self.desc or True:  # Always show status for multi-snake
            lines = body.splitlines()
            if self.desc or True:  # Always show status for multi-snake
                # Get terminal width and pass to _format_status to prevent wrapping
                term_cols, _ = _terminal_size()
                max_status_width = max(40, term_cols - 2*self.pad_x)  # Leave some margin
                title = self._format_status(max_status_width)
                lines = [title] + lines
            if self.pad_x or self.pad_y:
                side = " " * self.pad_x
                lines = ([""] * self.pad_y) + [side + ln + side for ln in lines] + ([""] * self.pad_y)
            body = "\n".join(lines)
        return body

    def _repaint(self, force=False):
        # Skip if nothing changed
        if not self._dirty and not force:
            return

        # Rate limit repaints to max 60 FPS (16.67ms between frames) unless forced
        current_time = time.perf_counter()
        if not force and (current_time - self._last_repaint) < 0.0167:
            return

        self._last_repaint = current_time
        self._dirty = False

        sys.stdout.write(self._home)
        sys.stdout.write(self._render_canvas())
        sys.stdout.flush()

    def _update_snake_internal(self, snake_idx: int, n: int) -> bool:
        """Internal: Update snake canvas state without repainting. Returns True if changed."""
        n = max(0, n)
        done = min(self.total, self._progress[snake_idx] + n)
        self._progress[snake_idx] = done

        # Each snake draws through its own segment
        segment = self.snake_segments[snake_idx]
        total_pts = len(segment)

        if done >= self.total:
            target_upto = total_pts
        else:
            frac = done / self.total
            target_upto = int(math.ceil(frac * total_pts))

        # Draw new points in this snake's segment - don't overwrite other snakes
        if target_upto > self._drawn_upto[snake_idx]:
            for k in range(self._drawn_upto[snake_idx], target_upto):
                y, x = segment[k]
                if 0 <= y < len(self.canvas) and 0 <= x < len(self.canvas[0]):
                    # Only draw if cell is empty
                    if self.canvas_snake_idx[y][x] == 0:
                        self.canvas[y][x] = self.ch
                        self.canvas_colors[y][x] = self.colors[snake_idx]
                        self.canvas_snake_idx[y][x] = snake_idx + 1

            self._drawn_upto[snake_idx] = target_upto
            self._dirty = True
            return True

        return False

    def update(self, n: int = 1):
        """Update all snakes together by n steps - each draws independently"""
        # Batch update: update all snakes' canvas state first, then repaint once
        any_changes = False
        last_snake_at_100 = False

        for snake_idx in range(self.n_snakes):
            changed = self._update_snake_internal(snake_idx, n)
            any_changes = any_changes or changed

            # Check if last snake reached 100%
            if snake_idx == self.n_snakes - 1 and self._progress[snake_idx] >= self.total:
                last_snake_at_100 = True

        # Special handling for last snake at 100% - fill remaining empty cells
        if last_snake_at_100:
            # Only check the last snake's segment range (where empty cells are likely to be)
            last_seg_start = self.segment_boundaries[self.n_snakes - 1]
            for k in range(last_seg_start, len(self.draw_seq)):
                y, x = self.draw_seq[k]
                if 0 <= y < len(self.canvas) and 0 <= x < len(self.canvas[0]):
                    if self.canvas[y][x] == self.bg:
                        self.canvas[y][x] = self.ch
                        self.canvas_colors[y][x] = self.colors[self.n_snakes - 1]
                        self.canvas_snake_idx[y][x] = self.n_snakes
                        self._dirty = True

        # Repaint once at the end if anything changed
        if any_changes or self._dirty:
            if last_snake_at_100:
                self._repaint(force=True)  # Force final repaint
            else:
                self._repaint()

    def update_snake(self, snake_idx: int, n: int = 1):
        """Update a specific snake - draws independently through its segment"""
        if snake_idx < 1 or snake_idx > self.n_snakes:
            raise ValueError(f"Invalid snake index: {snake_idx} (must be between 1 and {self.n_snakes})")

        # Convert to 0-indexed
        idx = snake_idx - 1

        # Update the snake's state
        changed = self._update_snake_internal(idx, n)

        # Special handling for last snake at 100% - ensure it fills to the very end
        if idx == self.n_snakes - 1 and self._progress[idx] >= self.total:
            # Only check the last snake's segment range (where empty cells are likely to be)
            last_seg_start = self.segment_boundaries[self.n_snakes - 1]
            for k in range(last_seg_start, len(self.draw_seq)):
                y, x = self.draw_seq[k]
                if 0 <= y < len(self.canvas) and 0 <= x < len(self.canvas[0]):
                    if self.canvas[y][x] == self.bg:
                        self.canvas[y][x] = self.ch
                        self.canvas_colors[y][x] = self.colors[idx]
                        self.canvas_snake_idx[y][x] = idx + 1
                        self._dirty = True
            if self._dirty:
                self._repaint(force=True)  # Force repaint to bypass rate limiting
        elif changed:
            self._repaint()

    def set_description(self, desc: str):
        self.desc = desc
        self._dirty = True
        self._repaint(force=True)

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
        raise ValueError("snake_bar requires a sized iterable; or use SnakeBAR(total=...) manually.")
    bar = SnakeBAR(total=total, **kwargs)
    return bar.wrap(iterable)


def multi_snake_bar(iterable: Iterable, n_snakes: int, **kwargs) -> Iterator:
    """Multi-snake progress bar convenience function"""
    total = len(iterable) if hasattr(iterable, "__len__") else None
    if total is None:
        raise ValueError("multi_snake_bar requires a sized iterable; or use MultiSnakeBAR(total=...) manually.")
    bar = MultiSnakeBAR(total=total, n_snakes=n_snakes, **kwargs)
    return bar.wrap(iterable)


def main():
    """
    CLI entry point: renders a snake progress bar for a dummy loop,
    or can be used to visualize the bar with custom settings.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="snakebar", description="tqdm-like progress bar that snakes across your terminal")
    parser.add_argument("-n", "--total", type=int, default=200, help="total number of steps")
    parser.add_argument("-s", "--snakes", type=int, default=1, help="number of snakes (default: 1, uses colors when > 1)")
    parser.add_argument("--desc", type=str, default="Snaking...", help="label printed above the bar")
    parser.add_argument("--seed", type=int, default=None, help="random seed for the path")
    parser.add_argument("--ch", type=str, default="█", help="character used for the snake (default: full block)")
    parser.add_argument("--bg", type=str, default=" ", help="background character")
    parser.add_argument("--sleep", type=float, default=0.01, help="sleep per step (seconds) for the demo")
    args = parser.parse_args()

    # Demo: iterate for the given total
    if args.snakes > 1:
        # Multi-snake mode
        with MultiSnakeBAR(total=args.total, n_snakes=args.snakes, desc=args.desc, seed=args.seed, bg=args.bg) as bar:
            for _ in range(args.total):
                time.sleep(max(0.0, args.sleep))
                bar.update(1)
    else:
        # Single snake mode
        with SnakeBAR(total=args.total, desc=args.desc, seed=args.seed, ch=args.ch, bg=args.bg) as bar:
            for _ in range(args.total):
                time.sleep(max(0.0, args.sleep))
                bar.update(1)

if __name__ == "__main__":
    main()
