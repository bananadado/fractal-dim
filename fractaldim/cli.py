"""Command line interface."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from . import library
from .plotting import DEFAULT_DPI, DEFAULT_SIZE
from .turtle import bounds, diameter, trace

#: Renders land here when -o is not given. Gitignored, so the directory is
#: scratch space: everything in it can be regenerated from a command line.
DEFAULT_OUTPUT_DIR = "output"


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def _open_in_viewer(path: str) -> bool:
    """Hand ``path`` to the system image viewer.

    Preferred over an interactive matplotlib window under WSL, where the GUI
    backend depends on WSLg being reachable and fails in ways that are tedious
    to diagnose. Writing a PNG and letting Windows open it always works.
    """
    if shutil.which("wslview"):                       # wslu, if installed
        commands = [["wslview", path]]
    elif _is_wsl() and shutil.which("explorer.exe"):
        windows_path = path
        if shutil.which("wslpath"):
            try:
                windows_path = subprocess.run(
                    ["wslpath", "-w", os.path.abspath(path)],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
            except (subprocess.SubprocessError, OSError):
                pass
        # explorer.exe reports failure even when it succeeds, so its exit
        # status tells us nothing and is deliberately ignored below.
        commands = [["explorer.exe", windows_path]]
    elif shutil.which("xdg-open"):
        commands = [["xdg-open", path]]
    else:
        return False

    for command in commands:
        try:
            subprocess.run(command, check=False, capture_output=True, timeout=15)
            return True
        except (subprocess.SubprocessError, OSError):
            continue
    return False


def _cmd_list(args: argparse.Namespace) -> int:
    print(f"{'name':20s} {'delta':>7s} {'dimension':>10s}  note")
    for name in library.names():
        fractal = library.get(name)
        dim = "  unknown" if fractal.dimension is None else f"{fractal.dimension:10.6f}"
        note = fractal.note
        if not args.full and len(note) > 64:
            note = note[:61] + "..."
        print(f"{name:20s} {fractal.angle:7g} {dim}  {note}")
        if args.full:
            print(f"{'':20s} {'':7s} {'':10s}  {fractal.system}")
    return 0


def _traced(args: argparse.Namespace):
    """Trace the requested level and report what came out."""
    fractal = library.get(args.name)
    segments = trace(
        fractal.system,
        args.level,
        step=args.step,
        commands=fractal.commands,
        start_heading=args.heading if args.heading is not None
        else fractal.start_heading,
    )
    xmin, ymin, xmax, ymax = bounds(segments)
    print(
        f"{fractal.name} level {args.level}: {len(segments)} segments, "
        f"span {xmax - xmin:g} x {ymax - ymin:g}, diameter {diameter(segments):.4f}"
    )
    return fractal, segments


def _style(args: argparse.Namespace, fractal, suffix: str = "") -> dict:
    title = None if args.no_title else f"{fractal.name}, level {args.level}{suffix}"
    return dict(
        color=args.color,
        cmap=args.cmap,
        title=title,
        size=args.size,
        linewidth_scale=args.linewidth_scale,
        background=args.background,
    )


def _ensure_parent(path: str) -> str:
    """Create the directory ``path`` lives in, if it names one."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _output_path(args: argparse.Namespace, fractal, suffix: str) -> str:
    """Where to write: the requested path, or a default under the output dir."""
    if args.output:
        return _ensure_parent(args.output)
    return _ensure_parent(
        os.path.join(args.output_dir, f"{fractal.name}-{args.level}{suffix}")
    )


def _deliver(path: str, do_open: bool) -> None:
    print(f"wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    if do_open and not _open_in_viewer(path):
        print("could not find an image viewer; open the file above by hand",
              file=sys.stderr)


def _cmd_draw(args: argparse.Namespace) -> int:
    if args.animate:
        # 'animate' is a sibling subcommand, not a flag on 'draw'. Easy mistake
        # to make, so point at the real spelling rather than just rejecting it.
        raise ValueError(
            "'animate' is a subcommand, not a flag on 'draw'. Try:\n"
            f"  fractaldim animate {args.name} -n {args.level} --open"
        )

    fractal, segments = _traced(args)

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    from .plotting import plot_segments, save_segments

    options = _style(args, fractal)
    # --show puts the figure on screen, so a file is only written if one was
    # actually asked for; otherwise the default output path applies.
    path = args.output if args.show else _output_path(args, fractal, ".png")

    if path:
        save_segments(segments, path, dpi=args.dpi, **options)
        _deliver(path, args.open)

    if args.show:
        import matplotlib.pyplot as plt

        plot_segments(segments, **options)
        plt.show()
    return 0


def _cmd_animate(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")

    fractal, segments = _traced(args)
    from .plotting import animate_segments

    path = _output_path(args, fractal, ".gif")

    def progress(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            print(f"\r  rendering frame {done}/{total}", end="", file=sys.stderr)
        if done == total:
            print(file=sys.stderr)

    animate_segments(
        segments, path,
        frames=args.frames, fps=args.fps, hold=args.hold, dpi=args.dpi,
        head=not args.no_head, progress=None if args.quiet else progress,
        **_style(args, fractal),
    )
    _deliver(path, args.open)
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options shared by every command that traces and renders a curve."""
    parser.add_argument("name", help="catalogue name, e.g. hilbert")
    parser.add_argument("-n", "--level", type=int, default=4,
                        help="number of rewriting steps (default: 4)")
    parser.add_argument("-o", "--output",
                        help="file to write; defaults to "
                             f"{DEFAULT_OUTPUT_DIR}/<name>-<level>.<ext>")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="directory for default filenames "
                             f"(default: {DEFAULT_OUTPUT_DIR}/)")
    parser.add_argument("--open", action="store_true",
                        help="open the result in the system image viewer; more "
                             "reliable than --show under WSL")
    parser.add_argument("--step", type=float, default=1.0,
                        help="turtle step length")
    parser.add_argument("--heading", type=float, default=None,
                        help="initial heading in degrees, overriding the catalogue")
    parser.add_argument("--color", default="index",
                        help="'index' to shade along the curve, or a matplotlib colour")
    parser.add_argument("--cmap", default="viridis",
                        help="colormap used when --color=index (default: viridis)")
    parser.add_argument("--background", default="white", help="figure background")
    parser.add_argument("--size", type=float, default=DEFAULT_SIZE,
                        help=f"figure edge in inches (default: {DEFAULT_SIZE:g}); "
                             "output pixels are roughly size x dpi")
    parser.add_argument("--linewidth-scale", type=float, default=1.0,
                        help="multiply the automatic stroke width")
    parser.add_argument("--no-title", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractaldim",
        description="L-systems and fractal dimension estimation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="show the catalogue of known systems")
    listing.add_argument("--full", action="store_true",
                         help="print untruncated notes and each system's productions")
    listing.set_defaults(func=_cmd_list)

    draw = sub.add_parser("draw", help="trace a catalogued system and render it")
    _add_common(draw)
    draw.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                      help=f"dots per inch (default: {DEFAULT_DPI})")
    draw.add_argument("--show", action="store_true",
                      help="open an interactive matplotlib window (needs a "
                           "working GUI backend)")
    # Accepted only so it can be redirected to the 'animate' subcommand.
    draw.add_argument("--animate", action="store_true", help=argparse.SUPPRESS)
    draw.set_defaults(func=_cmd_draw)

    animate = sub.add_parser(
        "animate", help="render a GIF of the turtle drawing the curve")
    _add_common(animate)
    animate.add_argument("--dpi", type=int, default=150,
                         help="dots per inch (default: 150; GIFs get large fast)")
    animate.add_argument("--frames", type=int, default=150,
                         help="reveal steps in the animation (default: 150)")
    animate.add_argument("--fps", type=int, default=20,
                         help="frames per second (default: 20)")
    animate.add_argument("--hold", type=int, default=12,
                         help="still frames on the finished curve (default: 12)")
    animate.add_argument("--no-head", action="store_true",
                         help="omit the marker showing the turtle's position")
    animate.add_argument("--quiet", action="store_true",
                         help="suppress the per-frame progress line")
    animate.set_defaults(func=_cmd_animate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
