import argparse
import os
import signal
import subprocess
import sys
import time

from tools.mu2e_env import EnvError, Mu2eEnv, configure, current

from .server import run_server


def free_port(port: int) -> None:
    """Kill a leftover mcp_server still holding the port, then return.

    A server suspended with Ctrl+Z (instead of stopped with Ctrl+C) keeps the
    port bound and the next start fails with "address already in use". Only
    processes whose command line contains "analysis_mcp_server" are killed, so
    an unrelated app on the same port is left alone (you'll get the normal
    bind error instead).
    """
    try:
        lsof = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True
        )
    except FileNotFoundError:  # no lsof on this platform; let bind errors surface
        return
    freed = False
    for pid in lsof.stdout.split():
        ps = subprocess.run(
            ["ps", "-p", pid, "-o", "command="], capture_output=True, text=True
        )
        if "analysis_mcp_server" in ps.stdout:
            # SIGKILL: a Ctrl+Z-suspended process would never handle SIGTERM.
            os.kill(int(pid), signal.SIGKILL)
            print(f"Freed port {port}: killed leftover analysis_mcp_server (pid {pid}).",
                  flush=True)
            freed = True
    if freed:
        time.sleep(0.5)  # give the kernel a moment to release the socket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis_mcp_server",
        description="Run the Mu2e analysis MCP server wrapper.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport. Use stdio for subprocess agents or streamable-http for a URL endpoint.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for streamable-http transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for streamable-http transport.",
    )

    # Where Offline comes from. One of these, or the MU2E_WORK_AREA /
    # MU2E_MUSING / MU2E_CODE_TARBALL environment variables; the work area
    # the server ships with is the fallback.
    code = parser.add_mutually_exclusive_group()
    code.add_argument(
        "--work-area",
        help="A muse work area to set up in: `cd <area> && muse setup`. "
             "What locally built analysis modules need.",
    )
    code.add_argument(
        "--musing",
        metavar="'NAME VERSION'",
        help="A published Musing to set up instead, e.g. --musing "
             "'SimJob MDC2025au' (`muse setup SimJob MDC2025au`).",
    )
    code.add_argument(
        "--code-tarball",
        help="A code tarball to unpack and set up in. Unpacked once and "
             "reused.",
    )
    parser.add_argument(
        "--code-dir",
        help="Where --code-tarball is unpacked. Default: a 'code' directory "
             "beside each job's output, so runs stay self-contained.",
    )
    parser.add_argument(
        "--code-subdir",
        help="Path inside the unpacked tarball to run `muse setup` in. "
             "Default: its root, or its single top-level directory.",
    )
    return parser


def environment_from(args: argparse.Namespace) -> Mu2eEnv | None:
    """The environment the flags ask for, or None to fall back."""
    if args.work_area:
        return Mu2eEnv.for_work_area(args.work_area)
    if args.musing:
        return Mu2eEnv.for_musing(args.musing)
    if args.code_tarball:
        return Mu2eEnv.for_tarball(args.code_tarball, args.code_dir, args.code_subdir)
    return None


def main() -> int:
    args = build_parser().parse_args()

    # Settled before the server starts, so a bad path or Musing is a startup
    # error naming itself rather than a failure inside the first job.
    try:
        env = environment_from(args)
    except EnvError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if env is not None:
        configure(env)
    print(f"mu2e jobs will run against {current().describe()}.",
          file=sys.stderr, flush=True)

    if args.transport == "streamable-http":
        free_port(args.port)
    run_server(
        transport=args.transport,
        host=args.host,
        port=args.port,
    )
    return 0
