#!/usr/bin/env python3
"""A minimal MCP client: start the server, list the analyses, run one (or two).

What it does, in the order an agent would:

  1. connect to the server and list its tools
  2. `list_analyses`  -- the catalogue: input kinds, metrics, parameters
  3. `run_analysis`   -- `edep` over the art file you name
  4. `run_analysis`   -- `approx_ce_sensitivity` over the nts.*.root step 3
                         wrote, which is what `produced_by` is for

Environment (the `ana` python already has the mcp SDK; no installs needed):

    source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
    pyenv ana 2.7.0

Run it -- the client starts the server itself over stdio. The path may be
relative or use ~; it is resolved here, because the server takes only
absolute paths (it runs the job from its own working directory):

    python3 examples/simple_client.py ../dts.mmackenz.CeEndpoint.....art

A quick smoke test that does not wait for a full mu2e job (per-gen-event
metrics are meaningless with --max-events, so skip the chained sensitivity):

    python3 examples/simple_client.py path/to/some.art \
        --max-events 200 --no-chain

Against a server you started by hand, which is the same code over HTTP:

    python3 -m analysis_mcp_server --transport streamable-http --port 8000 &
    python3 examples/simple_client.py path/to/some.art \
        --url http://127.0.0.1:8000/mcp

There is no default input file on purpose: `edep` runs over whatever art file
you point it at, and `approx_ce_sensitivity` only means anything for a CE
signal sample (it needs events leaving >10 MeV in the calorimeter).
"""

import argparse
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the Mu2e analysis MCP server over one art file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_file",
        help="Path to the input art file. A relative path (or ~) is "
             "resolved against the current directory before the call: the "
             "server only takes absolute paths, since it runs the job from "
             "its own working directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "output" / "example"),
        help="Where the runs write job output, logs and figures.",
    )
    parser.add_argument(
        "--url",
        help="Connect to an already-running streamable-http server "
             "(e.g. http://127.0.0.1:8000/mcp). Without it the client starts "
             "the server itself over stdio.",
    )
    parser.add_argument(
        "--max-events", type=int,
        help="Cap events in the mu2e job for a quick check. Any 'per gen "
             "event' metric is meaningless when this is set.",
    )
    parser.add_argument(
        "--sig-eff", type=float, default=0.1,
        help="Signal efficiency handed to approx_ce_sensitivity.",
    )
    parser.add_argument(
        "--timeout-s", type=int, default=1800,
        help="Kill the mu2e job after this many seconds.",
    )
    parser.add_argument(
        "--no-chain", action="store_true",
        help="Stop after edep instead of feeding its ROOT file to "
             "approx_ce_sensitivity.",
    )
    return parser.parse_args()


def payload(result: Any) -> dict[str, Any]:
    """The ArtifactResult a tool returned, as a plain dict.

    FastMCP sends the model back as structured content and, for older
    clients, as a JSON text block; accept either.
    """
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    raise RuntimeError(f"tool returned nothing usable: {result!r}")


def show_catalogue(analyses: dict[str, Any]) -> None:
    for name, entry in analyses.items():
        print(f"\n  {name}  [{entry['input_kind']}]")
        print(f"    {entry['description']}")
        if entry.get("produced_by"):
            print(f"    input comes from: {', '.join(entry['produced_by'])}")
        units = entry.get("units", {})
        metrics = ", ".join(
            f"{m} ({units[m]})" if m in units else m for m in entry["metrics"]
        )
        print(f"    metrics: {metrics}")
        for param, info in entry.get("parameters", {}).items():
            need = "required" if info["required"] else f"default {info['default']}"
            print(f"    param {param} [{need}]: {info['description']}")


def show_result(result: dict[str, Any], metrics: list[str]) -> None:
    """Print the tool's message, the metrics it advertises, and its files."""
    meta = result["metadata"]
    print(f"  status : {result['status']}")
    print(f"  message: {result['message']}")
    if result["status"] != "success":
        tail = meta.get("stdout_tail")
        if tail:
            print("  --- last lines of the job log ---")
            for line in tail.splitlines():
                print(f"  | {line}")
        print(f"  log    : {meta.get('log_path', '(none)')}")
        return
    for name in metrics:
        if name in meta:
            print(f"  {name:38s} {meta[name]:g}")
    for path in result["files"]:
        print(f"  wrote  : {path}")
    print(f"  log    : {meta.get('log_path', '(none)')}")


async def connect(stack: AsyncExitStack, url: str | None) -> ClientSession:
    """Open a session, either against `url` or a server we start ourselves."""
    if url:
        print(f"Connecting to {url} ...")
        read, write, _ = await stack.enter_async_context(streamablehttp_client(url))
    else:
        print("Starting the server over stdio ...")
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "analysis_mcp_server", "--transport", "stdio"],
            cwd=str(REPO),
        )
        read, write = await stack.enter_async_context(stdio_client(server))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir).expanduser().resolve()

    # The tool rejects relative paths, and rightly so -- the mu2e job runs
    # in output_dir, not here. Resolve ours so a relative argument works.
    data_file = Path(args.data_file).expanduser().resolve()
    if not data_file.exists():
        print(f"No such input file: {data_file}", file=sys.stderr)
        return 2

    async with AsyncExitStack() as stack:
        session = await connect(stack, args.url)

        tools = await session.list_tools()
        print(f"Tools: {', '.join(tool.name for tool in tools.tools)}")

        # 1. What can this server do?
        print("\n=== list_analyses ===")
        catalogue = payload(await session.call_tool("list_analyses", {}))
        print(catalogue["message"])
        analyses = catalogue["metadata"]["analyses"]
        show_catalogue(analyses)

        # 2. Energy deposition over the art file.
        print(f"\n=== run_analysis: edep on {data_file} ===")
        print("(a real mu2e job -- this can take a while)")
        edep = payload(await session.call_tool("run_analysis", {
            "analysis": "edep",
            "data_file": str(data_file),
            "output_dir": str(outdir / "edep"),
            "timeout_s": args.timeout_s,
            **({"max_events": args.max_events} if args.max_events else {}),
        }))
        show_result(edep, analyses["edep"]["metrics"])
        if edep["status"] != "success":
            return 1

        if args.no_chain:
            return 0

        # 3. Chain: the nts.*.root edep wrote is what the sensitivity reads.
        ntuples = [f for f in edep["files"] if Path(f).name.startswith("nts.")]
        if not ntuples:
            print("\nNo nts.*.root in edep's output, nothing to chain.")
            return 1

        print(f"\n=== run_analysis: approx_ce_sensitivity on {ntuples[0]} ===")
        sens = payload(await session.call_tool("run_analysis", {
            "analysis": "approx_ce_sensitivity",
            "data_file": ntuples[0],
            "output_dir": str(outdir / "sensitivity"),
            "parameters": {"sig_eff": args.sig_eff},
        }))
        show_result(sens, analyses["approx_ce_sensitivity"]["metrics"])
        return 0 if sens["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
