#!/usr/bin/env python3
"""A minimal MCP client: start the server, list the analyses, run one (or two).

What it does, in the order an agent would:

  1. connect to the server and list its tools
  2. `list_analyses`  -- the catalogue: input kinds, metrics, parameters
  3. `run_analysis`   -- `edep` over the art file you name
  4. `run_analysis`   -- `muon_stop_rate`, if you pass --stops-file: another
                         analysis over its own input, so it takes its own file
  5. `run_analysis`   -- `approx_ce_sensitivity` over the nts.*.root step 3
                         wrote, which is what `produced_by` is for

Environment (the `ana` python already has the mcp SDK; no installs needed):

    source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
    pyenv ana 2.7.0

Run it -- the client starts the server itself over stdio. The path may be
relative or use ~; it is resolved here, because the server takes only
absolute paths (it runs the job from its own working directory):

    python3 examples/simple_client.py ../dts.mmackenz.CeEndpoint.....art

With the stopping rate too, which needs a target-stop sim file of its own:

    python3 examples/simple_client.py ../dts.mmackenz.CeEndpoint.....art \
        --stops-file ../sim.mmackenz.TargetStops.....art

A quick smoke test that does not wait for a full mu2e job (per-gen-event
metrics are meaningless with --max-events, so skip the chained sensitivity):

    python3 examples/simple_client.py path/to/some.art \
        --max-events 200 --no-chain

Against a server you started by hand, which is the same code over HTTP:

    python3 -m analysis_mcp_server --transport streamable-http --port 8000 \
        --work-area /exp/mu2e/app/users/mmackenz/mu2eopt &
    python3 examples/simple_client.py path/to/some.art \
        --url http://127.0.0.1:8000/mcp

Where Offline comes from is the server's own setting, and this client starts
the server with --work-area pointing at the local muse area above. Pass
--musing 'SimJob MDC2025au' or --code-tarball <file> to set up a published
Musing or a code tarball instead; with --url the running server's setting
applies and these are ignored.

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

# The muse work area these examples were written against: a local build of
# Mu2eOptAna over a Run-1B backing release, which is what `edep` (the locally
# built EdepAna module) needs. --musing or --code-tarball point the server
# somewhere else instead.
WORK_AREA = "/exp/mu2e/app/users/mmackenz/mu2eopt"


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
             "the server itself over stdio. That server carries its own "
             "Offline setup, so the code options below do not apply to it.",
    )
    code = parser.add_mutually_exclusive_group()
    code.add_argument(
        "--work-area", default=WORK_AREA,
        help="Muse work area the server sets up in.",
    )
    code.add_argument(
        "--musing", metavar="'NAME VERSION'",
        help="Set up a published Musing instead, e.g. 'SimJob MDC2025au'. "
             "Note edep needs the locally built EdepAna, which a bare Musing "
             "does not have.",
    )
    code.add_argument(
        "--code-tarball",
        help="Set up an unpacked code tarball instead.",
    )
    parser.add_argument(
        "--code-dir",
        help="Where --code-tarball is unpacked. Worth setting to reuse one "
             "unpacking across runs; the default unpacks beside each job.",
    )
    parser.add_argument(
        "--code-subdir",
        help="Directory inside the unpacked tarball to run muse setup in.",
    )
    parser.add_argument(
        "--max-events", type=int,
        help="Cap events in the mu2e job for a quick check. Any 'per gen "
             "event' metric is meaningless when this is set.",
    )
    parser.add_argument(
        "--sig-eff", type=float, default=2.5e-4,
        help="Signal efficiency handed to approx_ce_sensitivity.",
    )
    parser.add_argument(
        "--stops-file",
        help="A target-stop sim file (sim.*.TargetStops.*.art). Given one, "
             "the client also runs muon_stop_rate over it.",
    )
    parser.add_argument(
        "--upstream-eff", type=float, default=0.012,
        help="Efficiency of everything upstream of the stopping target "
             "(generated events per POT), handed to muon_stop_rate.",
    )
    parser.add_argument(
        "--prescale-filter",
        help="Prescale filter label for muon_stop_rate. Left out, the server "
             "falls back to its default, the target-stop stream.",
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


def code_args(args: argparse.Namespace) -> list[str]:
    """The server flags saying where Offline comes from: one of the three."""
    if args.musing:
        return ["--musing", args.musing]
    if args.code_tarball:
        return ["--code-tarball", args.code_tarball,
                *(["--code-dir", args.code_dir] if args.code_dir else []),
                *(["--code-subdir", args.code_subdir] if args.code_subdir else [])]
    return ["--work-area", args.work_area]


async def connect(stack: AsyncExitStack, args: argparse.Namespace) -> ClientSession:
    """Open a session, either against `--url` or a server we start ourselves."""
    if args.url:
        print(f"Connecting to {args.url} ...")
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(args.url)
        )
    else:
        server_args = ["-m", "analysis_mcp_server", "--transport", "stdio",
                       *code_args(args)]
        print(f"Starting the server over stdio: {' '.join(code_args(args))} ...")
        server = StdioServerParameters(
            command=sys.executable,
            args=server_args,
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
    stops_file = (Path(args.stops_file).expanduser().resolve()
                  if args.stops_file else None)
    for path in (data_file, stops_file):
        if path is not None and not path.exists():
            print(f"No such input file: {path}", file=sys.stderr)
            return 2

    async with AsyncExitStack() as stack:
        session = await connect(stack, args)

        tools = await session.list_tools()
        print(f"Tools: {', '.join(tool.name for tool in tools.tools)}")

        # 1. What can this server do?
        print("\n=== list_analyses ===")
        catalogue = payload(await session.call_tool("list_analyses", {}))
        print(catalogue["message"])
        print(f"Offline environment: {catalogue['metadata']['environment']}")
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

        # 3. A second analysis, over its own input. Its prescale_filter
        #    parameter is optional, so it is passed only when you set one and
        #    the server's default -- the target-stop stream -- runs otherwise.
        if stops_file is not None:
            print(f"\n=== run_analysis: muon_stop_rate on {stops_file} ===")
            parameters: dict[str, Any] = {"upstream_eff": args.upstream_eff}
            if args.prescale_filter:
                parameters["prescale_filter"] = args.prescale_filter
            stops = payload(await session.call_tool("run_analysis", {
                "analysis": "muon_stop_rate",
                "data_file": str(stops_file),
                "output_dir": str(outdir / "muon_stop_rate"),
                "parameters": parameters,
                "timeout_s": args.timeout_s,
            }))
            show_result(stops, analyses["muon_stop_rate"]["metrics"])
            if stops["status"] == "success":
                meta = stops["metadata"]
                print(f"  prescale filter used: {meta['prescale_filter']} "
                      f"(of {', '.join(sorted(meta['prescale_filters']))})")

        if args.no_chain:
            return 0

        # 4. Chain: the nts.*.root edep wrote is what the sensitivity reads.
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
