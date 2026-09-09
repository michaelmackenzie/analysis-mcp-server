import importlib
from pathlib import Path
import tomllib
from typing import Literal

from mcp.server.fastmcp import FastMCP

Transport = Literal["stdio", "streamable-http"]


def pyproject_toml() -> Path:
    for directory in Path(__file__).resolve().parents:
        path = directory / "pyproject.toml"
        if path.exists():
            return path
    raise FileNotFoundError("Could not find pyproject.toml")


def configured_tool_module_names() -> list[str]:
    path = pyproject_toml()
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        return list(config["tool"]["mcp-server"]["tool_modules"])
    except KeyError as exc:
        raise RuntimeError(
            f"{path} must contain a [tool.mcp-server] section with a "
            'tool_modules list, e.g.\n\n[tool.mcp-server]\ntool_modules = ["tools"]'
        ) from exc


def load_tool_modules():
    return [
        importlib.import_module(module_name)
        for module_name in configured_tool_module_names()
    ]


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP:
    instructions = (
        "Mu2e analysis tools: run an analysis fcl over art data file(s) with "
        "the mu2e executable and get its numbers back. Call list_analyses to "
        "see what is available (energy deposition is one) and which metrics "
        "each reports, then run_analysis to run one. Each run is a real mu2e "
        "job and can take a while: pass absolute paths to the input art "
        "file(s), a directory the job may write into, and max_events for a "
        "quick check before a full run."
    )

    mcp = FastMCP(
        "Mu2e Analysis MCP Server",
        instructions=instructions,
        host=host,
        port=port,
    )

    for tool_module in load_tool_modules():
        if not hasattr(tool_module, "__all__"):
            raise RuntimeError(
                f"Tool module '{tool_module.__name__}' must define __all__ "
                "listing the functions to expose as MCP tools."
            )
        for name in tool_module.__all__:
            mcp.tool()(getattr(tool_module, name))
    return mcp


def run_server(
    *,
    transport: Transport = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    mcp = create_server(host=host, port=port)
    mcp.run(transport=transport)
