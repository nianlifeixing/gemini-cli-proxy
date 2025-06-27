"""
CLI entry module

Uses Click to handle command line arguments and application startup
"""

import click
import uvicorn


@click.command()
@click.option(
    "--host",
    default="127.0.0.1",
    help="Server host address"
)
@click.option(
    "--port",
    default=8765,
    type=int,
    help="Server port"
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="Log level"
)
@click.option(
    "--rate-limit",
    default=60,
    type=int,
    help="Maximum requests per minute"
)
@click.option(
    "--max-concurrency",
    default=4,
    type=int,
    help="Maximum concurrent subprocesses"
)
@click.option(
    "--timeout",
    default=30.0,
    type=float,
    help="Gemini CLI command timeout in seconds"
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode"
)
def main(
    host: str,
    port: int,
    log_level: str,
    rate_limit: int,
    max_concurrency: int,
    timeout: float,
    debug: bool
):
    """Start Gemini CLI Proxy server"""
    
    # Set configuration
    from .config import config
    config.host = host
    config.port = port
    config.log_level = log_level
    config.rate_limit = rate_limit
    config.max_concurrency = max_concurrency
    config.timeout = timeout
    config.debug = debug
    
    # Start server
    uvicorn.run(
        "gemini_cli_proxy.server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=debug
    )


if __name__ == "__main__":
    main() 