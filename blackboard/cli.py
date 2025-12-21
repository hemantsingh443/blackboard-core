"""
Blackboard CLI

Command-line interface for the Blackboard SDK.
"""

import argparse
import sys
import importlib
import logging

logger = logging.getLogger("blackboard.cli")


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the API server."""
    try:
        from blackboard.serve import create_app
    except ImportError as e:
        print(
            f"Error: Missing serve dependencies. Install with: pip install blackboard-core[serve]\n{e}",
            file=sys.stderr
        )
        return 1
    
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn not installed. Install with: pip install blackboard-core[serve]",
            file=sys.stderr
        )
        return 1
    
    # Parse module:attribute format
    module_path, _, attr_name = args.orchestrator.partition(":")
    if not attr_name:
        print(
            f"Error: Invalid format '{args.orchestrator}'. Use 'module:attribute' (e.g., 'my_app:orchestrator')",
            file=sys.stderr
        )
        return 1
    
    # Create the FastAPI app with the orchestrator factory
    app = create_app(
        orchestrator_path=args.orchestrator,
        title=args.title or f"Blackboard API ({module_path})",
        sessions_dir=args.sessions_dir
    )
    
    print(f"Starting Blackboard API server on http://{args.host}:{args.port}")
    print(f"  Orchestrator: {args.orchestrator}")
    print(f"  Sessions: {args.sessions_dir}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload
    )
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Print version information."""
    from blackboard import __version__
    print(f"blackboard-core {__version__}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the Streamlit UI."""
    import subprocess
    import os
    from pathlib import Path
    
    # Find the UI app module
    ui_app_path = Path(__file__).parent / "ui" / "app.py"
    
    if not ui_app_path.exists():
        print(
            f"Error: UI app not found at {ui_app_path}",
            file=sys.stderr
        )
        return 1
    
    # Check for streamlit
    try:
        import streamlit
    except ImportError:
        print(
            "Error: Streamlit not installed. Install with: pip install streamlit",
            file=sys.stderr
        )
        return 1
    
    print(f"Launching Blackboard UI...")
    print(f"  API URL: {args.api_url}")
    print(f"  Port: {args.port}")
    print()
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ui_app_path),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.headless else "false",
        "--", "--api-url", args.api_url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nUI stopped.")
        return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="blackboard",
        description="Blackboard SDK - Multi-agent orchestration framework"
    )
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="Show version and exit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the API server"
    )
    serve_parser.add_argument(
        "orchestrator",
        help="Orchestrator factory path in 'module:attribute' format (e.g., 'my_app:create_orchestrator')"
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    serve_parser.add_argument(
        "--title",
        default=None,
        help="API title (default: auto-generated)"
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    serve_parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
    )
    serve_parser.add_argument(
        "--sessions-dir",
        default="./api_sessions",
        help="Directory to store session state (default: ./api_sessions)"
    )
    serve_parser.set_defaults(func=cmd_serve)
    
    # version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)
    
    # ui command
    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Streamlit UI dashboard"
    )
    ui_parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="URL of the Blackboard API server (default: http://localhost:8000)"
    )
    ui_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port for Streamlit UI (default: 8501)"
    )
    ui_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no browser auto-open)"
    )
    ui_parser.set_defaults(func=cmd_ui)
    
    args = parser.parse_args()
    
    if args.version:
        return cmd_version(args)
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
