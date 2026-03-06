"""Main command-line entrypoint for adk-deepagents."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from adk_deepagents import __version__
from adk_deepagents.cli.config import ensure_config
from adk_deepagents.cli.models import DEFAULT_AGENT_NAME
from adk_deepagents.cli.non_interactive import (
    combine_non_interactive_prompt,
    read_piped_stdin,
    run_non_interactive,
)
from adk_deepagents.cli.paths import (
    bootstrap_agent_profile,
    get_sessions_db_path,
    list_agent_profiles,
    reset_agent_memory,
)
from adk_deepagents.cli.session_store import (
    ThreadInfo,
    create_thread,
    delete_thread,
    get_cli_user_id,
    get_latest_thread,
    get_thread,
    list_threads,
)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="adk-deepagents",
        description=("adk-deepagents CLI foundation. Interactive runtime wiring is in progress."),
    )
    parser.add_argument(
        "-a",
        "--agent",
        default=DEFAULT_AGENT_NAME,
        help="Agent profile name.",
    )
    parser.add_argument(
        "-M",
        "--model",
        help="Override the model for this run.",
    )
    parser.add_argument(
        "-n",
        "--non-interactive",
        dest="non_interactive_prompt",
        metavar="PROMPT",
        help="Run a single non-interactive task.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-essential non-interactive status lines.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Buffer non-interactive output and print only after completion.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Print installed adk-deepagents version.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        nargs="?",
        const="latest",
        metavar="THREAD_ID",
        help="Resume latest thread (or a specific THREAD_ID).",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "list",
        help="List configured agent profiles.",
    )

    reset_parser = subparsers.add_parser(
        "reset",
        help="Reset AGENTS.md for an agent profile.",
    )
    reset_parser.add_argument(
        "--agent",
        dest="target_agent",
        help="Agent profile to reset. Defaults to --agent value.",
    )

    threads_parser = subparsers.add_parser(
        "threads",
        help="Manage persisted CLI threads.",
    )
    threads_subparsers = threads_parser.add_subparsers(dest="threads_command", required=True)

    threads_list_parser = threads_subparsers.add_parser(
        "list",
        help="List persisted threads.",
    )
    threads_list_parser.add_argument(
        "--agent",
        dest="target_agent",
        help="Filter listed threads by agent profile.",
    )
    threads_list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of threads to show.",
    )

    threads_delete_parser = threads_subparsers.add_parser(
        "delete",
        help="Delete a persisted thread by ID.",
    )
    threads_delete_parser.add_argument("thread_id", help="Thread/session ID.")

    return parser


def _resolve_or_create_thread(
    *,
    user_id: str,
    agent_name: str,
    model: str,
    resume: str | None,
) -> tuple[ThreadInfo, str] | None:
    """Resolve the target thread for a run.

    Returns ``(thread, action)`` where action is ``"Created"`` or ``"Resumed"``.
    Returns ``None`` when explicit resume target does not exist.
    """
    db_path = get_sessions_db_path()

    if resume is None:
        thread = create_thread(
            db_path=db_path,
            user_id=user_id,
            agent_name=agent_name,
            model=model,
        )
        return thread, "Created"

    if resume == "latest":
        thread = get_latest_thread(
            db_path=db_path,
            user_id=user_id,
            agent_name=agent_name,
        )
        if thread is not None:
            return thread, "Resumed"

        thread = create_thread(
            db_path=db_path,
            user_id=user_id,
            agent_name=agent_name,
            model=model,
        )
        return thread, "Created"

    thread = get_thread(
        db_path=db_path,
        user_id=user_id,
        session_id=resume,
    )
    if thread is None:
        print(f"Thread not found: {resume}", file=sys.stderr)
        return None
    return thread, "Resumed"


def _cmd_list() -> int:
    profiles = list_agent_profiles()
    if not profiles:
        print("No agent profiles configured yet.")
        return 0

    for profile in profiles:
        print(profile)
    return 0


def _cmd_reset(agent_name: str) -> int:
    memory_path = reset_agent_memory(agent_name)
    print(f"Reset memory template at {memory_path}")
    return 0


def _cmd_threads_list(
    *,
    user_id: str,
    target_agent: str | None,
    limit: int,
) -> int:
    threads = list_threads(
        db_path=get_sessions_db_path(),
        user_id=user_id,
        agent_name=target_agent,
        limit=limit,
    )
    if not threads:
        print("No threads found.")
        return 0

    print("THREAD_ID  AGENT  MODEL  UPDATED_AT")
    for thread in threads:
        model = thread.model or "-"
        print(f"{thread.session_id}  {thread.agent_name}  {model}  {thread.updated_at_iso}")
    return 0


def _cmd_threads_delete(*, user_id: str, thread_id: str) -> int:
    deleted = delete_thread(
        db_path=get_sessions_db_path(),
        user_id=user_id,
        session_id=thread_id,
    )
    if not deleted:
        print(f"Thread not found: {thread_id}")
        return 1

    print(f"Deleted thread: {thread_id}")
    return 0


def _run_interactive_scaffold(
    *,
    user_id: str,
    agent_name: str,
    resume: str | None,
    model: str,
) -> int:
    profile_dir = bootstrap_agent_profile(agent_name)

    resolved = _resolve_or_create_thread(
        user_id=user_id,
        agent_name=agent_name,
        model=model,
        resume=resume,
    )
    if resolved is None:
        return 2
    thread, action = resolved

    print("adk-deepagents CLI foundation is ready.")
    print(f"Agent profile: {agent_name}")
    print(f"Profile directory: {profile_dir}")
    print(f"{action} thread: {thread.session_id}")
    print(f"Sessions DB: {get_sessions_db_path()}")
    print("Interactive runtime (sessions, streaming, HITL) is the next milestone.")
    return 0


def _run_non_interactive_mode(
    *,
    user_id: str,
    agent_name: str,
    resume: str | None,
    model: str,
    prompt: str,
    quiet: bool,
    no_stream: bool,
) -> int:
    bootstrap_agent_profile(agent_name)

    resolved = _resolve_or_create_thread(
        user_id=user_id,
        agent_name=agent_name,
        model=model,
        resume=resume,
    )
    if resolved is None:
        return 2
    thread, action = resolved

    if not quiet:
        print(
            f"[{action.lower()} thread {thread.session_id}] running non-interactive task",
            file=sys.stderr,
        )

    return run_non_interactive(
        prompt=prompt,
        model=model,
        agent_name=agent_name,
        user_id=user_id,
        session_id=thread.session_id,
        db_path=get_sessions_db_path(),
        no_stream=no_stream,
    )


def cli_main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by console scripts and module execution."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        print(f"adk-deepagents {__version__}")
        return 0

    _, config = ensure_config()
    user_id = get_cli_user_id()
    model = args.model or config.models.default

    if args.command == "list":
        return _cmd_list()
    if args.command == "reset":
        target_agent = args.target_agent or args.agent
        return _cmd_reset(target_agent)
    if args.command == "threads":
        if args.threads_command == "list":
            target_agent = args.target_agent or None
            return _cmd_threads_list(user_id=user_id, target_agent=target_agent, limit=args.limit)
        if args.threads_command == "delete":
            return _cmd_threads_delete(user_id=user_id, thread_id=args.thread_id)

    piped_text = read_piped_stdin()
    prompt = combine_non_interactive_prompt(args.non_interactive_prompt, piped_text)

    if prompt is None and (args.quiet or args.no_stream):
        print(
            "-q/--quiet and --no-stream require -n/--non-interactive or piped stdin.",
            file=sys.stderr,
        )
        return 2

    if prompt is not None:
        return _run_non_interactive_mode(
            user_id=user_id,
            agent_name=args.agent,
            resume=args.resume,
            model=model,
            prompt=prompt,
            quiet=args.quiet,
            no_stream=args.no_stream,
        )

    return _run_interactive_scaffold(
        user_id=user_id,
        agent_name=args.agent,
        resume=args.resume,
        model=model,
    )
