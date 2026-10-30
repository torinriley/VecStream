"""Command line: ``python -m vecstream serve|keys``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from typing import List, Optional

from .auth import KeyStore
from .server import VecStreamServer


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(f"VECSTREAM_{name}", default)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vecstream")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--data-dir", default=_env("DATA_DIR"), required=_env("DATA_DIR") is None)
    serve.add_argument("--host", default=_env("HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(_env("PORT", "8080") or 8080))
    serve.add_argument("--keys-file", default=_env("KEYS_FILE"))
    serve.add_argument(
        "--insecure-no-auth", action="store_true", help="disable authentication (development only)"
    )
    serve.add_argument("--tls-cert", default=_env("TLS_CERT"))
    serve.add_argument("--tls-key", default=_env("TLS_KEY"))
    serve.add_argument(
        "--drain-timeout", type=float, default=float(_env("DRAIN_TIMEOUT", "30") or 30)
    )
    serve.add_argument(
        "--checkpoint-ops", type=int, default=int(_env("CHECKPOINT_OPS", "10000") or 10000)
    )
    serve.add_argument("--log-level", default=_env("LOG_LEVEL", "INFO"))

    keys = commands.add_parser("keys", help="manage API keys")
    keys.add_argument("--keys-file", default=_env("KEYS_FILE"))
    actions = keys.add_subparsers(dest="action", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--keys-file", default=argparse.SUPPRESS)  # also accepted after the action
    create = actions.add_parser("create", parents=[common])
    create.add_argument("--tenant")
    create.add_argument("--role", required=True, choices=["reader", "writer", "admin", "operator"])
    create.add_argument(
        "--collections", nargs="+", default=["*"], help="name patterns, e.g. 'docs-*'"
    )
    create.add_argument("--ttl-days", type=float)
    create.add_argument("--label", default="")
    revoke = actions.add_parser("revoke", parents=[common])
    revoke.add_argument("key_id")
    actions.add_parser("list", parents=[common])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "keys":
        if not args.keys_file:
            print("error: --keys-file is required", file=sys.stderr)
            return 2
        store = KeyStore(args.keys_file)
        if args.action == "create":
            ttl = args.ttl_days * 86400 if args.ttl_days else None
            key_id, key = store.create_key(
                args.tenant, args.role, args.collections, ttl, args.label
            )
            print(f"key_id: {key_id}\nAPI key (shown once, store it securely): {key}")
        elif args.action == "revoke":
            store.revoke(args.key_id)
            print(f"revoked {args.key_id}")
        else:
            print(json.dumps(store.list_keys(), indent=2))
        return 0

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    if not args.keys_file and not args.insecure_no_auth:
        print("error: provide --keys-file, or --insecure-no-auth for development", file=sys.stderr)
        return 2
    server = VecStreamServer(
        args.data_dir,
        args.host,
        args.port,
        KeyStore(args.keys_file) if args.keys_file else None,
        allow_anonymous=args.insecure_no_auth,
        tls_cert=args.tls_cert,
        tls_key=args.tls_key,
        drain_timeout=args.drain_timeout,
        checkpoint_ops=args.checkpoint_ops,
    )
    signal.signal(signal.SIGTERM, lambda *_: server.request_stop())
    signal.signal(signal.SIGINT, lambda *_: server.request_stop())
    server.start()
    server.wait_for_stop_request()
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
