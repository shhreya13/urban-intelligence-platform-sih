#!/usr/bin/env python3
"""
scripts/manage_users.py
=======================

Manage platform accounts from the command line, for deployments where
``auth.seed_demo_users`` is false and nobody can sign in yet, or when
the only Central Command account is locked out.

Passwords are never taken as command-line arguments (they would end up
in shell history and in ``ps`` output): you are prompted for them, or
they are read from the ``UIP_NEW_PASSWORD`` environment variable for
scripted setups.

    python scripts/manage_users.py list
    python scripts/manage_users.py roles
    python scripts/manage_users.py create --name "Priya R" --email priya@city.gov \
        --role FIELD_OFFICER --department ROADS_INFRASTRUCTURE
    python scripts/manage_users.py set-password priya@city.gov
    python scripts/manage_users.py deactivate priya@city.gov
    python scripts/manage_users.py activate priya@city.gov

Changing a password or deactivating an account signs that user out of
every web and mobile session immediately.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import HTTPException                     # noqa: E402

from backend import rbac                              # noqa: E402
from backend.auth import AuthStore                    # noqa: E402
from backend.db import Database                       # noqa: E402
from core.config import load_config                   # noqa: E402


def _password(confirm: bool = True) -> str:
    env = os.environ.get("UIP_NEW_PASSWORD")
    if env:
        return env
    first = getpass.getpass("New password: ")
    if confirm and getpass.getpass("Repeat password: ") != first:
        raise SystemExit("passwords do not match")
    return first


def _find(auth: AuthStore, login: str):
    user = auth.find_by_login(login)
    if user is None:
        raise SystemExit(f"no user with email/username {login!r}")
    return user


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manage Urban Intelligence platform users")
    parser.add_argument("--db", help="database (default: backend.database from config)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list every account")
    sub.add_parser("roles", help="list valid roles and departments")
    c = sub.add_parser("create", help="create an account (prompts for the password)")
    c.add_argument("--name", required=True)
    c.add_argument("--email", required=True)
    c.add_argument("--username", help="defaults to the part of the email before @")
    c.add_argument("--role", required=True, choices=rbac.ROLES)
    c.add_argument("--department", help="required for FIELD_OFFICER; defaults to the role's own department")
    c.add_argument("--can-manage-users", action="store_true")
    for name in ("set-password", "deactivate", "activate"):
        s = sub.add_parser(name)
        s.add_argument("login", help="email or username")
    args = parser.parse_args(argv)

    overrides = {"auth": {"seed_demo_users": False}}          # never seed demo users from this tool
    if args.db:
        overrides["backend"] = {"database": args.db}
    config = load_config(overrides=overrides)
    database = Database(config.path("backend.database")).initialise()
    auth = AuthStore(database, config).initialise()

    try:
        if args.cmd == "list":
            rows = auth.list_users()
            if not rows:
                print("no users yet -- create one with:  manage_users.py create ... --role CENTRAL_COMMAND")
            for u in rows:
                state = "active" if u["is_active"] else "DEACTIVATED"
                print(f"{u['user_id']}  {u['email']:<32} {u['role']:<22} {u['department']:<22} {state}")
        elif args.cmd == "roles":
            for role in rbac.ROLES:
                print(f"{role:<22} {rbac.ROLE_LABEL.get(role, role)}")
            print("\nFIELD_OFFICER needs --department set to one of the operating departments.")
        elif args.cmd == "create":
            department = args.department or (None if args.role == rbac.FIELD_OFFICER else args.role)
            if department is None:
                raise SystemExit("--department is required for FIELD_OFFICER")
            user = auth.create_user(
                name=args.name, email=args.email, username=args.username or args.email.split("@", 1)[0],
                password=_password(),
                role=args.role, department=department, created_by="cli",
                can_manage_users=args.can_manage_users or args.role == rbac.CENTRAL_COMMAND,
            )
            print(f"created {user['user_id']} {user['email']} ({user['role']} / {user['department']})")
        elif args.cmd == "set-password":
            user = _find(auth, args.login)
            auth.set_password(user["user_id"], _password())
            print(f"password changed for {user['email']}; existing sessions signed out")
        elif args.cmd in ("deactivate", "activate"):
            user = _find(auth, args.login)
            auth.update_user(user["user_id"], is_active=args.cmd == "activate")
            print(f"{user['email']} {args.cmd}d")
    except HTTPException as exc:
        raise SystemExit(f"error: {exc.detail}")
    finally:
        database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
