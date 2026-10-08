"""
backend.auth
============

Users, password hashing and sessions -- shared by the web dashboard and the
mobile app, which both talk to this same backend and this same database.

* Passwords are stored as ``pbkdf2_sha256$<iterations>$<salt>$<hash>``
  (standard library only, no extra dependency to install on Windows).
* A login returns an opaque random token.  Only its SHA-256 is stored, so a
  copy of the database does not hand out live sessions.
* The token is accepted either as ``Authorization: Bearer <token>`` or as the
  ``uip_session`` HttpOnly cookie.  The cookie is what lets the existing
  ``<img src="/evidence/...">`` and ``<video src="/videos/...">`` tags keep
  working unchanged behind authentication.
* Deactivating a user revokes every session they hold.

Demo accounts are only created when ``auth.seed_demo_users`` is true AND the
users table is empty.  Nothing here contains a password: the demo password
comes from config / the environment, and if neither provides one a random
password is generated and written to ``outputs/demo_credentials.txt``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request

from backend import rbac

SESSION_COOKIE = "uip_session"

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id          TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    email            TEXT NOT NULL UNIQUE COLLATE NOCASE,
    username         TEXT UNIQUE COLLATE NOCASE,
    password_hash    TEXT NOT NULL,
    role             TEXT NOT NULL,
    department       TEXT NOT NULL,
    is_active        INTEGER NOT NULL DEFAULT 1,
    can_manage_users INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT,
    updated_at       TEXT,
    created_by       TEXT,
    last_login_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_department ON users(department, role);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    client        TEXT,
    created_at    REAL,
    expires_at    REAL,
    last_seen_at  REAL,
    user_agent    TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""

DEMO_ACCOUNTS = [
    ("admin@urbanintel.local", "Central Command Admin", rbac.CENTRAL_COMMAND, rbac.CENTRAL_COMMAND),
    ("traffic@urbanintel.local", "Traffic Control Officer", rbac.TRAFFIC_DEPARTMENT, rbac.TRAFFIC_DEPARTMENT),
    ("roads@urbanintel.local", "Roads Engineer", rbac.ROADS_INFRASTRUCTURE, rbac.ROADS_INFRASTRUCTURE),
    ("safety@urbanintel.local", "Public Safety Officer", rbac.PUBLIC_SAFETY, rbac.PUBLIC_SAFETY),
    ("transport@urbanintel.local", "Transport Controller", rbac.TRANSPORT_AUTHORITY, rbac.TRANSPORT_AUTHORITY),
    ("field@urbanintel.local", "Field Officer (Roads)", rbac.FIELD_OFFICER, rbac.ROADS_INFRASTRUCTURE),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ hashing

def hash_password(password: str, iterations: int = 600_000) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_b64, hash_b64 = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations)
        )
        return hmac.compare_digest(digest, expected)
    except Exception:                                            # noqa: BLE001
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- store

class AuthStore:
    """Users + sessions on top of the platform's single SQLite database."""

    def __init__(self, database, config, log=None):
        self.db = database
        self.config = config
        self.log = log
        section = config.section("auth")
        self.enabled = bool(section.get("enabled", True))
        self.iterations = int(section.get("pbkdf2_iterations", 600_000))
        self.web_ttl_s = float(section.get("session_ttl_hours", 12)) * 3600
        self.mobile_ttl_s = float(section.get("mobile_session_ttl_hours", 24 * 7)) * 3600
        self.min_password_length = int(section.get("min_password_length", 8))
        self.max_failures = int(section.get("max_failed_logins", 5))
        self.lockout_s = float(section.get("lockout_seconds", 60))
        self._failures: Dict[str, List[float]] = {}
        self._locked_until: Dict[str, float] = {}
        self._throttle_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def initialise(self) -> "AuthStore":
        self.db.executescript(AUTH_SCHEMA)
        self._seed_demo_users()
        return self

    def _seed_demo_users(self) -> None:
        section = self.config.section("auth")
        if not section.get("seed_demo_users", False):
            return
        existing = self.db.query_one("SELECT COUNT(*) AS n FROM users")
        if existing and existing["n"]:
            return
        password = section.get("demo_password")
        password = str(password) if password not in (None, "") else None
        generated = False
        if not password:
            password = secrets.token_urlsafe(9)
            generated = True
        for email, name, role, department in DEMO_ACCOUNTS:
            self.create_user(
                name=name, email=email, username=email.split("@", 1)[0],
                password=password, role=role, department=department,
                created_by="seed",
                can_manage_users=(role == rbac.CENTRAL_COMMAND),
            )
        target = None
        if generated:
            target = self.config.path("auth.demo_credentials_file", "outputs/demo_credentials.txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "Demo accounts (development only)\n"
                + "\n".join(f"{e}  /  {password}" for e, *_ in DEMO_ACCOUNTS) + "\n",
                encoding="utf-8",
            )
        if self.log:
            self.log.warning(
                "DEMO ACCOUNTS SEEDED (%s users)%s -- set auth.seed_demo_users: "
                "false for any real deployment",
                len(DEMO_ACCOUNTS),
                f"; password written to {target}" if target else "",
            )

    # -- users --------------------------------------------------------------

    @staticmethod
    def public(user: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if user is None:
            return None
        return {
            "user_id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
            "username": user.get("username"),
            "role": user["role"],
            "role_label": rbac.ROLE_LABEL.get(user["role"], user["role"]),
            "department": user["department"],
            "department_label": rbac.DEPARTMENT_LABEL.get(user["department"], user["department"]),
            "is_active": bool(user["is_active"]),
            "can_manage_users": bool(user.get("can_manage_users")) or user["role"] == rbac.CENTRAL_COMMAND,
            "created_at": user.get("created_at"),
            "last_login_at": user.get("last_login_at"),
        }

    def _next_user_id(self) -> str:
        row = self.db.query_one(
            "SELECT MAX(CAST(SUBSTR(user_id, 5) AS INTEGER)) AS n FROM users "
            "WHERE user_id LIKE 'USR-%'"
        )
        return f"USR-{(row['n'] or 0) + 1:04d}"

    def validate_password(self, password: Optional[str]) -> None:
        if not password or len(password) < self.min_password_length:
            raise HTTPException(
                422, f"password must be at least {self.min_password_length} characters"
            )

    def create_user(
        self, *, name: str, email: str, password: str, role: str, department: str,
        username: Optional[str] = None, created_by: Optional[str] = None,
        can_manage_users: bool = False, is_active: bool = True,
    ) -> Dict[str, Any]:
        error = rbac.valid_role_department(role, department)
        if error:
            raise HTTPException(422, error)
        email = (email or "").strip()
        if "@" not in email:
            raise HTTPException(422, "a valid email is required")
        if not (name or "").strip():
            raise HTTPException(422, "name is required")
        self.validate_password(password)
        username = (username or "").strip() or None
        clash = self.db.query_one(
            "SELECT user_id FROM users WHERE email = ? OR (username IS NOT NULL AND username = ?)",
            (email, username or "\x00"),
        )
        if clash:
            raise HTTPException(409, "a user with this email or username already exists")
        user_id = self._next_user_id()
        now = _now_iso()
        self.db.execute(
            "INSERT INTO users(user_id, name, email, username, password_hash, role, "
            "department, is_active, can_manage_users, created_at, updated_at, created_by) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                user_id, name.strip(), email, username,
                hash_password(password, self.iterations), role, department,
                1 if is_active else 0, 1 if can_manage_users else 0,
                now, now, created_by,
            ),
        )
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.db.query_one("SELECT * FROM users WHERE user_id = ?", (user_id,))

    def find_by_login(self, login: str) -> Optional[Dict[str, Any]]:
        login = (login or "").strip()
        if not login:
            return None
        return self.db.query_one(
            "SELECT * FROM users WHERE email = ? OR username = ?", (login, login)
        )

    def list_users(self, department: Optional[str] = None) -> List[Dict[str, Any]]:
        if department:
            rows = self.db.query(
                "SELECT * FROM users WHERE department = ? ORDER BY department, role, name",
                (department,),
            )
        else:
            rows = self.db.query("SELECT * FROM users ORDER BY department, role, name")
        return rows

    def update_user(self, user_id: str, **fields) -> Dict[str, Any]:
        user = self.get_user(user_id)
        if user is None:
            raise HTTPException(404, "user not found")
        role = fields.get("role") or user["role"]
        department = fields.get("department") or user["department"]
        error = rbac.valid_role_department(role, department)
        if error:
            raise HTTPException(422, error)
        updates: Dict[str, Any] = {"role": role, "department": department}
        if fields.get("name") is not None:
            if not str(fields["name"]).strip():
                raise HTTPException(422, "name cannot be empty")
            updates["name"] = str(fields["name"]).strip()
        if fields.get("is_active") is not None:
            updates["is_active"] = 1 if fields["is_active"] else 0
        if fields.get("can_manage_users") is not None:
            updates["can_manage_users"] = 1 if fields["can_manage_users"] else 0
        updates["updated_at"] = _now_iso()
        assignments = ", ".join(f"{k} = ?" for k in updates)
        self.db.execute(
            f"UPDATE users SET {assignments} WHERE user_id = ?",
            (*updates.values(), user_id),
        )
        if updates.get("is_active") == 0 or role != user["role"] or department != user["department"]:
            self.revoke_sessions(user_id)
        return self.get_user(user_id)

    def set_password(self, user_id: str, password: str) -> None:
        self.validate_password(password)
        self.db.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE user_id = ?",
            (hash_password(password, self.iterations), _now_iso(), user_id),
        )
        self.revoke_sessions(user_id)

    # -- throttling ---------------------------------------------------------

    def _throttle_key(self, ip: str, login: str) -> str:
        return f"{ip}|{(login or '').strip().lower()}"

    def check_locked(self, ip: str, login: str) -> Optional[int]:
        key = self._throttle_key(ip, login)
        with self._throttle_lock:
            until = self._locked_until.get(key, 0.0)
            remaining = until - time.time()
            return int(remaining) + 1 if remaining > 0 else None

    def record_failure(self, ip: str, login: str) -> None:
        key = self._throttle_key(ip, login)
        now = time.time()
        with self._throttle_lock:
            recent = [t for t in self._failures.get(key, []) if now - t < 300]
            recent.append(now)
            self._failures[key] = recent
            if len(recent) >= self.max_failures:
                self._locked_until[key] = now + self.lockout_s
                self._failures[key] = []

    def clear_failures(self, ip: str, login: str) -> None:
        key = self._throttle_key(ip, login)
        with self._throttle_lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)

    # -- sessions -----------------------------------------------------------

    def authenticate(self, login: str, password: str) -> Optional[Dict[str, Any]]:
        user = self.find_by_login(login)
        if user is None:
            # Burn comparable time so a missing account is not faster.
            verify_password(password or "", hash_password("x", 1_000))
            return None
        if not verify_password(password or "", user["password_hash"]):
            return None
        return user

    def create_session(self, user: Dict[str, Any], client: str = "web", user_agent: str = "") -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        ttl = self.mobile_ttl_s if client == "mobile" else self.web_ttl_s
        self.db.execute(
            "INSERT INTO sessions(token_hash, user_id, client, created_at, expires_at, "
            "last_seen_at, user_agent) VALUES (?,?,?,?,?,?,?)",
            (_token_hash(token), user["user_id"], client, now, now + ttl, now, (user_agent or "")[:200]),
        )
        self.db.execute(
            "UPDATE users SET last_login_at = ? WHERE user_id = ?",
            (_now_iso(), user["user_id"]),
        )
        self.db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        return token

    def resolve(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        digest = _token_hash(token)
        row = self.db.query_one(
            "SELECT s.token_hash, s.expires_at, s.last_seen_at, s.client, u.* "
            "FROM sessions s JOIN users u ON u.user_id = s.user_id "
            "WHERE s.token_hash = ?",
            (digest,),
        )
        if row is None:
            return None
        now = time.time()
        if row["expires_at"] < now or not row["is_active"]:
            self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (digest,))
            return None
        if now - (row["last_seen_at"] or 0) > 60:
            self.db.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?", (now, digest)
            )
        return row

    def revoke(self, token: Optional[str]) -> None:
        if token:
            self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))

    def revoke_sessions(self, user_id: str) -> None:
        self.db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


# ------------------------------------------------------ request dependencies

@dataclass
class CurrentUser:
    user: Dict[str, Any]
    scope: rbac.Scope
    token: Optional[str] = None

    @property
    def id(self) -> str:
        return self.user["user_id"]

    @property
    def name(self) -> str:
        return self.user["name"]


def request_token(request: Request) -> Optional[str]:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return request.cookies.get(SESSION_COOKIE)


def scope_for(user: Dict[str, Any]) -> rbac.Scope:
    return rbac.Scope(
        user_id=user["user_id"],
        role=user["role"],
        department=user["department"],
        can_manage_users=bool(user.get("can_manage_users")) or user["role"] == rbac.CENTRAL_COMMAND,
    )


_SYSTEM_USER = {
    "user_id": "USR-SYSTEM", "name": "Auth disabled", "email": "system@local",
    "username": "system", "role": rbac.CENTRAL_COMMAND,
    "department": rbac.CENTRAL_COMMAND, "is_active": 1, "can_manage_users": 1,
}


def optional_user(request: Request) -> Optional[CurrentUser]:
    auth: AuthStore = request.app.state.auth
    if not auth.enabled:
        return CurrentUser(user=_SYSTEM_USER, scope=scope_for(_SYSTEM_USER))
    token = request_token(request)
    user = auth.resolve(token)
    if user is None:
        return None
    return CurrentUser(user=user, scope=scope_for(user), token=token)


def current_user(request: Request) -> CurrentUser:
    who = optional_user(request)
    if who is None:
        raise HTTPException(401, "login required")
    return who


def require_roles(*roles: str):
    allowed = set(roles)

    def dependency(request: Request) -> CurrentUser:
        who = current_user(request)
        if who.scope.role not in allowed:
            raise HTTPException(403, "your role cannot access this resource")
        return who

    return dependency


def require_types(*event_types: str):
    """Allow the request when the user may see ANY of these event types."""

    def dependency(request: Request) -> CurrentUser:
        who = current_user(request)
        if not who.scope.sees_any(event_types):
            raise HTTPException(403, "this data belongs to another department")
        return who

    return dependency


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"
