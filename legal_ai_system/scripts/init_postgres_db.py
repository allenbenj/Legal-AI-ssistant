#!/usr/bin/env python3
"""Initialize a PostgreSQL database if it does not already exist."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import asyncpg


async def ensure_database_exists(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    admin_db: str = "postgres",
) -> None:
    """Create the target database if it does not exist.

    Parameters
    ----------
    host: str
        PostgreSQL server host name.
    port: int
        Server port.
    user: str
        Username with privileges to create databases.
    password: str
        Corresponding password.
    database: str
        Name of the database to create if missing.
    admin_db: str
        Database to connect to for checking/creating the target database.
    """
    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=admin_db,
    )
    exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname=$1", database)
    if not exists:
        await conn.execute(f'CREATE DATABASE "{database}"')
    await conn.close()


async def main() -> None:
    """Entry point for command-line execution."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DB", "legal_ai")
    admin_db = os.getenv("POSTGRES_ADMIN_DB", "postgres")

    await ensure_database_exists(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        admin_db=admin_db,
    )


if __name__ == "__main__":
    asyncio.run(main())
