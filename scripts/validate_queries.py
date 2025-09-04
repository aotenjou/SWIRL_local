#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import psycopg2


def parse_args():
    parser = argparse.ArgumentParser(description="Validate SQL query files by running EXPLAIN on each.")
    parser.add_argument(
        "--dir",
        default="/home/baiyutao/SWIRL/query_files/BASKETBALL",
        help="Directory containing query files (default: %(default)s)",
    )
    parser.add_argument(
        "--pattern",
        default="BASKETBALL_*.txt",
        help="Filename glob pattern (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of files to validate (sorted lexicographically) (default: %(default)s)",
    )
    parser.add_argument("--db", required=True, help="Postgres database name to connect to")
    parser.add_argument("--host", default=os.getenv("DATABASE_HOST", "localhost"), help="Postgres host")
    parser.add_argument("--port", default=os.getenv("DATABASE_PORT", "54321"), help="Postgres port")
    parser.add_argument("--user", default=os.getenv("PGUSER", os.getenv("USER", "postgres")), help="Postgres user")
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", None), help="Postgres password (optional)")
    return parser.parse_args()


def make_connection(args):
    conn_params = {
        "dbname": args.db,
        "host": args.host,
        "port": args.port,
        "user": args.user,
    }
    if args.password:
        conn_params["password"] = args.password
    return psycopg2.connect(**conn_params)


def main():
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        print(f"No files matched in {args.dir} with pattern {args.pattern}")
        sys.exit(2)

    files = files[: args.limit]

    try:
        conn = make_connection(args)
    except Exception as e:
        print(f"Failed to connect to Postgres: {e}")
        sys.exit(2)

    conn.autocommit = True
    cur = conn.cursor()

    total = len(files)
    ok = 0
    failed = 0

    for path in files:
        rel = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                sql = f.read().strip()
        except Exception as e:
            print(f"[FAIL] {rel}: unable to read file: {e}")
            failed += 1
            continue

        if not sql:
            print(f"[FAIL] {rel}: file is empty")
            failed += 1
            continue

        try:
            cur.execute("EXPLAIN " + sql)
            plan_rows = cur.fetchall()
            if plan_rows:
                print(f"[OK]   {rel}")
            else:
                print(f"[OK]   {rel} (no rows returned by EXPLAIN)")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {rel}: {e}")
            failed += 1

    cur.close()
    conn.close()

    print(f"\nSummary: {ok} OK, {failed} FAIL, {total} total")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()


