import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "bookings.db"


def initialize_store():
    with closing(sqlite3.connect(DB_PATH)) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_at TEXT NOT NULL,
                end_at TEXT NOT NULL,
                CHECK (start_at < end_at)
            )
        """)


def save_booking(start, end):
    start_text = start.isoformat(timespec="minutes")
    end_text = end.isoformat(timespec="minutes")

    with closing(sqlite3.connect(DB_PATH)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        overlap = db.execute(
            "SELECT 1 FROM bookings WHERE start_at < ? AND end_at > ? LIMIT 1",
            (end_text, start_text),
        ).fetchone()

        if overlap:
            return None

        row = db.execute(
            "INSERT INTO bookings (start_at, end_at) VALUES (?, ?)",
            (start_text, end_text),
        )
        return f"M{row.lastrowid}"


def lookup_booking(booking_id):
    code = booking_id.strip().upper()
    digits = code[1:]
    if not code.startswith("M") or not digits.isascii() or not digits.isdigit():
        return "Use a booking ID like M2."

    with closing(sqlite3.connect(DB_PATH)) as db:
        row = db.execute(
            "SELECT start_at, end_at FROM bookings WHERE id = ?",
            (digits,),
        ).fetchone()

    if row is None:
        return f"I couldn't find booking {code}."

    start, end = (datetime.fromisoformat(value) for value in row)
    return (
        f"Booking {code}: microscope, "
        f"{start:%d %b %Y at %I:%M %p} to {end:%d %b %Y at %I:%M %p}."
    )


def cancel_booking(booking_id):
    code = booking_id.strip().upper()
    digits = code[1:]
    if not code.startswith("M") or not digits.isascii() or not digits.isdigit():
        return "Use a booking ID like M2."

    with closing(sqlite3.connect(DB_PATH)) as db, db:
        result = db.execute("DELETE FROM bookings WHERE id = ?", (digits,))

    if result.rowcount == 0:
        return f"Booking {code} was not found. It may already be cancelled."

    return f"Booking {code} cancelled. The slot is available again."