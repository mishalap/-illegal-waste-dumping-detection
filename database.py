import sqlite3
import json
import os
from datetime import datetime
from config import DB_PATH, DATA_DIR


def get_connection():
    """Get a connection to the SQLite database."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_url TEXT,
                device_index INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                snapshot_path TEXT,
                description TEXT,
                objects_detected TEXT,
                incident_type TEXT DEFAULT 'person_dump',
                FOREIGN KEY (camera_id) REFERENCES cameras(id)
            )
        """)
        # Migration: add incident_type if table already existed without it
        try:
            cursor.execute(
                "ALTER TABLE incidents ADD COLUMN incident_type TEXT DEFAULT 'person_dump'"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT,
                status TEXT DEFAULT 'queued',
                total_frames INTEGER,
                processed_frames INTEGER DEFAULT 0,
                incidents_found INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER,
                frame_number INTEGER,
                timestamp_in_video REAL,
                confidence REAL,
                snapshot_path TEXT,
                objects_detected TEXT,
                incident_type TEXT DEFAULT 'person_dump',
                FOREIGN KEY (upload_id) REFERENCES video_uploads(id)
            )
        """)
        try:
            cursor.execute(
                "ALTER TABLE video_detections ADD COLUMN incident_type TEXT DEFAULT 'person_dump'"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass

        # Ensure "Video Upload" camera exists so incidents from uploads use a constant camera_id
        cursor.execute(
            "INSERT INTO cameras (name, source_type, source_url, device_index) "
            "SELECT 'Video Upload', 'video_upload', NULL, 0 "
            "WHERE NOT EXISTS (SELECT 1 FROM cameras WHERE source_type = 'video_upload')"
        )

        conn.commit()
    finally:
        conn.close()


# ============================================================
# Camera CRUD
# ============================================================

def insert_camera(name, source_type, source_url=None, device_index=0):
    """Add a new camera source."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO cameras (name, source_type, source_url, device_index) VALUES (?, ?, ?, ?)",
            (name, source_type, source_url, device_index)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


# Source type for the synthetic "Video Upload" camera (used for incidents from uploaded videos)
VIDEO_UPLOAD_SOURCE_TYPE = "video_upload"


def get_video_upload_camera_id():
    """
    Return the camera_id for the 'Video Upload' source. This constant camera is used
    so incidents from uploaded videos appear in Recent Incidents and Incident Viewer.
    Creates the camera record if it does not exist.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM cameras WHERE source_type = ? LIMIT 1",
            (VIDEO_UPLOAD_SOURCE_TYPE,)
        ).fetchone()
        if row:
            return row["id"]
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO cameras (name, source_type, source_url, device_index) VALUES (?, ?, ?, ?)",
            ("Video Upload", VIDEO_UPLOAD_SOURCE_TYPE, None, 0)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_cameras():
    """Get all registered cameras."""
    conn = get_connection()
    try:
        cameras = conn.execute("SELECT * FROM cameras ORDER BY created_at DESC").fetchall()
        return [dict(c) for c in cameras]
    finally:
        conn.close()


def get_camera(camera_id):
    """Get a single camera by ID."""
    conn = get_connection()
    try:
        camera = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        return dict(camera) if camera else None
    finally:
        conn.close()


def delete_camera(camera_id):
    """Delete a camera source."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        conn.commit()
    finally:
        conn.close()


# ============================================================
# Incident CRUD
# ============================================================

def insert_incident(camera_id, confidence, snapshot_path, description, objects_detected,
                   incident_type="person_dump"):
    """Log a new detected incident. incident_type: 'person_dump' or 'car_litter'."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        objects_json = json.dumps(objects_detected) if isinstance(objects_detected, list) else objects_detected
        cursor.execute(
            "INSERT INTO incidents (camera_id, confidence, snapshot_path, description, objects_detected, incident_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, confidence, snapshot_path, description, objects_json, incident_type)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_incidents(limit=50, camera_id=None, date_from=None, date_to=None):
    """Get incidents with optional filters."""
    conn = get_connection()
    try:
        query = "SELECT * FROM incidents WHERE 1=1"
        params = []

        if camera_id is not None:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if date_from is not None:
            query += " AND timestamp >= ?"
            params.append(date_from)
        if date_to is not None:
            query += " AND timestamp < ?"
            params.append(date_to)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        incidents = conn.execute(query, params).fetchall()
        return [dict(i) for i in incidents]
    finally:
        conn.close()


def get_incident(incident_id):
    """Get a single incident by ID."""
    conn = get_connection()
    try:
        incident = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        return dict(incident) if incident else None
    finally:
        conn.close()


def get_incident_count(today_only=False):
    """Get total incident count, optionally for today only."""
    conn = get_connection()
    try:
        if today_only:
            today = datetime.now().strftime("%Y-%m-%d")
            result = conn.execute(
                "SELECT COUNT(*) as count FROM incidents WHERE date(timestamp) = ?", (today,)
            ).fetchone()
        else:
            result = conn.execute("SELECT COUNT(*) as count FROM incidents").fetchone()
        return result["count"]
    finally:
        conn.close()


def get_incident_count_by_camera(today_only=False):
    """
    Get incident count per location (camera_id). Returns list of dicts with
    camera_id, camera_name, count. Includes all registered cameras (count 0 if none).
    Also includes one row for incidents with NULL camera_id as 'Unknown / Video'.
    """
    conn = get_connection()
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        date_filter = " AND date(i.timestamp) = ?" if today_only else ""
        params = [today] if today_only else []

        # Counts per camera (only cameras that have incidents)
        sql = """
            SELECT c.id AS camera_id, c.name AS camera_name, COUNT(i.id) AS count
            FROM cameras c
            LEFT JOIN incidents i ON i.camera_id = c.id
            """ + (" AND date(i.timestamp) = ?" if today_only else "") + """
            GROUP BY c.id, c.name
        """
        if today_only:
            # For LEFT JOIN with date filter we need the condition in ON or WHERE carefully
            sql = """
                SELECT c.id AS camera_id, c.name AS camera_name,
                       (SELECT COUNT(*) FROM incidents WHERE camera_id = c.id AND date(timestamp) = ?) AS count
                FROM cameras c
            """
            params = [today]

        rows = conn.execute(sql, params).fetchall()
        result = [{"camera_id": r["camera_id"], "camera_name": r["camera_name"], "count": r["count"]} for r in rows]

        # Add row for incidents with NULL camera_id (e.g. from video uploads)
        null_sql = "SELECT COUNT(*) AS count FROM incidents WHERE camera_id IS NULL"
        if today_only:
            null_sql += " AND date(timestamp) = ?"
        null_params = [today] if today_only else []
        null_row = conn.execute(null_sql, null_params).fetchone()
        if null_row and null_row["count"] > 0:
            result.append({"camera_id": None, "camera_name": "Unknown / Video", "count": null_row["count"]})

        # Sort by count descending so highest appears first
        result.sort(key=lambda x: x["count"], reverse=True)
        return result
    finally:
        conn.close()


# ============================================================
# Video Upload CRUD
# ============================================================

def insert_video_upload(filename, original_name, total_frames=0):
    """Create a new video upload record."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO video_uploads (filename, original_name, total_frames) VALUES (?, ?, ?)",
            (filename, original_name, total_frames)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_video_upload(upload_id, status=None, processed_frames=None, incidents_found=None):
    """Update a video upload's processing status."""
    conn = get_connection()
    try:
        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if processed_frames is not None:
            updates.append("processed_frames = ?")
            params.append(processed_frames)
        if incidents_found is not None:
            updates.append("incidents_found = ?")
            params.append(incidents_found)

        if updates:
            params.append(upload_id)
            conn.execute(
                f"UPDATE video_uploads SET {', '.join(updates)} WHERE id = ?",
                params
            )
            conn.commit()
    finally:
        conn.close()


def get_video_uploads():
    """Get all video uploads."""
    conn = get_connection()
    try:
        uploads = conn.execute("SELECT * FROM video_uploads ORDER BY created_at DESC").fetchall()
        return [dict(u) for u in uploads]
    finally:
        conn.close()


def get_video_upload(upload_id):
    """Get a single video upload by ID."""
    conn = get_connection()
    try:
        upload = conn.execute("SELECT * FROM video_uploads WHERE id = ?", (upload_id,)).fetchone()
        return dict(upload) if upload else None
    finally:
        conn.close()


# ============================================================
# Video Detection CRUD
# ============================================================

def insert_video_detection(upload_id, frame_number, timestamp_in_video, confidence,
                           snapshot_path, objects_detected, incident_type="person_dump"):
    """Log a detection found in an uploaded video. incident_type: 'person_dump' or 'car_litter'."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        objects_json = json.dumps(objects_detected) if isinstance(objects_detected, list) else objects_detected
        cursor.execute(
            "INSERT INTO video_detections "
            "(upload_id, frame_number, timestamp_in_video, confidence, snapshot_path, objects_detected, incident_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (upload_id, frame_number, timestamp_in_video, confidence, snapshot_path, objects_json, incident_type)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_video_detections(upload_id):
    """Get all detections for a given video upload."""
    conn = get_connection()
    try:
        detections = conn.execute(
            "SELECT * FROM video_detections WHERE upload_id = ? ORDER BY frame_number",
            (upload_id,)
        ).fetchall()
        return [dict(d) for d in detections]
    finally:
        conn.close()

