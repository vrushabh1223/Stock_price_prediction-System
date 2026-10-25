"""
StockAI — Database Backup Script
Run manually: python backup_db.py
Or runs automatically every day via the app scheduler
Saves backups to /backups/ folder with timestamp
"""

import subprocess
import os
from datetime import datetime

# ── CONFIG ──────────────────────────────
DB_HOST     = "localhost"
DB_USER     = "root"
DB_PASSWORD = "root"
DB_NAME     = "stock_prediction"
BACKUP_DIR  = os.path.join(os.path.dirname(__file__), "backups")
KEEP_LAST   = 7   # keep last 7 backups
# ────────────────────────────────────────

def backup():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"stockai_backup_{timestamp}.sql")

    # Try mysqldump
    cmd = [
        "mysqldump",
        f"-h{DB_HOST}",
        f"-u{DB_USER}",
        f"-p{DB_PASSWORD}",
        "--single-transaction",
        "--routines",
        "--triggers",
        DB_NAME
    ]

    try:
        with open(backup_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            size = os.path.getsize(backup_file)
            print(f"✅ Backup saved: {backup_file} ({size} bytes)")
            cleanup_old_backups()
            return backup_file
        else:
            print(f"❌ mysqldump error: {result.stderr}")
            os.remove(backup_file)
            return None
    except FileNotFoundError:
        print("❌ mysqldump not found. Using Python fallback...")
        return python_backup(backup_file, timestamp)
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return None


def python_backup(backup_file, timestamp):
    """Pure Python backup using mysql-connector — no mysqldump needed."""
    try:
        import mysql.connector, json
        db = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        cursor = db.cursor(dictionary=True)

        cursor.execute("SHOW TABLES")
        tables = [list(row.values())[0] for row in cursor.fetchall()]

        all_data = {}
        for table in tables:
            cursor.execute(f"SELECT * FROM `{table}`")
            rows = cursor.fetchall()
            # Convert non-serializable types
            clean_rows = []
            for row in rows:
                clean_row = {}
                for k, v in row.items():
                    if hasattr(v, 'isoformat'):
                        clean_row[k] = v.isoformat()
                    else:
                        clean_row[k] = v
                clean_rows.append(clean_row)
            all_data[table] = clean_rows

        backup_json = backup_file.replace('.sql', '.json')
        with open(backup_json, 'w') as f:
            json.dump(all_data, f, indent=2, default=str)

        size = os.path.getsize(backup_json)
        print(f"✅ JSON Backup saved: {backup_json} ({size} bytes)")
        for table, rows in all_data.items():
            print(f"   {table}: {len(rows)} rows")

        cursor.close()
        db.close()
        cleanup_old_backups()
        return backup_json
    except Exception as e:
        print(f"❌ Python backup failed: {e}")
        return None


def cleanup_old_backups():
    """Keep only the last N backups."""
    try:
        files = sorted([
            os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR)
            if f.startswith("stockai_backup_")
        ])
        while len(files) > KEEP_LAST:
            os.remove(files.pop(0))
            print(f"🗑 Removed old backup")
    except Exception:
        pass


def restore(backup_file):
    """Restore from a JSON backup file."""
    try:
        import mysql.connector, json
        with open(backup_file, 'r') as f:
            all_data = json.load(f)

        db = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        cursor = db.cursor()

        for table, rows in all_data.items():
            if not rows:
                continue
            cols    = list(rows[0].keys())
            placeholders = ', '.join(['%s'] * len(cols))
            col_names    = ', '.join([f'`{c}`' for c in cols])
            sql = f"INSERT IGNORE INTO `{table}` ({col_names}) VALUES ({placeholders})"
            for row in rows:
                try:
                    cursor.execute(sql, list(row.values()))
                except Exception:
                    pass
            db.commit()
            print(f"✅ Restored {table}: {len(rows)} rows")

        cursor.close()
        db.close()
        print("\n✅ Restore complete!")
    except Exception as e:
        print(f"❌ Restore failed: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        if len(sys.argv) > 2:
            restore(sys.argv[2])
        else:
            # List available backups
            if os.path.exists(BACKUP_DIR):
                files = sorted([f for f in os.listdir(BACKUP_DIR) if f.startswith("stockai_backup_")])
                if files:
                    print("Available backups:")
                    for i, f in enumerate(files):
                        path = os.path.join(BACKUP_DIR, f)
                        size = os.path.getsize(path)
                        print(f"  [{i+1}] {f} ({size} bytes)")
                    print(f"\nUsage: python backup_db.py restore backups/<filename>")
                else:
                    print("No backups found.")
            else:
                print("No backups directory found.")
    else:
        backup()