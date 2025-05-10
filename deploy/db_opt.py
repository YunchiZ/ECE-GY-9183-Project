import os, sqlite3, logging, time
from typing import List, Dict, Any

DEPLOY_DATA_DIR = '/app/deploy_data'
SERVING_DB      = os.path.join(DEPLOY_DATA_DIR, "serving.db")
CANDIDATE_DB    = os.path.join(DEPLOY_DATA_DIR, "candidate.db")


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        logging.info("Created dir: %s", path)


def _open_conn(db_path: str) -> sqlite3.Connection:
    """Unified open method with WAL + busy_timeout"""
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)  # autocommit
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")        # ms
    return conn


# ---------- 1. Initialization ----------

def init_db() -> None:
    """
    Pre-create `serving.db` and `candidate.db`, each containing three tables: task0~2.
    """
    _ensure_dir(DEPLOY_DATA_DIR)

    schema = {
        0: "pred TEXT",
        1: "pred INTEGER",
        2: "pred INTEGER"
    }

    for db in (SERVING_DB, CANDIDATE_DB):
        try:
            with _open_conn(db) as conn:
                cur = conn.cursor()
                for idx, pred_col in schema.items():
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS task{idx} (
                            id   INTEGER PRIMARY KEY,
                            text TEXT,
                            {pred_col},
                            time REAL
                        )
                    """)
            logging.info("Initialized database %s", db)
        except sqlite3.Error as e:
            logging.error("Init DB %s failed: %s", db, e)


def reset_table(index: int, db_path: str) -> None:
    if index not in (0, 1, 2):
        logging.error("reset_table: invalid index %s", index)
        return

    pred_col = "pred TEXT" if index == 0 else "pred INTEGER"
    try:
        with _open_conn(db_path) as conn:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS task{index}")
            cur.execute(f"""
                CREATE TABLE task{index} (
                    id   INTEGER PRIMARY KEY,
                    text TEXT,
                    {pred_col},
                    time REAL
                )
            """)
        logging.info("Reset table task%s in %s", index, db_path)
    except sqlite3.Error as e:
        logging.error("Reset table task%s failed: %s", index, e)


def batch_write(records: List[Dict[str, Any]],
                index: int,
                db_path: str,
                *,
                retries: int = 3,
                retry_delay: float = 1.0) -> None:
    """
        Batch write records into an existing database table.
        Assumes that the table has been created previously using a reset_table function:
          - If index == 0, the table schema is assumed to be:
                CREATE TABLE task0 (
                    id INTEGER PRIMARY KEY,
                    text TEXT,
                    pred TEXT,
                    time REAL
                )
          - Otherwise (if index == 1 or 2), the table schema is assumed to be:
                CREATE TABLE taskX (
                    id INTEGER PRIMARY KEY,
                    text TEXT,
                    pred INTEGER,
                    time REAL
                )
    """
    if index not in (0, 1, 2):
        logging.error("batch_write: invalid index %s", index)
        return

    is_text_task = (index == 0)
    table = f"task{index}"
    sql   = f"""
        INSERT OR REPLACE INTO {table} (id, text, pred, time)
        VALUES (?, ?, ?, ?)
    """

    # ---------- Organize Data ----------
    data: list[tuple] = []
    for item in records:
        try:
            if is_text_task:
                pred_val = str(item["pred"]) if item["pred"] is not None else None
            else:
                pred_val = int(item["pred"]) if isinstance(item["pred"], int) else None

            row = (
                int(item["id"]),
                str(item.get("text", "")),
                pred_val,
                float(item.get("time", 0.0))
            )
            data.append(row)
        except Exception as e:
            logging.warning("Skip malformed record %s: %s", item, e)

    if not data:
        logging.warning("batch_write: no valid rows for %s", table)
        return

    # ---------- Write + Retry ----------
    for attempt in range(1, retries + 1):
        try:
            with _open_conn(db_path) as conn:
                conn.executemany(sql, data)
            logging.info("Inserted %d rows into %s (try %d)",
                         len(data), table, attempt)
            return

        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < retries:
                logging.warning("DB locked, retry %d/%d in %.1fs",
                                attempt, retries, retry_delay)
                time.sleep(retry_delay)
                continue
            logging.error("batch_write OperationalError on %s: %s", table, e)
            break

        except Exception as e:
            logging.error("batch_write error on %s: %s", table, e)
            break

    logging.error("batch_write: give up after %d retries on %s", retries, table)




