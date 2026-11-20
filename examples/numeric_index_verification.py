"""Reproduction and regression verification for numeric equality-index tokens.

The regression: codec.scalar_token() collapsed distinct JSON numbers into the
same index token (integers above 2**53 were rounded through float(), floats
were formatted with only 15 significant digits). Unique indexes then reported
false conflicts, while ordinary equality queries still looked correct because
the final predicate re-check filtered the wrong candidates out.

Usage, from the repository root:

  python3 -m examples.numeric_index_verification generate <dir>
      Run with the PRE-FIX code. Creates file1.db and file2.db plus
      manifest.json inside <dir> and refuses to overwrite an existing one.
      file1.db: loose "docs" collection, a=2**53, b=2**53+1, c=0.1,
      d=nextafter(0.1, inf), plain equality index on "value".
      file2.db: unique index on "value" created first, only a written.
  python3 -m examples.numeric_index_verification demonstrate <dir>
      Run with the PRE-FIX code. Copies the saved files to a scratch area and
      shows the false unique conflicts; exits nonzero when the bug is absent.
  python3 -m examples.numeric_index_verification verify <dir>
      Run with the FIXED code against the ORIGINAL files (hash-checked against
      the manifest). Checks the automatic projection rebuild, equality and
      uniqueness semantics against an independent oracle, rollback
      completeness, request replay, historical reads and non-numeric indexes.
      Safe to re-run; file1 mutations are restored to the canonical state at
      the end of every run.
"""
import json
import math
import shutil
import sqlite3
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from strataleaf import Archives, Conflict, Database

A = 9007199254740992             # 2**53
B = 9007199254740993             # 2**53 + 1; float(B) rounds back to A
C = 0.1
D = math.nextafter(C, math.inf)  # the float immediately above 0.1
SEED_REQUEST = "file1_seed"
TABLES = ("meta", "collections", "index_defs", "versions", "commits",
          "receipts", "index_values", "pins", "consumers")
MISSING = object()
CHECKS = [0]


def check(condition, message):
    CHECKS[0] += 1
    if not condition:
        raise AssertionError(message)


def oracle_equal(left, right):
    """Independent equality oracle: JSON numbers compare by exact value."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if left is None or right is None:
        return left is None and right is None
    return type(left) is type(right) and left == right


def expected_token(value):
    """Independent reimplementation of the required index-token semantics."""
    if value is MISSING or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "b:" + str(int(value))
    if isinstance(value, (int, float)):
        ratio = Fraction(value)
        return "n:" + str(ratio.numerator) + "/" + str(ratio.denominator)
    return "v:" + json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)


def field_value(document, path):
    value = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return MISSING
        value = value[part]
    return value


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_truncate(path):
    raw = sqlite3.connect(str(path))
    raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw.close()
    check(not Path(str(path) + "-wal").exists(), "WAL file remains after checkpoint: " + str(path))


def dump_tables(path):
    raw = sqlite3.connect(str(path))
    try:
        return {table: [list(row) for row in raw.execute("SELECT * FROM " + table).fetchall()]
                for table in TABLES}
    finally:
        raw.close()


def table_snapshot(db):
    with db._store.read() as connection:
        return {table: sorted((tuple(row) for row in connection.execute("SELECT * FROM " + table)),
                              key=repr)
                for table in TABLES}


def visible_documents(versions_rows):
    latest = {}
    for collection, key, rev, body, _expires in versions_rows:
        identity = (collection, key)
        if identity not in latest or rev > latest[identity][0]:
            latest[identity] = (rev, body)
    return {identity: json.loads(body) for identity, (rev, body) in latest.items() if body is not None}


def projection_exact(db):
    """index_values must equal the projection recomputed from versions+defs."""
    with db._store.read() as connection:
        definitions = [tuple(row) for row in connection.execute(
            "SELECT collection,name,field,unique_flag FROM index_defs")]
        versions = [tuple(row) for row in connection.execute(
            "SELECT collection,key,rev,body,expires FROM versions")]
        actual = {tuple(row) for row in connection.execute(
            "SELECT collection,name,token,key FROM index_values")}
    visible = visible_documents(versions)
    expected = set()
    for collection, name, path, unique_flag in definitions:
        seen = {}
        for (doc_collection, key), document in visible.items():
            if doc_collection != collection:
                continue
            token = expected_token(field_value(document, path))
            if token is None:
                continue
            if unique_flag:
                check(token not in seen or seen[token] == key,
                      "unique index %s holds duplicate token for keys %s and %s"
                      % (name, seen.get(token), key))
                seen[token] = key
            expected.add((collection, name, token, key))
    check(actual == expected, "index_values differ from recomputed projection: %s"
          % sorted(expected.symmetric_difference(actual), key=repr)[:4])


def expect_eq(db, collection, path, probe, expected_keys, plan_index="__any__"):
    """Equality query must match the oracle, via the index and via a full scan."""
    predicate = {"field": path, "op": "eq", "value": probe}
    page = db.query(collection, predicate, limit=1000)
    keys = sorted(record.key for record in page.records)
    check(keys == sorted(expected_keys),
          "eq %r returned %s, expected %s" % (probe, keys, sorted(expected_keys)))
    for record in page.records:
        check(oracle_equal(record.document[path], probe),
              "record %s value %r is not oracle-equal to probe %r"
              % (record.key, record.document[path], probe))
    scanned = db.query(collection, predicate, limit=1000, force_scan=True)
    check(sorted(record.key for record in scanned.records) == keys,
          "force_scan disagrees with index path for probe %r" % (probe,))
    scalar = expected_token(probe) is not None
    expected_access = "equality_index" if scalar and plan_index is not None else "version_scan"
    check(page.plan["access"] == expected_access,
          "plan %s, expected access %s for probe %r" % (page.plan, expected_access, probe))
    if scalar and plan_index not in (None, "__any__"):
        check(page.plan["index"] == plan_index,
              "plan %s, expected index %s" % (page.plan, plan_index))
    return page


def expect_conflict(call, db, label):
    """A rejected write/index change must roll back every table completely."""
    before = table_snapshot(db)
    try:
        call()
    except Conflict:
        pass
    else:
        check(False, label + ": operation unexpectedly succeeded")
    after = table_snapshot(db)
    check(before == after, label + ": rejected operation left state changes behind")


def facts_preserved(db, entry, label):
    """Migration may only add derived state; stored facts are append-only."""
    snap = table_snapshot(db)
    old = entry["tables"]
    for table in ("versions", "commits", "receipts", "collections", "index_defs", "pins", "consumers"):
        check({tuple(row) for row in old[table]} <= set(snap[table]),
              "%s: %s rows from the pre-fix file were altered or lost" % (label, table))
    meta_old = {row[0]: row[1] for row in old["meta"]}
    meta_new = {row[0]: row[1] for row in snap["meta"]}
    check(meta_new.get("token_format") == "2", label + ": token_format marker missing after open")
    for key in ("format", "floor", "database_id", "cursor_secret", "mode", "upstream"):
        check(meta_new[key] == meta_old[key], "%s: meta %s changed" % (label, key))
    check(int(meta_new["head"]) >= int(meta_old["head"]), label + ": head moved backwards")
    check(int(meta_new["catalog_epoch"]) >= int(meta_old["catalog_epoch"]),
          label + ": catalog_epoch moved backwards")


def file_hash_ready(path):
    if Path(str(path) + "-wal").exists():
        checkpoint_truncate(path)
    return sha256(path)


def load_manifest(folder):
    with open(folder / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    check(float.fromhex(manifest["values"]["D_hex"]) == D and D != C,
          "manifest float boundary mismatch")
    check(manifest["values"]["A"] == A and manifest["values"]["B"] == B, "manifest integer mismatch")
    return manifest


def check_file_lineage(path, label, folder, manifest, state):
    digest = file_hash_ready(path)
    known = {manifest["files"][label]["sha256"]}
    known.update(state["hashes"].get(label, []))
    check(digest in known, "%s: file hash %s matches neither the pre-fix manifest nor a "
          "recorded post-verify state; refusing to use a regenerated file" % (label, digest))
    return digest


# ---------------------------------------------------------------- generate

def generate(folder):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit("refusing to overwrite existing directory: " + str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    file1, file2 = folder / "file1.db", folder / "file2.db"
    with Database(file1) as db:
        db.catalog.create("docs")
        batch = db.batch(SEED_REQUEST)
        for key, value in (("a", A), ("b", B), ("c", C), ("d", D)):
            batch.put("docs", key, {"value": value})
        result = batch.commit()
        check(result.revision == 1 and not result.replayed, "unexpected seed commit result")
        db.indexes.create("docs", "by_value", "value")
    with Database(file2) as db:
        db.catalog.create("docs")
        db.indexes.create("docs", "value_unique", "value", unique=True)
        db.batch().put("docs", "a", {"value": A}).commit()
    manifest = {"values": {"A": A, "B": B, "C": repr(C), "D": repr(D), "D_hex": D.hex()},
                "seed_request": SEED_REQUEST, "files": {}}
    for label, path in (("file1", file1), ("file2", file2)):
        checkpoint_truncate(path)
        entry = {"sha256": sha256(path), "tables": dump_tables(path)}
        tokens = {}
        for _collection, _name, token, key in entry["tables"]["index_values"]:
            tokens[key] = token
        if label == "file1":
            check(tokens["a"] == tokens["b"] and tokens["c"] == tokens["d"],
                  "on-disk token collision missing; was this file written by the regressing version?")
            check(tokens["a"] != tokens["c"], "unexpected token layout")
        if label == "file2":
            check(tokens["a"] == "n:" + str(A), "unexpected pre-fix token for a: " + tokens["a"])
        manifest["files"][label] = entry
    shutil.copyfile(file1, folder / "file1.crashseed.db")
    manifest["crashseed"] = {"sha256": sha256(folder / "file1.crashseed.db")}
    with open(folder / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print(json.dumps({"phase": "generate", "dir": str(folder), "file1_head": 1, "file2_head": 1,
                      "colliding_tokens": {"file1": [["a", "b"], ["c", "d"]]}}, sort_keys=True))


# ------------------------------------------------------------ demonstrate

def demonstrate(folder):
    folder = Path(folder)
    manifest = load_manifest(folder)
    observations = {"queries_correct_despite_bug": None, "deep_check_ok_despite_bug": None,
                    "unique_create_falsely_rejected": None, "unique_write_falsely_rejected": None,
                    "legitimate_conflict_still_detected": None}
    with tempfile.TemporaryDirectory(prefix="strataleaf-demonstrate-") as scratch:
        copy1, copy2 = Path(scratch) / "copy1.db", Path(scratch) / "copy2.db"
        shutil.copyfile(folder / "file1.db", copy1)
        shutil.copyfile(folder / "file2.db", copy2)
        check(sha256(copy1) == manifest["files"]["file1"]["sha256"], "file1 changed since generate")
        check(sha256(copy2) == manifest["files"]["file2"]["sha256"], "file2 changed since generate")
        with Database(copy1) as db:
            found = {}
            for key, value in (("a", A), ("b", B), ("c", C), ("d", D)):
                page = db.query("docs", {"field": "value", "op": "eq", "value": value})
                found[key] = [record.key for record in page.records]
            observations["queries_correct_despite_bug"] = (
                found == {"a": ["a"], "b": ["b"], "c": ["c"], "d": ["d"]})
            observations["deep_check_ok_despite_bug"] = db.diagnostics.inspect(deep=True)["ok"]
            try:
                db.indexes.create("docs", "value_unique", "value", unique=True)
                observations["unique_create_falsely_rejected"] = False
            except Conflict:
                observations["unique_create_falsely_rejected"] = True
            check([row["name"] for row in db.indexes.list("docs")] == ["by_value"],
                  "rejected index creation left a definition behind")
        with Database(copy2) as db:
            try:
                db.batch().put("docs", "b", {"value": B}).commit()
                observations["unique_write_falsely_rejected"] = False
            except Conflict:
                observations["unique_write_falsely_rejected"] = True
            try:
                db.batch().put("docs", "x", {"value": float(A)}).commit()
                observations["legitimate_conflict_still_detected"] = False
            except Conflict:
                observations["legitimate_conflict_still_detected"] = True
    print(json.dumps({"phase": "demonstrate", "observations": observations}, sort_keys=True))
    bug_seen = (observations["unique_create_falsely_rejected"]
                and observations["unique_write_falsely_rejected"]
                and observations["legitimate_conflict_still_detected"]
                and observations["queries_correct_despite_bug"])
    if not bug_seen:
        raise SystemExit("regression not reproduced; if the code is already fixed, run 'verify' instead")


# ---------------------------------------------------------------- verify

def verify_file1(folder, manifest, state):
    entry = manifest["files"]["file1"]
    check_file_lineage(folder / "file1.db", "file1", folder, manifest, state)
    with Database(folder / "file1.db") as db:
        facts_preserved(db, entry, "file1")
        projection_exact(db)
        check(db.revision >= 1, "file1 head lost")
        # Each original value selects exactly its own key through the index.
        expect_eq(db, "docs", "value", A, ["a"], plan_index="by_value")
        expect_eq(db, "docs", "value", B, ["b"], plan_index="by_value")
        expect_eq(db, "docs", "value", C, ["c"], plan_index="by_value")
        expect_eq(db, "docs", "value", D, ["d"], plan_index="by_value")
        expect_eq(db, "docs", "value", float(A), ["a"], plan_index="by_value")
        expect_eq(db, "docs", "value", float(B), ["a"], plan_index="by_value")  # float(B) rounds to A
        expect_eq(db, "docs", "value", 9007199254740994, [], plan_index="by_value")
        expect_eq(db, "docs", "value", 0.2, [], plan_index="by_value")
        expect_eq(db, "docs", "value", True, [], plan_index="by_value")
        expect_eq(db, "docs", "value", None, [], plan_index="by_value")
        # Creating the unique index on the migrated file must now succeed.
        if "value_unique" not in [row["name"] for row in db.indexes.list("docs")]:
            created = db.indexes.create("docs", "value_unique", "value", unique=True)
            check(created["unique"] is True, "unique index metadata wrong")
        definitions = {row["name"]: row for row in db.indexes.list("docs")}
        check(definitions["value_unique"]["unique_flag"] == 1, "unique index definition not preserved")
        # Unique enforcement is exact: equal values conflict, distinct values pass.
        expect_conflict(lambda: db.batch().put("docs", "e", {"value": float(A)}).commit(), db,
                        "file1 write float(A) must conflict with a")
        expect_conflict(lambda: db.batch().put("docs", "e", {"value": B}).commit(), db,
                        "file1 write B must conflict with b")
        db.batch().put("docs", "e", {"value": 9007199254740994}).commit()
        expect_eq(db, "docs", "value", 9007199254740994, ["e"], plan_index="by_value")
        # Updates and deletes behave on the migrated file.
        current_c = db.get("docs", "c")
        db.batch().put("docs", "c", {"value": 0.2}, expected=current_c.revision).commit()
        expect_eq(db, "docs", "value", C, [], plan_index="by_value")
        expect_eq(db, "docs", "value", 0.2, ["c"], plan_index="by_value")
        expect_conflict(lambda: db.batch().put("docs", "b", {"value": A}).commit(), db,
                        "file1 update b->A must conflict with a")
        check(db.get("docs", "b").document == {"value": B}, "file1 b changed by rejected write")
        db.batch().delete("docs", "d").commit()
        check(db.get("docs", "d") is None, "file1 d not deleted")
        db.batch().put("docs", "f", {"value": D}).commit()
        expect_eq(db, "docs", "value", D, ["f"], plan_index="by_value")
        # Historical reads are intact; history does not consume the current index.
        check(db.get("docs", "a", revision=1).document == {"value": A}, "file1 history for a lost")
        check(db.get("docs", "c", revision=1).document == {"value": C}, "file1 history for c lost")
        check(db.get("docs", "d", revision=1).document == {"value": D}, "file1 history for d lost")
        check(db.get("docs", "d", revision=1).revision == 1, "file1 historical revision changed")
        # The pre-fix request receipt still replays without a new commit.
        head_before = db.revision
        replay = db.batch(SEED_REQUEST)
        for key, value in (("a", A), ("b", B), ("c", C), ("d", D)):
            replay.put("docs", key, {"value": value})
        result = replay.commit()
        check(result.replayed and result.revision == 1 and result.count == 4,
              "file1 seed receipt did not replay: %r" % (result,))
        check(db.revision == head_before, "replay created a new commit")
        # The stored JSON body of d kept full precision (never shortened).
        with db._store.read() as connection:
            body = connection.execute(
                "SELECT body FROM versions WHERE collection='docs' AND key='d' AND rev=1").fetchone()[0]
        check(json.loads(body)["value"] == D and repr(D) in body,
              "file1 body of d lost float precision: " + body)
        # Restore the canonical state so every verify run sees the same file.
        current_c = db.get("docs", "c")
        db.batch().put("docs", "c", {"value": C}, expected=current_c.revision).commit()
        db.batch().delete("docs", "e").delete("docs", "f").commit()
        db.batch().put("docs", "d", {"value": D}).commit()
        expect_eq(db, "docs", "value", C, ["c"], plan_index="by_value")
        expect_eq(db, "docs", "value", D, ["d"], plan_index="by_value")
        expect_eq(db, "docs", "value", 0.2, [], plan_index="by_value")
        expect_eq(db, "docs", "value", 9007199254740994, [], plan_index="by_value")
        check(db.diagnostics.inspect(deep=True)["ok"], "file1 deep inspection failed")
        projection_exact(db)
    # Reopening a migrated file must not rewrite anything.
    first = file_hash_ready(folder / "file1.db")
    with Database(folder / "file1.db"):
        pass
    check(file_hash_ready(folder / "file1.db") == first, "reopening migrated file1 rewrote the file")
    # Backup/restore of the migrated file keeps the projection usable.
    with tempfile.TemporaryDirectory(prefix="strataleaf-verify-backup-") as scratch:
        with Database(folder / "file1.db") as db:
            fingerprint = db.diagnostics.fingerprint()["sha256"]
            receipt = db.archives.backup(Path(scratch) / "backup.db")
        Archives.restore(Path(scratch) / "backup.db", Path(scratch) / "restored.db", receipt["sha256"])
        with Database(Path(scratch) / "restored.db") as restored:
            check(restored.diagnostics.fingerprint()["sha256"] == fingerprint,
                  "restored file1 fingerprint differs")
            expect_eq(restored, "docs", "value", A, ["a"], plan_index="by_value")
            expect_eq(restored, "docs", "value", B, ["b"], plan_index="by_value")
            projection_exact(restored)
    return file_hash_ready(folder / "file1.db")


def verify_file2(folder, manifest, state):
    entry = manifest["files"]["file2"]
    check_file_lineage(folder / "file2.db", "file2", folder, manifest, state)
    with Database(folder / "file2.db") as db:
        facts_preserved(db, entry, "file2")
        projection_exact(db)
        definitions = {row["name"]: row for row in db.indexes.list("docs")}
        check(definitions["value_unique"]["unique_flag"] == 1, "file2 unique definition not preserved")
        # Writing the adjacent integer now succeeds; the rounded float conflicts with a.
        db.batch().put("docs", "b", {"value": B}).commit()
        expect_conflict(lambda: db.batch().put("docs", "x", {"value": float(A)}).commit(), db,
                        "file2 write float(A) must conflict with a")
        expect_conflict(lambda: db.batch().put("docs", "x", {"value": float(B)}).commit(), db,
                        "file2 write float(B) must conflict with a (float(B) rounds to A)")
        expect_conflict(lambda: db.batch().put("docs", "x", {"value": B}).commit(), db,
                        "file2 write B must conflict with b")
        expect_eq(db, "docs", "value", A, ["a"], plan_index="value_unique")
        expect_eq(db, "docs", "value", B, ["b"], plan_index="value_unique")
        expect_eq(db, "docs", "value", float(A), ["a"], plan_index="value_unique")
        expect_eq(db, "docs", "value", float(B), ["a"], plan_index="value_unique")
        expect_eq(db, "docs", "value", C, [], plan_index="value_unique")
        check(db.get("docs", "a", revision=1).document == {"value": A}, "file2 history for a lost")
        check(db.diagnostics.inspect(deep=True)["ok"], "file2 deep inspection failed")
        projection_exact(db)
    first = file_hash_ready(folder / "file2.db")
    with Database(folder / "file2.db"):
        pass
    check(file_hash_ready(folder / "file2.db") == first, "reopening migrated file2 rewrote the file")
    return first


def verify_crash_safety(folder, manifest):
    """An interrupted rebuild cannot persist; reopening completes it atomically."""
    check(sha256(folder / "file1.crashseed.db") == manifest["crashseed"]["sha256"],
          "crashseed copy was modified")
    with tempfile.TemporaryDirectory(prefix="strataleaf-verify-crash-") as scratch:
        work = Path(scratch) / "work.db"
        shutil.copyfile(folder / "file1.crashseed.db", work)
        pristine = sha256(work)
        raw = sqlite3.connect(str(work), isolation_level=None)
        raw.execute("BEGIN IMMEDIATE")
        raw.execute("DELETE FROM index_values")
        raw.execute("INSERT INTO index_values VALUES ('docs','by_value','n:partial','a')")
        raw.close()  # close with an open transaction == crash: everything rolls back
        check(not Path(str(work) + "-wal").exists(), "rolled-back crash simulation left a WAL")
        check(sha256(work) == pristine, "interrupted rebuild changed the database file")
        with Database(work) as db:
            projection_exact(db)
            expect_eq(db, "docs", "value", A, ["a"], plan_index="by_value")
            expect_eq(db, "docs", "value", B, ["b"], plan_index="by_value")
            expect_eq(db, "docs", "value", D, ["d"], plan_index="by_value")
            check(db.diagnostics.inspect(deep=True)["ok"], "crash-recovered file failed inspection")
        migrated = file_hash_ready(work)
        with Database(work):
            pass
        check(file_hash_ready(work) == migrated, "reopening crash-recovered file rewrote it")


# Fresh-database matrix: values inside one class are oracle-equal, classes are
# mutually distinct. Covers negatives, huge integers, float boundaries, null,
# strings, booleans, objects and arrays.
CLASSES = [
    ("zero", [0, -0.0, 0.0]),
    ("one", [1, 1.0]),
    ("true", [True]),
    ("false", [False]),
    ("two53", [A, float(A), float(B)]),
    ("two53p1", [B]),
    ("two53p2", [9007199254740994]),
    ("neg_one", [-1, -1.0]),
    ("neg_two53p1", [-9007199254740993]),
    ("neg_two53", [-9007199254740992]),
    ("point_one", [C]),
    ("point_one_next", [D]),
    ("neg_point_one", [-0.1]),
    ("neg_point_one_next", [math.nextafter(-0.1, -math.inf)]),
    ("int64max", [2**63 - 1]),
    ("int64", [2**63]),
    ("uint64", [2**64]),
    ("big_int", [10**30]),
    ("big_int_adj", [10**30 + 1]),
    ("huge_int", [10**400]),
    ("huge_int_adj", [10**400 + 1]),
    ("neg_huge", [-(10**400)]),
    ("two62", [2**62, float(2**62)]),
    ("max_float", [sys.float_info.max]),
    ("below_max", [math.nextafter(sys.float_info.max, 0.0)]),
    ("neg_max_float", [-sys.float_info.max]),
    ("min_normal", [sys.float_info.min]),
    ("below_normal", [math.nextafter(sys.float_info.min, 0.0)]),
    ("min_denorm", [math.nextafter(0.0, 1.0)]),
    ("null", [None]),
    ("str_one", ["1"]),
    ("str_text", ["east"]),
    ("object", [{"x": 1}]),
    ("array", [[1, 2]]),
]


def class_keys():
    keys = {}
    for class_index, (name, members) in enumerate(CLASSES):
        keys[name] = ["k%02d_%d" % (class_index, member_index) for member_index in range(len(members))]
    return keys


def verify_fresh(tmpdir):
    keys = class_keys()
    with Database(Path(tmpdir) / "fresh.db") as db:
        db.catalog.create("docs")
        # Order 1: write every document first, build the indexes afterwards.
        seed = db.batch("fresh_seed")
        for class_index, (name, members) in enumerate(CLASSES):
            for member_index, value in enumerate(members):
                seed.put("docs", keys[name][member_index], {"value": value, "tag": name})
        seed.put("docs", "missing_value", {"other": 7, "tag": "missing"})
        seed_revision = seed.commit().revision
        db.indexes.create("docs", "by_value", "value")
        db.indexes.create("docs", "by_tag", "tag")
        # A unique index over duplicated classes must fail and roll back fully.
        expect_conflict(lambda: db.indexes.create("docs", "u_bad", "value", unique=True), db,
                        "unique index over duplicated values must fail")
        check([row["name"] for row in db.indexes.list("docs")] == ["by_tag", "by_value"],
              "failed unique index creation left a definition behind")
        # Every probe selects exactly its oracle-equivalence class.
        all_docs = {}
        for name, members in CLASSES:
            for member_index, value in enumerate(members):
                all_docs[keys[name][member_index]] = value
        for name, members in CLASSES:
            for value in members:
                plan = None if isinstance(value, (dict, list)) else "by_value"
                expect_eq(db, "docs", "value", value, keys[name], plan_index=plan)
        for absent in (0.2, "zzz", {"x": 2}, [9], 3.5, 10**30 + 2):
            expect_eq(db, "docs", "value", absent, [])
        page = db.query("docs", {"field": "value", "op": "exists", "value": False})
        check([record.key for record in page.records] == ["missing_value"], "exists false mismatch")
        page = db.query("docs", {"field": "value", "op": "exists", "value": True}, limit=1000)
        check(sorted(record.key for record in page.records) == sorted(all_docs), "exists true mismatch")
        # Non-numeric index still serves equality on strings.
        expect_eq(db, "docs", "tag", "zero", keys["zero"], plan_index="by_tag")
        expect_eq(db, "docs", "tag", "missing", ["missing_value"], plan_index="by_tag")
        # Range predicates never treat booleans as numbers.
        page = db.query("docs", {"field": "value", "op": "lt", "value": 1}, limit=1000)
        expected = sorted(key for key, value in all_docs.items()
                          if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 1)
        check(sorted(record.key for record in page.records) == expected, "lt 1 mismatch")
        # Request replay on the fresh database.
        replay = db.batch("fresh_seed")
        head_before = db.revision
        for class_index, (name, members) in enumerate(CLASSES):
            for member_index, value in enumerate(members):
                replay.put("docs", keys[name][member_index], {"value": value, "tag": name})
        replay.put("docs", "missing_value", {"other": 7, "tag": "missing"})
        result = replay.commit()
        check(result.replayed and result.revision == seed_revision, "fresh replay failed")
        check(db.revision == head_before, "fresh replay created a commit")
        # Updates and deletes move values between classes correctly.
        db.batch().delete("docs", keys["one"][1]).commit()  # remove the 1.0 member
        expect_eq(db, "docs", "value", 1, [keys["one"][0]], plan_index="by_value")
        check(db.get("docs", keys["one"][1], revision=seed_revision).document["value"] == 1.0,
              "historical read of deleted member lost")
        db.batch().put("docs", keys["str_text"][0], {"value": 42, "tag": "str_text"}).commit()
        expect_eq(db, "docs", "value", "east", [], plan_index="by_value")
        expect_eq(db, "docs", "value", 42, [keys["str_text"][0]], plan_index="by_value")
        check(db.diagnostics.inspect(deep=True)["ok"], "fresh docs inspection failed")
        projection_exact(db)

        # Order 2: create the unique index first, then write one representative
        # of every class; duplicates conflict, distinct neighbours pass.
        db.catalog.create("uniq")
        db.indexes.create("uniq", "u_value", "value", unique=True)
        representatives = db.batch()
        rep_keys = {}
        for index, (name, members) in enumerate(CLASSES):
            rep_keys[name] = "u%02d" % index
            representatives.put("uniq", rep_keys[name], {"value": members[0]})
        representatives.commit()
        db.batch().put("uniq", "u_obj2", {"value": {"x": 1}}).commit()   # objects are not indexed
        db.batch().put("uniq", "u_arr2", {"value": [1, 2]}).commit()     # arrays are not indexed
        for probe, label in ((1.0, "1.0 conflicts with 1"), (-0.0, "-0.0 conflicts with 0"),
                             (float(A), "float(A) conflicts with A"), (float(B), "float(B) rounds to A"),
                             (2.0**62, "2.0**62 conflicts with 2**62"), (None, "second null conflicts"),
                             (True, "second true conflicts"), ("1", "second string '1' conflicts"),
                             (float(D), "float(D) conflicts with D")):
            expect_conflict(lambda probe=probe: db.batch().put("uniq", "dup", {"value": probe}).commit(),
                            db, "uniq duplicate: " + label)
        for index, probe in enumerate(("01", 0.11, -9007199254740993.5, 10**400 - 1)):
            db.batch().put("uniq", "ok_%d" % index, {"value": probe}).commit()
        # Distinct neighbours that the regression used to merge now coexist.
        expect_eq(db, "uniq", "value", A, [rep_keys["two53"]], plan_index="u_value")
        expect_eq(db, "uniq", "value", B, [rep_keys["two53p1"]], plan_index="u_value")
        expect_eq(db, "uniq", "value", C, [rep_keys["point_one"]], plan_index="u_value")
        expect_eq(db, "uniq", "value", D, [rep_keys["point_one_next"]], plan_index="u_value")
        expect_eq(db, "uniq", "value", True, [rep_keys["true"]], plan_index="u_value")
        expect_eq(db, "uniq", "value", 1, [rep_keys["one"]], plan_index="u_value")
        expect_eq(db, "uniq", "value", None, [rep_keys["null"]], plan_index="u_value")
        # Deleting frees the unique value for a different key.
        db.batch().delete("uniq", rep_keys["two53p1"]).commit()
        db.batch().put("uniq", "b_moved", {"value": B}).commit()
        expect_eq(db, "uniq", "value", B, ["b_moved"], plan_index="u_value")
        check(db.get("uniq", rep_keys["two53p1"]) is None, "deleted unique key still visible")
        check(db.diagnostics.inspect(deep=True)["ok"], "fresh uniq inspection failed")
        projection_exact(db)
        # Backup/restore round trip of the fresh database.
        receipt = db.archives.backup(Path(tmpdir) / "fresh-backup.db")
    Archives.restore(Path(tmpdir) / "fresh-backup.db", Path(tmpdir) / "fresh-restored.db", receipt["sha256"])
    with Database(Path(tmpdir) / "fresh-restored.db") as restored:
        expect_eq(restored, "uniq", "value", B, ["b_moved"], plan_index="u_value")
        expect_eq(restored, "docs", "value", D, keys["point_one_next"], plan_index="by_value")
        projection_exact(restored)


def verify(folder):
    folder = Path(folder)
    manifest = load_manifest(folder)
    state = {"runs": 0, "hashes": {}}
    sidecar = folder / "verify-state.json"
    if sidecar.exists():
        with open(sidecar, encoding="utf-8") as stream:
            state = json.load(stream)
    hashes = {"file1": verify_file1(folder, manifest, state),
              "file2": verify_file2(folder, manifest, state)}
    verify_crash_safety(folder, manifest)
    with tempfile.TemporaryDirectory(prefix="strataleaf-verify-fresh-") as scratch:
        verify_fresh(scratch)
    state["runs"] += 1
    state["hashes"] = {label: [digest] for label, digest in hashes.items()}
    with open(sidecar, "w", encoding="utf-8") as stream:
        json.dump(state, stream, indent=1, sort_keys=True)
    print(json.dumps({"phase": "verify", "run": state["runs"], "checks": CHECKS[0], "ok": True},
                     sort_keys=True))


if __name__ == "__main__":
    commands = {"generate": generate, "demonstrate": demonstrate, "verify": verify}
    if len(sys.argv) != 3 or sys.argv[1] not in commands:
        raise SystemExit("usage: python3 -m examples.numeric_index_verification {generate|demonstrate|verify} <dir>")
    commands[sys.argv[1]](sys.argv[2])
