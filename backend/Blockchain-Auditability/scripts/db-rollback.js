#!/usr/bin/env node
/**
 * Restores the SQLite audit store from the backup taken before the v1 -> v2 schema migration
 * (src/db/database.js writes <db>.bak-v1-<timestamp> with VACUUM INTO).
 *
 * Stop the service first: SQLite must not be open while its file is replaced.
 *
 *   node scripts/db-rollback.js [path/to/audits.sqlite] [--backup path/to/backup]
 *
 * The current file is kept as <db>.pre-rollback-<timestamp>, so a rollback can itself be undone.
 * The Postgres store needs no rollback: v2 stored its new fields inside each JSONB record and
 * added two tables, leaving every v1 row as it was.
 */
const fs = require("fs");
const path = require("path");

function latestBackup(dbPath) {
  const dir = path.dirname(dbPath);
  const prefix = `${path.basename(dbPath)}.bak-v1-`;
  const candidates = fs.readdirSync(dir).filter((f) => f.startsWith(prefix)).sort();
  return candidates.length ? path.join(dir, candidates[candidates.length - 1]) : null;
}

function rollback(dbPath, backupPath = null) {
  const backup = backupPath || latestBackup(dbPath);
  if (!backup || !fs.existsSync(backup)) {
    throw new Error(`No v1 backup found next to ${dbPath}`);
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const kept = `${dbPath}.pre-rollback-${stamp}`;
  if (fs.existsSync(dbPath)) fs.copyFileSync(dbPath, kept);
  for (const suffix of ["-wal", "-shm"]) {
    if (fs.existsSync(dbPath + suffix)) fs.unlinkSync(dbPath + suffix);
  }
  fs.copyFileSync(backup, dbPath);
  return { restoredFrom: backup, previousKeptAs: fs.existsSync(kept) ? kept : null };
}

if (require.main === module) {
  const args = process.argv.slice(2);
  const i = args.indexOf("--backup");
  const backup = i >= 0 ? args[i + 1] : null;
  const dbPath = args.find((a, idx) => !a.startsWith("--") && idx !== i + 1)
    || process.env.DB_PATH
    || path.join(__dirname, "../data/audits.sqlite");
  try {
    const result = rollback(path.resolve(dbPath), backup && path.resolve(backup));
    console.log(`Restored ${dbPath} from ${result.restoredFrom}`);
    if (result.previousKeptAs) console.log(`The migrated file was kept as ${result.previousKeptAs}`);
  } catch (err) {
    console.error(err.message);
    process.exitCode = 1;
  }
}

module.exports = { rollback, latestBackup };
