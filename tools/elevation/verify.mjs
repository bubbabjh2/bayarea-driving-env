// Verify the elevation rebuild bit-for-bit against the hashes pinned in tools/elevation/expected.sha256
// (taken from the original project's outputs).
//   node tools/elevation/verify.mjs            (npm run verify:elevation)
// Line format:  <sha256>  <path relative to the repo>[  ignore=<key,key>]
// For JSON files with `ignore=`, the hash is of JSON.stringify(parsed object without those top-level keys), so
// build timestamps do not count. Files that are not present are reported as SKIP (e.g. ground10 before build_conform).
import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

export function hashFile(file, ignore = null) {
  if (ignore) {
    let txt = readFileSync(file); if (file.endsWith('.gz')) txt = gunzipSync(txt);
    const o = JSON.parse(txt.toString('utf8')); for (const k of ignore) delete o[k];
    return Promise.resolve(createHash('sha256').update(JSON.stringify(o)).digest('hex'));
  }
  return new Promise((res, rej) => {
    const h = createHash('sha256');
    const s = file.endsWith('.gz') ? null : createReadStream(file);
    if (!s) { h.update(gunzipSync(readFileSync(file))); return res(h.digest('hex')); }   // .gz: hash the content
    s.on('data', (d) => h.update(d)).on('end', () => res(h.digest('hex'))).on('error', rej);
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const lines = readFileSync(path.join(ROOT, 'tools/elevation/expected.sha256'), 'utf8').split('\n').filter((l) => l.trim() && !l.startsWith('#'));
  let ok = 0, bad = 0, skip = 0;
  for (const l of lines) {
    const [sha, rel, opt] = l.trim().split(/\s{2,}/);
    const ignore = opt && opt.startsWith('ignore=') ? opt.slice(7).split(',') : null;
    const f = path.join(ROOT, rel);
    if (!existsSync(f)) { console.log('SKIP    ', rel, '(not built)'); skip++; continue; }
    const got = await hashFile(f, ignore);
    if (got === sha) { console.log('MATCH   ', rel); ok++; } else { console.log('MISMATCH', rel, got); bad++; }
  }
  console.log(`${ok} match, ${bad} mismatch, ${skip} not built`);
  process.exit(bad ? 1 : 0);
}
