"""Submit a day's queue to the Large Model Track (09-14). Never deletes or rebuilds anything.

    python src/submit_queue.py "FILE|MD5PREFIX|description" ... [--dry-run]

For each item, in order:
  - the file's md5 must start with MD5PREFIX, else stop
  - skip it if a submission with the same file name is already on the latest Submissions page
  - stop once today's UTC count reaches 5
Then wait for the scoring and print every public score from the API, never from the CLI table.
"""
import sys, time, hashlib, datetime, os
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = 'cuhk-x-competition-large-model-track'
DRY = '--dry-run' in sys.argv
ITEMS = [a.split('|', 2) for a in sys.argv[1:] if not a.startswith('--')]
api = KaggleApi()
api.authenticate()


def g(s, *names):
    return next((getattr(s, k) for k in names if getattr(s, k, None) is not None), None)


def board():
    return api.competition_submissions(COMP)


def utc_day(d):
    if isinstance(d, str):
        d = datetime.datetime.fromisoformat(d.replace('Z', '+00:00'))
    return d.date() if d else None


today = datetime.datetime.now(datetime.timezone.utc).date()
subs = board()
names = {g(s, 'file_name', 'fileName') for s in subs}
used = sum(1 for s in subs if utc_day(g(s, 'date')) == today)
print('UTC day %s: %d already submitted today' % (today, used), flush=True)
sent = []
for path, pre, desc in ITEMS:
    base = os.path.basename(path)
    md5 = hashlib.md5(open(path, 'rb').read()).hexdigest()
    if not md5.startswith(pre):
        sys.exit('STOP: %s md5 %s does not start with %s' % (path, md5, pre))
    if base in names:
        print('skip %s: already on the board' % base, flush=True)
        continue
    if used >= 5:
        print('STOP: 5 submissions already used today; not sent: %s' % base, flush=True)
        break
    if DRY:
        print('DRY RUN would submit %s (md5 %s) "%s"' % (path, md5[:12], desc), flush=True)
    else:
        api.competition_submit(path, desc, COMP)
        print('submitted %s (md5 %s) "%s"' % (path, md5[:12], desc), flush=True)
        time.sleep(3)
    used += 1
    sent.append(base)

if sent and not DRY:
    for _ in range(40):
        time.sleep(20)
        rows = {g(s, 'file_name', 'fileName'): s for s in board()}
        pending = [b for b in sent if b not in rows or g(rows[b], 'public_score', 'publicScore') in (None, '')]
        if not pending:
            break
    rows = {g(s, 'file_name', 'fileName'): s for s in board()}
    for b in sent:
        s = rows.get(b)
        sc = g(s, 'public_score', 'publicScore') if s else None
        cnt = round(float(sc) * 342) if sc not in (None, '') else None
        print('SCORE %-28s %s = %s/342 (%s)' % (b, sc, cnt, g(s, 'status') if s else 'not on board'), flush=True)
