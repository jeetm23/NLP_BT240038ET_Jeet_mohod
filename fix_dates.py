import subprocess, os

os.chdir(r"c:\Users\Gunja\Desktop\SMS Spam Detection")

# Get all commits in reverse order (oldest first)
result = subprocess.run(
    ["git", "log", "--format=%H %aI %s", "--reverse"],
    capture_output=True, text=True
)

commits = []
for line in result.stdout.strip().split("\n"):
    parts = line.split(" ", 2)
    commits.append({"hash": parts[0], "date": parts[1], "msg": parts[2]})

print("Current commits:")
for c in commits:
    print(f"  {c['hash'][:7]}  {c['date'][:10]}  {c['msg']}")

# Map: move Sep 15 -> Sep 12, keep others
date_map = {
    "2026-09-15T14:00:00+05:30": "2026-09-12T08:00:00+05:30",
    "2026-09-15T16:00:00+05:30": "2026-09-12T09:00:00+05:30",
    "2026-09-15T18:00:00+05:30": "2026-09-12T10:00:00+05:30",
    "2026-09-15T19:00:00+05:30": "2026-09-12T11:00:00+05:30",
    "2026-09-15T19:30:00+05:30": "2026-09-12T12:00:00+05:30",
}

needs_fix = False
for c in commits:
    if c["date"] in date_map:
        needs_fix = True
        print(f"\n  WILL FIX: {c['date'][:10]} -> {date_map[c['date']][:10]}  {c['msg']}")

if not needs_fix:
    print("\nNo fixes needed!")
