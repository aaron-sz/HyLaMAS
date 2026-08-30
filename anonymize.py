"""Strip machine identifiers from run manifests before release.
"""

import argparse
import glob
import json
import os
import re

# hostname -> constant; platform keeps the OS and libc, drops the kernel build
PLACEHOLDER_HOST = "anonymized"


def clean_platform(value):
    if not isinstance(value, str):
        return value, False
    m = re.match(r"^(Linux)-.*?(x86_64-with-.*)$", value)
    if m:
        return f"{m.group(1)}-{m.group(2)}", True
    return value, False


def scrub(path, write):
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    env = man.get("environment")
    if not isinstance(env, dict):
        return []
    changed = []
    if env.get("hostname") not in (None, PLACEHOLDER_HOST):
        changed.append(f"hostname: {env['hostname']} -> {PLACEHOLDER_HOST}")
        env["hostname"] = PLACEHOLDER_HOST
    new_platform, hit = clean_platform(env.get("platform"))
    if hit and new_platform != env.get("platform"):
        changed.append(f"platform: {env['platform']} -> {new_platform}")
        env["platform"] = new_platform
    if changed and write:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "manifest.json"),
                             recursive=True))
    total = 0
    for p in paths:
        for line in scrub(p, write=not args.check):
            total += 1
            print(f"{os.path.relpath(p):60s} {line}")
    verb = "would change" if args.check else "changed"
    print(f"\n{len(paths)} manifests scanned, {verb} {total} fields")


if __name__ == "__main__":
    main()
