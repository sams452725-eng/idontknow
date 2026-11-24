"""VSCode settings.json(JSONC) 문법 검사: 주석/끝 쉼표 제거 후 json 파싱."""

import json
import os
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars(r"%APPDATA%\Code\User\settings.json")
text = open(path, encoding="utf-8-sig").read()

out, i, n, in_str = [], 0, len(text), False
while i < n:
    c = text[i]
    if in_str:
        out.append(c)
        if c == "\\":
            out.append(text[i + 1])
            i += 1
        elif c == '"':
            in_str = False
    elif c == '"':
        in_str = True
        out.append(c)
    elif text.startswith("//", i):
        while i < n and text[i] != "\n":
            i += 1
        continue
    elif text.startswith("/*", i):
        i = text.index("*/", i) + 2
        continue
    else:
        out.append(c)
    i += 1

clean = re.sub(r",(\s*[}\]])", r"\1", "".join(out))
try:
    data = json.loads(clean)
except json.JSONDecodeError as e:
    print(f"INVALID: {e}")
    sys.exit(1)
print(f"VALID ({len(data)} keys)")
for k in (
    "workbench.colorTheme",
    "workbench.iconTheme",
    "indentRainbow.indicatorStyle",
    "better-comments.multilineComments",
):
    print(f"  {k} = {data.get(k)}")
