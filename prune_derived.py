import json, shutil
src = "derived.json"
shutil.copy(src, "derived.backup.json")
d = json.load(open(src))
scn_list = d["scn_list|||"]["value"]
def keep(k, v):
    parts = (k.split("|") + ["", "", "", ""])[:4]
    return parts[0] == "scn_list" or (v.get("cell") == "N2" and parts[1] in scn_list)
out = {k: v for k, v in d.items() if keep(k, v)}
json.dump(out, open(src, "w"), indent=1)
print(f"kept {len(out)} of {len(d)} entries")
