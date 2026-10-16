                                                                              
from __future__ import annotations
import csv, io, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "datasets" / "benchmarks"
SOLOMON_URL = "https://codeload.github.com/ashishshukla77/solomon-vrptw-optimization/zip/refs/heads/main"
CORDEAU_URL = "https://codeload.github.com/fboliveira/MDVRP-Instances/zip/refs/heads/master"
HEADERS = {"User-Agent": "Trajecta-reproducibility/1.0"}

def get_zip(url):
    req = urllib.request.Request(url, headers=HEADERS)
    return zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=90).read()))

def solomon_rows(text, name):
    lines = text.splitlines()
    vehicle_count = capacity = None
    for i, line in enumerate(lines):
        t = line.split()
        if len(t) == 2 and all(x.isdigit() for x in t):
            vehicle_count, capacity = map(int, t); break
    rows = []
    for line in lines:
        t = line.split()
        if len(t) >= 7:
            try: vals = [float(x) for x in t[:7]]
            except ValueError: continue
            idx, x, y, demand, ready, due, service = vals
            if idx.is_integer():
                rows.append({"family":"Solomon", "instance":name, "node_id":int(idx),
                    "x":x,"y":y,"demand":demand,"ready":ready,"due":due,
                    "service":service,"depot_id":0,"capacity":capacity,
                    "vehicle_limit":vehicle_count,"source_format":"Solomon VRPTW"})
    if len(rows) != 101: raise ValueError(f"{name}: expected depot + 100 customers, found {len(rows)}")
    return rows

def cordeau_rows(text, name):
    lines = [line.split() for line in text.splitlines() if line.strip()]
    typ, vehicles_per_depot, n, depots = map(int, lines[0][:4])
    caps = [float(lines[1+i][1]) for i in range(depots)]
    data = lines[1+depots:1+depots+n+depots]
    if len(data) != n + depots: raise ValueError(f"{name}: unexpected Cordeau row count")
    cust, dep = data[:n], data[n:]
    rows=[]
    for line in cust:
        idx,x,y,service,demand = int(line[0]),float(line[1]),float(line[2]),float(line[3]),float(line[4])
        rows.append({"family":"Cordeau-MDVRP","instance":name,"node_id":idx,
            "x":x,"y":y,"demand":demand,"ready":0,"due":float("inf"),
            "service":service,"depot_id":"nearest","capacity":max(caps),
            "vehicle_limit":vehicles_per_depot*depots,"source_format":"Cordeau MDVRP"})
    for j,line in enumerate(dep):
        idx,x,y = int(line[0]),float(line[1]),float(line[2])
        rows.append({"family":"Cordeau-MDVRP","instance":name,"node_id":f"D{j+1}",
            "x":x,"y":y,"demand":0,"ready":0,"due":float("inf"),
            "service":0,"depot_id":j+1,"capacity":caps[j],
            "vehicle_limit":vehicles_per_depot,"source_format":"Cordeau MDVRP depot"})
    return rows

def main():
    raw_sol = ROOT / "raw" / "solomon"; raw_cor = ROOT / "raw" / "cordeau"
    raw_sol.mkdir(parents=True, exist_ok=True); raw_cor.mkdir(parents=True, exist_ok=True)
    solz = get_zip(SOLOMON_URL); corz = get_zip(CORDEAU_URL)
    selected_sol = ["c101", "r101", "rc101"]
    selected_cor = ["p01", "p02"]
    rows=[]
    for stem in selected_sol:
        name = next(x for x in solz.namelist() if x.endswith(f"/data/solomon/{stem}.txt"))
        data = solz.read(name)
        (raw_sol / f"{stem}.txt").write_bytes(data)
        rows.extend(solomon_rows(data.decode("utf-8"), stem))
    for stem in selected_cor:
        name = f"MDVRP-Instances-master/dat/{stem}"
        data = corz.read(name)
        (raw_cor / stem).write_bytes(data)
        rows.extend(cordeau_rows(data.decode("utf-8"), stem))
    with (ROOT / "benchmark_instances.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT / "PROVENANCE.txt").write_text(
        "Solomon instances c101, r101, rc101: SINTEF Solomon 100-customer benchmark; files mirrored by https://github.com/ashishshukla77/solomon-vrptw-optimization\n"
        "Cordeau MDVRP p01, p02: Cordeau benchmark files mirrored by https://github.com/fboliveira/MDVRP-Instances; see repository LICENSE and DESCRIPTION.md.\n"
        "The raw instance files are kept in raw/ and normalized customer/depot rows in benchmark_instances.csv.\n",
        encoding="utf-8")
    print(f"Wrote {len(rows)} benchmark rows to {ROOT / 'benchmark_instances.csv'}")

if __name__ == "__main__": main()
