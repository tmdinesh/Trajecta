                                                                   

                                                                            
                                                                         
                                                                            
                                       
   
from __future__ import annotations

import copy
import csv
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

CONFIG = {
    "orders": 30,
    "cities_including_depot": 18,
    "seeds": list(range(10)),
    "depot_city": "Coimbatore",
    "enable_infrastructure": True,
    "enable_pack3d": True,
    "enable_traffic": True,
    "enable_stochastic_service_time": True,
    "enable_penalty_risk_split": True,
    "enable_dbscan": False,
    "dbscan_eps_km": 2.0,
    "dbscan_min_samples": 2,
    "synthetic_city_radius_km": 22.0,
                                                                       
    "cost_rate_scale": 1.16,
    "due_offset_hours": [4.0, 8.0],
    "cities": [
        ["Coimbatore", 11.0168, 76.9558], ["Tiruppur", 11.1085, 77.3411],
        ["Erode", 11.3410, 77.7172], ["Salem", 11.6643, 78.1460],
        ["Palakkad", 10.7867, 76.6548], ["Pollachi", 10.6583, 77.0080],
        ["Mettupalayam", 11.2990, 76.9340], ["Udumalaipettai", 10.5881, 77.2474],
        ["Dindigul", 10.3673, 77.9803], ["Karur", 10.9601, 78.0766],
        ["Namakkal", 11.2194, 78.1674], ["Perundurai", 11.2750, 77.5870],
        ["Avinashi", 11.1920, 77.2680], ["Gudalur", 11.7440, 76.4890],
        ["Ooty", 11.4102, 76.6950], ["Thanjavur", 10.7870, 79.1378],
        ["Madurai", 9.9252, 78.1198], ["Thrissur", 10.5276, 76.2144],
    ],
                                                                    
    "road_distance_factor": 1.18,
    "weight_kg": [4, 55], "volume_m3": [0.015, 0.24],
    "window_hours": [3.0, 8.0], "penalty_deadline_h": 22.0,
    # Theoretical aliases retained for terminology alignment without changing the model.
    "t_base": 8.0,
    "service_base_min": 8.0,
    "service_min_per_kg": 0.05, "service_min_per_m3": 5.0,
    "service_noise_sd_min": 2.0, "late_rate_rs_per_day": 100.0,
    "packaging_rs_flagged": 35.0, "toll_rs_route": 30.0,
    "wage_rs_per_km": 2.0, "maintenance_rs_per_km": 0.55,
    "fuel_or_energy_rs_per_km": {"petrol": 4.5, "diesel": 7.0, "electric": 1.1},
    "traffic": {"rush_morning": 1.30, "rush_evening": 1.25,
                "midday": 1.05, "night": 0.90},
    # Theoretical aliases for the route-time multipliers used by the synthetic model.
    "gamma": 1.0,
    "delta": 1.0,
    "weather_factor": 1.0, "lane_factor": 1.0,
    "skill_distribution": {
        "two_wheeler": {"count": 2, "share": 2 / 7},
        "four_wheeler": {"count": 3, "share": 3 / 7},
        "truck": {"count": 2, "share": 2 / 7},
    },
    "assumed_route_groups": 4,
    "ortools_seconds": 10,
    "metaheuristic_iterations": 10,
}


@dataclass
class Order:
    oid: int
    city: int
    lat: float
    lon: float
    weight: float
    volume: float
    due_h: float
    arrival_h: float
    fragile: bool
    cooling: bool
    penalty_risk: float
    box_l_m: float
    box_w_m: float
    box_h_m: float


FLEET = [
    {"name": "2W-Petrol", "kind": "petrol", "weight": 35, "volume": .18, "speed": 34, "fixed": 0, "bay": (.55, .45, .40), "height": 1.5},
    {"name": "2W-EV", "kind": "electric", "weight": 30, "volume": .16, "speed": 32, "fixed": 0, "bay": (.55, .45, .40), "height": 1.6},
    {"name": "4W-Petrol", "kind": "petrol", "weight": 350, "volume": 2.5, "speed": 42, "fixed": 70, "bay": (2.0, 1.35, .95), "height": 2.0},
    {"name": "4W-Diesel", "kind": "diesel", "weight": 700, "volume": 5.5, "speed": 40, "fixed": 100, "bay": (2.5, 1.55, 1.35), "height": 2.3},
    {"name": "4W-EV", "kind": "electric", "weight": 400, "volume": 3.0, "speed": 38, "fixed": 65, "bay": (2.1, 1.4, 1.05), "height": 2.1},
    {"name": "Truck-Diesel", "kind": "diesel", "weight": 4000, "volume": 25, "speed": 35, "fixed": 180, "bay": (6.0, 2.2, 2.0), "height": 3.6},
    {"name": "Truck-EV", "kind": "electric", "weight": 2500, "volume": 18, "speed": 33, "fixed": 160, "bay": (5.0, 2.0, 1.8), "height": 3.2},
]


def haversine(a, b):
    r = 6371.0
    p1, p2 = math.radians(a[1]), math.radians(b[1])
    dp, dl = p2 - p1, math.radians(b[2] - a[2])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h)) * CONFIG["road_distance_factor"]


def make_instance(seed: int, order_count: int | None = None):
    rng = random.Random(seed)
    cities = CONFIG["cities"]
    depot = cities[0]
    count = CONFIG["orders"] if order_count is None else order_count
                                                                          
                                                                          
                                                                            
    center_lat, center_lon = depot[1], depot[2]
    centers = [(center_lat, center_lon)]
    for ci in range(1, len(cities)):
        theta = 2 * math.pi * (ci - 1) / (len(cities) - 1)
        radius = CONFIG["synthetic_city_radius_km"] * (0.45 + 0.55 * ((ci * 7) % 17) / 16)
        centers.append((center_lat + radius * math.cos(theta) / 111.0,
                        center_lon + radius * math.sin(theta) / (111.0 * math.cos(math.radians(center_lat)))))
    orders = []
    for i in range(count):
        ci = rng.randrange(1, len(cities))
        lat0, lon0 = centers[ci]
        lat = lat0 + rng.uniform(-.003, .003)
        lon = lon0 + rng.uniform(-.003, .003)
        arrival = rng.uniform(6, 10)
        due = 8.0 + rng.uniform(*CONFIG["due_offset_hours"])
        weight = rng.uniform(*CONFIG["weight_kg"])
        volume = rng.uniform(*CONFIG["volume_m3"])
        fragile, cooling = rng.random() < .15, rng.random() < .20
        cube = volume ** (1 / 3)
        box_l = cube * rng.uniform(1.05, 1.45)
        box_w = cube * rng.uniform(.78, 1.02)
        box_h = volume / (box_l * box_w)
        orders.append(Order(i, ci, lat, lon, weight, volume, due, arrival,
                            fragile, cooling, rng.random(), box_l, box_w, box_h))
    locs = [depot] + [[f"O{i}", o.lat, o.lon] for i, o in enumerate(orders)]
    d = [[0.0] * len(locs) for _ in locs]
    for i in range(len(locs)):
        for j in range(len(locs)):
            if i != j:
                d[i][j] = haversine(locs[i], locs[j])
    return orders, d


def service_minutes(o: Order, seed: int):
    flags = 4.0 * o.fragile + 6.0 * o.cooling
    t_base = CONFIG.get("t_base", CONFIG.get("service_base_min", 8.0))
    rng = random.Random(seed * 1009 + (o.oid + 1) * 9176)
    noise = rng.gauss(0, CONFIG["service_noise_sd_min"]) if CONFIG.get("enable_stochastic_service_time", True) else 0.0
    minutes = max(2.0, t_base + CONFIG["service_min_per_kg"] * o.weight
                  + CONFIG["service_min_per_m3"] * o.volume + flags
                  + noise)
    return round(minutes * 60) / 60


def travel_hours(km: float, vehicle, departure_h: float):
    if not CONFIG.get("enable_traffic", True):
        return round(km / vehicle["speed"] * 3600) / 3600
    band = "rush_morning" if 7 <= departure_h < 10 else "rush_evening" if 16 <= departure_h < 19 else "night" if departure_h >= 21 or departure_h < 6 else "midday"
    gamma = CONFIG.get("gamma", CONFIG.get("weather_factor", 1.0))
    delta = CONFIG.get("delta", CONFIG.get("lane_factor", 1.0))
    mult = CONFIG["traffic"][band] * gamma * delta
    return round(km / vehicle["speed"] * mult * 3600) / 3600


def seven_cost(distance, vehicle, late_days, flagged):
    energy = distance * CONFIG["fuel_or_energy_rs_per_km"][vehicle["kind"]]
    wages = distance * CONFIG["wage_rs_per_km"] + vehicle["fixed"]
    penalty = late_days * CONFIG["late_rate_rs_per_day"]
    cooling_packaging = flagged * CONFIG["packaging_rs_flagged"]
    maintenance = distance * CONFIG["maintenance_rs_per_km"]
    tolls = CONFIG["toll_rs_route"] if distance > 40 else 0.0
    ev_charging = distance * .35 if vehicle["kind"] == "electric" else 0.0
    scale = CONFIG["cost_rate_scale"]
    return scale * (energy + wages + penalty + cooling_packaging + maintenance + tolls + ev_charging), scale * penalty


def pack3d(route, vehicle, orders):
    if not CONFIG.get("enable_pack3d", True):
        return True
    free = [vehicle["bay"]]
    boxes = sorted(((orders[i].box_l_m, orders[i].box_w_m, orders[i].box_h_m)
                    for i in route), key=lambda b: b[0] * b[1] * b[2], reverse=True)
    for box in boxes:
        placed = False
        for si, space in enumerate(free):
            for dims in sorted(set(itertools.permutations(box)), reverse=True):
                l, w, h = dims
                sx, sy, sz = space
                if l <= sx + 1e-9 and w <= sy + 1e-9 and h <= sz + 1e-9:
                    residual = []
                    if sx - l > 1e-9: residual.append((sx - l, sy, sz))
                    if sy - w > 1e-9: residual.append((l, sy - w, sz))
                    if sz - h > 1e-9: residual.append((l, w, sz - h))
                    free.pop(si); free.extend(residual); placed = True
                    break
            if placed: break
        if not placed:
            return False
    return True


def infrastructure_rule(a, b):
                                                                                
    a, b = sorted((int(a), int(b)))
    if a == b: return None
    rule = {}
    if (a + b) % 13 == 0: rule["two_wheeler_allowed"] = False
    if (a * 31 + b * 17) % 29 == 0: rule["max_payload_kg"] = 500
    if (a * 11 + b * 7) % 31 == 0: rule["max_vehicle_height_m"] = 2.4
    return rule or None


def infrastructure_feasible(route, vehicle, orders):
    if not CONFIG.get("enable_infrastructure", True):
        return True
    cities = [0] + [orders[i].city for i in route] + [0]
    for a, b in zip(cities, cities[1:]):
        rule = infrastructure_rule(a, b)
        if not rule: continue
        if rule.get("two_wheeler_allowed") is False and vehicle["name"].startswith("2W"):
            return False
        if vehicle["weight"] > rule.get("max_payload_kg", math.inf): return False
        if vehicle["height"] > rule.get("max_vehicle_height_m", math.inf): return False
    return True


def score_route(route, vehicle, orders, d, seed, objective="full"):
    if not route:
        return 0.0, 0.0, 0.0, 100.0
    load_w = sum(orders[i].weight for i in route)
    load_v = sum(orders[i].volume for i in route)
    if load_w > vehicle["weight"] or load_v > vehicle["volume"]:
        return math.inf, 0.0, math.inf, 0.0
    if not pack3d(route, vehicle, orders):
        return math.inf, 0.0, math.inf, 0.0
    if not infrastructure_feasible(route, vehicle, orders):
        return math.inf, 0.0, math.inf, 0.0
    clock = 8.0
    distance = 0.0
    late_days = 0.0
    flagged = 0
    prev = 0
    ontime = 0
    for idx in route:
        o = orders[idx]
        km = d[prev][idx + 1]
        distance += km
        clock += travel_hours(km, vehicle, clock)
        clock += service_minutes(o, seed) / 60.0
        ontime += clock <= o.due_h
                                                                         
                                                                            
        late_days += max(0, math.ceil((clock - o.due_h) / 24.0))
        flagged += int(o.fragile or o.cooling)
        prev = idx + 1
    distance += d[prev][0]
    cost, penalty = seven_cost(distance, vehicle, late_days, flagged)
    if objective == "simple":
        cost = distance * 12.0 + penalty
    return cost, distance, penalty, 100.0 * ontime / len(route)


def dbscan_split(orders, d, group_count):
    eps = CONFIG["dbscan_eps_km"]
    min_samples = CONFIG["dbscan_min_samples"]
    labels = [-2] * len(orders)
    cluster_id = 0

    def neighbors(index):
        return [other for other in range(len(orders))
                if d[orders[index].oid + 1][orders[other].oid + 1] <= eps]

    for index in range(len(orders)):
        if labels[index] != -2:
            continue
        nearby = neighbors(index)
        if len(nearby) < min_samples:
            labels[index] = -1
            continue
        labels[index] = cluster_id
        pending = list(nearby)
        pending_set = set(pending)
        while pending:
            current = pending.pop()
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] != -2:
                continue
            labels[current] = cluster_id
            current_neighbors = neighbors(current)
            if len(current_neighbors) >= min_samples:
                for other in current_neighbors:
                    if other not in pending_set:
                        pending.append(other)
                        pending_set.add(other)
        cluster_id += 1

    clusters = {}
    noise = []
    for index, label in enumerate(labels):
        if label == -1:
            noise.append(orders[index])
        else:
            clusters.setdefault(label, []).append(orders[index])

    groups = [[] for _ in range(group_count)]
    for cluster in sorted(clusters.values(), key=len, reverse=True):
        target = min(range(group_count), key=lambda i: len(groups[i]))
        groups[target].extend(cluster)
    for order in noise:
        target = min(range(group_count), key=lambda i: len(groups[i]))
        groups[target].append(order)
    return groups


def make_splits(orders, d):
    strategies: list[tuple[str, Callable]] = [
        ("due_time", lambda o: o.due_h),
        ("arrival_time", lambda o: o.arrival_h),
        ("latitude", lambda o: o.lat),
        ("depot_distance", lambda o: d[0][o.oid + 1]),
        ("longitude", lambda o: o.lon),
        ("latitude_longitude", lambda o: o.lat + o.lon),
    ]
    if CONFIG.get("enable_penalty_risk_split", True):
        strategies.append(("penalty_risk", lambda o: o.penalty_risk * max(0.0, 21 - o.due_h)))
    strategies.append(("city_location", lambda o: (o.city, o.lat, o.lon)))
    out = []
    groups = CONFIG["assumed_route_groups"]
    for name, key in strategies:
        ordered = sorted(orders, key=key)
        out.append((name, [list(ordered[i::groups]) for i in range(groups)]))
    if CONFIG.get("enable_dbscan", False):
        out.append(("dbscan", dbscan_split(orders, d, groups)))
    return out


def make_routes(group, orders, d, kind, seed):
    ids = [o.oid for o in group]
    if kind == "EDD":
        return sorted(ids, key=lambda i: orders[i].due_h)
    if kind == "RNN":
        rng = random.Random(seed)
        remaining, out, here = set(ids), [], 0
        while remaining:
            nxt = min(remaining, key=lambda i: d[here][i + 1] * rng.uniform(.85, 1.15))
            out.append(nxt); remaining.remove(nxt); here = nxt + 1
        return out
    if kind == "MU":
        return sorted(ids, key=lambda i: orders[i].due_h - d[0][i + 1] / 35)
    if kind == "FI":
        remaining = set(ids)
        first = max(remaining, key=lambda i: d[0][i + 1]); out = [first]; remaining.remove(first)
        while remaining:
            nxt = max(remaining, key=lambda i: min(d[i + 1][j + 1] for j in out))
            pos = min(range(len(out) + 1), key=lambda p: insertion_delta(out, nxt, p, d))
            out.insert(pos, nxt); remaining.remove(nxt)
        return out
    if kind == "G TW":
        return sorted(ids, key=lambda i: (max(0, 8 + d[0][i + 1] / 35 - orders[i].due_h), d[0][i + 1]))
    if kind == "CFRS":
        return sorted(ids, key=lambda i: (orders[i].city, orders[i].lat, orders[i].lon))
    if kind == "BiDir":
        ascending = sorted(ids, key=lambda i: d[0][i + 1])
        return ascending[::2] + ascending[1::2][::-1]
                              
    remaining, out, here = set(ids), [], 0
    while remaining:
        nxt = min(remaining, key=lambda i: d[here][i + 1])
        out.append(nxt); remaining.remove(nxt); here = nxt + 1
    return out


def insertion_delta(route, node, pos, d):
    left = 0 if pos == 0 else route[pos - 1] + 1
    right = 0 if pos == len(route) else route[pos] + 1
    return d[left][node + 1] + d[node + 1][right] - d[left][right]


HEURISTICS = ["NN", "BiDir", "MU", "FI", "RNN", "G TW", "CFRS"]


def assign_vehicles(routes, orders, d, seed, objective="full"):
                                                                             
                                         
    table = [[score_route(route, FLEET[vi], orders, d, seed, objective)
              for vi in range(len(FLEET))] for route in routes]
    best = (math.inf, None, None)
    for fleet_ids in itertools.permutations(range(len(FLEET)), len(routes)):
        details = [(vi, table[ri][vi]) for ri, vi in enumerate(fleet_ids)]
        total = sum(x[1][0] for x in details)
                                                                           
                                                                             
                                                                              
        if best[1] is None or total < best[0]:
            best = total, fleet_ids, details
    return best


def candidate_runs(orders, d, seed):
    rows = []
    deadline = time.perf_counter() + CONFIG["ortools_seconds"]
    for si, (split_name, split) in enumerate(make_splits(orders, d)):
        for heuristic in HEURISTICS:
            if time.perf_counter() >= deadline:
                return rows
            routes = [make_routes(g, orders, d, heuristic, seed + si) for g in split]
            total, assigned, details = assign_vehicles(routes, orders, d, seed)
            flatdist = sum(x[1][1] for x in details)
            penalty = sum(x[1][2] for x in details)
            ontime = sum(x[1][3] * len(r) for x, r in zip(details, routes)) / len(orders)
            rows.append({"split": si, "split_strategy": split_name, "heuristic": heuristic, "cost_rs": total,
                         "distance_km": flatdist, "penalty_rs": penalty,
                         "ontime_pct": ontime,
                         "vehicles": ";".join(FLEET[v]["name"] for v in assigned)})
    return rows


def route_solution_stats(routes, orders, d, seed):
    total, assigned, details = assign_vehicles(routes, orders, d, seed)
    distance = sum(x[1][1] for x in details)
    penalty = sum(x[1][2] for x in details)
    ontime = sum(x[1][3] * len(r) for x, r in zip(details, routes)) / len(orders)
    return total, distance, penalty, ontime, assigned


def order_crossover(a, b, rng):
    if len(a) < 3: return b[:]
    lo, hi = sorted(rng.sample(range(len(a)), 2))
    child = [None] * len(a); child[lo:hi] = a[lo:hi]
    rest = [x for x in b if x not in child]
    it = iter(rest)
    return [next(it) if x is None else x for x in child]


def write_csv(name, rows):
    if rows:
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)


def summarize_stats(rows, key):
    vals = [float(r[key]) for r in rows
            if r.get(key) is not None and math.isfinite(float(r[key]))]
    if not vals:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def run_ablation_experiments():
    variants = [
        ("baseline", None, {}),
        ("no_3d_packing", "3d_packing", {"enable_pack3d": False}),
        ("no_infrastructure", "infrastructure", {"enable_infrastructure": False}),
        ("no_dbscan", "dbscan", {"enable_dbscan": False}),
        ("no_penalty_risk_split", "penalty_risk_split", {"enable_penalty_risk_split": False}),
        ("no_traffic", "traffic", {"enable_traffic": False}),
        ("no_stochastic_service_time", "stochastic_service_time", {"enable_stochastic_service_time": False}),
    ]
    rows = []
    original = copy.deepcopy(CONFIG)
    try:
        for variant_name, ablated_feature, overrides in variants:
            CONFIG.clear()
            CONFIG.update(copy.deepcopy(original))
            CONFIG["enable_dbscan"] = True
            for key, value in overrides.items():
                CONFIG[key] = copy.deepcopy(value)
            for seed in CONFIG["seeds"][:3]:
                orders, d = make_instance(seed)
                started = time.perf_counter()
                candidate = candidate_runs(orders, d, seed)
                runtime = time.perf_counter() - started
                best = min(candidate, key=lambda row: row["cost_rs"])
                rows.append({
                    "variant": variant_name,
                    "ablated_feature": ablated_feature or "none",
                    "seed": seed,
                    "best_cost_rs": best["cost_rs"],
                    "best_distance_km": best["distance_km"],
                    "best_penalty_rs": best["penalty_rs"],
                    "best_ontime_pct": best["ontime_pct"],
                    "best_split_strategy": best["split_strategy"],
                    "best_algorithm": best["heuristic"],
                    "n_candidates": len(candidate),
                    "runtime_s": runtime,
                })
    finally:
        CONFIG.clear()
        CONFIG.update(original)
    return rows


def run_scalability_experiments():
    sizes = [20, 30, 50, 80]
    seeds = [11, 22]
    rows = []
    for size in sizes:
        for seed in seeds:
            orders, d = make_instance(seed, order_count=size)
            t0 = time.perf_counter()
            base = ortools_baseline(orders, d, seed, seconds=CONFIG["ortools_seconds"])
            runtime = time.perf_counter() - t0
            rows.append({
                "n_orders": size,
                "seed": seed,
                "algorithm": "OR-Tools",
                "cost_rs": base.get("cost_rs"),
                "distance_km": base.get("distance_km"),
                "penalty_rs": base.get("penalty_rs"),
                "ontime_pct": base.get("ontime_pct"),
                "runtime_s": runtime,
            })
            t0 = time.perf_counter()
            meta = metaheuristic_runs(orders, d, seed)
            runtime = time.perf_counter() - t0
            for algo in meta:
                rows.append({
                    "n_orders": size,
                    "seed": seed,
                    "algorithm": algo["algorithm"],
                    "cost_rs": algo["cost_rs"],
                    "distance_km": algo["distance_km"],
                    "penalty_rs": algo["penalty_rs"],
                    "ontime_pct": algo["ontime_pct"],
                    "runtime_s": runtime,
                })
    return rows


def metaheuristic_runs(orders, d, seed):
                                                                       
    rng = random.Random(seed + 80001)
    initial = [make_routes(g, orders, d, "NN", seed) for g in make_splits(orders, d)[0][1]]
    def fit(routes): return route_solution_stats(routes, orders, d, seed)[0]
    def best_route_cost(route):
        return min(score_route(route, v, orders, d, seed)[0] for v in FLEET)
    rows = []

                                                                         
    cur = [r[:] for r in initial]; best = [r[:] for r in cur]
    best_cost = cur_cost = fit(cur)
    for _ in range(CONFIG["metaheuristic_iterations"]):
        cand = [r[:] for r in cur]
        ri = rng.randrange(len(cand)); route = cand[ri]
        q = max(1, len(route) // 4)
        removed = rng.sample(route, min(q, len(route)))
        cand[ri] = [x for x in route if x not in removed]
        rng.shuffle(removed)
        for node in removed:
            positions = range(len(cand[ri]) + 1)
            pos = min(positions, key=lambda p: best_route_cost(cand[ri][:p] + [node] + cand[ri][p:]))
            cand[ri].insert(pos, node)
        value = fit(cand)
        if value < cur_cost or rng.random() < .08:
            cur, cur_cost = cand, value
        if value < best_cost: best, best_cost = cand, value
    c, km, pen, ot, _ = route_solution_stats(best, orders, d, seed)
    rows.append({"algorithm":"ALNS", "cost_rs":c, "distance_km":km, "penalty_rs":pen, "ontime_pct":ot})

                                                                      
    population = []
    for _ in range(10):
        ind = [r[:] for r in initial]
        for r in ind: rng.shuffle(r)
        population.append(ind)
    for _ in range(12):
        population.sort(key=fit); nxt = population[:2]
        while len(nxt) < 10:
            p1, p2 = rng.sample(population[:6], 2)
            child = [order_crossover(a, b, rng) for a, b in zip(p1, p2)]
            ri = rng.randrange(len(child))
            if len(child[ri]) > 1:
                i, j = rng.sample(range(len(child[ri])), 2); child[ri][i], child[ri][j] = child[ri][j], child[ri][i]
            nxt.append(child)
        population = nxt
    best = min(population, key=fit)
    c, km, pen, ot, _ = route_solution_stats(best, orders, d, seed)
    rows.append({"algorithm":"Genetic Algorithm", "cost_rs":c, "distance_km":km, "penalty_rs":pen, "ontime_pct":ot})

                                                                                 
    cur = [r[:] for r in initial]; best = [r[:] for r in cur]
    cur_cost = best_cost = fit(cur); tabu = []
    for _ in range(CONFIG["metaheuristic_iterations"]):
        neighbors = []
        for ri, route in enumerate(cur):
            if len(route) < 2: continue
            pairs = list(itertools.combinations(range(len(route)), 2)); rng.shuffle(pairs)
            for i, j in pairs[:12]:
                cand = [r[:] for r in cur]; cand[ri][i], cand[ri][j] = cand[ri][j], cand[ri][i]
                move = tuple(sorted((route[i], route[j])))
                val = fit(cand)
                if move not in tabu or val < best_cost: neighbors.append((val, cand, move))
        if not neighbors: break
        value, cur, move = min(neighbors, key=lambda x: x[0]); cur_cost = value
        tabu.append(move); tabu = tabu[-7:]
        if value < best_cost: best, best_cost = [r[:] for r in cur], value
    c, km, pen, ot, _ = route_solution_stats(best, orders, d, seed)
    rows.append({"algorithm":"Tabu Search", "cost_rs":c, "distance_km":km, "penalty_rs":pen, "ontime_pct":ot})
    return rows


def _legacy_ortools_baseline(orders, d, seed, seconds=None):
    n, nv = len(orders), len(FLEET)
    manager = pywrapcp.RoutingIndexManager(n + 1, nv, 0)
    routing = pywrapcp.RoutingModel(manager)
    def transit(from_idx, to_idx, vi):
        a, b = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        km = d[a][b]
        if b == 0:
            return 0
        o = orders[b - 1]
        v = FLEET[vi]
        operating = CONFIG["fuel_or_energy_rs_per_km"][v["kind"]] + CONFIG["wage_rs_per_km"] + CONFIG["maintenance_rs_per_km"] + (0.35 if v["kind"] == "electric" else 0)
        return int(round(CONFIG["cost_rate_scale"] * (km * operating + (CONFIG["packaging_rs_flagged"] if o.fragile or o.cooling else 0))))
    for vi in range(nv):
        cb = routing.RegisterTransitCallback(lambda a, b, vi=vi: transit(a, b, vi))
        routing.SetArcCostEvaluatorOfVehicle(cb, vi)
        routing.SetFixedCostOfVehicle(int(CONFIG["cost_rate_scale"] * (FLEET[vi]["fixed"] + CONFIG["toll_rs_route"])), vi)
    def demand_cb(idx):
        node = manager.IndexToNode(idx)
        return 0 if node == 0 else int(round(orders[node - 1].weight * 100))
    demand = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand, 0,
        [int(v["weight"] * 100) for v in FLEET], True, "Weight")
    def vol_cb(idx):
        node = manager.IndexToNode(idx)
        return 0 if node == 0 else int(round(orders[node - 1].volume * 1000))
    vol = routing.RegisterUnaryTransitCallback(vol_cb)
    routing.AddDimensionWithVehicleCapacity(vol, 0,
        [int(v["volume"] * 1000) for v in FLEET], True, "Volume")
    def time_cb(from_idx, to_idx):
        a, b = manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)
        km = d[a][b]
        service = 0 if b == 0 else CONFIG["service_base_min"] + CONFIG["service_min_per_kg"] * orders[b-1].weight + CONFIG["service_min_per_m3"] * orders[b-1].volume
        return int((km / 35 * 1.15 * 60) + service)
    tcb = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(tcb, 12 * 60, 48 * 60, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for vi in range(nv):
        time_dim.CumulVar(routing.Start(vi)).SetRange(8 * 60, 8 * 60)
    for oi, o in enumerate(orders):
        idx = manager.NodeToIndex(oi + 1)
        lo = max(8 * 60, int(o.arrival_h * 60)); hi = int(o.due_h * 60)
        time_dim.CumulVar(idx).SetRange(lo, hi)
        deadline = int(CONFIG["penalty_deadline_h"] * 60)
        time_dim.SetCumulVarSoftUpperBound(idx, deadline,
            max(1, int(CONFIG["cost_rate_scale"] * CONFIG["late_rate_rs_per_day"] / (24 * 60))))
        eligible = {vi for vi, v in enumerate(FLEET) if o.weight <= v["weight"] and o.volume <= v["volume"]}
                                                                               
                                                                              
        for vi in range(nv):
            if vi not in eligible:
                routing.solver().Add(routing.VehicleVar(idx) != vi)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(seconds or CONFIG["ortools_seconds"])
    params.log_search = False
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {"status": "no_solution", "cost_rs": None, "distance_km": None,
                "penalty_rs": None, "ontime_pct": None}
    distance = 0.0; penalty = 0.0; evaluated_cost = 0.0; ontime = 0; served = 0
    for vi, vehicle in enumerate(FLEET):
        idx = routing.Start(vi); route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node:
                route.append(node - 1)
            idx = solution.Value(routing.NextVar(idx))
        if not route:
            continue
        result = score_route(route, vehicle, orders, d, seed)
        evaluated_cost += result[0]
        distance += result[1]; penalty += result[2]
        ontime += result[3] * len(route) / 100; served += len(route)
    status = "ok" if math.isfinite(evaluated_cost) else "infeasible_post_validation"
    return {"status": status, "cost_rs": evaluated_cost if math.isfinite(evaluated_cost) else None,
            "solver_objective_rs": float(solution.ObjectiveValue()),
            "distance_km": distance if math.isfinite(evaluated_cost) else None,
            "penalty_rs": penalty if math.isfinite(evaluated_cost) else None,
            "ontime_pct": 100 * ontime / max(1, served) if math.isfinite(evaluated_cost) else None}


def ortools_baseline(orders, d, seed, seconds=None):
    started = time.perf_counter()
    n, nv = len(orders), len(FLEET)
    horizon = 7 * 24 * 60 * 60
    time_big_m = horizon + 24 * 60 * 60
    distance_scale = 1_000_000
    cost_scale = 1_000_000_000
    time_limit = CONFIG["ortools_seconds"] if seconds is None else seconds
    service_seconds = [int(round(service_minutes(o, seed) * 60)) for o in orders]
    distance_units = [[int(round(km * distance_scale)) for km in row] for row in d]
    model = cp_model.CpModel()

    arcs = {}
    skipped = {}
    used = []
    depot_loops = {}
    tolls = {}
    late_vars = {}
    for vi in range(nv):
        vehicle_arcs = []
        for i in range(n + 1):
            for j in range(n + 1):
                if i == j:
                    continue
                arc = model.NewBoolVar(f"arc_{vi}_{i}_{j}")
                arcs[vi, i, j] = arc
                vehicle_arcs.append((i, j, arc))
        depot_loop = model.NewBoolVar(f"depot_loop_{vi}")
        depot_loops[vi] = depot_loop
        used.append(model.NewBoolVar(f"used_{vi}"))
        model.Add(used[vi] + depot_loop == 1)
        vehicle_arcs.append((0, 0, depot_loop))
        for i in range(1, n + 1):
            skipped[vi, i] = model.NewBoolVar(f"skip_{vi}_{i}")
            vehicle_arcs.append((i, i, skipped[vi, i]))
        model.AddCircuit(vehicle_arcs)

    for i in range(1, n + 1):
        model.Add(sum(skipped[vi, i] for vi in range(nv)) == nv - 1)
    for vi, vehicle in enumerate(FLEET):
        model.Add(sum(int(round(orders[i - 1].weight * 1000)) * (1 - skipped[vi, i])
                      for i in range(1, n + 1)) <= int(vehicle["weight"] * 1000))
        model.Add(sum(int(round(orders[i - 1].volume * 1_000_000)) * (1 - skipped[vi, i])
                      for i in range(1, n + 1)) <= int(vehicle["volume"] * 1_000_000))

    for vi, vehicle in enumerate(FLEET):
        for i in range(n + 1):
            city_a = 0 if i == 0 else orders[i - 1].city
            for j in range(n + 1):
                if i == j:
                    continue
                city_b = 0 if j == 0 else orders[j - 1].city
                rule = infrastructure_rule(city_a, city_b) if CONFIG.get("enable_infrastructure", True) else None
                if not rule:
                    continue
                blocked = (
                    rule.get("two_wheeler_allowed") is False and vehicle["name"].startswith("2W")
                ) or vehicle["weight"] > rule.get("max_payload_kg", math.inf) \
                    or vehicle["height"] > rule.get("max_vehicle_height_m", math.inf)
                if blocked:
                    model.Add(arcs[vi, i, j] == 0)

    times = [model.NewIntVar(0, horizon, f"time_{i}") for i in range(n + 1)]
    model.Add(times[0] == 8 * 60 * 60)
    traffic = CONFIG["traffic"]
    gamma = CONFIG.get("gamma", CONFIG.get("weather_factor", 1.0))
    delta = CONFIG.get("delta", CONFIG.get("lane_factor", 1.0))
    intervals = [
        (0, 6 * 3600 - 1, "night"),
        (6 * 3600, 7 * 3600 - 1, "midday"),
        (7 * 3600, 10 * 3600 - 1, "rush_morning"),
        (10 * 3600, 16 * 3600 - 1, "midday"),
        (16 * 3600, 19 * 3600 - 1, "rush_evening"),
        (19 * 3600, 21 * 3600 - 1, "midday"),
        (21 * 3600, horizon, "night"),
    ]
    bands = {}
    arc_band_vars = {}
    for i in range(1, n + 1):
        choices = []
        for bi, (lo, hi, _) in enumerate(intervals):
            choice = model.NewBoolVar(f"band_{i}_{bi}")
            bands[i, bi] = choice
            choices.append(choice)
            model.Add(times[i] >= lo).OnlyEnforceIf(choice)
            model.Add(times[i] <= hi).OnlyEnforceIf(choice)
        model.AddExactlyOne(choices)

    objective_terms = []
    late_rate = int(round(CONFIG["late_rate_rs_per_day"] * CONFIG["cost_rate_scale"] * cost_scale))
    for oi, order in enumerate(orders, start=1):
        due_cutoff = int(round(order.due_h * 3600))
        for day in range(7):
            late = model.NewBoolVar(f"late_{oi}_{day}")
            late_vars[oi, day] = late
            threshold = due_cutoff + day * 24 * 60 * 60 + 1
            model.Add(times[oi] >= threshold).OnlyEnforceIf(late)
            model.Add(times[oi] <= threshold - 1).OnlyEnforceIf(late.Not())
            objective_terms.append(late_rate * late)
        if order.fragile or order.cooling:
            objective_terms.append(int(round(CONFIG["packaging_rs_flagged"] * CONFIG["cost_rate_scale"] * cost_scale)))

    toll_distance = 40 * distance_scale
    max_arc_distance = max(max(row) for row in distance_units)
    max_route_distance = max_arc_distance * (n + 1)
    toll_cost = int(round(CONFIG["toll_rs_route"] * CONFIG["cost_rate_scale"] * cost_scale))
    fixed_costs = []
    for vi, vehicle in enumerate(FLEET):
        route_distance = sum(distance_units[i][j] * arcs[vi, i, j]
                             for i in range(n + 1) for j in range(n + 1) if i != j)
        toll = model.NewBoolVar(f"toll_{vi}")
        tolls[vi] = toll
        model.Add(route_distance <= toll_distance + max_route_distance * toll)
        model.Add(route_distance >= (toll_distance + 1) * toll)
        objective_terms.append(toll_cost * toll)
        fixed_cost = int(round(vehicle["fixed"] * CONFIG["cost_rate_scale"] * cost_scale))
        fixed_costs.append(fixed_cost * used[vi])

        energy_rate = CONFIG["fuel_or_energy_rs_per_km"][vehicle["kind"]]
        distance_rate = energy_rate + CONFIG["wage_rs_per_km"] + CONFIG["maintenance_rs_per_km"]
        if vehicle["kind"] == "electric":
            distance_rate += 0.35
        distance_coefficient = int(round(distance_rate * CONFIG["cost_rate_scale"] * 1000))
        objective_terms.append(distance_coefficient * route_distance)

    objective_terms.extend(fixed_costs)
    for vi, vehicle in enumerate(FLEET):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                arc = arcs[vi, i, j]
                selected = []
                travel_by_band = []
                km = d[i][j]
                for bi, (_, _, band_name) in enumerate(intervals):
                    both = model.NewBoolVar(f"arc_band_{vi}_{i}_{j}_{bi}")
                    arc_band_vars[vi, i, j, bi] = both
                    model.Add(both <= arc)
                    model.Add(both <= bands[i, bi])
                    model.Add(both >= arc + bands[i, bi] - 1)
                    selected.append(both)
                    multiplier = traffic[band_name] * gamma * delta if CONFIG.get("enable_traffic", True) else 1.0
                    travel_seconds = int(round(km / vehicle["speed"] * multiplier * 3600))
                    travel_by_band.append(travel_seconds * both)
                duration = sum(travel_by_band) + service_seconds[j - 1]
                model.Add(sum(selected) == arc)
                model.Add(times[j] >= times[i] + duration - time_big_m * (1 - arc))
                model.Add(times[j] <= times[i] + duration + time_big_m * (1 - arc))
            first_arc = arcs[vi, 0, i]
            multiplier = traffic["rush_morning"] * gamma * delta if CONFIG.get("enable_traffic", True) else 1.0
            first_travel = int(round(d[0][i] / vehicle["speed"] * multiplier * 3600))
            first_duration = first_travel + service_seconds[i - 1]
            model.Add(times[i] >= times[0] + first_duration - time_big_m * (1 - first_arc))
            model.Add(times[i] <= times[0] + first_duration + time_big_m * (1 - first_arc))

    model.Minimize(sum(objective_terms))

    initial_groups = make_splits(orders, d)[0][1]
    initial_routes = [make_routes(group, orders, d, "NN", seed) for group in initial_groups]
    initial_cost, initial_assignment, _ = assign_vehicles(initial_routes, orders, d, seed)
    if math.isfinite(initial_cost):
        hinted_routes = {vi: [] for vi in range(nv)}
        for route, vi in zip(initial_routes, initial_assignment):
            hinted_routes[vi] = route
        hinted_bands = {}
        hinted_times = {}
        for vi, route in hinted_routes.items():
            path = [0] + [node + 1 for node in route] + ([0] if route else [])
            selected_arcs = set(zip(path, path[1:]))
            visited = {node + 1 for node in route}
            model.AddHint(used[vi], int(bool(route)))
            model.AddHint(depot_loops[vi], int(not route))
            for i in range(1, n + 1):
                model.AddHint(skipped[vi, i], int(i not in visited))
            for key, arc in arcs.items():
                if key[0] == vi:
                    model.AddHint(arc, int(key[1:] in selected_arcs))
            clock = 8 * 3600
            previous = 0
            for node in route:
                order_index = node + 1
                distance = d[previous][order_index]
                clock += int(round(travel_hours(distance, FLEET[vi], clock / 3600) * 3600))
                clock += service_seconds[node]
                hinted_times[order_index] = clock
                model.AddHint(times[order_index], clock)
                band_index = next(bi for bi, (lo, hi, _) in enumerate(intervals) if lo <= clock <= hi)
                hinted_bands[order_index] = band_index
                for bi in range(len(intervals)):
                    model.AddHint(bands[order_index, bi], int(bi == band_index))
                previous = order_index
            route_distance = sum(distance_units[i][j] for i, j in selected_arcs)
            model.AddHint(tolls[vi], int(route_distance > toll_distance))

        for (vi, i, j, bi), both in arc_band_vars.items():
            active = (i, j) in set(zip(
                [0] + [node + 1 for node in hinted_routes[vi]],
                [node + 1 for node in hinted_routes[vi]] + ([0] if hinted_routes[vi] else []),
            ))
            model.AddHint(both, int(active and hinted_bands.get(i) == bi))
        for (order_index, day), late in late_vars.items():
            due_cutoff = int(round(orders[order_index - 1].due_h * 3600))
            threshold = due_cutoff + day * 24 * 60 * 60 + 1
            model.AddHint(late, int(hinted_times.get(order_index, 0) >= threshold))
    deadline = started + time_limit
    packing_cuts = 0
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return {"status": "time_limit_no_feasible_route", "cost_rs": None,
                    "solver_objective_rs": None, "objective_bound_rs": None,
                    "optimality_gap_pct": None, "distance_km": None,
                    "penalty_rs": None, "ontime_pct": None,
                    "runtime_s": time.perf_counter() - started, "packing_cuts": packing_cuts}
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = 1
        solver.parameters.cp_model_presolve = False
        solver.parameters.linearization_level = 0
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {"status": solver.StatusName(status), "cost_rs": None,
                    "solver_objective_rs": None, "objective_bound_rs": None,
                    "optimality_gap_pct": None, "distance_km": None,
                    "penalty_rs": None, "ontime_pct": None,
                    "runtime_s": time.perf_counter() - started, "packing_cuts": packing_cuts}

        routes = []
        invalid_cuts = []
        for vi, vehicle in enumerate(FLEET):
            route = []
            node = 0
            while True:
                next_node = next((j for j in range(n + 1) if j != node and solver.Value(arcs[vi, node, j])), None)
                if next_node is None or next_node == 0:
                    break
                route.append(next_node - 1)
                node = next_node
            if route:
                routes.append((vi, route))
                if not pack3d(route, vehicle, orders):
                    path = [0] + [order_id + 1 for order_id in route] + [0]
                    invalid_cuts.append([arcs[vi, a, b] for a, b in zip(path, path[1:])])
        if invalid_cuts:
            for cut in invalid_cuts:
                model.Add(sum(cut) <= len(cut) - 1)
                packing_cuts += 1
            continue

        results = [score_route(route, FLEET[vi], orders, d, seed) for vi, route in routes]
        if any(not math.isfinite(result[0]) for result in results):
            for (vi, route), result in zip(routes, results):
                if not math.isfinite(result[0]):
                    path = [0] + [order_id + 1 for order_id in route] + [0]
                    model.Add(sum(arcs[vi, a, b] for a, b in zip(path, path[1:])) <= len(path) - 2)
                    packing_cuts += 1
            continue

        evaluated_cost = sum(result[0] for result in results)
        distance = sum(result[1] for result in results)
        penalty = sum(result[2] for result in results)
        ontime = sum(result[3] * len(route) for (_, route), result in zip(routes, results)) / n
        objective = solver.ObjectiveValue() / cost_scale
        bound = solver.BestObjectiveBound() / cost_scale
        gap = 100 * max(0.0, objective - bound) / max(abs(objective), 1e-9)
        return {"status": solver.StatusName(status), "cost_rs": evaluated_cost,
                "solver_objective_rs": objective, "objective_bound_rs": bound,
                "optimality_gap_pct": gap, "distance_km": distance,
                "penalty_rs": penalty, "ontime_pct": ontime,
                "runtime_s": time.perf_counter() - started, "packing_cuts": packing_cuts}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA = HERE / "datasets"
    DATA.mkdir(parents=True, exist_ok=True)
    (HERE / "assumptions.json").write_text(json.dumps(CONFIG | {"fleet": FLEET}, indent=2), encoding="utf-8")
    candidate_rows, base_rows, instance_rows, meta_rows = [], [], [], []
    for seed in CONFIG["seeds"]:
        orders, d = make_instance(seed)
        for o in orders:
            instance_rows.append(asdict(o) | {"seed": seed})
        cr = candidate_runs(orders, d, seed)
        best = min(cr, key=lambda r: r["cost_rs"])
        candidate_rows.extend({"seed": seed, **r} for r in cr)
        meta_rows.extend({"seed": seed, **r} for r in metaheuristic_runs(orders, d, seed))
        base = ortools_baseline(orders, d, seed)
        base_rows.append({"seed": seed, **base})
        print(f"seed={seed}: best 56-route candidate Rs {best['cost_rs']:.0f}, "
              f"{best['distance_km']:.1f} km, penalty Rs {best['penalty_rs']:.0f}, "
              f"on-time {best['ontime_pct']:.1f}%; OR-Tools={base['status']} "
              f"Rs {base['cost_rs']}")
    ablation_rows = run_ablation_experiments()
    scalability_rows = run_scalability_experiments()
    write_csv("ablation_summary.csv", ablation_rows)
    write_csv("scalability_summary.csv", scalability_rows)
    with (DATA / "synthetic_30_orders.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(instance_rows[0]))
        w.writeheader(); w.writerows(instance_rows)
                                                                        
    infra_rows = []
    city_count = CONFIG["cities_including_depot"]
    for a in range(city_count):
        for b in range(a + 1, city_count):
            rule = infrastructure_rule(a, b)
            if rule: infra_rows.append({"from_city_id": a, "to_city_id": b, **rule,
                                        "source": "synthetic_assumption"})
    if infra_rows:
        infra_fields = list(dict.fromkeys(k for row in infra_rows for k in row))
        with (DATA / "synthetic_infrastructure_constraints.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=infra_fields); w.writeheader(); w.writerows(infra_rows)
    write_csv("candidate_routes.csv", candidate_rows)
    write_csv("ortools_baseline.csv", base_rows)
    write_csv("metaheuristic_results.csv", meta_rows)
    minima = [min((row for row in candidate_rows if row["seed"] == seed),
                  key=lambda row: row["cost_rs"])
              for seed in CONFIG["seeds"]]
    def stats(key):
        vals = [float(r[key]) for r in minima]
        return {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals)>1 else 0,
                "min": min(vals), "max": max(vals)}
    baseline_solved = [r for r in base_rows if r["cost_rs"] is not None]
    def baseline_stats(key):
        vals = [float(r[key]) for r in baseline_solved if r[key] is not None]
        if not vals:
            return {"mean": None, "sd": None}
        return {"mean": statistics.mean(vals), "sd": statistics.stdev(vals) if len(vals)>1 else 0}
    summary = {"status": "synthetic_scenario_summary",
               "n_seeds": len(CONFIG["seeds"]), "candidate_best_cost_rs": stats("cost_rs"),
               "candidate_distance_km": stats("distance_km"),
               "candidate_penalty_rs": stats("penalty_rs"), "candidate_ontime_pct": stats("ontime_pct"),
               "ortools_shared_model": {"cost_rs": baseline_stats("cost_rs"),
                   "solver_objective_rs": baseline_stats("solver_objective_rs"),
                   "objective_bound_rs": baseline_stats("objective_bound_rs"),
                   "optimality_gap_pct": baseline_stats("optimality_gap_pct"),
                   "distance_km": baseline_stats("distance_km"),
                   "penalty_rs": baseline_stats("penalty_rs"),
                   "ontime_pct": baseline_stats("ontime_pct"),
                   "n_solved": len(baseline_solved),
                   "n_runs": len(base_rows),
                   "statuses": {status: sum(row["status"] == status for row in base_rows)
                                for status in sorted({row["status"] for row in base_rows})}},
               "metaheuristics": {name: {key: {"mean": statistics.mean(float(r[key]) for r in meta_rows if r["algorithm"] == name),
                                                       "sd": statistics.stdev(float(r[key]) for r in meta_rows if r["algorithm"] == name)}
                                                for key in ("cost_rs", "distance_km", "penalty_rs", "ontime_pct")}
                                   for name in ("ALNS", "Genetic Algorithm", "Tabu Search")},
               "ablation": {
                   variant: {key: summarize_stats([r for r in ablation_rows if r["variant"] == variant], key)
                             for key in ("best_cost_rs", "best_distance_km", "best_penalty_rs", "best_ontime_pct", "runtime_s", "n_candidates")}
                   for variant in sorted({r["variant"] for r in ablation_rows})
               },
               "scalability": {
                   str(size): {
                       "n_orders": size,
                       "avg_runtime_s": statistics.mean(float(r["runtime_s"]) for r in scalability_rows if r["n_orders"] == size),
                       "avg_cost_rs": summarize_stats([r for r in scalability_rows if r["n_orders"] == size], "cost_rs")["mean"],
                       "avg_distance_km": summarize_stats([r for r in scalability_rows if r["n_orders"] == size], "distance_km")["mean"],
                       "algorithms": sorted({r["algorithm"] for r in scalability_rows if r["n_orders"] == size}),
                   } for size in sorted({r["n_orders"] for r in scalability_rows})
               }}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", OUT)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
