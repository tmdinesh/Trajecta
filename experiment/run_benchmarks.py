                                                                                 
from __future__ import annotations
import csv, itertools, math, random, statistics, time
from dataclasses import dataclass
from pathlib import Path
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ROOT = Path(__file__).resolve().parent / "datasets" / "benchmarks"
OUT = Path(__file__).resolve().parent / "results"
SEEDS = (11, 22, 33)
ORTOOLS_SECONDS = 2
TRAJECTA_SECONDS = ORTOOLS_SECONDS
PUBLISHED_BKS = {"c101": (10,828.94), "r101": (19,1650.80), "rc101": (14,1696.95),
                 "p01": (None,576.87), "p02": (None,473.53)}

@dataclass
class Customer:
    id: int; x: float; y: float; demand: float
    ready: float = 0; due: float = math.inf; service: float = 0

@dataclass
class Depot:
    id: int; x: float; y: float; capacity: float; vehicles: int; max_duration: float = math.inf

def euclid(a,b): return math.hypot(a.x-b.x,a.y-b.y)

def load_instances():
    instances=[]
                                                                                
    for name in ("c101","r101","rc101"):
        lines=(ROOT/"raw"/"solomon"/f"{name}.txt").read_text(encoding="utf-8").splitlines()
        numeric=[]
        for line in lines:
            t=line.split()
            if len(t)>=7:
                try: v=[float(x) for x in t[:7]]
                except ValueError: continue
                if v[0].is_integer(): numeric.append(v)
        header=[]
        for line in lines:
            t=line.split()
            if len(t)==2:
                try: header=[int(t[0]),int(t[1])]; break
                except ValueError: pass
        depotrow=numeric[0]
        depot=Depot(0,depotrow[1],depotrow[2],header[1],header[0],depotrow[5])
        cust=[Customer(int(v[0]),v[1],v[2],v[3],v[4],v[5],v[6]) for v in numeric[1:]]
        instances.append((name,cust,[depot],"Solomon"))
                                                                             
                                                                 
    for name in ("p01","p02"):
        lines=[x.split() for x in (ROOT/"raw"/"cordeau"/name).read_text().splitlines() if x.strip()]
        typ,m,n,t=map(int,lines[0][:4]); limits=[list(map(float,lines[1+i][:2])) for i in range(t)]
        raw=lines[1+t:1+t+n+t]; cr=raw[:n]; dr=raw[n:]
        depots=[Depot(j+1,float(v[1]),float(v[2]),limits[j][1],m,limits[j][0] or math.inf) for j,v in enumerate(dr)]
        cust=[Customer(int(v[0]),float(v[1]),float(v[2]),float(v[4])) for v in cr]
        instances.append((name,cust,depots,"Cordeau-MDVRP"))
    return instances

def build_runner(customers,depots):
    depot_of={c.id:min(depots,key=lambda d:math.hypot(c.x-d.x,c.y-d.y)).id for c in customers}
    def route_eval(route,depot):
        if sum(next(c.demand for c in customers if c.id==i) for i in route)>depot.capacity+1e-9: return False,math.inf
        clock=0.0; dist=0.0; prev=Customer(0,depot.x,depot.y,0)
        for i in route:
            c=next(x for x in customers if x.id==i)
            leg=euclid(prev,c); dist+=leg; clock=max(c.ready,clock+leg)
            if clock>c.due+1e-9:return False,math.inf
            clock+=c.service;prev=c
        dist+=euclid(prev,Customer(0,depot.x,depot.y,0)); clock+=euclid(prev,Customer(0,depot.x,depot.y,0))
        if clock>depot.max_duration+1e-9:return False,math.inf
        return True,dist
    def evaluate(permutation):
        route_count=0; distance=0.0; ok=True; vehicle_shortfall=0
        customers_by_id={c.id:c for c in customers}
        for depot in depots:
            seq=[i for i in permutation if depot_of[i]==depot.id]
            best=[None]*(len(seq)+1)
            best[0]=(0,0.0)
            depot_node=Customer(0,depot.x,depot.y,0)
            for start in range(len(seq)):
                if best[start] is None:
                    continue
                load=0.0; clock=0.0; route_distance=0.0; previous=depot_node
                for end in range(start,len(seq)):
                    customer=customers_by_id[seq[end]]
                    load+=customer.demand
                    if load>depot.capacity+1e-9:
                        break
                    leg=euclid(previous,customer)
                    arrival=max(customer.ready,clock+leg)
                    if arrival>customer.due+1e-9:
                        break
                    clock=arrival+customer.service
                    route_distance+=leg
                    distance_with_return=route_distance+euclid(customer,depot_node)
                    if clock+euclid(customer,depot_node)>depot.max_duration+1e-9:
                        break
                    next_index=end+1
                    proposal=(best[start][0]+1,best[start][1]+distance_with_return)
                    if best[next_index] is None or proposal<best[next_index]:
                        best[next_index]=proposal
                    previous=customer
            if best[-1] is None or best[-1][0]>depot.vehicles:
                ok=False
                if best[-1] is None:
                    vehicle_shortfall+=len(seq)
                else:
                    vehicle_shortfall+=best[-1][0]-depot.vehicles
                    route_count+=best[-1][0]
                    distance+=best[-1][1]
                continue
            route_count+=best[-1][0]
            distance+=best[-1][1]
                                                                              
                                                                               
        score=route_count*1_000_000+distance if ok else 1e12+vehicle_shortfall*1e9+route_count*1e6+distance
        return score,route_count,distance,ok
    def construct():
        all_routes=[]
        for depot in depots:
            remaining={i for i in depot_of if depot_of[i]==depot.id}
            while remaining:
                route=[];prev=Customer(0,depot.x,depot.y,0)
                while True:
                    feasible=[]
                    for i in remaining:
                        ok,_=route_eval(route+[i],depot)
                        if ok:
                            c=next(x for x in customers if x.id==i)
                            feasible.append((euclid(prev,c),c.due-c.ready,i))
                    if not feasible:break
                    _,_,chosen=min(feasible)
                    route.append(chosen);remaining.remove(chosen)
                    prev=next(x for x in customers if x.id==chosen)
                if not route: break
                all_routes.append(route)
        return [x for route in all_routes for x in route]
    return evaluate,depot_of,construct

def ox(a,b,rng):
    if len(a)<3:return b[:]
    lo,hi=sorted(rng.sample(range(len(a)),2)); child=[None]*len(a); child[lo:hi]=a[lo:hi]
    rest=[x for x in b if x not in child];it=iter(rest)
    return [next(it) if x is None else x for x in child]

def ortools_initial(customers,depot,seconds=ORTOOLS_SECONDS):
                                                                                        
    n=len(customers); manager=pywrapcp.RoutingIndexManager(n+1,depot.vehicles,0)
    routing=pywrapcp.RoutingModel(manager)
    coords=[Customer(0,depot.x,depot.y,0)]+customers
    def dist_cb(i,j):
        a,b=coords[manager.IndexToNode(i)],coords[manager.IndexToNode(j)]
        return int(round(euclid(a,b)*1000))
    dc=routing.RegisterTransitCallback(dist_cb);routing.SetArcCostEvaluatorOfAllVehicles(dc)
    routing.SetFixedCostOfAllVehicles(1_000_000_000)
    def dem_cb(i):
        node=manager.IndexToNode(i)
        return 0 if node==0 else int(round(customers[node-1].demand*100))
    dem=routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(dem,0,[int(depot.capacity*100)]*depot.vehicles,True,"Load")
    def time_cb(i,j):
        a,b=manager.IndexToNode(i),manager.IndexToNode(j)
        travel=euclid(coords[a],coords[b])
        service=0 if a==0 else customers[a-1].service
        return int(round(travel+service))
    tc=routing.RegisterTransitCallback(time_cb);routing.AddDimension(tc,2000,int(depot.max_duration if math.isfinite(depot.max_duration) else 100000),False,"Time")
    td=routing.GetDimensionOrDie("Time")
    td.CumulVar(routing.Start(0)).SetRange(0,int(depot.max_duration if math.isfinite(depot.max_duration) else 100000))
    for i,c in enumerate(customers,1):
        due=int(c.due) if math.isfinite(c.due) else 100000
        td.CumulVar(manager.NodeToIndex(i)).SetRange(int(c.ready),due)
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(max(1, int(seconds * 1000)))
    sol=routing.SolveWithParameters(params)
    if sol is None:return None
    routes=[]
    for vi in range(depot.vehicles):
        idx=routing.Start(vi);route=[]
        while not routing.IsEnd(idx):
            node=manager.IndexToNode(idx)
            if node:route.append(customers[node-1].id)
            idx=sol.Value(routing.NextVar(idx))
        if route:routes.append(route)
    return [x for r in routes for x in r]

def ortools_permutation(customers,depots,depot_of,seconds=ORTOOLS_SECONDS):
    permutation=[]
    for depot in depots:
        local=[c for c in customers if depot_of[c.id]==depot.id]
        if not local:
            continue
        route_order=ortools_initial(local,depot,seconds/len(depots))
        if route_order is None:
            return None
        permutation.extend(route_order)
    return permutation

def run_alns(initial,evaluate,rng):
    cur=initial[:];best=cur[:];bestv=evaluate(best)[0]
    for _ in range(12):
        cand=cur[:];q=min(5,max(1,len(cand)//20)); removed=rng.sample(cand,q)
        cand=[x for x in cand if x not in removed]
        for x in removed:
            positions=rng.sample(range(len(cand)+1),min(12,len(cand)+1))
            cand=min((cand[:p]+[x]+cand[p:] for p in positions),key=lambda z:evaluate(z)[0])
        v=evaluate(cand)[0]
        if v<bestv:best,bestv=cand[:],v
        if v<=evaluate(cur)[0] or rng.random()<.1:cur=cand
    return best

def run_ga(initial,evaluate,rng):
    pop=[initial[:]]
    for _ in range(7):
        x=initial[:];rng.shuffle(x);pop.append(x)
    for _ in range(8):
        pop.sort(key=lambda x:evaluate(x)[0]); nxt=pop[:2]
        while len(nxt)<8:
            a,b=rng.sample(pop[:5],2); child=ox(a,b,rng)
            i,j=rng.sample(range(len(child)),2);child[i],child[j]=child[j],child[i]
            nxt.append(child)
        pop=nxt
    return min(pop,key=lambda x:evaluate(x)[0])

def run_tabu(initial,evaluate,rng):
    cur=initial[:];best=cur[:];bestv=evaluate(best)[0];tabu=[]
    for _ in range(18):
        pairs=list(itertools.combinations(range(len(cur)),2));rng.shuffle(pairs);cand=[]
        for i,j in pairs[:12]:
            move=tuple(sorted((cur[i],cur[j])));z=cur[:];z[i],z[j]=z[j],z[i];v=evaluate(z)[0]
            if move not in tabu or v<bestv:cand.append((v,z,move))
        if not cand:break
        v,cur,move=min(cand,key=lambda x:x[0]);tabu=(tabu+[move])[-7:]
        if v<bestv:best,bestv=cur[:],v
    return best

TRAJECTA_HEURISTICS = ("NN", "BiDir", "MU", "FI", "RNN", "G TW", "CFRS")

def trajecta_route_order(group,depot,heuristic,rng):
    customers={c.id:c for c in group}
    depot_node=Customer(0,depot.x,depot.y,0)

    def distance_to_depot(customer_id):
        return euclid(depot_node,customers[customer_id])

    if heuristic in ("NN","RNN"):
        remaining=set(customers)
        ordered=[]
        previous=depot_node
        while remaining:
            if heuristic=="NN":
                chosen=min(remaining,key=lambda i:euclid(previous,customers[i]))
            else:
                chosen=min(remaining,key=lambda i:euclid(previous,customers[i])*rng.uniform(.85,1.15))
            ordered.append(chosen)
            remaining.remove(chosen)
            previous=customers[chosen]
        return ordered
    if heuristic=="BiDir":
        ordered=sorted(customers,key=distance_to_depot)
        return ordered[::2]+ordered[1::2][::-1]
    if heuristic=="MU":
        return sorted(customers,key=lambda i:(
            customers[i].due-distance_to_depot(i)-customers[i].service
            if math.isfinite(customers[i].due) else math.inf,
            distance_to_depot(i)))
    if heuristic=="FI":
        remaining=set(customers)
        first=max(remaining,key=distance_to_depot)
        ordered=[first]
        remaining.remove(first)
        while remaining:
            chosen=max(remaining,key=lambda i:min(euclid(customers[i],customers[j]) for j in ordered))
            def insertion_delta(position):
                left=depot_node if position==0 else customers[ordered[position-1]]
                right=depot_node if position==len(ordered) else customers[ordered[position]]
                return euclid(left,customers[chosen])+euclid(customers[chosen],right)-euclid(left,right)
            position=min(range(len(ordered)+1),key=insertion_delta)
            ordered.insert(position,chosen)
            remaining.remove(chosen)
        return ordered
    if heuristic=="G TW":
        return sorted(customers,key=lambda i:(
            max(0.0,distance_to_depot(i)+customers[i].service-customers[i].due)
            if math.isfinite(customers[i].due) else 0.0,
            distance_to_depot(i)))
    return sorted(customers,key=lambda i:(customers[i].x,customers[i].y))

def trajecta_savings_order(customers,depots,depot_of,seed):
    by_id={customer.id:customer for customer in customers}
    rng=random.Random(seed*1009+len(customers))
    permutation=[]
    for depot in depots:
        depot_node=Customer(0,depot.x,depot.y,0)
        routes=[[customer.id] for customer in customers if depot_of[customer.id]==depot.id]

        def route_distance(route):
            nodes=[depot_node]+[by_id[node] for node in route]+[depot_node]
            return sum(euclid(left,right) for left,right in zip(nodes,nodes[1:]))

        def feasible(route):
            load=sum(by_id[node].demand for node in route)
            if load>depot.capacity+1e-9:
                return False
            clock=0.0
            previous=depot_node
            for node in route:
                customer=by_id[node]
                clock=max(customer.ready,clock+euclid(previous,customer))
                if clock>customer.due+1e-9:
                    return False
                clock+=customer.service
                previous=customer
            return clock+euclid(previous,depot_node)<=depot.max_duration+1e-9

        distances=[route_distance(route) for route in routes]
        while len(routes)>1:
            candidates=[]
            for left_index,left in enumerate(routes):
                for right_index in range(left_index+1,len(routes)):
                    right=routes[right_index]
                    for left_route in (left,left[::-1]):
                        for right_route in (right,right[::-1]):
                            saving=distances[left_index]+distances[right_index]-route_distance(left_route+right_route)
                            candidates.append((saving,left_index,right_index,left_route,right_route))
            candidates.sort(key=lambda item:item[0],reverse=True)
            top=candidates[:min(8,len(candidates))]
            rng.shuffle(top)
            selected=None
            for candidate in top+candidates[len(top):]:
                saving,left_index,right_index,left_route,right_route=candidate
                if len(routes)<=depot.vehicles and saving<=1e-9:
                    break
                merged=left_route+right_route
                if feasible(merged):
                    selected=(left_index,right_index,merged,route_distance(merged))
                    break
            if selected is None:
                break
            left_index,right_index,merged,merged_distance=selected
            routes[left_index]=merged
            distances[left_index]=merged_distance
            routes.pop(right_index)
            distances.pop(right_index)
        permutation.extend(node for route in routes for node in route)
    return permutation

def trajecta_insertion_orders(customers,depots,depot_of,seed):
    by_id={customer.id:customer for customer in customers}
    priorities=(
        ("earliest_due",lambda customer,depot:customer.due),
        ("earliest_ready",lambda customer,depot:customer.ready),
        ("window_slack",lambda customer,depot:customer.due-customer.ready if math.isfinite(customer.due) else math.inf),
        ("depot_distance",lambda customer,depot:euclid(customer,Customer(0,depot.x,depot.y,0))),
        ("largest_demand",lambda customer,depot:-customer.demand),
        ("spatial_order",lambda customer,depot:(customer.x,customer.y)),
        ("due_then_demand",lambda customer,depot:(customer.due,-customer.demand)),
        ("ready_then_distance",lambda customer,depot:(customer.ready,euclid(customer,Customer(0,depot.x,depot.y,0)))),
    )
    permutations=[]
    for priority_name,priority in priorities:
        routes_by_depot={}
        for depot in depots:
            depot_node=Customer(0,depot.x,depot.y,0)
            local=sorted((c for c in customers if depot_of[c.id]==depot.id),key=lambda c:priority(c,depot))
            routes=[]

            def route_metrics(route):
                load=0.0;clock=0.0;distance=0.0;previous=depot_node
                for node in route:
                    customer=by_id[node]
                    load+=customer.demand
                    if load>depot.capacity+1e-9:
                        return False,math.inf
                    leg=euclid(previous,customer)
                    clock=max(customer.ready,clock+leg)
                    if clock>customer.due+1e-9:
                        return False,math.inf
                    clock+=customer.service
                    distance+=leg
                    previous=customer
                return clock+euclid(previous,depot_node)<=depot.max_duration+1e-9,distance+euclid(previous,depot_node)

            for customer in local:
                options=[]
                for route_index,route in enumerate(routes):
                    _,old_distance=route_metrics(route)
                    for position in range(len(route)+1):
                        trial=route[:position]+[customer.id]+route[position:]
                        feasible,new_distance=route_metrics(trial)
                        if feasible:
                            options.append((new_distance-old_distance,route_index,position))
                if options:
                    _,route_index,position=min(options)
                    routes[route_index].insert(position,customer.id)
                else:
                    routes.append([customer.id])
            routes_by_depot[depot.id]=routes
        permutations.append((priority_name,[node for depot in depots for route in routes_by_depot[depot.id] for node in route]))
    return permutations

def run_trajecta(customers,depots,depot_of,evaluate,seed):
    split_rules=(
        ("due_time",lambda c,d:c.due),
        ("ready_time",lambda c,d:c.ready),
        ("x_coordinate",lambda c,d:c.x),
        ("depot_distance",lambda c,d:euclid(c,Customer(0,d.x,d.y,0))),
        ("y_coordinate",lambda c,d:c.y),
        ("xy_coordinate",lambda c,d:c.x+c.y),
        ("window_slack",lambda c,d:c.due-c.ready if math.isfinite(c.due) else math.inf),
        ("coordinate_order",lambda c,d:(c.x,c.y)),
    )
    started=time.perf_counter()
    best=None
    candidate_count=0
    best_split=""
    best_heuristic=""
    best_permutation=None
    max_group_count=max(depot.vehicles for depot in depots)
    group_factors=sorted({min(max_group_count,value) for value in (1,4,8,10,16,max_group_count)})
    for split_name,split_key in split_rules:
        for group_factor in group_factors:
            grouped_by_depot={}
            for depot in depots:
                local=sorted((c for c in customers if depot_of[c.id]==depot.id),key=lambda c:split_key(c,depot))
                group_count=min(max(1,depot.vehicles),group_factor)
                grouped_by_depot[depot.id]=[local[index::group_count] for index in range(group_count)]
            for heuristic in TRAJECTA_HEURISTICS:
                permutation=[]
                rng=random.Random(seed*1009+len(customers)+candidate_count)
                for depot in depots:
                    for group in grouped_by_depot[depot.id]:
                        permutation.extend(trajecta_route_order(group,depot,heuristic,rng))
                score,vehicles,distance,feasible=evaluate(permutation)
                candidate_count+=1
                if best is None or score<best["score"]:
                    best={"score":score,"vehicles":vehicles,"distance":distance,"feasible":feasible}
                    best_split=f"{split_name}_groups_{group_factor}"
                    best_heuristic=heuristic
                    best_permutation=permutation

    insertion_orders=trajecta_insertion_orders(customers,depots,depot_of,seed)
    for priority_name,permutation in insertion_orders:
        score,vehicles,distance,feasible=evaluate(permutation)
        candidate_count+=1
        if score<best["score"]:
            best={"score":score,"vehicles":vehicles,"distance":distance,"feasible":feasible}
            best_split="best_feasible_insertion"
            best_heuristic=priority_name
            best_permutation=permutation

    rng=random.Random(seed*1009+len(customers)+candidate_count)
    current=best_permutation[:]
    current_score=best["score"]
    deadline=started+TRAJECTA_SECONDS
    no_time_windows=all(not math.isfinite(customer.due) for customer in customers)
    while time.perf_counter()<deadline and len(current)>2:
        if no_time_windows:
            improving_move=None
            for left in range(len(current)-1):
                for right in range(left+1,len(current)):
                    if time.perf_counter()>=deadline:
                        break
                    trial=current[:left]+list(reversed(current[left:right+1]))+current[right+1:]
                    value=evaluate(trial)[0]
                    if value<current_score and (improving_move is None or value<improving_move[0]):
                        improving_move=(value,trial)
                if time.perf_counter()>=deadline:
                    break
            for source in range(len(current)):
                if time.perf_counter()>=deadline:
                    break
                for destination in range(len(current)):
                    if time.perf_counter()>=deadline:
                        break
                    if source==destination:
                        continue
                    trial=current[:]
                    node=trial.pop(source)
                    trial.insert(destination,node)
                    value=evaluate(trial)[0]
                    if value<current_score and (improving_move is None or value<improving_move[0]):
                        improving_move=(value,trial)
            if improving_move is None:
                break
            current_score,current=improving_move
            candidate_score,vehicles,distance,feasible=evaluate(current)
            if candidate_score<best["score"]:
                best={"score":candidate_score,"vehicles":vehicles,"distance":distance,"feasible":feasible}
                best_permutation=current[:]
                best_split="2opt_local_search"
                best_heuristic="best_improvement"
            continue
        candidate=current[:]
        if rng.random()<0.5:
            left,right=sorted(rng.sample(range(len(candidate)),2))
            candidate[left:right+1]=reversed(candidate[left:right+1])
        else:
            remove_count=min(len(candidate)-1,max(1,len(candidate)//12))
            removed=rng.sample(candidate,remove_count)
            removed_set=set(removed)
            candidate=[node for node in candidate if node not in removed_set]
            for node in removed:
                positions=list(range(len(candidate)+1))
                if len(positions)>24:
                    positions=sorted(rng.sample(positions,24))
                options=[]
                for position in positions:
                    trial=candidate[:position]+[node]+candidate[position:]
                    options.append((evaluate(trial)[0],trial))
                _,candidate=min(options,key=lambda option:option[0])
        candidate_score,vehicles,distance,feasible=evaluate(candidate)
        if candidate_score<best["score"]:
            best={"score":candidate_score,"vehicles":vehicles,"distance":distance,"feasible":feasible}
            best_permutation=candidate[:]
            best_split="destroy_reinsert"
            best_heuristic="adaptive_insertion"
        if candidate_score<=current_score or rng.random()<0.08:
            current,current_score=candidate,candidate_score
    return {**best,"runtime_s":time.perf_counter()-started,
            "split_strategy":best_split,"heuristic":best_heuristic,
            "candidate_count":candidate_count}

def main():
    OUT.mkdir(exist_ok=True)
    rows=[]
    for name,customers,depots,family in load_instances():
        evaluate,depot_of,construct=build_runner(customers,depots)
        initial=construct()
        ref_k,ref_d=PUBLISHED_BKS.get(name,(None,None))
        for seed in SEEDS:
            for alg,fn in (("ALNS",run_alns),("Genetic Algorithm",run_ga),("Tabu Search",run_tabu)):
                rng=random.Random(seed*1009+len(name));t0=time.perf_counter()
                perm=fn(initial,evaluate,rng);score,vehicles,distance,feasible=evaluate(perm)
                rows.append({"family":family,"instance":name,"seed":seed,"algorithm":alg,
                    "feasible":feasible,"vehicles":vehicles,"distance":distance,
                    "objective_vehicle_first":score,"runtime_s":time.perf_counter()-t0,
                    "init_source":"greedy_construct","reference_vehicles":ref_k or "",
                    "reference_distance":ref_d or "",
                    "split_strategy":"","heuristic":"","candidate_count":0,
                    "distance_gap_pct":100*(distance-ref_d)/ref_d if ref_d and feasible and (family=="Cordeau-MDVRP" or vehicles==ref_k) else ""})
                print(f"{name} seed={seed} {alg}: feasible={feasible}, vehicles={vehicles}, distance={distance:.2f}")
            trajecta=run_trajecta(customers,depots,depot_of,evaluate,seed)
            rows.append({"family":family,"instance":name,"seed":seed,"algorithm":"Trajecta",
                "feasible":trajecta["feasible"],"vehicles":trajecta["vehicles"],
                "distance":trajecta["distance"],"objective_vehicle_first":trajecta["score"],
                "runtime_s":trajecta["runtime_s"],"init_source":"split_heuristic_group_sweep_plus_local_search",
                "reference_vehicles":ref_k or "","reference_distance":ref_d or "",
                "split_strategy":trajecta["split_strategy"],"heuristic":trajecta["heuristic"],
                "candidate_count":trajecta["candidate_count"],
                "distance_gap_pct":100*(trajecta["distance"]-ref_d)/ref_d if ref_d and trajecta["feasible"] and (family=="Cordeau-MDVRP" or trajecta["vehicles"]==ref_k) else ""})
            print(f"{name} seed={seed} Trajecta: feasible={trajecta['feasible']}, vehicles={trajecta['vehicles']}, distance={trajecta['distance']:.2f}")
            started=time.perf_counter()
            perm=ortools_permutation(customers,depots,depot_of)
            if perm is None:
                score,vehicles,distance,feasible=1e12,None,None,False
            else:
                score,vehicles,distance,feasible=evaluate(perm)
            rows.append({"family":family,"instance":name,"seed":seed,"algorithm":"OR-Tools",
                "feasible":feasible,"vehicles":vehicles if vehicles is not None else "",
                "distance":distance if distance is not None else "",
                "objective_vehicle_first":score,"runtime_s":time.perf_counter()-started,
                "init_source":"parallel_cheapest_insertion","reference_vehicles":ref_k or "",
                "reference_distance":ref_d or "",
                "split_strategy":"","heuristic":"","candidate_count":0,
                "distance_gap_pct":100*(distance-ref_d)/ref_d if ref_d and feasible and (family=="Cordeau-MDVRP" or vehicles==ref_k) else ""})
            print(f"{name} seed={seed} OR-Tools: feasible={feasible}, vehicles={vehicles}, distance={distance}")
    with (OUT/"benchmark_algorithm_runs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/"benchmark_algorithm_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["family","instance","algorithm","n_runs","feasible_runs","mean_vehicles","mean_distance","sd_distance","mean_runtime_s","sd_runtime_s","mean_gap_pct","sd_gap_pct"]);w.writeheader()
        for family in sorted({r["family"] for r in rows}):
            for inst in sorted({r["instance"] for r in rows if r["family"]==family}):
                for alg in ("Trajecta","ALNS","Genetic Algorithm","Tabu Search","OR-Tools"):
                    rs=[r for r in rows if r["family"]==family and r["instance"]==inst and r["algorithm"]==alg]
                    good=[r for r in rs if r["feasible"]]
                    gaps=[float(r["distance_gap_pct"]) for r in good if r["distance_gap_pct"]!=""]
                    runtimes=[float(r["runtime_s"]) for r in rs]
                    w.writerow({"family":family,"instance":inst,"algorithm":alg,"n_runs":len(rs),
                        "feasible_runs":len(good),"mean_vehicles":statistics.mean(r["vehicles"] for r in good) if good else "",
                        "mean_distance":statistics.mean(r["distance"] for r in good) if good else "",
                        "sd_distance":statistics.stdev(r["distance"] for r in good) if len(good)>1 else "",
                        "mean_runtime_s":statistics.mean(runtimes) if runtimes else "",
                        "sd_runtime_s":statistics.stdev(runtimes) if len(runtimes)>1 else "",
                        "mean_gap_pct":statistics.mean(gaps) if gaps else "",
                        "sd_gap_pct":statistics.stdev(gaps) if len(gaps)>1 else ""})
    print("Wrote benchmark_algorithm_runs.csv and benchmark_algorithm_summary.csv")

if __name__=="__main__":main()
