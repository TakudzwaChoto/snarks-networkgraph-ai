import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

try:
    import networkx as nx
    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False

CACHE_FILE = "models/topology_cache.parquet"


def load_topology(topology_xlsx: str) -> pd.DataFrame:
    if not os.path.exists(topology_xlsx):
        raise FileNotFoundError(f"Topology file not found: {topology_xlsx}")
    df = pd.read_excel(topology_xlsx)
    # Expected columns from README: Subbasin, FROM_NODE, TO_NODE, FLOW_OUTcms, AreaC, Len2, Slo2, Wid2, Dep2
    return df


def build_graph(df: pd.DataFrame) -> "nx.DiGraph":
    if not NX_AVAILABLE:
        raise RuntimeError("networkx not available")
    G = nx.DiGraph()
    # Use node id as Subbasin
    for _, r in df.iterrows():
        sid = r.get("Subbasin")
        if pd.isna(sid):
            continue
        props = {
            "flow_out": float(r.get("FLOW_OUTcms", 0.0)),
            "area": float(r.get("AreaC", 0.0)),
            "length": float(r.get("Len2", 0.0)),
            "slope": float(r.get("Slo2", 0.0)),
            "width": float(r.get("Wid2", 0.0)),
            "depth": float(r.get("Dep2", 0.0)),
        }
        G.add_node(int(sid), **props)
        fnode = r.get("FROM_NODE")
        tnode = r.get("TO_NODE")
        if not pd.isna(fnode) and not pd.isna(tnode):
            G.add_edge(int(fnode), int(tnode))
    return G


def compute_pair_features(G: "nx.DiGraph", buyer: int, seller: int) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    # distance along directed path if exists; else undirected shortest path; else large value
    dist = np.inf
    if G.has_node(buyer) and G.has_node(seller):
        try:
            dist = float(nx.shortest_path_length(G, buyer, seller))
            feats["upstream_downstream"] = 1  # buyer upstream of seller
        except Exception:
            try:
                dist = float(nx.shortest_path_length(G.to_undirected(), buyer, seller))
                feats["upstream_downstream"] = 0
            except Exception:
                dist = 1e6
                feats["upstream_downstream"] = -1
        # Node attributes
        battrs = G.nodes[buyer]
        sattrs = G.nodes[seller]
        for k in ["flow_out","area","length","slope","width","depth"]:
            feats[f"buyer_{k}"] = float(battrs.get(k, 0.0))
            feats[f"seller_{k}"] = float(sattrs.get(k, 0.0))
        feats["distance"] = dist
    else:
        for k in ["flow_out","area","length","slope","width","depth"]:
            feats[f"buyer_{k}"] = 0.0
            feats[f"seller_{k}"] = 0.0
        feats["distance"] = 1e6
        feats["upstream_downstream"] = -1
    return feats


def get_or_build_cache(topology_xlsx: str) -> Tuple[pd.DataFrame, "nx.DiGraph"]:
    df = load_topology(topology_xlsx)
    G = build_graph(df)
    return df, G