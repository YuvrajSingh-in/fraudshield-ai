"""
Graph-based fraud ring detection.

BUG FIXED: The original code added country as a graph node and connected every
user to it. This caused all users from the same country (e.g., all 302 Nigerian
users) to share one connected component, which immediately exceeded the cluster
threshold. Every Nigerian user was falsely flagged as a fraud ring member.

FIX: Country is no longer a graph node. The graph only tracks relationships
between users, merchants, and vendors. A fraud ring is detected when a tightly
connected cluster of users share merchants AND vendors simultaneously.
"""

import networkx as nx
from threading import Lock

from config import CONFIG

_graph = nx.Graph()
_lock  = Lock()

CLUSTER_THRESHOLD: int = CONFIG["network"]["cluster_threshold"]


def _update_graph(transaction: dict) -> None:
    user     = str(transaction.get("user_id",     "?"))
    merchant = f"M:{transaction.get('merchant_id', '?')}"
    vendor   = f"V:{transaction.get('vendor_id',   '?')}"

    _graph.add_node(user,     node_type="user")
    _graph.add_node(merchant, node_type="merchant")
    _graph.add_node(vendor,   node_type="vendor")

    _graph.add_edge(user, merchant)
    _graph.add_edge(user, vendor)


def detect_fraud_ring(transaction: dict) -> float:
    """
    Adds the transaction to the in-memory graph and checks whether
    the user belongs to a suspicious cluster (user-to-user connections
    via shared merchants or vendors exceed the threshold).

    Returns 0.40 if a fraud ring is detected, else 0.0.
    """
    user = str(transaction.get("user_id", "?"))

    with _lock:
        _update_graph(transaction)

        if user not in _graph:
            return 0.0

        component = nx.node_connected_component(_graph, user)

        # Count only user-type nodes in the component (not merchants/vendors)
        user_nodes = [
            n for n in component
            if _graph.nodes[n].get("node_type") == "user"
        ]

    return 0.40 if len(user_nodes) > CLUSTER_THRESHOLD else 0.0
