"""
Ontology-based keyword set generation utilities.

This module provides functions to generate keyword sets from ontologies/hierarchies
by expanding from seed nodes using N-hop neighbors. This allows you to leverage
hierarchical structure while maintaining flexibility for relationships not in the ontology.

The key idea:
- Use ontology to guide keyword selection (seed nodes + N-hop neighbors)
- Use Qwen/LLM embeddings (not hierarchical embeddings) to maintain flexibility
- Train with many expanded keyword sets - hierarchical structure emerges naturally
- Relationships not in ontology can still be learned from data

Example:
    >>> from phenoref.ontology import expand_keyword_sets_from_ontology
    >>> 
    >>> # Define ontology (or load from SNOMED/UMLS)
    >>> ontology = {
    ...     "dementia": ["alzheimer's disease", "vascular dementia"],
    ...     "alzheimer's disease": ["early onset alzheimer's"],
    ... }
    >>> 
    >>> # Start with seeds
    >>> seeds = {"dementia": ["dementia"]}
    >>> 
    >>> # Expand to 2-hop neighbors
    >>> expanded = expand_keyword_sets_from_ontology(
    ...     seed_keywords=seeds,
    ...     ontology_dict=ontology,
    ...     n_hops=2
    ... )
    >>> # Result: {"dementia": ["dementia", "alzheimer's disease", "vascular dementia", 
    >>> #                       "early onset alzheimer's", ...]}
"""

from typing import Dict, List, Set, Optional, Tuple, Union, Any, Callable
import random
import os
import functools
# Heavy/optional plotting deps -- import lazily so a bare ``import phenoref``
# (or any non-plotting code path) does not require them.
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    DATASHADER_AVAILABLE = True
except ImportError:
    DATASHADER_AVAILABLE = False
    ds = None
    tf = None
import pandas as pd
from .utils import count_tokens
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
try:
    from owlready2 import get_ontology, Ontology
    OWLREADY2_AVAILABLE = True
except ImportError:
    OWLREADY2_AVAILABLE = False
    get_ontology = None
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    Ontology = None
try:
    from scipy import sparse
    from scipy.sparse import csgraph
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    sparse = None
    csgraph = None
from collections import deque, Counter


def expand_keyword_sets_from_ontology(
    seed_keywords: Dict[str, List[str]],
    ontology_graph: Optional[nx.DiGraph] = None,
    ontology_dict: Optional[Dict[str, List[str]]] = None,
    n_hops: int = 2,
    max_keywords_per_set: Optional[int] = None,
    include_seeds: bool = True,
    bidirectional: bool = False,
) -> Dict[str, List[str]]:
    """
    Expand keyword sets from seed nodes using N-hop neighbors in an ontology graph.
    
    This function takes seed keywords and expands them by including all terms
    that are N steps away in the ontology. This creates natural hierarchical
    structure while allowing the model to learn relationships beyond the ontology.
    
    Parameters
    ----------
    seed_keywords : Dict[str, List[str]]
        Dictionary mapping keyword set names to lists of seed keywords.
        Example: {"dementia": ["dementia", "cognitive decline"], ...}
    ontology_graph : nx.DiGraph, optional
        NetworkX directed graph representing the ontology.
        Nodes are terms, edges represent relationships (e.g., "is_a", "part_of").
        If None, will try to construct from ontology_dict.
    ontology_dict : Dict[str, List[str]], optional
        Dictionary mapping terms to their related terms (children, parts, etc.).
        Example: {"dementia": ["alzheimer's disease", "vascular dementia"], ...}
        If ontology_graph is None, this will be used to construct a graph.
    n_hops : int, default 2
        Number of hops to expand from seed nodes. 
        - n_hops=1: Include direct neighbors only
        - n_hops=2: Include neighbors of neighbors
        - Higher values include more distant terms
    max_keywords_per_set : int, optional
        Maximum number of keywords per set. If None, no limit.
        If set, will prioritize closer neighbors (fewer hops).
    include_seeds : bool, default True
        Whether to include the original seed keywords in the expanded sets.
    bidirectional : bool, default False
        If True, traverse edges in both directions (parent->child and child->parent).
        If False, only traverse in the direction of edges (typically parent->child).
    
    Returns
    -------
    Dict[str, List[str]]
        Expanded keyword sets with the same structure as seed_keywords.
        Each set contains seed keywords plus all terms within n_hops.
    
    Examples
    --------
    >>> # Using ontology dictionary
    >>> ontology = {
    ...     "dementia": ["alzheimer's disease", "vascular dementia"],
    ...     "alzheimer's disease": ["early onset alzheimer's", "late onset alzheimer's"],
    ...     "cognitive decline": ["mild cognitive impairment", "dementia"]
    ... }
    >>> seeds = {"dementia": ["dementia", "cognitive decline"]}
    >>> expanded = expand_keyword_sets_from_ontology(
    ...     seed_keywords=seeds,
    ...     ontology_dict=ontology,
    ...     n_hops=2
    ... )
    >>> # Result includes: dementia, cognitive decline, alzheimer's disease, 
    >>> # vascular dementia, early onset alzheimer's, late onset alzheimer's, etc.
    
    >>> # Using NetworkX graph
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edge("dementia", "alzheimer's disease")
    >>> G.add_edge("alzheimer's disease", "early onset alzheimer's")
    >>> expanded = expand_keyword_sets_from_ontology(
    ...     seed_keywords={"dementia": ["dementia"]},
    ...     ontology_graph=G,
    ...     n_hops=2
    ... )
    """
    # Check networkx availability
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "networkx is required for ontology expansion. "
            "Please install it: pip install networkx"
        )
    
    # Construct graph if needed
    if ontology_graph is None:
        if ontology_dict is None:
            raise ValueError("Either ontology_graph or ontology_dict must be provided")
        
        # Build graph from dictionary
        ontology_graph = nx.DiGraph()
        for term, related_terms in ontology_dict.items():
            for related in related_terms:
                ontology_graph.add_edge(term, related)
                if bidirectional:
                    ontology_graph.add_edge(related, term)
    
    expanded_sets = {}
    
    for set_name, seed_list in seed_keywords.items():
        # Collect all terms within n_hops from any seed
        expanded_terms: Set[str] = set()
        term_distances: Dict[str, int] = {}  # Track distance from nearest seed
        
        # BFS from each seed
        for seed in seed_list:
            if seed not in ontology_graph:
                # Seed not in ontology, just add it
                if include_seeds:
                    expanded_terms.add(seed)
                    term_distances[seed] = 0
                continue
            
            # BFS to find all nodes within n_hops
            queue = deque([(seed, 0)])  # (node, distance)
            visited = {seed}
            
            if include_seeds:
                expanded_terms.add(seed)
                term_distances[seed] = 0
            
            while queue:
                current, distance = queue.popleft()
                
                if distance >= n_hops:
                    continue
                
                # Get neighbors
                neighbors = list(ontology_graph.successors(current))
                if bidirectional:
                    neighbors.extend(ontology_graph.predecessors(current))
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_distance = distance + 1
                        
                        # Only add if within n_hops
                        if new_distance <= n_hops:
                            expanded_terms.add(neighbor)
                            # Track minimum distance from any seed
                            if neighbor not in term_distances or new_distance < term_distances[neighbor]:
                                term_distances[neighbor] = new_distance
                            queue.append((neighbor, new_distance))
        
        # Convert to sorted list, prioritizing closer terms
        expanded_list = sorted(
            expanded_terms,
            key=lambda x: (term_distances.get(x, n_hops + 1), x)  # Sort by distance, then alphabetically
        )
        
        # Limit size if requested
        if max_keywords_per_set is not None and len(expanded_list) > max_keywords_per_set:
            # Keep seeds first, then closest neighbors
            seed_set = set(seed_list) if include_seeds else set()
            seed_terms = [t for t in expanded_list if t in seed_set]
            non_seed_terms = [t for t in expanded_list if t not in seed_set]
            
            # Prioritize by distance
            non_seed_terms = sorted(
                non_seed_terms,
                key=lambda x: term_distances.get(x, n_hops + 1)
            )
            
            expanded_list = seed_terms + non_seed_terms[:max_keywords_per_set - len(seed_terms)]
        
        expanded_sets[set_name] = expanded_list
    
    return expanded_sets


def create_hierarchical_keyword_sets(
    seed_keywords: Dict[str, List[str]],
    ontology_graph: Optional[nx.DiGraph] = None,
    ontology_dict: Optional[Dict[str, List[str]]] = None,
    hop_levels: List[int] = [1, 2, 3],
    max_keywords_per_level: Optional[int] = None,
) -> Dict[str, Dict[int, List[str]]]:
    """
    Create hierarchical keyword sets at multiple hop levels.
    
    This allows you to train with different levels of specificity:
    - Level 1: Very specific (direct neighbors)
    - Level 2: Moderate (2-hop neighbors)
    - Level 3: Broad (3-hop neighbors)
    
    Parameters
    ----------
    seed_keywords : Dict[str, List[str]]
        Dictionary mapping keyword set names to lists of seed keywords.
    ontology_graph : nx.DiGraph, optional
        NetworkX directed graph representing the ontology.
    ontology_dict : Dict[str, List[str]], optional
        Dictionary mapping terms to their related terms.
    hop_levels : List[int], default [1, 2, 3]
        List of hop levels to generate. Each level creates a separate keyword set.
    max_keywords_per_level : int, optional
        Maximum keywords per level. If None, no limit.
    
    Returns
    -------
    Dict[str, Dict[int, List[str]]]
        Nested dictionary: {set_name: {hop_level: [keywords]}}
        Example: {"dementia": {1: [...], 2: [...], 3: [...]}}
    
    Examples
    --------
    >>> seeds = {"dementia": ["dementia"]}
    >>> hierarchical = create_hierarchical_keyword_sets(
    ...     seed_keywords=seeds,
    ...     ontology_dict=ontology,
    ...     hop_levels=[1, 2, 3]
    ... )
    >>> # Use level 1 for specific, level 3 for broad
    >>> specific_sets = {name: sets[1] for name, sets in hierarchical.items()}
    >>> broad_sets = {name: sets[3] for name, sets in hierarchical.items()}
    """
    hierarchical_sets = {}
    
    for set_name in seed_keywords.keys():
        hierarchical_sets[set_name] = {}
        
        for hop_level in hop_levels:
            expanded = expand_keyword_sets_from_ontology(
                seed_keywords={set_name: seed_keywords[set_name]},
                ontology_graph=ontology_graph,
                ontology_dict=ontology_dict,
                n_hops=hop_level,
                max_keywords_per_set=max_keywords_per_level,
                include_seeds=True,
            )
            hierarchical_sets[set_name][hop_level] = expanded[set_name]
    
    return hierarchical_sets


def flatten_hierarchical_sets(
    hierarchical_sets: Dict[str, Dict[int, List[str]]],
    level: Optional[int] = None,
    combine_levels: bool = False,
) -> Dict[str, List[str]]:
    """
    Flatten hierarchical keyword sets into a simple dictionary.
    
    Parameters
    ----------
    hierarchical_sets : Dict[str, Dict[int, List[str]]]
        Hierarchical sets from create_hierarchical_keyword_sets.
    level : int, optional
        If specified, return only this level. If None and combine_levels=False,
        returns the highest level.
    combine_levels : bool, default False
        If True, combines all levels into one set per keyword set name.
        If False, returns only the specified level (or highest if level=None).
    
    Returns
    -------
    Dict[str, List[str]]
        Flattened keyword sets.
    """
    if combine_levels:
        # Combine all levels
        flattened = {}
        for set_name, levels_dict in hierarchical_sets.items():
            combined = set()
            for keywords in levels_dict.values():
                combined.update(keywords)
            flattened[set_name] = sorted(list(combined))
        return flattened
    else:
        # Use specific level or highest
        if level is None:
            # Use highest level
            flattened = {}
            for set_name, levels_dict in hierarchical_sets.items():
                if levels_dict:
                    max_level = max(levels_dict.keys())
                    flattened[set_name] = levels_dict[max_level]
            return flattened
        else:
            # Use specified level
            flattened = {}
            for set_name, levels_dict in hierarchical_sets.items():
                if level in levels_dict:
                    flattened[set_name] = levels_dict[level]
            return flattened


def random_seed_keyword_sets(
    ontology_graph: Optional[nx.DiGraph] = None,
    ontology_dict: Optional[Dict[str, List[str]]] = None,
    n_seeds: int = 10,
    n_keywords_per_seed: int = 1,
    seed_nodes: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    min_degree: int = 1,
    max_degree: Optional[int] = None,
    min_seed_depth: int = 0,
) -> Dict[str, List[str]]:
    """
    Generate keyword sets by randomly sampling seed nodes from an ontology.
    
    This function randomly selects nodes from the ontology as seeds, then you can
    expand from them using expand_keyword_sets_from_ontology(). This creates
    diverse keyword sets covering different parts of the ontology.
    
    Parameters
    ----------
    ontology_graph : nx.DiGraph, optional
        NetworkX directed graph representing the ontology.
        If None, will construct from ontology_dict.
    ontology_dict : Dict[str, List[str]], optional
        Dictionary mapping terms to their related terms.
        If ontology_graph is None, this will be used to construct a graph.
    n_seeds : int, default 10
        Number of seed keyword sets to generate.
    n_keywords_per_seed : int, default 1
        Number of keywords to include in each seed set.
        If 1, uses the seed node itself as the only keyword.
        If > 1, randomly samples multiple nodes per set.
    seed_nodes : List[str], optional
        If provided, randomly samples from this list instead of all ontology nodes.
        Useful for filtering (e.g., only disease nodes, only high-degree nodes).
    random_seed : int, optional
        Random seed for reproducibility.
    min_degree : int, default 1
        Minimum degree (number of connections) for a node to be eligible as a seed.
        Filters out isolated nodes.
    max_degree : int, optional
        Maximum degree for a node to be eligible as a seed.
        If None, no maximum. Useful for avoiding overly connected hub nodes.
    min_seed_depth : int, default 0
        Minimum depth (levels from root/top of ontology) for a node to be eligible as a seed.
        Depth 0 = root nodes, depth 1 = direct children of roots, etc.
        Useful for avoiding top-level nodes and focusing on more specific concepts.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping seed set names to lists of seed keywords.
        Set names are "seed_0", "seed_1", etc., or use the node name if n_keywords_per_seed=1.
    
    Examples
    --------
    >>> # Random seeds from entire ontology
    >>> seeds = random_seed_keyword_sets(
    ...     ontology_dict=ontology,
    ...     n_seeds=20,
    ...     n_keywords_per_seed=1
    ... )
    >>> 
    >>> # Expand each seed
    >>> expanded = expand_keyword_sets_from_ontology(
    ...     seed_keywords=seeds,
    ...     ontology_dict=ontology,
    ...     n_hops=2
    ... )
    >>> 
    >>> # Filter to high-degree nodes only
    >>> high_degree_nodes = [n for n, d in G.degree() if d >= 5]
    >>> seeds = random_seed_keyword_sets(
    ...     ontology_graph=G,
    ...     seed_nodes=high_degree_nodes,
    ...     n_seeds=15
    ... )
    """
    # Check networkx availability
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "networkx is required for random seed generation. "
            "Please install it: pip install networkx"
        )
    
    # Set random seed
    if random_seed is not None:
        random.seed(random_seed)
    
    # Construct graph if needed
    if ontology_graph is None:
        if ontology_dict is None:
            raise ValueError("Either ontology_graph or ontology_dict must be provided")
        
        # Build graph from dictionary
        ontology_graph = nx.DiGraph()
        for term, related_terms in ontology_dict.items():
            for related in related_terms:
                ontology_graph.add_edge(term, related)
    
    # Get candidate nodes
    if seed_nodes is None:
        # Use all nodes, filtered by degree
        candidates = list(ontology_graph.nodes())
    else:
        # Use provided seed nodes
        candidates = [n for n in seed_nodes if n in ontology_graph]
    
    # Filter by degree
    # For directed graphs, we need outgoing edges for expansion to work
    # So we check out_degree (successors) in addition to total degree
    if min_degree > 0 or max_degree is not None:
        filtered_candidates = []
        for node in candidates:
            degree = ontology_graph.degree(node)
            out_degree = ontology_graph.out_degree(node)  # Outgoing edges (needed for expansion)
            
            # Node must have at least min_degree total connections
            # AND at least 1 outgoing edge (so expansion can find neighbors)
            if degree >= min_degree and out_degree >= 1:
                if max_degree is None or degree <= max_degree:
                    filtered_candidates.append(node)
        candidates = filtered_candidates
    
    # Filter by depth (distance from root nodes)
    if min_seed_depth > 0:
        # Find root nodes (nodes with no incoming edges)
        root_nodes = [node for node in ontology_graph.nodes() if ontology_graph.in_degree(node) == 0]
        
        # If no root nodes found (e.g., cycles), use nodes with minimum in-degree
        if not root_nodes:
            min_in_degree = min(ontology_graph.in_degree(node) for node in ontology_graph.nodes())
            root_nodes = [node for node in ontology_graph.nodes() if ontology_graph.in_degree(node) == min_in_degree]
        
        # Compute depth for all nodes efficiently using single_source_shortest_path_length
        # This is much faster than checking each node individually
        node_depths = {}
        
        # For each root, compute shortest paths to all reachable nodes
        # Then take minimum depth across all roots
        for root in root_nodes:
            try:
                # Get all nodes reachable from this root with their distances
                path_lengths = nx.single_source_shortest_path_length(ontology_graph, root)
                for node, distance in path_lengths.items():
                    # Track minimum distance from any root
                    if node not in node_depths or distance < node_depths[node]:
                        node_depths[node] = distance
            except (nx.NetworkXError, KeyError):
                # Skip if root not in graph or other error
                continue
        
        # For nodes not reached by any root, assign depth 0
        # Filter by minimum depth
        candidates = [
            node for node in candidates 
            if node_depths.get(node, 0) >= min_seed_depth
        ]
    
    if len(candidates) < n_seeds:
        raise ValueError(
            f"Not enough candidate nodes: {len(candidates)} available, "
            f"but {n_seeds} seeds requested. Try reducing n_seeds or "
            f"relaxing min_degree/max_degree filters."
        )
    
    # Randomly sample seeds
    seed_keywords = {}
    
    if n_keywords_per_seed == 1:
        # One keyword per set - use node name as set name
        selected = random.sample(candidates, n_seeds)
        for i, node in enumerate(selected):
            # Use node name as set name, or "seed_N" if name is too long
            set_name = node if len(node) < 50 else f"seed_{i}"
            seed_keywords[set_name] = [node]
    else:
        # Multiple keywords per set
        for i in range(n_seeds):
            selected = random.sample(candidates, min(n_keywords_per_seed, len(candidates)))
            seed_keywords[f"seed_{i}"] = selected
    
    return seed_keywords


def extract_relationships_for_keyword_set(
    keyword_set: List[str],
    ontology_graph: Optional[nx.DiGraph],
) -> List[Tuple[str, str, str]]:
    """
    Extract relationships between keywords in a set from the ontology graph.
    
    Only includes relationships where both source and target are in the keyword set.
    
    Parameters
    ----------
    keyword_set : List[str]
        List of keywords in the set.
    ontology_graph : nx.DiGraph, optional
        NetworkX directed graph representing the ontology.
        Edges should have a 'relationship' attribute.
    
    Returns
    -------
    List[Tuple[str, str, str]]
        List of tuples (source, target, relationship_type) for relationships
        between keywords in the set.
    """
    if ontology_graph is None:
        return []
    
    if not NETWORKX_AVAILABLE:
        return []
    
    keyword_set_set = set(keyword_set)
    relationships = []
    
    # Much faster: only check edges from nodes in the keyword set
    # Instead of iterating all edges, iterate only nodes in set and their out-edges
    for source in keyword_set_set:
        if source in ontology_graph:
            # Get all outgoing edges from this source
            for target in ontology_graph.successors(source):
                # Only include if target is also in the keyword set
                if target in keyword_set_set:
                    # Get relationship type from edge data
                    edge_data = ontology_graph.get_edge_data(source, target, {})
                    relationship_type = edge_data.get('relationship', 'unknown')
                    relationships.append((source, target, relationship_type))
    
    return relationships


def format_relationship_as_text(
    source: str,
    target: str,
    relationship_type: str,
    formatter: Optional[Callable[[str, str, str], str]] = None,
) -> str:
    """
    Format a relationship as natural language text.
    
    Parameters
    ----------
    source : str
        Source term in the relationship.
    target : str
        Target term in the relationship.
    relationship_type : str
        Type of relationship (e.g., "is_a", "causes", "part_of").
    formatter : Callable[[str, str, str], str], optional
        Custom formatter function that takes (source, target, relationship_type)
        and returns a formatted string. If None, uses default natural language mapping.
    
    Returns
    -------
    str
        Formatted relationship text, e.g., "pheno 2 is a subtype of pheno 1".
    """
    if formatter is not None:
        return formatter(source, target, relationship_type)
    
    # Default natural language mapping
    relationship_mapping = {
        "is_a": "is a subtype of",
        "reverse_is_a": "is a subtype of",  # Same semantics, just reverse edge direction
        "causes": "causes",
        "part_of": "is part of",
        "has_part": "has part",
        "disease_has_location": "has location",
        "has_location": "has location",
    }
    
    # Get the natural language description
    if relationship_type in relationship_mapping:
        rel_text = relationship_mapping[relationship_type]
    else:
        # Default: replace underscores with spaces and use as-is
        rel_text = relationship_type.replace("_", " ")
        # Handle reverse_ prefix
        if rel_text.startswith("reverse "):
            rel_text = rel_text.replace("reverse ", "is reverse of ")
    
    # Handle relationship direction correctly
    # For "is_a": edge (parent, child) means "child is_a parent" = "child is a subtype of parent"
    # So we need to swap source and target for "is_a" relationships
    if relationship_type == "is_a":
        # source is parent (broader), target is child (more specific)
        # We want: "target is a subtype of source"
        return f"{target} {rel_text} {source}"
    elif relationship_type == "reverse_is_a":
        # reverse_is_a: edge (child, parent) with reverse direction
        # Semantically still means "child is_a parent" = "child is a subtype of parent"
        # source is child (more specific), target is parent (broader)
        # We want: "source is a subtype of target"
        return f"{source} {rel_text} {target}"
    else:
        # For other relationships, edge direction is correct as stored
        # e.g., (cls, value) with "causes" means "cls causes value"
        return f"{source} {rel_text} {target}"


def generate_keyword_sets_from_ontology(
    ontology_owl: Optional[Any] = None,
    ontology_graph: Optional[nx.DiGraph] = None,
    ontology_dict: Optional[Dict[str, List[str]]] = None,
    n_seeds: int = 20,
    n_hops: Union[int, List[int]] = 2,
    n_keywords_per_seed: int = 1,
    max_keywords_per_set: Optional[int] = None,
    min_degree: int = 1,
    max_degree: Optional[int] = None,
    random_seed: Optional[int] = None,
    include_seeds: bool = True,
    bidirectional: bool = False,
    flatten_indices: bool = False,
    min_seed_depth: int = 0,
    resample_times: Optional[int] = None,
    resample_frac: Optional[float] = None,
    resample_min: int = 2,
    verbose: int = 1,
    show_progress: bool = True,
    include_synonyms: bool = False,
    include_relationships: bool = True,
    relationship_formatter: Optional[Callable[[str, str, str], str]] = None,
) -> Union[
    Tuple[Dict[str, List[str]], List[str], List[str], Dict[str, List[str]]],
    Tuple[Dict[str, List[str]], np.ndarray, List[str], List[str], Dict[str, List[str]]]
]:
    """
    One-step function: randomly sample seeds from ontology and expand them.
    
    This is a convenience function that combines random_seed_keyword_sets()
    and expand_keyword_sets_from_ontology() into one call.
    
    Parameters
    ----------
    ontology_graph : nx.DiGraph, optional
        NetworkX directed graph representing the ontology.
    ontology_dict : Dict[str, List[str]], optional
        Dictionary mapping terms to their related terms.
    n_seeds : int, default 20
        Number of seed keyword sets to generate.
    n_hops : int or List[int], default 2
        Number of hops to expand from each seed.
        If an int, expands all seeds with that number of hops.
        If a list, creates separate keyword sets for each hop value.
        Set names will be "{seed_name}_{n_hops}" when n_hops is a list.
    n_keywords_per_seed : int, default 1
        Number of keywords per seed set.
        If 1, uses the seed node itself as the only keyword.
        If > 1, selects this many seed nodes per set, then expands from them.
        The expansion will then find all keywords meeting n_hops and other criteria.
    max_keywords_per_set : int, optional
        Maximum keywords per expanded set.
    min_degree : int, default 1
        Minimum degree for seed nodes.
    max_degree : int, optional
        Maximum degree for seed nodes.
    random_seed : int, optional
        Random seed for reproducibility.
    include_seeds : bool, default True
        Whether to include seed keywords in expanded sets.
    bidirectional : bool, default False
        Whether to traverse ontology edges bidirectionally.
    flatten_indices : bool, default False
        If True, also return indices mapping flattened keywords to their sets.
        When True, returns (keyword_sets, set_indices) where set_indices is a numpy array
        of shape (n_keywords,) where each element indicates which set (by index) that keyword
        belongs to in the flattened keyword list.
        Useful when you want to embed all keywords at once and then aggregate by set.
    min_seed_depth : int, default 0
        Minimum depth (levels from root/top of ontology) for seed nodes.
        Depth 0 = root nodes, depth 1 = direct children of roots, etc.
        Useful for avoiding top-level nodes and focusing on more specific concepts.
    resample_times : int, optional
        If provided and > 0, resample each keyword set this many times.
        Each resample randomly selects a fraction of keywords from the original set.
        New sets will be named "{original_name}_resample_{i}" where i is 1 to resample_times.
    resample_frac : float, optional
        Fraction of keywords to sample for each resample.
        Must be between 0 and 1. If None and resample_times is set, a random fraction
        will be chosen for each resample (between 0.1 and 0.9).
        If resample_times is None, this parameter is ignored.
        If provided, must be > 0 and <= 1.
    resample_min : int, default 2
        Minimum set size required to attempt resampling.
        Sets with fewer keywords than this will be skipped during resampling.
        This helps avoid warnings when sets are too small to generate unique resamples.
    verbose : int, default 1
        Verbosity level:
        - 0: No output
        - 1: Basic output (default)
        - 2: Include warnings about resampling (e.g., when not enough unique resamples can be generated)
    show_progress : bool, default True
        Whether to show progress bars during processing.
    include_synonyms : bool, default False
        If True, includes synonyms for each term in the keyword sets.
        Requires ontology_owl to be provided to extract synonyms.
    ontology_owl : owlready2.Ontology, optional
        OWL ontology object (from owlready2). Required if include_synonyms=True.
        Can be obtained from load_mondo_ontology(return_owl=True).
    include_relationships : bool, default True
        If True, extracts relationships between keywords in each set from the ontology graph
        and formats them as text documents. Only relationships where both source and target
        are in the same keyword set are included. Requires ontology_graph to be provided.
    relationship_formatter : Callable[[str, str, str], str], optional
        Custom function to format relationships as text. Takes (source, target, relationship_type)
        and returns a formatted string. If None, uses default natural language mapping
        (e.g., "is_a" -> "is a subtype of", "causes" -> "causes").
    
    Returns
    -------
    Tuple[Dict[str, List[str]], List[str], List[str], Dict[str, List[str]]] or Tuple[Dict[str, List[str]], np.ndarray, List[str], List[str], Dict[str, List[str]]]
        - If flatten_indices=False: (keyword_sets, all_keywords, used_keywords, relationship_docs) where:
          - keyword_sets: Expanded keyword sets dictionary
          - all_keywords: List of ALL keywords in the entire ontology (including synonyms if include_synonyms=True)
          - used_keywords: List of keywords actually used in the returned keyword sets
          - relationship_docs: Dictionary mapping set names to lists of relationship text strings.
            Each string describes a relationship between keywords in that set, e.g.,
            "pheno 2 is a subtype of pheno 1" or "pheno 2 causes disease A".
            Empty list if no relationships found or include_relationships=False.
        - If flatten_indices=True: (keyword_sets, set_indices, all_keywords, used_keywords, relationship_docs) where:
          - keyword_sets: Expanded keyword sets dictionary
          - set_indices: numpy array of shape (n_keywords,) with set indices for each keyword
          - all_keywords: List of ALL keywords in the entire ontology (including synonyms if include_synonyms=True)
          - used_keywords: List of keywords actually used in the returned keyword sets
          - relationship_docs: Dictionary mapping set names to lists of relationship text strings.
            Each string describes a relationship between keywords in that set.
            Empty list if no relationships found or include_relationships=False.
    
    Examples
    --------
    >>> # Generate 20 random keyword sets from ontology with relationships
    >>> keyword_sets, all_keywords, used_keywords, relationship_docs = generate_keyword_sets_from_ontology(
    ...     ontology_graph=ontology_graph,
    ...     n_seeds=20,
    ...     n_hops=2,
    ...     max_keywords_per_set=50
    ... )
    >>> # relationship_docs contains text descriptions of relationships, e.g.:
    >>> # {"seed_0": ["pheno 2 is a subtype of pheno 1", "pheno 2 causes disease A"], ...}
    >>> 
    >>> # Generate sets with multiple hop levels
    >>> # Creates sets like "seed_0_1", "seed_0_2", "seed_1_1", "seed_1_2", etc.
    >>> keyword_sets, all_keywords, used_keywords, relationship_docs = generate_keyword_sets_from_ontology(
    ...     ontology_graph=ontology_graph,
    ...     n_seeds=20,
    ...     n_hops=[1, 2, 3],  # Create sets for 1-hop, 2-hop, and 3-hop
    ...     max_keywords_per_set=50
    ... )
    >>> 
    >>> # Use with your system
    >>> keyword_sets_obj = create_keyword_sets(
    ...     keyword_sets=keyword_sets,
    ...     event_dataset=event_dataset
    ... )
    >>> 
    >>> # Get indices for aggregating embeddings by set
    >>> keyword_sets, set_indices, all_keywords, used_keywords, relationship_docs = generate_keyword_sets_from_ontology(
    ...     ontology_graph=ontology_graph,
    ...     n_seeds=20,
    ...     flatten_indices=True
    ... )
    >>> # Use used_keywords to embed only keywords actually used in sets
    >>> embeddings = embed_keywords(used_keywords)
    >>> # Aggregate by set using indices
    >>> set_names = list(keyword_sets.keys())
    >>> for set_idx, set_name in enumerate(set_names):
    ...     mask = set_indices == set_idx
    ...     set_embeddings = embeddings[mask]
    ...     aggregated = set_embeddings.mean(axis=0)
    >>> 
    >>> # Generate sets with resampling and synonyms, including relationships
    >>> # Creates original sets plus resampled versions, with synonyms included
    >>> keyword_sets, all_keywords, used_keywords, relationship_docs = generate_keyword_sets_from_ontology(
    ...     ontology_graph=ontology_graph,
    ...     ontology_owl=ontology_owl,
    ...     n_seeds=20,
    ...     n_hops=2,
    ...     resample_times=3,  # Create 3 resamples of each set
    ...     resample_frac=0.5,  # Use 50% of keywords for each resample
    ...     include_synonyms=True,  # Include synonyms in sets
    ...     include_relationships=True  # Extract relationship documents
    ... )
    >>> # Result includes: "seed_0", "seed_0_resample_1", "seed_0_resample_2", "seed_0_resample_3", etc.
    >>> # all_keywords includes ALL terms and synonyms in the entire ontology
    >>> # used_keywords includes only keywords actually used in the returned sets
    >>> # relationship_docs contains relationship text for each set
    >>> 
    >>> # Use custom relationship formatter
    >>> def custom_formatter(source, target, rel_type):
    ...     return f"{source} [REL: {rel_type}] {target}"
    >>> keyword_sets, all_keywords, used_keywords, relationship_docs = generate_keyword_sets_from_ontology(
    ...     ontology_graph=ontology_graph,
    ...     n_seeds=20,
    ...     relationship_formatter=custom_formatter
    ... )
    """
    # Validate parameters at the beginning
    if include_synonyms and ontology_owl is None:
        raise ValueError(
            "ontology_owl must be provided when include_synonyms=True. "
            "Use load_mondo_ontology(return_owl=True) to get the OWL object."
        )
    
    # Generate random seeds
    seeds = random_seed_keyword_sets(
        ontology_graph=ontology_graph,
        ontology_dict=ontology_dict,
        n_seeds=n_seeds,
        n_keywords_per_seed=n_keywords_per_seed,
        random_seed=random_seed,
        min_degree=min_degree,
        max_degree=max_degree,
        min_seed_depth=min_seed_depth,
    )
    
    # Handle n_hops as list or int
    if isinstance(n_hops, list):
        # If n_hops is a list, create sets for each hop value
        expanded = {}
        # Setup progress bar for hop expansion
        try:
            from tqdm.auto import tqdm
            hop_iterator = tqdm(n_hops, desc="Expanding hops", disable=not show_progress)
        except ImportError:
            hop_iterator = n_hops
        
        for hop_value in hop_iterator:
            # Expand from seeds with this hop value
            hop_expanded = expand_keyword_sets_from_ontology(
                seed_keywords=seeds,
                ontology_graph=ontology_graph,
                ontology_dict=ontology_dict,
                n_hops=hop_value,
                max_keywords_per_set=max_keywords_per_set,
                include_seeds=include_seeds,
                bidirectional=bidirectional,
            )
            # Rename sets to include hop value: {seed_name}_{hop_value}
            for seed_name, keywords in hop_expanded.items():
                new_name = f"{seed_name}_{hop_value}"
                expanded[new_name] = keywords
    else:
        # Single hop value - use original behavior
        expanded = expand_keyword_sets_from_ontology(
            seed_keywords=seeds,
            ontology_graph=ontology_graph,
            ontology_dict=ontology_dict,
            n_hops=n_hops,
            max_keywords_per_set=max_keywords_per_set,
            include_seeds=include_seeds,
            bidirectional=bidirectional,
        )
    
    # Include synonyms if requested (BEFORE resampling, so synonyms are available for sampling)
    if include_synonyms:
        if verbose >= 1:
            print("Extracting synonyms from ontology...")
        
        # Get synonyms for all terms
        try:
            synonyms_dict = get_ontology_synonyms(ontology_owl)
            
            if verbose >= 1:
                print(f"Found synonyms for {len(synonyms_dict)} terms")
            
            # Add synonyms to each keyword set and ensure uniqueness
            expanded_with_synonyms = {}
            for set_name, keywords in expanded.items():
                # Start with original keywords as a set for uniqueness
                expanded_keywords_set = set(keywords)
                
                # Add synonyms for each keyword in the set
                for keyword in keywords:
                    if keyword in synonyms_dict:
                        synonyms = synonyms_dict[keyword]
                        # Add synonyms to the set (automatically handles uniqueness)
                        expanded_keywords_set.update(synonyms)
                
                # Convert back to list (unique values only, preserving some order)
                expanded_with_synonyms[set_name] = list(expanded_keywords_set)
            
            expanded = expanded_with_synonyms
            
            if verbose >= 1:
                total_synonyms = sum(len(synonyms_dict.get(kw, [])) for keywords in expanded.values() for kw in keywords)
                print(f"Added synonyms to keyword sets (total synonyms found: {total_synonyms:,})")
        except Exception as e:
            if verbose >= 1:
                print(f"Warning: Could not extract synonyms: {e}")
            # Continue without synonyms if extraction fails
    
    # Apply resampling if requested (synonyms are now available in expanded sets)
    if resample_times is not None and resample_times > 0:
        # Validate resample_frac if provided
        if resample_frac is not None:
            if resample_frac <= 0 or resample_frac > 1:
                raise ValueError(
                    f"resample_frac must be between 0 and 1, got {resample_frac}"
                )
        
        # Set random seed for reproducibility if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        resampled_sets = {}
        # Setup progress bar for resampling
        try:
            from tqdm.auto import tqdm
            resample_iterator = tqdm(expanded.items(), desc="Resampling sets", disable=not show_progress, total=len(expanded))
        except ImportError:
            resample_iterator = expanded.items()
        
        for original_name, keywords in resample_iterator:
            # Add original set
            resampled_sets[original_name] = keywords
            
            # Skip resampling if set is too small
            if len(keywords) < resample_min:
                if verbose >= 2:
                    print(f"Skipping resampling for {original_name}: set size {len(keywords):,} < resample_min {resample_min:,}")
                continue
            
            # Track unique sets for this original set (using frozenset for hashability)
            seen_sets = {frozenset(keywords)}  # Start with original set
            
            # Create resamples
            resample_count = 0
            max_attempts = resample_times * 10  # Prevent infinite loop
            attempts = 0
            
            while resample_count < resample_times and attempts < max_attempts:
                attempts += 1
                
                # Determine fraction to sample
                if resample_frac is not None:
                    frac = resample_frac
                else:
                    # Random fraction between 0.1 and 0.9
                    frac = random.uniform(0.1, 0.9)
                
                # Calculate number of keywords to sample
                n_sample = max(1, int(len(keywords) * frac))
                
                # Ensure we don't sample more than available
                n_sample = min(n_sample, len(keywords))
                
                # Randomly sample keywords
                sampled_keywords = random.sample(keywords, n_sample)
                sampled_set = frozenset(sampled_keywords)
                
                # Only add if unique
                if sampled_set not in seen_sets:
                    seen_sets.add(sampled_set)
                    resample_count += 1
                    resample_name = f"{original_name}_resample_{resample_count}"
                    resampled_sets[resample_name] = sampled_keywords
            
            # Warn if we couldn't generate enough unique resamples (only if verbose >= 2)
            if resample_count < resample_times and verbose >= 2:
                print(f"Warning: Only generated {resample_count:,} unique resamples for {original_name} "
                      f"(requested {resample_times:,})")
        
        expanded = resampled_sets
    
    # Collect keywords actually used in the returned sets
    used_keywords_set = set()
    for keywords in expanded.values():
        used_keywords_set.update(keywords)
    used_keywords = sorted(list(used_keywords_set))
    
    # Get ALL keywords from the entire ontology (including synonyms if requested)
    all_keywords_set = set()
    
    # Get all terms from the ontology
    if ontology_graph is not None or ontology_dict is not None or ontology_owl is not None:
        all_terms = get_ontology_terms(
            ontology_graph=ontology_graph,
            ontology_owl=ontology_owl,
            ontology_dict=ontology_dict,
        )
        all_keywords_set.update(all_terms)
    
    # If synonyms are requested, get all synonyms from the entire ontology
    if include_synonyms and ontology_owl is not None:
        try:
            all_synonyms_dict = get_ontology_synonyms(ontology_owl)
            # Add all synonyms to the set
            for term_synonyms in all_synonyms_dict.values():
                all_keywords_set.update(term_synonyms)
            # Also add the terms that have synonyms
            all_keywords_set.update(all_synonyms_dict.keys())
        except Exception as e:
            if verbose >= 1:
                print(f"Warning: Could not extract all synonyms for all_keywords: {e}")
    
    all_keywords = sorted(list(all_keywords_set))
    
    # Extract and format relationships for each keyword set
    relationship_docs = {}
    if include_relationships and ontology_graph is not None:
        if verbose >= 1:
            print("Extracting relationships between keywords in sets...")
        
        # Pre-compute all relationships for all keywords that appear in any set
        # This is much faster than checking edges for each set separately
        all_keywords_in_sets = set()
        for keywords in expanded.values():
            all_keywords_in_sets.update(keywords)
        
        # Build a fast lookup: for each keyword, get all its relationships to other keywords in sets
        keyword_relationships = {}  # {source: [(target, rel_type), ...]}
        for source in all_keywords_in_sets:
            if source in ontology_graph:
                keyword_relationships[source] = []
                for target in ontology_graph.successors(source):
                    if target in all_keywords_in_sets:
                        edge_data = ontology_graph.get_edge_data(source, target, {})
                        relationship_type = edge_data.get('relationship', 'unknown')
                        keyword_relationships[source].append((target, relationship_type))
        
        # Now quickly extract relationships for each set
        # Setup progress bar for relationship extraction
        try:
            from tqdm.auto import tqdm
            relationship_iterator = tqdm(expanded.items(), desc="Extracting relationships", disable=not show_progress, total=len(expanded))
        except ImportError:
            relationship_iterator = expanded.items()
        
        for set_name, keywords in relationship_iterator:
            keyword_set_set = set(keywords)
            relationship_texts = []
            
            # For each keyword in the set, get its relationships
            for source in keywords:
                if source in keyword_relationships:
                    for target, rel_type in keyword_relationships[source]:
                        # Only include if target is also in this set
                        if target in keyword_set_set:
                            text = format_relationship_as_text(
                                source=source,
                                target=target,
                                relationship_type=rel_type,
                                formatter=relationship_formatter,
                            )
                            relationship_texts.append(text)
            
            # Only include sets with non-empty relationship lists
            if relationship_texts:
                relationship_docs[set_name] = relationship_texts
        
        if verbose >= 1:
            total_relationships = sum(len(docs) for docs in relationship_docs.values())
            print(f"Extracted {total_relationships:,} relationships across {len(relationship_docs):,} sets")
    elif include_relationships and ontology_graph is None:
        # Warn if relationships requested but no graph provided
        if verbose >= 1:
            print("Warning: include_relationships=True but ontology_graph is None. "
                  "Cannot extract relationships without graph. Returning empty relationship docs.")
        # Don't add empty lists - only return sets with relationships
    else:
        # Don't add empty lists - only return sets with relationships
        pass
    
    # Report statistics
    if verbose >= 1:
        print(f"\nKeyword statistics:")
        print(f"  Keywords used in sets: {len(used_keywords):,}")
        print(f"  All keywords in ontology: {len(all_keywords):,}")
        if include_synonyms:
            print(f"  (all_keywords includes synonyms from entire ontology)")
    
    if flatten_indices:
        # Build indices array: each element indicates which set (by index) that keyword belongs to
        import numpy as np
        
        set_indices_list = []
        for set_idx, (set_name, keywords) in enumerate(expanded.items()):
            n_keywords = len(keywords)
            set_indices_list.extend([set_idx] * n_keywords)
        
        set_indices = np.array(set_indices_list, dtype=np.int64)
        return expanded, set_indices, all_keywords, used_keywords, relationship_docs
    
    return expanded, all_keywords, used_keywords, relationship_docs


def load_mondo_ontology(
    mondo_file_path: Optional[str] = None,
    download_url: Optional[str] = None,
    cache_dir: Optional[str] = None,
    relationship_types: Optional[List[str]] = None,
    bidirectional: bool = False,
    verbose: bool = True,
    return_owl: bool = False,
) -> Union[nx.DiGraph, Tuple[nx.DiGraph, Any]]:
    """
    Load MONDO (Monarch Disease Ontology) using owlready2.
    
    MONDO is a comprehensive disease ontology that provides hierarchical
    relationships between diseases. This function loads it and converts it
    to a NetworkX graph for use with keyword expansion functions.
    
    Parameters
    ----------
    mondo_file_path : str, optional
        Path to local MONDO OWL file. If None, will try to download or use cache.
        Example: "/path/to/mondo.owl"
    download_url : str, optional
        URL to download MONDO OWL file if not found locally.
        Default: "http://purl.obolibrary.org/obo/mondo.owl"
    cache_dir : str, optional
        Directory to cache downloaded MONDO file.
        Default: ~/.cache/aou/mondo.owl
    relationship_types : List[str], optional
        Types of relationships to include in the graph.
        Common types: ["is_a", "part_of", "has_part", "disease_has_location"]
        If None, includes "is_a" (subclass) relationships by default.
    bidirectional : bool, default False
        If True, adds reverse edges for all relationships.
    verbose : bool, default True
        Print progress information.
    return_owl : bool, default False
        If True, also returns the owlready2 ontology object.
        When True, returns (graph, ontology) tuple.
    
    Returns
    -------
    nx.DiGraph or Tuple[nx.DiGraph, owlready2.Ontology]
        NetworkX directed graph representing MONDO ontology.
        Nodes are disease terms (with labels/IDs), edges are relationships.
        If return_owl=True, also returns the owlready2 ontology object.
    
    Examples
    --------
    >>> # Load MONDO from local file
    >>> mondo_graph = load_mondo_ontology(
    ...     mondo_file_path="/path/to/mondo.owl"
    ... )
    >>> 
    >>> # Generate keyword sets from MONDO
    >>> keyword_sets = generate_keyword_sets_from_ontology(
    ...     ontology_graph=mondo_graph,
    ...     n_seeds=50,
    ...     n_hops=2
    ... )
    >>> 
    >>> # Load with specific relationships only
    >>> mondo_graph = load_mondo_ontology(
    ...     mondo_file_path="/path/to/mondo.owl",
    ...     relationship_types=["is_a", "part_of"]
    ... )
    """
    if not OWLREADY2_AVAILABLE:
        raise ImportError(
            "owlready2 is required to load MONDO ontology. "
            "Please install it: pip install owlready2"
        )
    
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "networkx is required for ontology graphs. "
            "Please install it: pip install networkx"
        )
    
    # Default download URL for MONDO
    if download_url is None:
        download_url = "http://purl.obolibrary.org/obo/mondo.owl"
    
    # Determine file path
    if mondo_file_path is None:
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/aou")
        os.makedirs(cache_dir, exist_ok=True)
        mondo_file_path = os.path.join(cache_dir, "mondo.owl")
    
    # Download if needed
    if not os.path.exists(mondo_file_path):
        if verbose:
            print(f"MONDO file not found at {mondo_file_path}")
            print(f"Downloading from {download_url}...")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(download_url, mondo_file_path)
            if verbose:
                print(f"Downloaded MONDO to {mondo_file_path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not download MONDO ontology. "
                f"Please download it manually from {download_url} "
                f"and provide mondo_file_path, or check your internet connection. "
                f"Error: {e}"
            )
    
    if verbose:
        print(f"Loading MONDO ontology from {mondo_file_path}...")
    
    # Load OWL file using owlready2
    try:
        # owlready2 uses file:// protocol for local files
        file_url = f"file://{os.path.abspath(mondo_file_path)}"
        onto = get_ontology(file_url).load()
        if verbose:
            print(f"Loaded MONDO ontology successfully")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MONDO ontology with owlready2. "
            f"Make sure the file is valid OWL format. Error: {e}"
        )
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Default relationship types if not specified
    if relationship_types is None:
        # Focus on is_a (subclass) relationships - most important for hierarchy
        relationship_types = ["is_a"]
    
    if verbose:
        print(f"Extracting relationships: {relationship_types}")
    
    # Get all classes (diseases)
    all_classes = list(onto.classes())
    if verbose:
        print(f"Found {len(all_classes):,} classes in MONDO")
    
    # Extract relationships
    relationship_count = 0
    
    for cls in all_classes:
        # Get class label (human-readable name)
        cls_label = cls.label.first() if cls.label else str(cls).split('.')[-1]
        # Clean up label - remove namespace if present
        if ':' in cls_label:
            cls_label = cls_label.split(':')[-1]
        
        # Extract is_a relationships (subclass)
        if "is_a" in relationship_types:
            for parent in cls.is_a:
                # Skip if parent is a restriction or other non-class
                if not hasattr(parent, 'iri'):
                    continue
                
                parent_label = parent.label.first() if parent.label else str(parent).split('.')[-1]
                # Clean up label
                if ':' in parent_label:
                    parent_label = parent_label.split(':')[-1]
                
                # Add edge: parent -> child (is_a means child is_a parent)
                # In MONDO, is_a creates hierarchy: parent (broader) -> child (more specific)
                G.add_edge(parent_label, cls_label, relationship="is_a")
                relationship_count += 1
                
                if bidirectional:
                    G.add_edge(cls_label, parent_label, relationship="reverse_is_a")
        
        # Extract other relationship types
        for rel_type in relationship_types:
            if rel_type == "is_a":
                continue  # Already handled
            
            # Try to get property by name
            try:
                # Search for property in ontology
                prop = None
                for p in onto.properties():
                    if p.name == rel_type or rel_type.replace('_', ' ') in str(p.label).lower():
                        prop = p
                        break
                
                if prop is not None:
                    # Get values of this property for this class
                    values = getattr(cls, prop.name, [])
                    if not isinstance(values, list):
                        values = [values] if values else []
                    
                    for value in values:
                        if hasattr(value, 'label'):
                            value_label = value.label.first() if value.label else str(value).split('.')[-1]
                            if ':' in value_label:
                                value_label = value_label.split(':')[-1]
                            
                            G.add_edge(cls_label, value_label, relationship=rel_type)
                            relationship_count += 1
                            
                            if bidirectional:
                                G.add_edge(value_label, cls_label, relationship=f"reverse_{rel_type}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not extract {rel_type} relationships: {e}")
    
    if verbose:
        print(f"Loaded {len(G.nodes()):,} nodes and {len(G.edges()):,} edges from MONDO")
        if G.edges():
            rel_types = set(data.get('relationship', 'unknown') for _, _, data in G.edges(data=True))
            print(f"Relationship types found: {rel_types}")
    
    if return_owl:
        return G, onto
    return G


# Module-level LRU cache for graphs
# Key: (ontology_iri, relationship_types_tuple, bidirectional)
# Value: nx.DiGraph
_extract_graph_cache: Dict[Tuple[str, Optional[Tuple[str, ...]], bool], nx.DiGraph] = {}
# Maximum number of ontology graphs to keep in the LRU cache.
# When the number of cached graphs exceeds this, the oldest ones are evicted to save memory.
_extract_graph_cache_maxsize = 128

# Cache for relationship type discovery (when relationship_types=None)
# Key: ontology_iri
# Value: List[str] of discovered relationship types
_discovered_relationship_types_cache: Dict[str, List[str]] = {}


def extract_graph_from_owl(
    ontology_owl: Any,
    relationship_types: Optional[List[str]] = None,
    bidirectional: bool = False,
) -> nx.DiGraph:
    """
    Extract a NetworkX graph from an OWL ontology.
    
    This function converts an OWL ontology object into a NetworkX directed graph,
    where nodes are ontology classes (terms) and edges represent relationships
    between them. Edge attributes store the relationship type.
    
    Results are cached using LRU cache based on ontology IRI, relationship types,
    and bidirectional flag to avoid redundant graph extraction.
    
    Parameters
    ----------
    ontology_owl : owlready2.Ontology
        OWL ontology object (from owlready2).
    relationship_types : List[str], optional
        Specific relationship types to extract. If None, extracts all available
        relationship types from the ontology (including "is_a").
    bidirectional : bool, default False
        If True, adds reverse edges for all relationships.
        For example, if "is_a" creates edge (parent, child), also creates
        (child, parent) with relationship "reverse_is_a".
    
    Returns
    -------
    nx.DiGraph
        NetworkX directed graph extracted from the OWL ontology.
        Nodes are term labels (strings), edges have a 'relationship' attribute
        indicating the relationship type.
        
        Note: The returned graph is cached and should not be mutated.
        If you need to modify the graph, create a copy first.
    
    Examples
    --------
    >>> # Extract graph with all relationship types
    >>> mondo_graph, mondo_owl = load_mondo_ontology(return_owl=True)
    >>> graph = extract_graph_from_owl(ontology_owl=mondo_owl)
    >>> print(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    >>> 
    >>> # Extract graph with only specific relationship types
    >>> graph = extract_graph_from_owl(
    ...     ontology_owl=mondo_owl,
    ...     relationship_types=["is_a", "part_of"]
    ... )
    >>> 
    >>> # Extract with bidirectional edges
    >>> graph = extract_graph_from_owl(
    ...     ontology_owl=mondo_owl,
    ...     bidirectional=True
    ... )
    """
    if not OWLREADY2_AVAILABLE:
        raise ImportError(
            "owlready2 is required to extract graph from OWL ontology. "
            "Please install it: pip install owlready2"
        )
    
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "networkx is required to extract graph. "
            "Please install it: pip install networkx"
        )
    # Create cache key from ontology IRI and parameters
    try:
        ontology_iri = str(ontology_owl.base_iri) if hasattr(ontology_owl, 'base_iri') else str(id(ontology_owl))
    except:
        ontology_iri = str(id(ontology_owl))
    
    # Convert relationship_types to tuple for hashing
    relationship_types_tuple = tuple(sorted(relationship_types)) if relationship_types else None
    cache_key = (ontology_iri, relationship_types_tuple, bidirectional)
    
    # Check cache
    if cache_key in _extract_graph_cache:
        # Return cached graph directly (callers should not mutate it)
        return _extract_graph_cache[cache_key]
    
    # Get all available relationship types if not specified
    if relationship_types is None:
        # Check cache for discovered relationship types
        if ontology_iri in _discovered_relationship_types_cache:
            relationship_types = _discovered_relationship_types_cache[ontology_iri]
        else:
            relationship_types_set = set()
            relationship_types_set.add("is_a")  # Always include is_a
            
            # Get all object properties (relationships) from the ontology
            for prop in ontology_owl.properties():
                if hasattr(prop, 'name'):
                    relationship_types_set.add(prop.name)
            
            relationship_types = sorted(list(relationship_types_set))
            # Cache the discovered relationship types
            _discovered_relationship_types_cache[ontology_iri] = relationship_types
    
    # Build graph from OWL (similar to load_mondo_ontology logic)
    G = nx.DiGraph()
    all_classes = list(ontology_owl.classes())
    
    # Setup progress bar for graph extraction
    try:
        from tqdm.auto import tqdm
        class_iterator = tqdm(all_classes, desc="Extracting graph", unit="classes")
    except ImportError:
        class_iterator = all_classes
    
    for cls in class_iterator:
        # Get class label (human-readable name)
        label_obj = cls.label.first() if cls.label else None
        if label_obj is None:
            cls_label = str(cls).split('.')[-1]
        else:
            # Handle locstr objects from owlready2
            if hasattr(label_obj, '__str__'):
                cls_label = str(label_obj)
            else:
                cls_label = str(label_obj)
        # Clean up label - remove namespace if present, handle locstr format
        if isinstance(cls_label, str):
            if ':' in cls_label:
                cls_label = cls_label.split(':')[-1]
            # Handle locstr('text', 'en') format - extract the text
            import re
            locstr_match = re.match(r"locstr\('([^']+)'", cls_label)
            if locstr_match:
                cls_label = locstr_match.group(1)
        
        # Extract is_a relationships (subclass)
        if "is_a" in relationship_types:
            for parent in cls.is_a:
                # Skip if parent is a restriction or other non-class
                if not hasattr(parent, 'iri'):
                    continue
                
                parent_label = parent.label.first() if parent.label else str(parent).split('.')[-1]
                # Clean up label
                if ':' in parent_label:
                    parent_label = parent_label.split(':')[-1]
                
                # Add edge: parent -> child (is_a means child is_a parent)
                G.add_edge(parent_label, cls_label, relationship="is_a")
                
                if bidirectional:
                    G.add_edge(cls_label, parent_label, relationship="reverse_is_a")
        
        # Extract other relationship types
        for rel_type in relationship_types:
            if rel_type == "is_a":
                continue  # Already handled
            
            # Try to get property by name
            try:
                # Search for property in ontology
                prop = None
                for p in ontology_owl.properties():
                    if p.name == rel_type or rel_type.replace('_', ' ') in str(p.label).lower():
                        prop = p
                        break
                
                if prop is not None:
                    # Get values of this property for this class
                    values = getattr(cls, prop.name, [])
                    if not isinstance(values, list):
                        values = [values] if values else []
                    
                    for value in values:
                        if hasattr(value, 'label'):
                            value_label = value.label.first() if value.label else str(value).split('.')[-1]
                            if ':' in value_label:
                                value_label = value_label.split(':')[-1]
                            
                            G.add_edge(cls_label, value_label, relationship=rel_type)
                            
                            if bidirectional:
                                G.add_edge(value_label, cls_label, relationship=f"reverse_{rel_type}")
            except Exception:
                # Skip if property extraction fails
                pass
    
    # Store in cache (with LRU eviction if needed)
    if len(_extract_graph_cache) >= _extract_graph_cache_maxsize:
        # Remove oldest entry (FIFO - first in, first out)
        # In Python 3.7+, dicts maintain insertion order
        oldest_key = next(iter(_extract_graph_cache))
        del _extract_graph_cache[oldest_key]
    
    # Store the graph in cache
    _extract_graph_cache[cache_key] = G
    
    # Return graph directly (callers should not mutate it to avoid affecting cache)
    return G


def list_ontology_relationship_types(
    ontology_owl: Any,
    include_reverse: bool = True,
) -> List[str]:
    """
    List all relationship types present in an ontology.
    
    This function extracts a NetworkX graph from the OWL ontology and
    returns all unique relationship types found in the graph edges.
    
    Parameters
    ----------
    ontology_owl : owlready2.Ontology
        OWL ontology object (from owlready2).
    include_reverse : bool, default True
        If True, includes reverse relationship types (e.g., "reverse_is_a").
        If False, filters them out.
    
    Returns
    -------
    List[str]
        Sorted list of unique relationship types found in the ontology.
        Examples: ["is_a", "part_of", "has_part", "disease_has_location"]
    
    Examples
    --------
    >>> # From an OWL ontology object
    >>> mondo_graph, mondo_owl = load_mondo_ontology(return_owl=True)
    >>> rel_types = list_ontology_relationship_types(ontology_owl=mondo_owl)
    >>> print(rel_types)
    ['is_a', 'part_of', 'has_part', ...]
    >>> 
    >>> # Exclude reverse relationships
    >>> rel_types = list_ontology_relationship_types(
    ...     ontology_owl=mondo_owl,
    ...     include_reverse=False
    ... )
    """
    # Extract graph from OWL ontology
    G = extract_graph_from_owl(ontology_owl)
    
    # Extract relationship types from graph edges
    relationship_types = set()
    for _, _, data in G.edges(data=True):
        rel_type = data.get('relationship', None)
        if rel_type:
            relationship_types.add(rel_type)
    
    # Filter out reverse relationships if requested
    if not include_reverse:
        relationship_types = {
            rel_type for rel_type in relationship_types
            if not rel_type.startswith("reverse_")
        }
    
    # Return sorted list
    return sorted(list(relationship_types))


def count_ontology_relationship_types(
    ontology_owl: Any,
    include_reverse: bool = True,
    sort_by_count: bool = True,
    relationship_types: Optional[List[str]] = None,
    bidirectional: bool = False,
) -> Dict[str, int]:
    """
    Count occurrences of each relationship type in an ontology.
    
    This function extracts a NetworkX graph from the OWL ontology and counts
    relationship types from the graph edges.
    
    Parameters
    ----------
    ontology_owl : owlready2.Ontology
        OWL ontology object (from owlready2).
    include_reverse : bool, default True
        If True, includes reverse relationship types (e.g., "reverse_is_a").
        If False, filters them out.
    sort_by_count : bool, default True
        If True, returns dictionary sorted by count (descending).
        If False, returns dictionary sorted by relationship type name.
    relationship_types : List[str], optional
        Specific relationship types to extract. If None, extracts all available
        relationship types from the ontology.
    bidirectional : bool, default False
        If True, adds reverse edges for all relationships.
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping relationship type to count of occurrences.
        Examples: {"is_a": 78014, "part_of": 1234, "causes": 567}
    
    Examples
    --------
    >>> # Count all relationship types
    >>> mondo_graph, mondo_owl = load_mondo_ontology(return_owl=True)
    >>> rel_counts = count_ontology_relationship_types(ontology_owl=mondo_owl)
    >>> print(rel_counts)
    {'is_a': 78014, 'part_of': 1234, ...}
    >>> 
    >>> # Exclude reverse relationships and sort by type name
    >>> rel_counts = count_ontology_relationship_types(
    ...     ontology_owl=mondo_owl,
    ...     include_reverse=False,
    ...     sort_by_count=False
    ... )
    >>> 
    >>> # Count only specific relationship types
    >>> rel_counts = count_ontology_relationship_types(
    ...     ontology_owl=mondo_owl,
    ...     relationship_types=["is_a", "part_of"]
    ... )
    >>> 
    >>> # Print sorted by count
    >>> for rel_type, count in rel_counts.items():
    ...     print(f"{rel_type}: {count:,}")
    """
    if not OWLREADY2_AVAILABLE:
        raise ImportError(
            "owlready2 is required to count relationship types from OWL ontology. "
            "Please install it: pip install owlready2"
        )
    
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "networkx is required to count relationship types. "
            "Please install it: pip install networkx"
        )
    
    # Extract graph from OWL ontology
    G = extract_graph_from_owl(
        ontology_owl=ontology_owl,
        relationship_types=relationship_types,
        bidirectional=bidirectional
    )
    
    # Count relationship types from graph edges (highly optimized)
    num_edges = G.number_of_edges()
    
    # Setup progress bar for counting (only for large graphs)
    try:
        from tqdm.auto import tqdm
        show_progress = num_edges > 10000
        if show_progress:
            edge_iterator = tqdm(G.edges(data=True), desc="Counting relationships", total=num_edges, unit="edges")
        else:
            edge_iterator = G.edges(data=True)
    except ImportError:
        edge_iterator = G.edges(data=True)
        show_progress = False
    
    # Ultra-fast counting: extract relationship type once per edge
    if include_reverse:
        # Fast path: count all relationship types
        relationship_counts = Counter(
            rel_type
            for _, _, data in edge_iterator
            if (rel_type := data.get('relationship'))
        )
    else:
        # Filter out reverse relationships (optimized: check startswith only once)
        relationship_counts = Counter(
            rel_type
            for _, _, data in edge_iterator
            if (rel_type := data.get('relationship')) and not rel_type.startswith("reverse_")
        )
    
    # Convert Counter to dict for consistency
    relationship_counts = dict(relationship_counts)
    
    # Sort by count (descending) or by name
    if sort_by_count:
        # Sort by count descending, then by name for ties
        sorted_items = sorted(relationship_counts.items(), key=lambda x: (-x[1], x[0]))
        return dict(sorted_items)
    else:
        # Sort by name
        return dict(sorted(relationship_counts.items()))


def get_ontology_terms(
    ontology_graph: Optional[nx.DiGraph] = None,
    ontology_owl: Optional[Any] = None,
    ontology_dict: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Get all terms (nodes/classes) from an ontology.
    
    This function extracts all unique terms from either a NetworkX graph,
    an OWL ontology object, or a dictionary representation. If both graph
    and OWL are provided, the graph takes precedence.
    
    Parameters
    ----------
    ontology_graph : nx.DiGraph, optional
        NetworkX directed graph representing the ontology.
        If provided, terms are extracted from graph nodes.
        Takes precedence over ontology_owl if both are provided.
    ontology_owl : owlready2.Ontology, optional
        OWL ontology object (from owlready2).
        If provided, terms are extracted from ontology classes.
        The graph is extracted from the OWL internally.
        Only used if ontology_graph is not provided.
    ontology_dict : Dict[str, List[str]], optional
        Dictionary mapping terms to their related terms.
        If provided, terms are extracted from dictionary keys and values.
        Only used if neither graph nor owl are provided.
    
    Returns
    -------
    List[str]
        Sorted list of all unique terms found in the ontology.
    
    Examples
    --------
    >>> # From a NetworkX graph
    >>> mondo_graph, mondo_owl = load_mondo_ontology(return_owl=True)
    >>> terms = get_ontology_terms(ontology_graph=mondo_graph)
    >>> print(f"Found {len(terms)} terms")
    >>> 
    >>> # From an OWL ontology object
    >>> terms = get_ontology_terms(ontology_owl=mondo_owl)
    >>> 
    >>> # From a dictionary
    >>> ontology_dict = {"disease": ["cancer", "diabetes"], "cancer": ["lung cancer"]}
    >>> terms = get_ontology_terms(ontology_dict=ontology_dict)
    >>> # Result: ["cancer", "diabetes", "disease", "lung cancer"]
    """
    terms = set()
    
    # Extract from NetworkX graph if provided (takes precedence)
    if ontology_graph is not None:
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required to extract terms from graph. "
                "Please install it: pip install networkx"
            )
        
        # Get all nodes (terms) from the graph
        terms.update(ontology_graph.nodes())
    
    # Extract from OWL ontology if provided and no graph was given
    elif ontology_owl is not None:
        if not OWLREADY2_AVAILABLE:
            raise ImportError(
                "owlready2 is required to extract terms from OWL ontology. "
                "Please install it: pip install owlready2"
            )
        
        # Extract graph from OWL and get all nodes (terms)
        G = extract_graph_from_owl(ontology_owl)
        terms.update(G.nodes())
    
    # Extract from dictionary if provided and neither graph nor owl were given
    elif ontology_dict is not None:
        # Get all keys (terms)
        terms.update(ontology_dict.keys())
        # Get all values (related terms)
        for related_terms in ontology_dict.values():
            terms.update(related_terms)
    
    else:
        raise ValueError(
            "Either ontology_graph, ontology_owl, or ontology_dict must be provided"
        )
    
    # Return sorted list
    return sorted(list(terms))


def get_ontology_synonyms(
    ontology_owl: Any,
    synonym_types: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Get synonyms for all terms in an OWL ontology.
    
    MONDO and other OBO ontologies store synonyms as annotation properties:
    - oboInOwl:hasExactSynonym
    - oboInOwl:hasNarrowSynonym
    - oboInOwl:hasBroadSynonym
    - oboInOwl:hasRelatedSynonym
    
    Parameters
    ----------
    ontology_owl : owlready2.Ontology
        OWL ontology object (from owlready2).
    synonym_types : List[str], optional
        Types of synonyms to extract. Common types:
        - "hasExactSynonym" or "oboInOwl:hasExactSynonym"
        - "hasNarrowSynonym" or "oboInOwl:hasNarrowSynonym"
        - "hasBroadSynonym" or "oboInOwl:hasBroadSynonym"
        - "hasRelatedSynonym" or "oboInOwl:hasRelatedSynonym"
        If None, extracts all synonym types found.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping term labels to lists of synonyms.
        Format: {term_label: [synonym1, synonym2, ...]}
    
    Examples
    --------
    >>> # Get all synonyms from MONDO
    >>> mondo_graph, mondo_owl = load_mondo_ontology(return_owl=True)
    >>> synonyms = get_ontology_synonyms(mondo_owl)
    >>> 
    >>> # Get only exact synonyms
    >>> exact_synonyms = get_ontology_synonyms(
    ...     mondo_owl,
    ...     synonym_types=["hasExactSynonym"]
    ... )
    >>> 
    >>> # Check synonyms for a specific term
    >>> if "Alzheimer's disease" in synonyms:
    ...     print(f"Synonyms: {synonyms['Alzheimer's disease']}")
    """
    if not OWLREADY2_AVAILABLE:
        raise ImportError(
            "owlready2 is required to extract synonyms from OWL ontology. "
            "Please install it: pip install owlready2"
        )
    
    synonyms_dict = {}
    
    # Default synonym types if not specified
    if synonym_types is None:
        synonym_types = [
            "hasExactSynonym",
            "hasNarrowSynonym",
            "hasBroadSynonym",
            "hasRelatedSynonym",
        ]
    
    # Normalize synonym type names
    normalized_types = [st.split(":")[-1] if ":" in st else st for st in synonym_types]
    
    # Pre-find all synonym properties once (much faster than searching per class)
    synonym_props = {}
    obo_iri_base = "http://www.geneontology.org/formats/oboInOwl#"
    
    # Get all annotation properties
    all_props = list(ontology_owl.annotation_properties())
    
    # Build lookup map for synonym properties
    for prop in all_props:
        prop_name = prop.name if hasattr(prop, 'name') else str(prop)
        prop_iri = str(prop.iri) if hasattr(prop, 'iri') else ""
        
        # Check if this matches any synonym type
        for syn_type in normalized_types:
            if (syn_type in prop_name or 
                syn_type in prop_iri or
                prop_iri.endswith(f"#{syn_type}") or
                prop_iri == f"{obo_iri_base}{syn_type}"):
                if syn_type not in synonym_props:
                    synonym_props[syn_type] = prop
                break
    
    # Get all classes once
    all_classes = list(ontology_owl.classes())
    
    # Process classes efficiently
    for cls in all_classes:
        # Get class label (primary term) - cache this
        cls_label = cls.label.first() if cls.label else str(cls).split('.')[-1]
        if ':' in cls_label:
            cls_label = cls_label.split(':')[-1]
        
        # Collect all synonyms for this class
        class_synonyms = []
        
        # Try each synonym property we found
        for syn_type, prop in synonym_props.items():
            try:
                # Direct attribute access (fastest)
                if hasattr(cls, prop.name):
                    values = getattr(cls, prop.name, [])
                else:
                    # Try direct annotation access
                    try:
                        # owlready2 stores annotations in _AnnotationProperty__values
                        values = prop._AnnotationProperty__values.get(cls, [])
                    except (AttributeError, KeyError):
                        # Fallback: try get_values if available
                        values = list(prop.get_values(cls)) if hasattr(prop, 'get_values') else []
                
                # Normalize to list
                if not isinstance(values, list):
                    values = [values] if values else []
                
                # Extract synonym strings
                for val in values:
                    if val:
                        # Handle different value types
                        if hasattr(val, 'first'):
                            syn_str = val.first()
                        elif isinstance(val, str):
                            syn_str = val
                        else:
                            syn_str = str(val)
                        
                        # Clean up
                        syn_str = syn_str.strip('"\'')
                        if syn_str:
                            class_synonyms.append(syn_str)
            except (AttributeError, TypeError, KeyError):
                # Skip this property for this class
                continue
        
        # Only add to dict if we found synonyms
        if class_synonyms:
            synonyms_dict[cls_label] = class_synonyms
    
    return synonyms_dict


def mondo_to_dict(
    mondo_graph: nx.DiGraph,
    relationship_type: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Convert MONDO NetworkX graph to simple dictionary format.
    
    This is useful if you prefer the dict format over NetworkX graphs.
    
    Parameters
    ----------
    mondo_graph : nx.DiGraph
        MONDO graph from load_mondo_ontology().
    relationship_type : str, optional
        If specified, only include edges of this relationship type.
        If None, includes all relationships.
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping terms to their related terms.
        Format: {parent_term: [child_term1, child_term2, ...]}
    """
    ontology_dict = {}
    
    for parent, child, data in mondo_graph.edges(data=True):
        # Filter by relationship type if specified
        if relationship_type is not None:
            if data.get('relationship') != relationship_type:
                continue
        
        if parent not in ontology_dict:
            ontology_dict[parent] = []
        ontology_dict[parent].append(child)
    
    return ontology_dict


def plot_static(
    keyword_reduced_embeddings,  
    mondo_keyword_sets,
    plot_width=800, 
    plot_height=600, 
    color_col: str = "set_name",
    palette: str = "rainbow",
    color_scale: str = "eq_hist",
    show_colorbar: bool = "auto",
    spread_px: int = 0,
    figsize: tuple = (8, 6),
    title: str = None,
    background_data: Optional[Union[Any, Any]] = None,  # np.ndarray or torch.Tensor
    background_palette: str = "binary_r",
    background_alpha: float = 0.3,
):
    """
    Visualize reduced keyword embeddings using Datashader and matplotlib.

    Args:
        keyword_reduced_embeddings: np.ndarray of shape (n_keywords, 2)
        all_keywords: list of keyword strings
        set_indices: iterable of int, maps each keyword to a set index
        mondo_keyword_sets: dict, keys = set names
        color_col: str, column to use for coloring points (default "set_name")
            Options: "set_name", "tokens", "density" (for local density), or any numeric column in df
        palette: str, colormap/palette name (default "rainbow")
        color_scale: str, datashader shading method: 'linear', 'log', or 'eq_hist' (default "eq_hist")
        show_colorbar: bool or str, if True show colorbar, if "auto" show only for numeric columns (default "auto")
        spread_px: int, spread pixels in datashader mode to make points appear larger (default 0)
        figsize: tuple, figure size in inches (default (8, 6))
        title: str, plot title. If None, auto-generated.
        plot_width: int, width of plot in pixels (default 800)
        plot_height: int, height of plot in pixels (default 600)
        background_data: np.ndarray or torch.Tensor, optional
            Optional array or tensor with same 2D shape as main data (n_points, 2) to plot behind main data points.
            If a torch.Tensor is provided, it will be converted to numpy array automatically.
            These background points will be rendered first (behind the main data).
        background_palette: str, default "binary_r"
            Colormap/palette name for background data. Only used when background_data is provided.
        background_alpha: float, default 0.3
            Transparency (alpha) value for background data points. Range: 0.0 (fully transparent) to 1.0 (fully opaque).
            Only used when background_data is provided.
    """
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    
    # Convert torch tensor to numpy array if needed (for background_data)
    if background_data is not None:
        try:
            import torch
            if torch.is_tensor(background_data):
                background_data = background_data.detach().cpu().numpy()
        except ImportError:
            pass  # torch not available, assume it's already numpy
    
    # Prepare dataframe for Datashader
    df = pd.DataFrame(keyword_reduced_embeddings, columns=["Dim1", "Dim2"])

    df["set_name"] = list(mondo_keyword_sets.keys())
    df["set_terms"] = df["set_name"].apply(lambda x: mondo_keyword_sets[x])
    df["tokens"] = df["set_terms"].apply(lambda x: count_tokens(x, approximate=True))

    
    # Check if color_col is "density" (special case for density-based coloring)
    color_by_density = (color_col == "density")
    
    if not color_by_density:
        # Check if color_col exists
        if color_col not in df.columns:
            available = list(df.columns) + ["density"]
            raise ValueError(
                f"Color column '{color_col}' not found. Available: {available}"
            )
        
        # Determine if color column is numeric
        color_values = df[color_col].values
        is_numeric = pd.api.types.is_numeric_dtype(df[color_col])
    else:
        # For density coloring, we'll use count aggregation
        is_numeric = True  # Density is numeric (count values)
        color_values = None  # Will be computed from aggregation
    
    # Calculate aspect ratio and adjust plot dimensions
    # Include background_data in range calculation if provided
    if background_data is not None:
        # Validate background_data shape
        if background_data.shape[1] != 2:
            raise ValueError(f"background_data must have shape (n_points, 2), got {background_data.shape}")
        
        # Combine ranges from both main data and background
        x_min = min(df['Dim1'].min(), background_data[:, 0].min())
        x_max = max(df['Dim1'].max(), background_data[:, 0].max())
        y_min = min(df['Dim2'].min(), background_data[:, 1].min())
        y_max = max(df['Dim2'].max(), background_data[:, 1].max())
        x_range = (x_min, x_max)
        y_range = (y_min, y_max)
    else:
        x_range = (df['Dim1'].min(), df['Dim1'].max())
        y_range = (df['Dim2'].min(), df['Dim2'].max())
    
    aspect = (x_range[1] - x_range[0]) / (y_range[1] - y_range[0])
    if aspect > 1:
        plot_width = plot_width
        plot_height = int(plot_width / aspect)
    else:
        plot_height = plot_height
        plot_width = int(plot_height * aspect)
    
    # Create a Datashader canvas
    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                      x_range=x_range, y_range=y_range)
    
    # Aggregate by color column
    if color_by_density:
        # For density coloring, use count aggregation (points per pixel = local density)
        agg = canvas.points(df, 'Dim1', 'Dim2', ds.count())
    elif is_numeric:
        # For numeric columns, aggregate by mean
        agg = canvas.points(df, 'Dim1', 'Dim2', ds.mean(color_col))
    else:
        # For categorical columns, aggregate by count
        agg = canvas.points(df, 'Dim1', 'Dim2', ds.count())
    
    # Get colormap
    cmap = colormaps.get_cmap(palette)
    
    # Shade the aggregation
    img = tf.shade(agg, cmap=cmap, how=color_scale)
    
    # Spread pixels to make points appear larger
    if spread_px > 0:
        img = tf.spread(img, px=spread_px)
    
    # Convert to numpy array for matplotlib
    # Set background to white, then make white pixels transparent
    img = tf.set_background(img, 'white')
    img_array = np.array(img.to_pil())
    
    # Add alpha channel: make white pixels transparent
    if img_array.shape[2] == 3:  # RGB image
        alpha_channel = np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255
        white_mask = np.all(img_array >= 250, axis=2)  # Allow slight tolerance for white
        alpha_channel[white_mask] = 0
        img_array = np.dstack([img_array, alpha_channel])
    elif img_array.shape[2] == 4:  # Already RGBA
        white_mask = np.all(img_array[:, :, :3] >= 250, axis=2)
        img_array[:, :, 3][white_mask] = 0
    
    # Flip vertically to match scatter plot orientation
    img_array = np.flipud(img_array)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot background data first (behind main data)
    if background_data is not None:
        bg_df = pd.DataFrame({
            'x': background_data[:, 0],
            'y': background_data[:, 1],
        })
        bg_df = bg_df.dropna()
        
        if len(bg_df) > 0:
            # Use datashader for background if dataset is large
            n_bg_points = len(bg_df)
            use_ds_bg = n_bg_points > 10000
            
            if use_ds_bg:
                # Create canvas for background (use same resolution as main plot)
                canvas_bg = ds.Canvas(plot_width=plot_width, plot_height=plot_height,
                                     x_range=x_range, y_range=y_range)
                
                # Aggregate by count (density)
                agg_bg = canvas_bg.points(bg_df, 'x', 'y', ds.count())
                
                # Get background colormap
                bg_cmap = colormaps.get_cmap(background_palette)
                
                # Shade the aggregation with eq_hist for better visibility
                img_bg = tf.shade(agg_bg, cmap=bg_cmap, how='eq_hist')
                
                # Convert to numpy array for matplotlib
                # Set background to white, then we'll make white pixels transparent
                img_bg = tf.set_background(img_bg, 'white')
                img_array_bg = np.array(img_bg.to_pil())
                
                # Add alpha channel: make white pixels transparent, keep data pixels visible
                if img_array_bg.shape[2] == 3:  # RGB image
                    # Add alpha channel
                    alpha_channel = np.ones((img_array_bg.shape[0], img_array_bg.shape[1]), dtype=np.uint8) * int(background_alpha * 255)
                    # White pixels (background) should be transparent
                    white_mask = np.all(img_array_bg >= 250, axis=2)  # Allow slight tolerance for white
                    alpha_channel[white_mask] = 0
                    # Stack to create RGBA
                    img_array_bg = np.dstack([img_array_bg, alpha_channel])
                elif img_array_bg.shape[2] == 4:  # Already RGBA
                    # Make white pixels transparent
                    white_mask = np.all(img_array_bg[:, :, :3] >= 250, axis=2)
                    img_array_bg[:, :, 3][white_mask] = 0
                    # Apply background_alpha to non-white pixels
                    non_white_mask = ~white_mask
                    if np.any(non_white_mask):
                        img_array_bg[:, :, 3][non_white_mask] = (img_array_bg[:, :, 3][non_white_mask] * background_alpha).astype(np.uint8)
                
                # Flip vertically to match scatter plot orientation
                img_array_bg = np.flipud(img_array_bg)
                
                # Plot background (lower zorder so it's behind)
                ax.imshow(img_array_bg, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                         origin='lower', aspect='auto', zorder=0)
            else:
                # Use matplotlib scatter for small background datasets
                ax.scatter(bg_df['x'], bg_df['y'], c='gray', s=10, alpha=background_alpha, 
                          edgecolors='none', zorder=0)
    
    # Plot main data (higher zorder so it's on top)
    im = ax.imshow(img_array, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                origin='lower', aspect='auto', zorder=1)
    
    # Set title
    if title is None:
        if color_by_density:
            title = f"Keyword Reduced Embeddings colored by local density (n={len(df):,})"
        else:
            title = f"Keyword Reduced Embeddings colored by {color_col} (n={len(df):,})"
    ax.set_title(title)
    ax.set_xlabel("Reduced Embedding Dim 1")
    ax.set_ylabel("Reduced Embedding Dim 2")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add colorbar
    if show_colorbar:
        if show_colorbar == "auto":
            # Auto-show colorbar for numeric columns (including density)
            show_colorbar = is_numeric
        
        if show_colorbar:
            if is_numeric:
                if color_by_density:
                    # For density, use the aggregation values directly
                    # Get non-zero values from aggregation (these are the density counts)
                    density_vals = agg.values[agg.values > 0]
                    if len(density_vals) > 0:
                        c_min = float(density_vals.min())
                        c_max = float(density_vals.max())
                    else:
                        c_min, c_max = 0, 1
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=c_min, vmax=c_max))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, label="Local Density (points per pixel)")
                else:
                    c_min = color_values.min()
                    c_max = color_values.max()
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=c_min, vmax=c_max))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, label=color_col)
            else:
                # For categorical, show colorbar with unique values
                unique_values = df[color_col].unique()
                n_unique = len(unique_values)
                if n_unique <= 20:  # Only show colorbar for reasonable number of categories
                    # Create a mapping from category to color
                    norm = Normalize(vmin=0, vmax=n_unique - 1)
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, label=color_col)
                    # Set ticks to category names
                    tick_positions = np.linspace(0, n_unique - 1, n_unique)
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels(sorted(unique_values))
    
    plt.tight_layout()
    plt.show()


def aggregate_keyword_set_embeddings(
    keyword_embeddings: np.ndarray,
    used_keywords: List[str],
    keyword_sets: Dict[str, List[str]],
    method: str = "mean",
    show_progress: bool = True,
) -> Any:  # Returns torch.Tensor, but type hint as Any to avoid import error
    """
    Aggregate embeddings for each keyword set and stack into a tensor.
    
    Given embeddings for keywords and a mapping of sets to keywords,
    this function aggregates the embeddings for each set (e.g., by mean) and
    returns a stacked tensor of shape (n_sets, embedding_dim).
    
    Parameters
    ----------
    keyword_embeddings : np.ndarray
        Embeddings for keywords, shape (n_keywords, embedding_dim).
        Must correspond to used_keywords in the same order (i.e., keyword_embeddings[i] 
        corresponds to used_keywords[i]).
    used_keywords : List[str]
        List of unique keywords corresponding to keyword_embeddings.
        Must be in the same order as keyword_embeddings.
    keyword_sets : Dict[str, List[str]]
        Dictionary mapping set names to lists of keywords in each set.
        Keywords in the lists should be present in used_keywords.
    method : str, default "mean"
        Aggregation method to use:
        - "mean": Mean pooling across keywords in each set
        - "max": Element-wise maximum across keywords
        - "sum": Sum across keywords
    show_progress : bool, default True
        Whether to show a progress bar during aggregation.
    
    Returns
    -------
    torch.Tensor
        Stacked aggregated embeddings, shape (n_sets, embedding_dim).
        Each row corresponds to one keyword set in the order of keyword_sets.keys().
    
    Examples
    --------
    >>> # Embeddings for unique keywords
    >>> used_keywords = list(set([k for ks in mondo_keyword_sets.values() for k in ks]))
    >>> keyword_embeddings = embed_keywords(used_keywords)  # shape: (n_unique, dim)
    >>> 
    >>> aggregated_embeddings = aggregate_keyword_set_embeddings(
    ...     keyword_embeddings=keyword_embeddings,
    ...     used_keywords=used_keywords,
    ...     keyword_sets=mondo_keyword_sets,
    ...     method="mean"
    ... )
    >>> # Shape: (n_sets, embedding_dim)
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "torch is required for aggregate_keyword_set_embeddings. "
            "Please install it: pip install torch"
        )
    
    if not NUMPY_AVAILABLE:
        raise ImportError(
            "numpy is required for aggregate_keyword_set_embeddings. "
            "Please install it: pip install numpy"
        )
    
    # Create mapping from keyword to embedding index
    keyword_to_idx = {kw: idx for idx, kw in enumerate(used_keywords)}
    
    # Validate that keyword_embeddings matches used_keywords
    if len(keyword_embeddings) != len(used_keywords):
        raise ValueError(
            f"keyword_embeddings length ({len(keyword_embeddings)}) "
            f"does not match used_keywords length ({len(used_keywords)}). "
            f"Ensure embeddings correspond to unique keywords in the same order."
        )
    
    # Get embedding dimension
    embedding_dim = keyword_embeddings.shape[1]
    
    # Report input information
    n_sets = len(keyword_sets)
    n_keywords = len(used_keywords)
    total_keywords_in_sets = sum(len(kw_list) for kw_list in keyword_sets.values())
    
    print(f"\nAggregating keyword set embeddings:")
    print(f"  Input:")
    print(f"    Number of keyword sets: {n_sets:,}")
    print(f"    Unique keywords: {n_keywords:,}")
    print(f"    Total keywords across all sets: {total_keywords_in_sets:,}")
    print(f"    Embedding dimension: {embedding_dim}")
    print(f"    Aggregation method: {method}")
    
    # Aggregate embeddings for each set
    # IMPORTANT: Maintain exact order of keyword_sets.keys() to ensure output order matches input
    aggregated_list = []
    set_names = list(keyword_sets.keys())  # Preserve insertion order (Python 3.7+)
    
    # Setup progress bar
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(set_names, desc="Aggregating keyword sets", disable=not show_progress)
    except ImportError:
        iterator = set_names
    
    # Iterate in the same order as keyword_sets.keys() to maintain order
    for set_name in iterator:
        keywords = keyword_sets[set_name]
        
        # Find indices of keywords in this set
        keyword_indices = []
        for kw in keywords:
            if kw in keyword_to_idx:
                keyword_indices.append(keyword_to_idx[kw])
            # If keyword not found, skip it (could warn here)
        
        if len(keyword_indices) == 0:
            # No valid keywords found for this set, use zero vector
            aggregated_emb = np.zeros(embedding_dim, dtype=keyword_embeddings.dtype)
        else:
            # Get embeddings for keywords in this set
            set_embeddings = keyword_embeddings[keyword_indices]  # shape: (n_keywords_in_set, embedding_dim)
            
            # Aggregate based on method
            if method == "mean":
                aggregated_emb = np.mean(set_embeddings, axis=0)
            elif method == "max":
                aggregated_emb = np.max(set_embeddings, axis=0)
            elif method == "sum":
                aggregated_emb = np.sum(set_embeddings, axis=0)
            else:
                raise ValueError(
                    f"Unknown aggregation method: {method}. "
                    f"Use 'mean', 'max', or 'sum'."
                )
        
        aggregated_list.append(aggregated_emb)
    
    # Stack into tensor
    # Order is preserved: aggregated_list[i] corresponds to set_names[i] (i.e., keyword_sets.keys()[i])
    aggregated_array = np.stack(aggregated_list, axis=0)  # shape: (n_sets, embedding_dim)
    aggregated_tensor = torch.from_numpy(aggregated_array)
    
    # Verify order matches (sanity check)
    assert len(aggregated_list) == len(set_names), \
        f"Length mismatch: {len(aggregated_list)} aggregated embeddings vs {len(set_names)} sets"
    
    # Report output information
    print(f"  Output:")
    print(f"    Aggregated embeddings shape: {aggregated_tensor.shape}")
    print(f"    Type: {type(aggregated_tensor).__name__}")
    print(f"    Order: Matches keyword_sets.keys() order (first: {set_names[0] if set_names else 'N/A'}, "
          f"last: {set_names[-1] if set_names else 'N/A'})")
    
    return aggregated_tensor


def compute_event_concept_similarity_matrix(
    event_embeddings: np.ndarray,
    concept_embeddings: np.ndarray,
    normalize: bool = True,
    similarity_metric: str = "cosine",
    verbose: bool = True,
    batch_size: Optional[int] = None,
    save_path: Optional[str] = None,
    force: bool = False,
    top_k: Optional[int] = None,
    top_percentile: Optional[float] = None,
    use_sparse: bool = False,
    device: Optional[str] = None,  # GPU device (e.g., "cuda:0", "cuda:1") or None for CPU
) -> Union[np.ndarray, Any]:  # Returns np.ndarray or scipy.sparse matrix
    """
    Compute similarity matrix between event embeddings and ontology concept embeddings.
    
    This function computes a similarity matrix where each row represents an event
    and each column represents an ontology concept. The matrix can be used to guide
    event positioning in latent space based on their similarity to ontology concepts.
    
    **Memory-efficient implementation**: For large datasets, computes similarities in
    batches and optionally saves to disk as a memory-mapped array to avoid memory errors.
    
    The similarity matrix represents each event's "profile" of similarities to
    all ontology concepts. Events with similar concept profiles should be
    positioned close together in the latent space.
    
    Parameters
    ----------
    event_embeddings : np.ndarray
        Event embeddings, shape (n_events, embedding_dim).
        These are embeddings of EHR markdown documents (events).
    concept_embeddings : np.ndarray
        Ontology concept embeddings, shape (n_concepts, embedding_dim).
        These are embeddings of ontology keywords/concepts (e.g., MONDO, SNOMED, etc.).
    normalize : bool, default True
        Whether to L2-normalize embeddings before computing similarity.
        Recommended for cosine similarity.
    similarity_metric : str, default "cosine"
        Similarity metric to use:
        - "cosine": Cosine similarity (dot product for normalized embeddings)
        - "dot": Dot product (same as cosine for normalized embeddings)
        - "euclidean": Negative euclidean distance (as similarity)
    verbose : bool, default True
        Whether to print progress information.
    batch_size : int, optional
        Number of events to process per batch. If None, auto-computes based on available memory.
        For very large datasets (millions of events), use 10000-50000.
        Default: None (auto-compute, or 10000 if save_path provided)
    save_path : str, optional
        Path to save similarity matrix. Format depends on storage mode:
        - Full matrix: .npy file (memory-mapped)
        - Top-k sparse: .npz file with 'data', 'indices', 'indptr' (scipy.sparse format)
        If provided and file exists, loads from disk instead of recomputing (unless force=True).
    force : bool, default False
        If True, recompute even if save_path exists.
    top_k : int, optional
        If provided, only store top-k most similar concepts per event.
        Dramatically reduces storage: from O(n_events * n_concepts) to O(n_events * top_k).
        Example: top_k=100 reduces 3.8TB to ~2.4GB for 6M events.
        Recommended: 50-200 for most use cases.
    top_percentile : float, optional
        If provided, only store top percentile of similarities per event (0-100).
        More adaptive than absolute threshold - automatically adjusts to each event's
        similarity distribution. Example: top_percentile=10 keeps top 10% of concepts per event.
        Can be combined with top_k (applies percentile first, then takes top-k).
        Recommended: 5-20 for most use cases.
    use_sparse : bool, default False
        If True and (top_k or top_percentile provided), returns scipy.sparse matrix.
        Much more memory-efficient for sparse similarity matrices.
    device : str, optional
        Device for GPU acceleration. Options:
        - None or "cpu": Use CPU (default)
        - "cuda:0", "cuda:1", etc.: Use specific GPU
        - "cuda": Use default GPU (cuda:0)
        GPU acceleration can provide 10-100x speedup for large similarity computations.
        Requires PyTorch with CUDA support.
    
    Returns
    -------
    np.ndarray or scipy.sparse matrix
        Similarity matrix of shape (n_events, n_concepts).
        - If top_k or top_percentile: sparse matrix (CSR format) or dense with zeros
        - If save_path provided: memory-mapped array (read-only) or sparse matrix
        Each element [i, j] is the similarity between event i and concept j.
        For cosine similarity, values range from -1.0 to 1.0.
        For euclidean, values are negative distances (higher = more similar).
    
    Examples
    --------
    >>> # Compute similarity matrix between events and ontology concepts
    >>> event_embeddings = event_dataset.embeddings  # (n_events, 1024)
    >>> concept_embeddings = keyword_embeddings  # (n_concepts, 1024)
    >>> 
    >>> # For large datasets, use batching and save to disk
    >>> similarity_matrix = compute_event_concept_similarity_matrix(
    ...     event_embeddings=event_embeddings,
    ...     concept_embeddings=concept_embeddings,
    ...     normalize=True,
    ...     similarity_metric="cosine",
    ...     batch_size=10000,  # Process 10k events at a time
    ...     save_path="similarity_matrix.npy"  # Save as memory-mapped file
    ... )
    >>> # Shape: (n_events, n_concepts) - memory-mapped, doesn't load into RAM
    >>> 
    >>> # Each row is an event's similarity profile to all ontology concepts
    >>> # Events with similar profiles should be close in latent space
    >>> event_0_profile = similarity_matrix[0, :]  # Similarities for first event
    >>> event_1_profile = similarity_matrix[1, :]  # Similarities for second event
    >>> 
    >>> # Use in training: preserve these similarity profiles in latent space
    >>> # See concept_guided_similarity_loss() in autoencoder.py
    """
    if not NUMPY_AVAILABLE:
        raise ImportError(
            "numpy is required for compute_event_concept_similarity_matrix. "
            "Please install it: pip install numpy"
        )
    
    import os
    from .utils import l2_normalize, cosine_similarity, euclidean_similarity, dot_product_similarity
    
    # Setup device for GPU acceleration
    use_gpu = False
    use_multi_gpu = False
    torch_device = None
    gpu_list = []
    
    if device is not None and device != "cpu":
        try:
            import torch
            if device == "auto" or device == "all":
                # Use all available GPUs
                if torch.cuda.is_available():
                    n_gpus = torch.cuda.device_count()
                    if n_gpus > 0:
                        gpu_list = list(range(n_gpus))
                        use_multi_gpu = True
                        use_gpu = True
                        if verbose:
                            print(f"  Using {n_gpus} GPUs: {gpu_list}")
                    else:
                        if verbose:
                            print(f"  Warning: No GPUs available, using CPU")
                        use_gpu = False
                else:
                    if verbose:
                        print(f"  Warning: CUDA not available, using CPU")
                    use_gpu = False
            elif device == "cuda":
                # Use default GPU
                torch_device = torch.device("cuda:0")
                if torch.cuda.is_available() and torch_device.type == "cuda":
                    use_gpu = True
                    gpu_list = [0]
                    if verbose:
                        print(f"  Using GPU: {torch_device}")
                else:
                    if verbose:
                        print(f"  Warning: GPU requested but not available, using CPU")
                    use_gpu = False
                    torch_device = None
            else:
                # Specific GPU device
                torch_device = torch.device(device)
                if torch.cuda.is_available() and torch_device.type == "cuda":
                    use_gpu = True
                    gpu_id = torch_device.index if torch_device.index is not None else 0
                    gpu_list = [gpu_id]
                    if verbose:
                        print(f"  Using GPU: {torch_device}")
                else:
                    if verbose:
                        print(f"  Warning: GPU requested but not available, using CPU")
                    use_gpu = False
                    torch_device = None
        except ImportError:
            if verbose:
                print(f"  Warning: PyTorch not available, using CPU")
            use_gpu = False
            torch_device = None
    else:
        use_gpu = False
        torch_device = None
    
    # Check if we can load from disk
    if save_path is not None and os.path.exists(save_path) and not force:
        if verbose:
            print(f"Loading similarity matrix from {save_path}...")
        try:
            # Try loading as sparse matrix first (.npz)
            if save_path.endswith('.npz'):
                try:
                    from scipy import sparse
                    similarity_matrix = sparse.load_npz(save_path)
                    if verbose:
                        print(f"  Loaded sparse matrix: shape {similarity_matrix.shape}, nnz={similarity_matrix.nnz:,}")
                        print(f"  File size: {os.path.getsize(save_path) / 1e9:.2f} GB")
                    return similarity_matrix
                except ImportError:
                    if verbose:
                        print("  scipy not available, trying as dense array...")
                except Exception:
                    if verbose:
                        print("  Not a sparse matrix, trying as dense array...")
            
            # Load as memory-mapped array (read-only, doesn't load into RAM)
            similarity_matrix = np.load(save_path, mmap_mode='r')
            if verbose:
                print(f"  Loaded memory-mapped array: shape {similarity_matrix.shape}")
                print(f"  Memory-mapped (read-only) - file size: {os.path.getsize(save_path) / 1e9:.2f} GB")
            return similarity_matrix
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load from {save_path}: {e}")
                print(f"  Will recompute instead.")
    
    # Validate input shapes
    if len(event_embeddings.shape) != 2:
        raise ValueError(
            f"event_embeddings must be 2D array, got shape {event_embeddings.shape}"
        )
    if len(concept_embeddings.shape) != 2:
        raise ValueError(
            f"concept_embeddings must be 2D array, got shape {concept_embeddings.shape}"
        )
    if event_embeddings.shape[1] != concept_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions must match: "
            f"event_embeddings.shape[1]={event_embeddings.shape[1]} != "
            f"concept_embeddings.shape[1]={concept_embeddings.shape[1]}"
        )
    
    n_events = event_embeddings.shape[0]
    n_concepts = concept_embeddings.shape[0]
    embedding_dim = event_embeddings.shape[1]
    
    # Estimate memory requirements
    matrix_size_gb = (n_events * n_concepts * 4) / 1e9  # float32 = 4 bytes
    
    # Determine storage mode
    use_top_k = top_k is not None and top_k > 0
    use_percentile = top_percentile is not None and 0 < top_percentile <= 100
    sparse_mode = use_top_k or use_percentile
    
    # Validate percentile
    if use_percentile:
        if top_percentile <= 0 or top_percentile > 100:
            raise ValueError(
                f"top_percentile must be between 0 and 100, got {top_percentile}"
            )
    
    if use_top_k:
        # Top-k storage: n_events * top_k * 4 bytes (indices + values)
        sparse_size_gb = (n_events * top_k * 8) / 1e9  # indices (int32) + values (float32)
    else:
        sparse_size_gb = matrix_size_gb
    
    if verbose:
        print(f"\nComputing event-concept similarity matrix:")
        print(f"  Event embeddings: {n_events:,} events × {embedding_dim} dim")
        print(f"  Concept embeddings: {n_concepts:,} concepts × {embedding_dim} dim")
        print(f"  Similarity metric: {similarity_metric}")
        print(f"  Normalize: {normalize}")
        print(f"  Full matrix size: {matrix_size_gb:.2f} GB")
        if use_top_k:
            print(f"  Top-k mode: storing top {top_k} concepts per event")
            print(f"  Sparse storage size: ~{sparse_size_gb:.2f} GB ({sparse_size_gb/matrix_size_gb*100:.1f}% of full)")
        if use_percentile:
            print(f"  Percentile mode: storing top {top_percentile}% of similarities per event")
        if sparse_mode and use_sparse:
            print(f"  Using scipy.sparse format for efficient storage")
    
    # Auto-determine batch size if not provided
    if batch_size is None:
        if save_path is not None:
            # If saving to disk, use larger batches
            batch_size = 10000
        else:
            # Estimate based on available memory (assume 8GB available for computation)
            # Each batch needs: batch_size * n_concepts * 4 bytes
            available_gb = 8.0  # Conservative estimate
            max_batch_size = int((available_gb * 1e9) / (n_concepts * 4))
            batch_size = min(10000, max(1000, max_batch_size))
    
    if verbose:
        print(f"  Using batch size: {batch_size:,} events per batch")
        print(f"  Total batches: {(n_events + batch_size - 1) // batch_size}")
    
    # Normalize concept embeddings once (they're reused across all batches)
    concept_emb = np.ascontiguousarray(concept_embeddings, dtype=np.float32)
    if normalize:
        concept_emb = l2_normalize(concept_emb)
    
    # Move concept embeddings to GPU(s) if using GPU
    concept_emb_torch = None
    concept_emb_torch_list = []
    if use_gpu:
        import torch
        if use_multi_gpu:
            # Copy concept embeddings to all GPUs
            # Ensure writable copy to avoid PyTorch warning
            if not concept_emb.flags.writeable:
                concept_emb = concept_emb.copy()
            for gpu_id in gpu_list:
                gpu_device = torch.device(f"cuda:{gpu_id}")
                concept_emb_torch_gpu = torch.from_numpy(concept_emb).to(gpu_device)
                concept_emb_torch_list.append(concept_emb_torch_gpu)
            if verbose:
                print(f"  Loaded concept embeddings on {len(gpu_list)} GPUs")
        else:
            # Single GPU
            # Ensure writable copy to avoid PyTorch warning
            if not concept_emb.flags.writeable:
                concept_emb = concept_emb.copy()
            concept_emb_torch = torch.from_numpy(concept_emb).to(torch_device)
            concept_emb_torch_list = [concept_emb_torch]
            # concept_emb is already normalized if normalize=True, so no need to normalize again
    
    # Prepare output based on storage mode
    if sparse_mode:
        # Sparse storage: collect data, indices, indptr for CSR format
        if use_sparse:
            try:
                from scipy import sparse
                SCIPY_AVAILABLE = True
            except ImportError:
                SCIPY_AVAILABLE = False
                if verbose:
                    print("Warning: scipy not available, using dense array with zeros for sparse mode")
                use_sparse = False
        else:
            SCIPY_AVAILABLE = False
        
        if use_sparse and SCIPY_AVAILABLE:
            # Use scipy.sparse CSR format
            # For memory efficiency with save_path, write incrementally to disk
            if save_path is not None:
                # Use temporary files to accumulate sparse data incrementally
                import tempfile
                temp_dir = os.path.dirname(save_path) or '.'
                temp_data_file = os.path.join(temp_dir, f".similarity_data_{os.getpid()}.tmp")
                temp_indices_file = os.path.join(temp_dir, f".similarity_indices_{os.getpid()}.tmp")
                temp_indptr_file = os.path.join(temp_dir, f".similarity_indptr_{os.getpid()}.tmp")
                
                # Open files for binary writing
                data_fp = open(temp_data_file, 'wb')
                indices_fp = open(temp_indices_file, 'wb')
                indptr_fp = open(temp_indptr_file, 'wb')
                
                # Write initial indptr value
                np.array([0], dtype=np.int64).tofile(indptr_fp)
                
                data_list = None  # Don't accumulate in memory
                indices_list = None
                indptr_count = np.int64(0)
                use_incremental_disk = True
                
                if verbose:
                    print(f"  Using incremental disk writing for sparse data")
            else:
                # In-memory accumulation (for when no save_path)
                data_list = []
                indices_list = []
                indptr = [0]
                use_incremental_disk = False
            
            use_memmap = False  # Sparse matrices are not memory-mapped
            
            if verbose:
                mode_str = []
                if use_top_k:
                    mode_str.append(f"top_k={top_k}")
                if use_percentile:
                    mode_str.append(f"top_percentile={top_percentile}%")
                print(f"  Using sparse CSR format ({', '.join(mode_str)})")
        else:
            # Dense array with zeros (less efficient but works without scipy)
            if save_path is not None and not save_path.endswith('.npz'):
                # For sparse, we need .npz format
                save_path = save_path.replace('.npy', '.npz')
            similarity_matrix = np.zeros((n_events, n_concepts), dtype=np.float32)
            use_memmap = False
    else:
        # Dense storage: full matrix
        if save_path is not None:
            # Create memory-mapped file for output
            if verbose:
                print(f"  Creating memory-mapped output file: {save_path}")
            # Pre-allocate file
            similarity_matrix = np.lib.format.open_memmap(
                save_path,
                mode='w+',
                dtype=np.float32,
                shape=(n_events, n_concepts)
            )
            use_memmap = True
        else:
            # Check if we can fit in memory
            if matrix_size_gb > 10.0:  # More than 10GB
                raise MemoryError(
                    f"Similarity matrix would require {matrix_size_gb:.2f} GB of memory. "
                    f"Options:\n"
                    f"  1. Use top_k parameter (e.g., top_k=100) to store only top similarities\n"
                    f"  2. Use save_path to save as memory-mapped file\n"
                    f"  3. Use on-the-fly computation (don't store matrix, compute during training)"
                )
            similarity_matrix = np.zeros((n_events, n_concepts), dtype=np.float32)
            use_memmap = False
    
    # Process events in batches
    try:
        from tqdm.auto import tqdm
        batch_iterator = tqdm(
            range(0, n_events, batch_size),
            desc="Computing similarities",
            disable=not verbose
        )
    except ImportError:
        batch_iterator = range(0, n_events, batch_size)
    
    batch_count = 0
    
    # For multi-GPU, split batches across GPUs
    if use_multi_gpu:
        # Process batches in parallel across GPUs
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def process_batch_on_gpu(batch_start, batch_end, gpu_id, concept_emb_torch_gpu):
            """Process a single batch on a specific GPU."""
            import torch
            gpu_device = torch.device(f"cuda:{gpu_id}")
            
            # Slice event embeddings directly (lazy, no copy yet)
            event_batch = event_embeddings[batch_start:batch_end]
            # Convert to contiguous array only when needed, and ensure writable
            event_batch_array = np.ascontiguousarray(event_batch, dtype=np.float32)
            if not event_batch_array.flags.writeable:
                event_batch_array = event_batch_array.copy()
            event_batch_torch = torch.from_numpy(event_batch_array).to(gpu_device, non_blocking=True)
            
            if normalize:
                event_batch_torch = torch.nn.functional.normalize(event_batch_torch, p=2, dim=1)
            
            # Compute similarity on GPU
            if similarity_metric == "cosine" or similarity_metric == "dot":
                batch_similarities_torch = torch.mm(event_batch_torch, concept_emb_torch_gpu.t())
            elif similarity_metric == "euclidean":
                dot_products = torch.mm(event_batch_torch, concept_emb_torch_gpu.t())
                distances_sq = 2 - 2 * dot_products
                batch_similarities_torch = -torch.sqrt(distances_sq.clamp(min=0))
            else:
                raise ValueError(f"Unknown similarity_metric: {similarity_metric}")
            
            # Move to CPU asynchronously to avoid blocking
            batch_similarities_cpu = batch_similarities_torch.cpu().numpy()
            return batch_start, batch_end, batch_similarities_cpu
        
        # Process batches in parallel across GPUs with controlled concurrency
        # In multi-GPU mode, batch_size is per-GPU, so we can process multiple batches in parallel
        n_batches = (n_events + batch_size - 1) // batch_size
        total_events_per_round = batch_size * len(gpu_list)  # Events processed per round
        if verbose:
            print(f"  Processing {n_batches} batches across {len(gpu_list)} GPUs")
            print(f"  Batch size: {batch_size:,} events per GPU ({total_events_per_round:,} events per round)")
        
        # For sparse mode with in-order requirement, we must limit in-flight futures
        # because each batch result is ~batch_size * n_concepts * 4 bytes (~600 MB)
        # and ALL pending futures hold their results in CPU memory
        is_sparse_scipy = sparse_mode and use_sparse and SCIPY_AVAILABLE
        
        if is_sparse_scipy:
            # For sparse mode: balance memory usage with GPU utilization
            # Each in-flight batch = batch_size * n_concepts * 4 bytes
            # With 4 GPUs, we can process more batches in parallel while managing memory
            # Increase from 2 to min(4, len(gpu_list)) to better utilize all GPUs
            # This allows ~2.4 GB CPU RAM for in-flight batches (4 batches * 600MB)
            max_in_flight = min(len(gpu_list), 4)  # Use all GPUs, but cap at 4 for memory safety
            max_queue_size = max_in_flight
            if verbose:
                print(f"  Sparse mode: processing up to {max_in_flight} batches in parallel ({max_in_flight * batch_size * n_concepts * 4 / 1e9:.2f} GB CPU RAM)")
        else:
            # Dense memmap mode: can write out-of-order, more in flight is OK
            max_in_flight = len(gpu_list) * 2
            max_queue_size = 4
        
        # Use ThreadPoolExecutor to process batches in parallel
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
            pending_futures = {}  # future -> batch_start
            batch_idx = 0
            
            # For sparse mode: need in-order processing
            next_batch_to_process = 0
            batch_results_queue = {}  # Only used for sparse mode
            
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=n_batches, desc="Computing similarities", disable=not verbose)
            except ImportError:
                pbar = None
            
            def submit_batch():
                nonlocal batch_idx
                if batch_idx >= n_batches:
                    return False
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_events)
                if batch_start >= n_events:
                    return False
                gpu_id = gpu_list[batch_idx % len(gpu_list)]
                concept_emb_torch_gpu = concept_emb_torch_list[gpu_list.index(gpu_id)]
                future = executor.submit(process_batch_on_gpu, batch_start, batch_end, gpu_id, concept_emb_torch_gpu)
                pending_futures[future] = batch_start
                batch_idx += 1
                return True
            
            # Submit initial batch of work
            for _ in range(min(max_in_flight, n_batches)):
                if not submit_batch():
                    break
            
            # Process results as they complete
            completed_count = 0
            while completed_count < n_batches and pending_futures:
                # Wait for next completed future
                done_future = next(as_completed(pending_futures))
                batch_start = pending_futures.pop(done_future)
                
                try:
                    batch_start_result, batch_end, batch_similarities = done_future.result()
                    completed_count += 1
                    batch_size_actual = batch_similarities.shape[0]
                    
                    # MEMORY-MAPPED MODE: Write immediately at correct offset (no queue needed!)
                    if use_memmap and not is_sparse_scipy:
                        # Can write out-of-order directly to memmap
                        if sparse_mode:
                            # Dense storage with zeros
                            for i in range(batch_size_actual):
                                row_sims = batch_similarities[i, :].copy()
                                if use_percentile:
                                    percentile_threshold = np.percentile(row_sims, 100 - top_percentile)
                                    row_sims[row_sims < percentile_threshold] = 0.0
                                if use_top_k:
                                    top_k_actual = min(top_k, n_concepts)
                                    top_k_idx = np.argpartition(row_sims, -top_k_actual)[-top_k_actual:]
                                    mask = np.zeros(n_concepts, dtype=bool)
                                    mask[top_k_idx] = True
                                    row_sims[~mask] = 0.0
                                similarity_matrix[batch_start_result + i, :] = row_sims
                        else:
                            similarity_matrix[batch_start_result:batch_end, :] = batch_similarities
                        
                        del batch_similarities
                        similarity_matrix.flush()
                        
                        if pbar:
                            pbar.update(1)
                        
                        # Submit next batch immediately
                        submit_batch()
                        continue
                    
                    # SPARSE MODE: Need in-order processing, use limited queue
                    batch_results_queue[batch_start_result] = (batch_end, batch_similarities)
                    
                    # Process batches in order as they become available
                    while next_batch_to_process in batch_results_queue:
                        batch_start_process = next_batch_to_process
                        batch_end_process, batch_similarities_process = batch_results_queue.pop(batch_start_process)
                        batch_size_actual = batch_similarities_process.shape[0]
                        
                        # Process sparse data (scipy.sparse format)
                        for i in range(batch_size_actual):
                            row_sims = batch_similarities_process[i, :]
                            
                            # Apply percentile filtering if specified
                            if use_percentile:
                                percentile_threshold = np.percentile(row_sims, 100 - top_percentile)
                                mask = row_sims >= percentile_threshold
                            else:
                                mask = np.ones(n_concepts, dtype=bool)
                            
                            # Get top-k if specified
                            if use_top_k:
                                valid_indices = np.where(mask)[0]
                                if len(valid_indices) > 0:
                                    valid_sims = row_sims[valid_indices]
                                    top_k_actual = min(top_k, len(valid_indices))
                                    top_k_idx = np.argpartition(valid_sims, -top_k_actual)[-top_k_actual:]
                                    top_k_indices = valid_indices[top_k_idx]
                                    top_k_values = row_sims[top_k_indices]
                                    
                                    sort_idx = np.argsort(top_k_values)[::-1]
                                    if use_incremental_disk:
                                        top_k_values[sort_idx].astype(np.float32).tofile(data_fp)
                                        top_k_indices[sort_idx].astype(np.int32).tofile(indices_fp)
                                        indptr_count += len(top_k_values[sort_idx])
                                        np.array([indptr_count], dtype=np.int64).tofile(indptr_fp)
                                    else:
                                        data_list.extend(top_k_values[sort_idx])
                                        indices_list.extend(top_k_indices[sort_idx])
                                        indptr.append(len(data_list))
                                else:
                                    # No valid similarities - still need indptr entry
                                    if use_incremental_disk:
                                        np.array([indptr_count], dtype=np.int64).tofile(indptr_fp)
                                    else:
                                        indptr.append(len(data_list))
                            else:
                                # Only percentile, no top-k
                                valid_indices = np.where(mask)[0]
                                if use_incremental_disk:
                                    row_sims[valid_indices].astype(np.float32).tofile(data_fp)
                                    valid_indices.astype(np.int32).tofile(indices_fp)
                                    indptr_count += len(valid_indices)
                                    np.array([indptr_count], dtype=np.int64).tofile(indptr_fp)
                                else:
                                    data_list.extend(row_sims[valid_indices])
                                    indices_list.extend(valid_indices)
                                    indptr.append(len(data_list))
                        
                        # Free memory immediately
                        del batch_similarities_process
                        next_batch_to_process += batch_size
                        
                        if pbar:
                            pbar.update(1)
                    
                    # Submit next batch, but limit queue size to prevent memory explosion
                    if len(batch_results_queue) < max_queue_size:
                        submit_batch()
                    
                except Exception as e:
                    if verbose:
                        print(f"  Error processing batch starting at {batch_start}: {e}")
                    raise
            
            if pbar:
                pbar.close()
            
            # All batches should be processed by now, but handle any remaining
            # (This should be empty if the loop worked correctly)
            if len(batch_results_queue) > 0:
                if verbose:
                    print(f"  Warning: {len(batch_results_queue)} batches remaining in queue")
                # Process any remaining batches
                for batch_start in sorted(batch_results_queue.keys()):
                    batch_end, batch_similarities = batch_results_queue.pop(batch_start)
                    batch_size_actual = batch_similarities.shape[0]
                    
                    if sparse_mode:
                        if use_sparse and SCIPY_AVAILABLE:
                            for i in range(batch_size_actual):
                                row_sims = batch_similarities[i, :]
                                if use_percentile:
                                    percentile_threshold = np.percentile(row_sims, 100 - top_percentile)
                                    mask = row_sims >= percentile_threshold
                                else:
                                    mask = np.ones(n_concepts, dtype=bool)
                                if use_top_k:
                                    valid_indices = np.where(mask)[0]
                                    if len(valid_indices) > 0:
                                        valid_sims = row_sims[valid_indices]
                                        top_k_actual = min(top_k, len(valid_indices))
                                        top_k_idx = np.argpartition(valid_sims, -top_k_actual)[-top_k_actual:]
                                        top_k_indices = valid_indices[top_k_idx]
                                        top_k_values = row_sims[top_k_indices]
                                        sort_idx = np.argsort(top_k_values)[::-1]
                                        if use_incremental_disk:
                                            # Write to disk immediately
                                            top_k_values[sort_idx].astype(np.float32).tofile(data_fp)
                                            top_k_indices[sort_idx].astype(np.int32).tofile(indices_fp)
                                            indptr_count += len(top_k_values[sort_idx])
                                            np.array([indptr_count], dtype=np.int64).tofile(indptr_fp)
                                        else:
                                            data_list.extend(top_k_values[sort_idx])
                                            indices_list.extend(top_k_indices[sort_idx])
                                            indptr.append(len(data_list))
                                    else:
                                        # No valid similarities - still need indptr entry
                                        if use_incremental_disk:
                                            np.array([indptr_count], dtype=np.int64).tofile(indptr_fp)
                                        else:
                                            indptr.append(len(data_list))
                                else:
                                    valid_indices = np.where(mask)[0]
                                    if use_incremental_disk:
                                        # Write to disk immediately
                                        row_sims[valid_indices].astype(np.float32).tofile(data_fp)
                                        valid_indices.astype(np.int32).tofile(indices_fp)
                                        indptr_count += len(valid_indices)
                                        np.array([indptr_count], dtype=np.int64).tofile(indptr_fp)
                                    else:
                                        data_list.extend(row_sims[valid_indices])
                                        indices_list.extend(valid_indices)
                                        indptr.append(len(data_list))
                        else:
                            for i in range(batch_size_actual):
                                row_sims = batch_similarities[i, :].copy()
                                if use_percentile:
                                    percentile_threshold = np.percentile(row_sims, 100 - top_percentile)
                                    row_sims[row_sims < percentile_threshold] = 0.0
                                if use_top_k:
                                    top_k_actual = min(top_k, n_concepts)
                                    top_k_idx = np.argpartition(row_sims, -top_k_actual)[-top_k_actual:]
                                    mask = np.zeros(n_concepts, dtype=bool)
                                    mask[top_k_idx] = True
                                    row_sims[~mask] = 0.0
                                similarity_matrix[batch_start + i, :] = row_sims
                    else:
                        similarity_matrix[batch_start:batch_end, :] = batch_similarities
                    
                    del batch_similarities
                    if use_memmap:
                        similarity_matrix.flush()
        
        # Clear GPU cache
        if use_gpu:
            import torch
            for gpu_id in gpu_list:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()
    
    # Single GPU or CPU path
    else:
        # Initialize incremental disk writing flag for single-GPU path
        use_incremental_disk = False
        if sparse_mode and use_sparse and SCIPY_AVAILABLE and save_path is not None:
            # Use temporary files to accumulate sparse data incrementally
            import tempfile
            temp_dir = os.path.dirname(save_path) or '.'
            temp_data_file = os.path.join(temp_dir, f".similarity_data_{os.getpid()}.tmp")
            temp_indices_file = os.path.join(temp_dir, f".similarity_indices_{os.getpid()}.tmp")
            temp_indptr_file = os.path.join(temp_dir, f".similarity_indptr_{os.getpid()}.tmp")
            
            # Open files for binary writing
            data_fp = open(temp_data_file, 'wb')
            indices_fp = open(temp_indices_file, 'wb')
            indptr_fp = open(temp_indptr_file, 'wb')
            
            # Write initial indptr value
            np.array([0], dtype=np.int64).tofile(indptr_fp)
            
            data_list = None  # Don't accumulate in memory
            indices_list = None
            indptr_count = np.int64(0)
            use_incremental_disk = True
            
            if verbose:
                print(f"  Using incremental disk writing for sparse data")
        for batch_start in batch_iterator:
            batch_end = min(batch_start + batch_size, n_events)
            event_batch = event_embeddings[batch_start:batch_end]
            batch_count += 1
            
            # Compute similarity for this batch
            if use_gpu:
                import torch
                # GPU-accelerated computation
                event_batch_torch = torch.from_numpy(
                    np.ascontiguousarray(event_batch, dtype=np.float32)
                ).to(torch_device)
                
                if normalize:
                    event_batch_torch = torch.nn.functional.normalize(event_batch_torch, p=2, dim=1)
                
                # Compute similarity on GPU
                concept_emb_torch = concept_emb_torch_list[0]  # Single GPU
                if similarity_metric == "cosine" or similarity_metric == "dot":
                    # For normalized embeddings, cosine = dot product
                    batch_similarities_torch = torch.mm(event_batch_torch, concept_emb_torch.t())
                elif similarity_metric == "euclidean":
                    # Euclidean similarity: negative distance
                    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a@b
                    # For normalized: ||a||^2 = ||b||^2 = 1, so ||a-b||^2 = 2 - 2*a@b
                    dot_products = torch.mm(event_batch_torch, concept_emb_torch.t())
                    distances_sq = 2 - 2 * dot_products
                    batch_similarities_torch = -torch.sqrt(distances_sq.clamp(min=0))
                else:
                    raise ValueError(
                        f"Unknown similarity_metric: {similarity_metric}. "
                        f"Use 'cosine', 'dot', or 'euclidean'."
                    )
                
                # Convert back to numpy for sparse operations
                batch_similarities = batch_similarities_torch.cpu().numpy()
                
                # Clear GPU cache periodically to avoid OOM (every 10 batches)
                if batch_count % 10 == 0:
                    torch.cuda.empty_cache()
        else:
            # CPU computation (original path)
            event_batch = np.ascontiguousarray(event_batch, dtype=np.float32)
            if normalize:
                event_batch = l2_normalize(event_batch)
            
            if similarity_metric == "cosine":
                batch_similarities = cosine_similarity(event_batch, concept_emb)
            elif similarity_metric == "dot":
                batch_similarities = dot_product_similarity(event_batch, concept_emb)
            elif similarity_metric == "euclidean":
                batch_similarities = euclidean_similarity(event_batch, concept_emb)
            else:
                raise ValueError(
                    f"Unknown similarity_metric: {similarity_metric}. "
                    f"Use 'cosine', 'dot', or 'euclidean'."
                )
        
        # Apply threshold and/or top-k filtering
        if sparse_mode:
            batch_size_actual = batch_similarities.shape[0]
            
            if use_sparse and SCIPY_AVAILABLE:
                # Collect sparse data for CSR format
                for i in range(batch_size_actual):
                    row_sims = batch_similarities[i, :]
                    
                    # Apply percentile filtering if specified
                    if use_percentile:
                        # Compute percentile threshold for this event
                        percentile_threshold = np.percentile(row_sims, 100 - top_percentile)
                        mask = row_sims >= percentile_threshold
                    else:
                        mask = np.ones(n_concepts, dtype=bool)
                    
                    # Get top-k if specified
                    if use_top_k:
                        # Get top-k indices (after percentile filtering)
                        valid_indices = np.where(mask)[0]
                        if len(valid_indices) > 0:
                            valid_sims = row_sims[valid_indices]
                            top_k_actual = min(top_k, len(valid_indices))
                            top_k_idx = np.argpartition(valid_sims, -top_k_actual)[-top_k_actual:]
                            top_k_indices = valid_indices[top_k_idx]
                            top_k_values = row_sims[top_k_indices]
                            
                            # Sort by similarity (descending)
                            sort_idx = np.argsort(top_k_values)[::-1]
                            if use_incremental_disk:
                                # Write to disk immediately
                                top_k_values[sort_idx].astype(np.float32).tofile(data_fp)
                                top_k_indices[sort_idx].astype(np.int32).tofile(indices_fp)
                                indptr_count += len(top_k_values[sort_idx])
                                np.array([indptr_count], dtype=np.int64).tofile(indptr_fp)
                            else:
                                data_list.extend(top_k_values[sort_idx])
                                indices_list.extend(top_k_indices[sort_idx])
                                indptr.append(len(data_list))
                        else:
                            # No valid similarities - still need indptr entry
                            if use_incremental_disk:
                                np.array([indptr_count], dtype=np.int64).tofile(indptr_fp)
                            else:
                                indptr.append(len(data_list))
                    else:
                        # Only percentile, no top-k
                        valid_indices = np.where(mask)[0]
                        if use_incremental_disk:
                            # Write to disk immediately
                            row_sims[valid_indices].astype(np.float32).tofile(data_fp)
                            valid_indices.astype(np.int32).tofile(indices_fp)
                            indptr_count += len(valid_indices)
                            np.array([indptr_count], dtype=np.int32).tofile(indptr_fp)
                        else:
                            data_list.extend(row_sims[valid_indices])
                            indices_list.extend(valid_indices)
                            indptr.append(len(data_list))
            else:
                # Dense storage with zeros for non-top-k/non-percentile values
                for i in range(batch_size_actual):
                    row_sims = batch_similarities[i, :].copy()
                    
                    # Apply percentile filtering if specified
                    if use_percentile:
                        # Compute percentile threshold for this event
                        percentile_threshold = np.percentile(row_sims, 100 - top_percentile)
                        row_sims[row_sims < percentile_threshold] = 0.0
                    
                    # Apply top-k if specified
                    if use_top_k:
                        # Zero out all except top-k
                        top_k_actual = min(top_k, n_concepts)
                        top_k_idx = np.argpartition(row_sims, -top_k_actual)[-top_k_actual:]
                        mask = np.zeros(n_concepts, dtype=bool)
                        mask[top_k_idx] = True
                        row_sims[~mask] = 0.0
                    
                    similarity_matrix[batch_start + i, :] = row_sims
        else:
            # Dense storage: write full batch
            similarity_matrix[batch_start:batch_end, :] = batch_similarities
        
        # Flush if using memory-mapped file
        if use_memmap:
            similarity_matrix.flush()
    
    # Clear GPU cache if using GPU
    if use_gpu:
        import torch
        torch.cuda.empty_cache()
    
    # Finalize sparse matrix if using scipy.sparse
    if sparse_mode and use_sparse and SCIPY_AVAILABLE:
        if use_incremental_disk:
            # Close temporary files and load from disk
            data_fp.close()
            indices_fp.close()
            indptr_fp.close()
            
            # Load data from temporary files
            if verbose:
                print(f"  Loading sparse data from temporary files...")
            data_array = np.fromfile(temp_data_file, dtype=np.float32)
            indices_array = np.fromfile(temp_indices_file, dtype=np.int32)
            indptr_array = np.fromfile(temp_indptr_file, dtype=np.int64)
            
            # Clean up temporary files
            os.remove(temp_data_file)
            os.remove(temp_indices_file)
            os.remove(temp_indptr_file)
            
            if verbose:
                print(f"  Constructing sparse matrix from {len(data_array):,} non-zero elements...")
        else:
            # Convert from in-memory lists
            data_array = np.array(data_list, dtype=np.float32)
            indices_array = np.array(indices_list, dtype=np.int32)
            indptr_array = np.array(indptr, dtype=np.int64)
        
        # Construct CSR sparse matrix
        similarity_matrix = sparse.csr_matrix(
            (data_array, indices_array, indptr_array),
            shape=(n_events, n_concepts)
        )
        
        if save_path is not None:
            # Save sparse matrix
            if not save_path.endswith('.npz'):
                save_path = save_path.replace('.npy', '.npz')
            sparse.save_npz(save_path, similarity_matrix)
            if verbose:
                file_size_gb = os.path.getsize(save_path) / 1e9
                print(f"  Saved sparse matrix to: {save_path} ({file_size_gb:.2f} GB)")
                print(f"  Non-zero elements: {similarity_matrix.nnz:,} ({similarity_matrix.nnz/(n_events*n_concepts)*100:.2f}% dense)")
    
    if verbose:
        if sparse_mode and use_sparse and SCIPY_AVAILABLE:
            # Sparse matrix stats
            sample_size = min(1000, similarity_matrix.nnz)
            if sample_size > 0:
                sample_data = similarity_matrix.data[:sample_size]
                print(f"  Similarity range (sample): [{sample_data.min():.4f}, {sample_data.max():.4f}]")
                print(f"  Mean similarity (sample): {sample_data.mean():.4f}")
        else:
            print(f"  Output similarity matrix shape: {similarity_matrix.shape}")
            if not use_memmap:
                # Compute stats on non-zero values if sparse mode
                if sparse_mode:
                    non_zero = similarity_matrix[similarity_matrix != 0]
                    if len(non_zero) > 0:
                        print(f"  Non-zero elements: {len(non_zero):,} ({len(non_zero)/(n_events*n_concepts)*100:.2f}% dense)")
                        print(f"  Similarity range (non-zero): [{non_zero.min():.4f}, {non_zero.max():.4f}]")
                        print(f"  Mean similarity (non-zero): {non_zero.mean():.4f}")
                else:
                    print(f"  Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
                    print(f"  Mean similarity: {similarity_matrix.mean():.4f}")
            else:
                # For memory-mapped, compute stats on a sample
                sample_size = min(10000, n_events)
                sample_indices = np.random.choice(n_events, sample_size, replace=False)
                sample = similarity_matrix[sample_indices, :]
                print(f"  Similarity range (sample): [{sample.min():.4f}, {sample.max():.4f}]")
                print(f"  Mean similarity (sample): {sample.mean():.4f}")
                print(f"  Saved to: {save_path} ({os.path.getsize(save_path) / 1e9:.2f} GB)")
    
    return similarity_matrix


import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse

def get_event_concept_attention_weights(
    event_embeddings,  # (n_events, embedding_dim) - all event embeddings
    concept_embeddings,  # (n_concepts, embedding_dim) - all concept embeddings
    concept_similarity_matrix=None,  # Pre-computed similarity matrix (n_events, n_concepts) or path
    temperature=0.1,
    top_k=None,  # If provided, only return top-k concepts per event
    concept_indices=None,  # If provided, only return attention for these specific concept indices
    batch_size=1280,
    device="cuda:0",
    normalize=True,
    show_progress=True,
):
    """
    Extract attention weights for each event's top concepts.
    
    This function recomputes the attention weights that were used during training
    to enhance event embeddings with concept information.
    
    Parameters
    ----------
    event_embeddings : np.ndarray or torch.Tensor
        Event embeddings, shape (n_events, embedding_dim)
    concept_embeddings : np.ndarray or torch.Tensor
        Concept embeddings, shape (n_concepts, embedding_dim)
    concept_similarity_matrix : np.ndarray, scipy.sparse matrix, str, or None
        Pre-computed similarity matrix or path to saved matrix.
        If None, computes similarities on-the-fly.
    temperature : float, default 0.1
        Temperature for attention softmax (should match training)
    top_k : int, optional
        If provided, only return top-k concepts per event.
        Ignored if concept_indices is provided.
    concept_indices : array-like, optional
        If provided, only return attention weights for these specific concept indices.
        Shape: (n_selected_concepts,). Overrides top_k if both are provided.
    batch_size : int, default 1280
        Batch size for processing
    device : str, default "cuda:0"
        Device to use for computation
    normalize : bool, default True
        Whether to normalize embeddings before computing similarities
    show_progress : bool, default True
        Show progress bar
    
    Returns
    -------
    attention_weights : np.ndarray
        Attention weights:
        - If concept_indices provided: shape (n_events, len(concept_indices))
        - If top_k provided: shape (n_events, top_k)
        - Otherwise: shape (n_events, n_concepts)
    concept_indices_out : np.ndarray, optional
        Concept indices corresponding to the returned attention weights:
        - If concept_indices provided: returns concept_indices as array
        - If top_k provided: returns top-k indices per event, shape (n_events, top_k)
        - Otherwise: None
    """
    from tqdm.auto import tqdm
    
    # Handle concept_indices parameter
    if concept_indices is not None:
        # Convert to numpy array and validate
        if isinstance(concept_indices, (list, tuple)):
            concept_indices = np.array(concept_indices)
        elif isinstance(concept_indices, torch.Tensor):
            concept_indices = concept_indices.cpu().numpy()
        elif not isinstance(concept_indices, np.ndarray):
            concept_indices = np.array(concept_indices)
        
        # Ensure it's 1D
        concept_indices = concept_indices.flatten()
        
        # Validate indices are within bounds (will check n_concepts later)
        use_specific_indices = True
        # Override top_k if concept_indices is provided
        if top_k is not None:
            top_k = None
    else:
        use_specific_indices = False
    
    # Convert to tensors
    if isinstance(event_embeddings, np.ndarray):
        event_emb_tensor = torch.from_numpy(event_embeddings).float().to(device)
    else:
        event_emb_tensor = event_embeddings.float().to(device)
    
    if isinstance(concept_embeddings, np.ndarray):
        concept_emb_tensor = torch.from_numpy(concept_embeddings).float().to(device)
    else:
        concept_emb_tensor = concept_embeddings.float().to(device)
    
    # Normalize if needed
    if normalize:
        event_emb_tensor = F.normalize(event_emb_tensor, p=2, dim=1)
        concept_emb_tensor = F.normalize(concept_emb_tensor, p=2, dim=1)
    
    n_events = event_emb_tensor.shape[0]
    n_concepts = concept_emb_tensor.shape[0]
    
    # Validate concept_indices after we know n_concepts
    if use_specific_indices:
        if np.any(concept_indices < 0) or np.any(concept_indices >= n_concepts):
            raise ValueError(f"concept_indices must be in range [0, {n_concepts}), "
                           f"but found values outside this range")
        concept_indices_tensor = torch.from_numpy(concept_indices).long().to(device)
    
    # Load similarity matrix if provided
    similarity_matrix_tensor = None
    similarity_matrix_sparse = None
    similarity_matrix_mmap = None
    
    if concept_similarity_matrix is not None:
        if isinstance(concept_similarity_matrix, str):
            # Load from file
            if concept_similarity_matrix.endswith('.npz'):
                similarity_matrix_sparse = sparse.load_npz(concept_similarity_matrix)
            else:
                similarity_matrix_mmap = np.load(concept_similarity_matrix, mmap_mode='r')
        elif isinstance(concept_similarity_matrix, sparse.spmatrix):
            similarity_matrix_sparse = concept_similarity_matrix
        elif isinstance(concept_similarity_matrix, np.ndarray):
            if isinstance(concept_similarity_matrix, np.memmap):
                similarity_matrix_mmap = concept_similarity_matrix
            else:
                similarity_matrix_tensor = torch.from_numpy(concept_similarity_matrix).float().to(device)
        elif isinstance(concept_similarity_matrix, torch.Tensor):
            similarity_matrix_tensor = concept_similarity_matrix.float().to(device)
    
    # Process in batches
    all_attention_weights = []
    all_top_indices = [] if (top_k is not None and not use_specific_indices) else None
    
    n_batches = (n_events + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing attention weights")
    
    for batch_idx in iterator:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_events)
        batch_event_emb = event_emb_tensor[batch_start:batch_end]
        batch_size_actual = batch_end - batch_start
        
        # Get similarities for this batch
        if similarity_matrix_tensor is not None:
            # Dense tensor
            batch_similarities = similarity_matrix_tensor[batch_start:batch_end, :]
        elif similarity_matrix_sparse is not None:
            # Sparse matrix
            event_indices_np = np.arange(batch_start, batch_end)
            rows_batch = similarity_matrix_sparse[event_indices_np, :]
            rows_dense = rows_batch.toarray()
            batch_similarities = torch.from_numpy(rows_dense).float().to(device)
        elif similarity_matrix_mmap is not None:
            # Memory-mapped array
            event_indices_np = np.arange(batch_start, batch_end)
            rows_batch = similarity_matrix_mmap[event_indices_np, :]
            batch_similarities = torch.from_numpy(rows_batch).float().to(device)
        else:
            # Compute on-the-fly
            batch_similarities = torch.mm(batch_event_emb, concept_emb_tensor.t())
        
        # Compute attention weights
        if use_specific_indices:
            # Compute full attention weights, then extract specific indices
            full_attention = F.softmax(batch_similarities / temperature, dim=1)  # (batch_size, n_concepts)
            # Extract attention for specified concept indices
            selected_attention = full_attention[:, concept_indices_tensor]  # (batch_size, len(concept_indices))
            all_attention_weights.append(selected_attention.cpu().numpy())
        elif top_k is not None and top_k < n_concepts:
            # Get top-k
            top_k_values, top_k_indices = torch.topk(batch_similarities, k=top_k, dim=1)
            top_k_attention = F.softmax(top_k_values / temperature, dim=1)
            
            # Store top-k attention and indices
            all_attention_weights.append(top_k_attention.cpu().numpy())
            all_top_indices.append(top_k_indices.cpu().numpy())
        else:
            # Use all concepts
            attention_weights = F.softmax(batch_similarities / temperature, dim=1)
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    # Concatenate results
    attention_weights = np.concatenate(all_attention_weights, axis=0)
    
    # Return appropriate indices
    if use_specific_indices:
        return attention_weights, concept_indices
    elif top_k is not None:
        top_concept_indices = np.concatenate(all_top_indices, axis=0)
        return attention_weights, top_concept_indices
    else:
        return attention_weights


def analyze_attention_weights(
    attention_weights: np.ndarray,
    concept_indices: np.ndarray,
    concept_names: Optional[Union[List[str], np.ndarray]] = None,
    top_n: Optional[int] = None,
    bottom_n: Optional[int] = None,
    sort_by: str = "total_attention_weight",
) -> pd.DataFrame:
    """
    Analyze attention weights to find the most important concepts.
    
    This function aggregates attention weights across all events to identify
    which concepts receive the most attention overall. It computes:
    - Count: How many events have this concept in their top-k
    - Total attention weight: Sum of attention weights across all events
    - Mean attention weight: Average attention weight per event
    
    Parameters
    ----------
    attention_weights : np.ndarray
        Attention weights, shape (n_events, n_concepts_per_event)
        Typically from get_event_concept_attention_weights with top_k or concept_indices
    concept_indices : np.ndarray
        Concept indices corresponding to attention_weights, shape (n_events, n_concepts_per_event)
        or (n_events, top_k) if using top_k mode
    concept_names : list or np.ndarray, optional
        Names/labels for concepts. If provided, must have length >= max(concept_indices) + 1
        If None, concept_name column will contain string representations of indices
    top_n : int, optional, default None
        Number of top concepts to return (sorted by sort_by in descending order).
        If None, returns all concepts (unless bottom_n is specified, in which case only bottom_n is returned).
    bottom_n : int, optional, default None
        Number of bottom concepts to return (sorted by sort_by in ascending order).
        If provided, these will be appended to the top_n results.
        The combined result will have top_n rows (descending) followed by bottom_n rows (ascending).
        If both top_n and bottom_n are None, returns all concepts sorted by sort_by in descending order.
    sort_by : str, default "total_attention_weight"
        Column to sort by. Options: "total_attention_weight", "count", "mean_attention_weight", 
        "min_attention_weight", "max_attention_weight", "mean_attention_weight_scaled"
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - concept_index: Index of the concept
        - concept_name: Name of the concept (if concept_names provided)
        - count: Number of events that have this concept in their top-k
        - total_attention_weight: Sum of attention weights across all events
        - mean_attention_weight: Average attention weight per event (total / count)
        - min_attention_weight: Minimum attention weight across all events for this concept
        - max_attention_weight: Maximum attention weight across all events for this concept
        - mean_attention_weight_scaled: mean_attention_weight / log(count + 1). Higher scores indicate 
          concepts that are important when they appear but appear rarely (rare but important).
          This is mean attention weight scaled by log frequency to favor rare but important concepts.
        
        If only top_n is specified: sorted by sort_by in descending order, limited to top_n rows.
        If only bottom_n is specified: sorted by sort_by in ascending order, limited to bottom_n rows.
        If both are specified: top_n rows (descending) followed by bottom_n rows (ascending).
        If both are None: returns all concepts sorted by sort_by in descending order.
    
    Examples
    --------
    >>> attention_weights, top_indices = get_event_concept_attention_weights(
    ...     event_embeddings=event_embeddings,
    ...     concept_embeddings=concept_embeddings,
    ...     top_k=100
    ... )
    >>> # Get all concepts (default: both top_n and bottom_n are None)
    >>> all_concepts = analyze_attention_weights(
    ...     attention_weights=attention_weights,
    ...     concept_indices=top_indices,
    ...     concept_names=keyword_list
    ... )
    >>> # Result: all concepts sorted by sort_by in descending order
    >>> 
    >>> # Get top 50 concepts
    >>> top_concepts = analyze_attention_weights(
    ...     attention_weights=attention_weights,
    ...     concept_indices=top_indices,
    ...     concept_names=keyword_list,
    ...     top_n=50
    ... )
    >>> print(top_concepts.head())
    
    >>> # Get both top and bottom concepts
    >>> concepts = analyze_attention_weights(
    ...     attention_weights=attention_weights,
    ...     concept_indices=top_indices,
    ...     concept_names=keyword_list,
    ...     top_n=50,
    ...     bottom_n=20
    ... )
    >>> # Result: 50 top concepts (descending) + 20 bottom concepts (ascending)
    
    >>> # Get only bottom concepts
    >>> bottom_concepts = analyze_attention_weights(
    ...     attention_weights=attention_weights,
    ...     concept_indices=top_indices,
    ...     concept_names=keyword_list,
    ...     top_n=None,
    ...     bottom_n=20
    ... )
    >>> # Result: 20 bottom concepts (ascending)
    """
    # Print initial status
    n_events, n_concepts_per_event = attention_weights.shape
    print(f"📊 Analyzing attention weights:")
    print(f"   Input shape: {attention_weights.shape} ({n_events:,} events × {n_concepts_per_event} concepts per event)")
    print(f"   Total attention weight entries: {attention_weights.size:,}")
    
    # Validate inputs early (before any processing)
    if concept_names is not None:
        try:
            concept_names_array = np.array(concept_names)
            # Check if we can access it (will raise NameError if variable is not defined)
            _ = len(concept_names_array)
            print(f"   Concept names provided: {len(concept_names_array):,} concepts")
        except (NameError, TypeError) as e:
            raise NameError(
                f"concept_names parameter references an undefined variable. "
                f"Please ensure the variable is defined before calling this function. "
                f"Original error: {e}"
            ) from e
    else:
        print(f"   Concept names: None (will use 'concept_{idx}' format)")
    
    # Flatten arrays for aggregation
    print(f"🔄 Flattening arrays for aggregation...")
    flat_indices = concept_indices.ravel()  # shape (num_events * n_concepts_per_event,)
    flat_weights = attention_weights.ravel()  # shape (num_events * n_concepts_per_event,)
    print(f"   Flattened shape: {flat_indices.shape[0]:,} entries")
    
    # OPTIMIZED: Use bincount for count/sum (very fast), then efficient min/max using sort+reduceat
    print(f"📈 Computing aggregations (count, sum, min, max)...")
    max_idx = flat_indices.max()
    print(f"   Max concept index: {max_idx:,}")
    if max_idx < 2**31:
        # Fast path: use bincount for count and sum
        counts = np.bincount(flat_indices, minlength=max_idx+1)
        weights_sum = np.bincount(flat_indices, weights=flat_weights, minlength=max_idx+1)
        unique_indices = np.arange(len(counts))
        mask = counts > 0
        unique_indices = unique_indices[mask]
        counts = counts[mask]
        weights_sum = weights_sum[mask]
        
        n_unique_concepts = len(unique_indices)
        print(f"   Found {n_unique_concepts:,} unique concepts")
        
        # For very large datasets, use np.minimum.at / np.maximum.at instead of sorting
        # Sorting 6+ billion entries is extremely slow, while np.minimum.at is O(n) with C-level loops
        if flat_indices.size > 100_000_000:  # For datasets > 100M entries, use at operations
            print(f"   Computing min/max using vectorized operations (large dataset: {flat_indices.size:,} entries)...")
            # Initialize min/max arrays
            min_weights = np.full(max_idx + 1, np.inf, dtype=flat_weights.dtype)
            max_weights = np.full(max_idx + 1, -np.inf, dtype=flat_weights.dtype)
            
            # Use at operations (O(n) with C-level loops, much faster than sorting for large n)
            np.minimum.at(min_weights, flat_indices, flat_weights)
            np.maximum.at(max_weights, flat_indices, flat_weights)
            
            # Extract only for unique concepts and replace inf/-inf with NaN for concepts that don't exist
            min_weights = min_weights[unique_indices]
            max_weights = max_weights[unique_indices]
            # Replace inf/-inf with NaN (shouldn't happen since we mask by counts > 0, but safety check)
            min_weights = np.where(np.isinf(min_weights), np.nan, min_weights)
            max_weights = np.where(np.isinf(max_weights), np.nan, max_weights)
        else:
            # For smaller datasets, sorting + reduceat is still faster
            print(f"   Computing min/max (sorting {flat_indices.size:,} entries)...")
            # OPTIMIZED min/max: Sort by concept_index and use reduceat (faster for smaller datasets)
            sort_idx = np.argsort(flat_indices)
            sorted_indices = flat_indices[sort_idx]
            sorted_weights = flat_weights[sort_idx]
            
            # Find positions where each concept starts (for reduceat)
            _, unique_positions = np.unique(sorted_indices, return_index=True)
            
            # Use reduceat for efficient min/max computation
            min_weights_reduceat = np.minimum.reduceat(sorted_weights, unique_positions)
            max_weights_reduceat = np.maximum.reduceat(sorted_weights, unique_positions)
            
            # Handle the last group (reduceat doesn't include the end)
            if len(unique_positions) > 0:
                last_start = unique_positions[-1]
                min_weights_reduceat[-1] = np.min(sorted_weights[last_start:])
                max_weights_reduceat[-1] = np.max(sorted_weights[last_start:])
            
            # Map back to unique_indices order
            sorted_unique = sorted_indices[unique_positions]
            idx_map = np.searchsorted(sorted_unique, unique_indices)
            min_weights = min_weights_reduceat[idx_map]
            max_weights = max_weights_reduceat[idx_map]
    else:
        # Fallback for very large indices
        unique_indices, idx_inv, counts = np.unique(flat_indices, return_inverse=True, return_counts=True)
        weights_sum = np.zeros_like(unique_indices, dtype=flat_weights.dtype)
        np.add.at(weights_sum, idx_inv, flat_weights)
        
        # For min/max with large indices, use np.minimum.at / np.maximum.at for very large datasets
        if flat_indices.size > 100_000_000:  # For datasets > 100M entries, use at operations
            print(f"   Computing min/max using vectorized operations (large dataset: {flat_indices.size:,} entries)...")
            # Initialize min/max arrays (use max_idx+1 to cover all possible indices)
            max_global_idx = max(unique_indices.max(), max_idx) if len(unique_indices) > 0 else max_idx
            min_weights_full = np.full(max_global_idx + 1, np.inf, dtype=flat_weights.dtype)
            max_weights_full = np.full(max_global_idx + 1, -np.inf, dtype=flat_weights.dtype)
            
            # Use at operations directly on flat_indices
            np.minimum.at(min_weights_full, flat_indices, flat_weights)
            np.maximum.at(max_weights_full, flat_indices, flat_weights)
            
            # Extract only for unique_indices
            min_weights = min_weights_full[unique_indices]
            max_weights = max_weights_full[unique_indices]
            # Replace inf/-inf with NaN (shouldn't happen since we mask by counts > 0, but safety check)
            min_weights = np.where(np.isinf(min_weights), np.nan, min_weights)
            max_weights = np.where(np.isinf(max_weights), np.nan, max_weights)
        else:
            # For smaller datasets, sorting + reduceat is still faster
            print(f"   Computing min/max (sorting {flat_indices.size:,} entries)...")
            sort_idx = np.argsort(flat_indices)
            sorted_indices = flat_indices[sort_idx]
            sorted_weights = flat_weights[sort_idx]
            _, unique_positions = np.unique(sorted_indices, return_index=True)
            min_weights_reduceat = np.minimum.reduceat(sorted_weights, unique_positions)
            max_weights_reduceat = np.maximum.reduceat(sorted_weights, unique_positions)
            if len(unique_positions) > 0:
                last_start = unique_positions[-1]
                min_weights_reduceat[-1] = np.min(sorted_weights[last_start:])
                max_weights_reduceat[-1] = np.max(sorted_weights[last_start:])
            sorted_unique = sorted_indices[unique_positions]
            # Map to unique_indices order
            idx_map = np.searchsorted(sorted_unique, unique_indices)
            min_weights = min_weights_reduceat[idx_map]
            max_weights = max_weights_reduceat[idx_map]
    
    # Compute mean attention weight
    print(f"✅ Aggregation complete. Computing statistics...")
    mean_weights = weights_sum / counts
    
    # Compute scaled mean attention weight: mean_attention_weight / log(count + 1)
    # This favors concepts with high mean attention but low frequency (rare but important)
    mean_attention_weight_scaled = mean_weights / np.log(counts + 1)
    
    # Create temporary arrays for sorting
    print(f"🔍 Sorting by '{sort_by}'...")
    if sort_by == "total_attention_weight":
        sort_values = weights_sum
    elif sort_by == "count":
        sort_values = counts
    elif sort_by == "mean_attention_weight":
        sort_values = mean_weights
    elif sort_by == "min_attention_weight":
        sort_values = min_weights
    elif sort_by == "max_attention_weight":
        sort_values = max_weights
    elif sort_by == "mean_attention_weight_scaled":
        sort_values = mean_attention_weight_scaled
    else:
        raise ValueError(f"sort_by must be one of ['total_attention_weight', 'count', 'mean_attention_weight', 'min_attention_weight', 'max_attention_weight', 'mean_attention_weight_scaled'], got '{sort_by}'")
    
    # Get top N (descending order) if requested, or all if both top_n and bottom_n are None
    if top_n is not None and top_n > 0:
        top_n_actual = min(top_n, len(unique_indices))
        print(f"   Selecting top {top_n_actual:,} concepts (using optimized partial sort)...")
        # OPTIMIZED: Use argpartition for partial sort (much faster than full sort)
        # Get indices of top_n_actual largest values
        partition_idx = np.argpartition(sort_values, -top_n_actual)[-top_n_actual:]
        # Sort only the top_n_actual indices (small sort, very fast)
        top_sort_idx = partition_idx[np.argsort(sort_values[partition_idx])[::-1]]
        top_indices_sorted = unique_indices[top_sort_idx]
        top_counts_sorted = counts[top_sort_idx]
        top_weights_sorted = weights_sum[top_sort_idx]
        top_mean_weights_sorted = mean_weights[top_sort_idx]
        top_min_weights_sorted = min_weights[top_sort_idx]
        top_max_weights_sorted = max_weights[top_sort_idx]
        top_mean_attention_weight_scaled_sorted = mean_attention_weight_scaled[top_sort_idx]
    elif top_n is None and bottom_n is None:
        # Both are None: return all concepts sorted by sort_by in descending order
        print(f"   Sorting all {len(unique_indices):,} concepts...")
        top_sort_idx = np.argsort(sort_values)[::-1]  # All indices, descending
        top_indices_sorted = unique_indices[top_sort_idx]
        top_counts_sorted = counts[top_sort_idx]
        top_weights_sorted = weights_sum[top_sort_idx]
        top_mean_weights_sorted = mean_weights[top_sort_idx]
        top_min_weights_sorted = min_weights[top_sort_idx]
        top_max_weights_sorted = max_weights[top_sort_idx]
        top_mean_attention_weight_scaled_sorted = mean_attention_weight_scaled[top_sort_idx]
    else:
        top_indices_sorted = None
        top_sort_idx = None
        top_min_weights_sorted = None
        top_max_weights_sorted = None
        top_mean_attention_weight_scaled_sorted = None
    
    # Get bottom N if requested
    if bottom_n is not None and bottom_n > 0:
        bottom_n_actual = min(bottom_n, len(unique_indices))
        print(f"   Selecting bottom {bottom_n_actual:,} concepts...")
        # If top_n was also requested, exclude those indices from bottom_n
        if top_sort_idx is not None:
            remaining_mask = np.ones(len(unique_indices), dtype=bool)
            remaining_mask[top_sort_idx] = False
            remaining_indices = np.where(remaining_mask)[0]
        else:
            # No top_n, so all indices are available for bottom_n
            remaining_indices = np.arange(len(unique_indices))
        
        if len(remaining_indices) > 0:
            remaining_sort_values = sort_values[remaining_indices]
            # OPTIMIZED: Use argpartition for partial sort (much faster than full sort)
            partition_idx_local = np.argpartition(remaining_sort_values, bottom_n_actual)[:bottom_n_actual]
            # Sort only the bottom_n_actual indices (small sort, very fast)
            bottom_sort_idx_local = partition_idx_local[np.argsort(remaining_sort_values[partition_idx_local])]
            bottom_sort_idx = remaining_indices[bottom_sort_idx_local]
            
            bottom_indices_sorted = unique_indices[bottom_sort_idx]
            bottom_counts_sorted = counts[bottom_sort_idx]
            bottom_weights_sorted = weights_sum[bottom_sort_idx]
            bottom_mean_weights_sorted = mean_weights[bottom_sort_idx]
            bottom_min_weights_sorted = min_weights[bottom_sort_idx]
            bottom_max_weights_sorted = max_weights[bottom_sort_idx]
            bottom_mean_attention_weight_scaled_sorted = mean_attention_weight_scaled[bottom_sort_idx]
        else:
            # No remaining indices, skip bottom_n
            bottom_indices_sorted = np.array([], dtype=unique_indices.dtype)
            bottom_counts_sorted = np.array([], dtype=counts.dtype)
            bottom_weights_sorted = np.array([], dtype=weights_sum.dtype)
            bottom_mean_weights_sorted = np.array([], dtype=mean_weights.dtype)
            bottom_min_weights_sorted = np.array([], dtype=min_weights.dtype)
            bottom_max_weights_sorted = np.array([], dtype=max_weights.dtype)
            bottom_mean_attention_weight_scaled_sorted = np.array([], dtype=mean_attention_weight_scaled.dtype)
    else:
        bottom_indices_sorted = None
    
    # Get concept names if provided
    # NOTE: concept_names should be in the same order as concept_embeddings used in get_event_concept_attention_weights.
    # For example, if concept_embeddings[i] corresponds to concept_names[i], then concept_indices[j] = i
    # means that concept_names[i] is the name for concept index i.
    print(f"📝 Mapping concept names...")
    if concept_names is not None:
        concept_names_array = np.array(concept_names)
        if len(concept_names_array) <= max(unique_indices):
            raise ValueError(
                f"concept_names must have length > max(concept_indices)={max(unique_indices)}, "
                f"but got length {len(concept_names_array)}"
            )
        if top_indices_sorted is not None:
            top_concept_names = concept_names_array[top_indices_sorted]
        else:
            top_concept_names = None
        if bottom_indices_sorted is not None and len(bottom_indices_sorted) > 0:
            bottom_concept_names = concept_names_array[bottom_indices_sorted]
        else:
            bottom_concept_names = None
    else:
        if top_indices_sorted is not None:
            top_concept_names = [f"concept_{idx}" for idx in top_indices_sorted]
        else:
            top_concept_names = None
        if bottom_indices_sorted is not None and len(bottom_indices_sorted) > 0:
            bottom_concept_names = [f"concept_{idx}" for idx in bottom_indices_sorted]
        else:
            bottom_concept_names = None
    
    # Create DataFrame for top N if requested
    print(f"📋 Creating DataFrame...")
    if top_indices_sorted is not None:
        print(f"   Top concepts: {len(top_indices_sorted):,} rows")
        top_df = pd.DataFrame({
            "concept_index": top_indices_sorted,
            "concept_name": top_concept_names,
            "count": top_counts_sorted,
            "total_attention_weight": top_weights_sorted,
            "mean_attention_weight": top_mean_weights_sorted,
            "min_attention_weight": top_min_weights_sorted,
            "max_attention_weight": top_max_weights_sorted,
            "mean_attention_weight_scaled": top_mean_attention_weight_scaled_sorted,
        })
        # Already sorted by sort_by in descending order, no need to sort again
    else:
        top_df = None
    
    # Create DataFrame for bottom N if requested
    if bottom_indices_sorted is not None and len(bottom_indices_sorted) > 0:
        print(f"   Bottom concepts: {len(bottom_indices_sorted):,} rows")
        bottom_df = pd.DataFrame({
            "concept_index": bottom_indices_sorted,
            "concept_name": bottom_concept_names,
            "count": bottom_counts_sorted,
            "total_attention_weight": bottom_weights_sorted,
            "mean_attention_weight": bottom_mean_weights_sorted,
            "min_attention_weight": bottom_min_weights_sorted,
            "max_attention_weight": bottom_max_weights_sorted,
            "mean_attention_weight_scaled": bottom_mean_attention_weight_scaled_sorted,
        })
        # Already sorted by sort_by in ascending order, no need to sort again
    else:
        bottom_df = None
    
    # Combine results
    if top_df is not None and bottom_df is not None:
        # Concatenate top and bottom
        print(f"   Combining top and bottom results...")
        result_df = pd.concat([top_df, bottom_df], ignore_index=True)
        print(f"✅ Final result: {len(result_df):,} rows (top {len(top_df):,} + bottom {len(bottom_df):,})")
    elif top_df is not None:
        result_df = top_df
        print(f"✅ Final result: {len(result_df):,} rows")
    elif bottom_df is not None:
        result_df = bottom_df
        print(f"✅ Final result: {len(result_df):,} rows")
    else:
        # This shouldn't happen due to validation, but handle it gracefully
        result_df = pd.DataFrame(columns=["concept_index", "concept_name", "count", 
                                         "total_attention_weight", "mean_attention_weight"])
        print(f"⚠️  Warning: No results to return (empty DataFrame)")
    
    return result_df

def get_attention_weight_for_concept(
    attention_weights,
    concept_idx,
    top_indices=None,
    verbose=True
):
    """
    Computes a vector of attention weights for all events for a given concept.
    If top_indices is provided (event-by-k matrix of top-k concept indices per event),
    finds attention for that concept in the top-k columns, else attempts columnwise extraction.

    Parameters
    ----------
    attention_weights : np.ndarray
        Array of shape (n_events, k) or (n_events, n_concepts) with per-event attention weights.
    concept_idx : int
        Concept index to extract.
    top_indices : np.ndarray or None
        Array of shape (n_events, k) with the top-k concept indices for each event.
    verbose : bool
        Whether to print messages about found/missing concepts.

    Returns
    -------
    attn_weight_vec : np.ndarray
        Array of length n_events, with attention weights (zeros for events w/o this concept).
    """

    if top_indices is not None:
        row_pos, col_pos = np.where(top_indices == concept_idx)

        if len(row_pos) == 0:
            raise ValueError(f"Concept {concept_idx} not found in any event's top-k concepts")
        
        attn_weight_vec = np.zeros(attention_weights.shape[0])
        for event_idx, col_idx in zip(row_pos, col_pos):
            attn_weight_vec[event_idx] = attention_weights[event_idx, col_idx]
        
        if verbose:
            print(f"Found concept {concept_idx} in {len(row_pos)} events out of {attention_weights.shape[0]} total events")
    else:
        # No mapping provided; try direct positional indexing, but warn if index out of bounds
        if hasattr(attention_weights, "shape") and concept_idx < attention_weights.shape[1]:
            attn_weight_vec = attention_weights[:, concept_idx]
        else:
            raise ValueError(f"Concept {concept_idx} not found in attention_weights columns (no mapping available)")
    return attn_weight_vec


def ontology_to_gene_phenotype_matrix(
    ontology_owl: Optional[Any] = None,
    ontology_graph: Optional[nx.DiGraph] = None,
    ontology_dict: Optional[Dict[str, List[str]]] = None,
    gene_annotation_property: Optional[str] = None,
    phenotype_label_property: str = "label",
    gene_id_property: Optional[str] = None,
    phenotype_id_property: Optional[str] = None,
    relationship_types: Optional[List[str]] = None,
    include_ancestors: bool = True,
    include_descendants: bool = False,
    score_by_depth: bool = False,
    verbose: bool = True,
    return_sparse: bool = True,
) -> Tuple[Union[np.ndarray, Any], Dict[str, Any]]:
    """
    Convert an ontology with gene annotations into a gene-phenotype matrix.
    
    This function extracts gene-phenotype associations from an ontology and creates
    a sparse matrix where rows represent genes and columns represent phenotypes.
    The matrix can be used for downstream analysis, similarity computation, or
    integration with other gene-phenotype datasets.
    
    Parameters
    ----------
    ontology_owl : owlready2.Ontology, optional
        OWL ontology object (from owlready2) with gene annotations.
        If provided, gene annotations will be extracted from OWL properties.
    ontology_graph : nx.DiGraph, optional
        NetworkX directed graph representing the ontology with gene annotations.
        Nodes should have gene annotation attributes or edges should connect genes to phenotypes.
        If None and ontology_owl is provided, will extract graph from OWL.
    ontology_dict : Dict[str, List[str]], optional
        Dictionary mapping phenotype terms to lists of associated gene IDs.
        Format: {phenotype_id: [gene_id1, gene_id2, ...]}
        Only used if neither ontology_owl nor ontology_graph is provided.
    gene_annotation_property : str, optional
        Name of the OWL property that links phenotypes to genes.
        Common properties: "has_gene", "associated_with_gene", "gene_annotation", etc.
        If None, will attempt to auto-detect from common property names.
    phenotype_label_property : str, default "label"
        Name of the OWL property to use for phenotype labels/names.
        Default: "label" (standard OWL label property).
    gene_id_property : str, optional
        Name of the OWL property to use for gene identifiers.
        If None, will use the gene class name or IRI.
    phenotype_id_property : str, optional
        Name of the OWL property to use for phenotype identifiers.
        If None, will use the phenotype class name or IRI.
    relationship_types : List[str], optional
        Types of relationships to consider when extracting gene-phenotype associations.
        Only used when include_ancestors or include_descendants is True.
        Common types: ["is_a", "part_of", "has_part"]
        If None, uses ["is_a"] by default.
    include_ancestors : bool, default True
        If True, includes associations from ancestor phenotypes (broader terms).
        For example, if gene G is associated with "diabetes" and "diabetes" is_a "disease",
        then G will also be associated with "disease" if include_ancestors=True.
    include_descendants : bool, default False
        If True, includes associations from descendant phenotypes (more specific terms).
        For example, if gene G is associated with "disease" and "diabetes" is_a "disease",
        then G will also be associated with "diabetes" if include_descendants=True.
    score_by_depth : bool, default False
        If True, weights associations by ontology depth (deeper = higher score).
        Only applies when include_ancestors or include_descendants is True.
    verbose : bool, default True
        Whether to print progress information.
    return_sparse : bool, default True
        If True, returns scipy.sparse matrix (CSR format) for memory efficiency.
        If False, returns dense numpy array.
    
    Returns
    -------
    matrix : np.ndarray or scipy.sparse.spmatrix
        Gene-phenotype matrix of shape (n_genes, n_phenotypes).
        Values are binary (0/1) or weighted scores if score_by_depth=True.
        - If return_sparse=True: scipy.sparse.csr_matrix
        - If return_sparse=False: np.ndarray (float32)
    metadata : dict
        Dictionary containing:
        - 'gene_ids': List[str] - Gene identifiers (row order)
        - 'phenotype_ids': List[str] - Phenotype identifiers (column order)
        - 'phenotype_labels': List[str] - Phenotype labels/names (column order)
        - 'gene_id_to_idx': Dict[str, int] - Mapping from gene ID to row index
        - 'phenotype_id_to_idx': Dict[str, int] - Mapping from phenotype ID to column index
        - 'n_genes': int - Number of unique genes
        - 'n_phenotypes': int - Number of unique phenotypes
        - 'n_associations': int - Total number of gene-phenotype associations
        - 'sparsity': float - Matrix sparsity (fraction of zeros)
    
    Examples
    --------
    >>> # From OWL ontology with gene annotations
    >>> mondo_graph, mondo_owl = load_mondo_ontology(return_owl=True)
    >>> matrix, metadata = ontology_to_gene_phenotype_matrix(
    ...     ontology_owl=mondo_owl,
    ...     gene_annotation_property="has_gene",
    ...     include_ancestors=True
    ... )
    >>> print(f"Matrix shape: {matrix.shape}")
    >>> print(f"Genes: {metadata['n_genes']}, Phenotypes: {metadata['n_phenotypes']}")
    >>> 
    >>> # From NetworkX graph with gene annotations
    >>> matrix, metadata = ontology_to_gene_phenotype_matrix(
    ...     ontology_graph=mondo_graph,
    ...     include_ancestors=True
    ... )
    >>> 
    >>> # From dictionary format
    >>> ontology_dict = {
    ...     "diabetes": ["GENE1", "GENE2"],
    ...     "cancer": ["GENE2", "GENE3"]
    ... }
    >>> matrix, metadata = ontology_to_gene_phenotype_matrix(
    ...     ontology_dict=ontology_dict
    ... )
    """
    if not NUMPY_AVAILABLE:
        raise ImportError(
            "numpy is required for ontology_to_gene_phenotype_matrix. "
            "Please install it: pip install numpy"
        )
    
    try:
        from scipy import sparse
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        if return_sparse:
            if verbose:
                print("Warning: scipy not available, returning dense matrix instead")
            return_sparse = False
    
    # Extract gene-phenotype associations based on input type
    gene_phenotype_associations = []
    
    if ontology_owl is not None:
        if not OWLREADY2_AVAILABLE:
            raise ImportError(
                "owlready2 is required when using ontology_owl. "
                "Please install it: pip install owlready2"
            )
        
        if verbose:
            print("Extracting gene-phenotype associations from OWL ontology...")
        
        # Auto-detect gene annotation property if not provided
        if gene_annotation_property is None:
            common_property_names = [
                "has_gene", "associated_with_gene", "gene_annotation",
                "hasGene", "associatedWithGene", "geneAnnotation",
                "gene", "genes", "annotated_with_gene"
            ]
            for prop_name in common_property_names:
                for prop in ontology_owl.properties():
                    if prop.name == prop_name or prop_name in str(prop.label).lower():
                        gene_annotation_property = prop.name
                        if verbose:
                            print(f"  Auto-detected gene annotation property: {gene_annotation_property}")
                        break
                if gene_annotation_property:
                    break
        
        if gene_annotation_property is None:
            raise ValueError(
                "Could not find gene annotation property. "
                "Please specify gene_annotation_property parameter."
            )
        
        # Get all classes (phenotypes)
        all_classes = list(ontology_owl.classes())
        
        if verbose:
            print(f"  Processing {len(all_classes):,} ontology classes...")
        
        # Extract associations
        for cls in all_classes:
            # Get phenotype identifier
            if phenotype_id_property:
                try:
                    phenotype_id = getattr(cls, phenotype_id_property).first()
                    if not phenotype_id:
                        phenotype_id = str(cls).split('.')[-1]
                except:
                    phenotype_id = str(cls).split('.')[-1]
            else:
                phenotype_id = str(cls).split('.')[-1]
            
            # Get phenotype label
            try:
                phenotype_label = cls.label.first() if cls.label else phenotype_id
                if ':' in str(phenotype_label):
                    phenotype_label = str(phenotype_label).split(':')[-1]
            except:
                phenotype_label = phenotype_id
            
            # Get gene annotations for this phenotype
            try:
                gene_prop = getattr(ontology_owl, gene_annotation_property, None)
                if gene_prop is None:
                    # Try to find property by name
                    for prop in ontology_owl.properties():
                        if prop.name == gene_annotation_property:
                            gene_prop = prop
                            break
                
                if gene_prop is not None:
                    genes = getattr(cls, gene_annotation_property, [])
                    if not isinstance(genes, list):
                        genes = [genes] if genes else []
                    
                    for gene in genes:
                        # Get gene identifier
                        if gene_id_property:
                            try:
                                gene_id = getattr(gene, gene_id_property).first()
                                if not gene_id:
                                    gene_id = str(gene).split('.')[-1]
                            except:
                                gene_id = str(gene).split('.')[-1]
                        else:
                            gene_id = str(gene).split('.')[-1]
                        
                        gene_phenotype_associations.append((gene_id, phenotype_id, phenotype_label))
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not extract genes for {phenotype_id}: {e}")
                continue
        
        if verbose:
            print(f"  Found {len(gene_phenotype_associations):,} direct gene-phenotype associations")
        
        # Build ontology graph if needed for ancestor/descendant expansion
        if include_ancestors or include_descendants:
            if ontology_graph is None:
                if verbose:
                    print("  Building ontology graph for ancestor/descendant expansion...")
                ontology_graph = extract_graph_from_owl(
                    ontology_owl=ontology_owl,
                    relationship_types=relationship_types or ["is_a"],
                    bidirectional=False
                )
    
    elif ontology_graph is not None:
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required when using ontology_graph. "
                "Please install it: pip install networkx"
            )
        
        if verbose:
            print("Extracting gene-phenotype associations from NetworkX graph...")
        
        # Extract from graph nodes/edges
        for node in ontology_graph.nodes():
            # Check if node has gene annotations as attributes
            if 'genes' in ontology_graph.nodes[node]:
                genes = ontology_graph.nodes[node]['genes']
                if not isinstance(genes, list):
                    genes = [genes] if genes else []
                
                phenotype_id = node
                phenotype_label = ontology_graph.nodes[node].get('label', node)
                
                for gene_id in genes:
                    gene_phenotype_associations.append((gene_id, phenotype_id, phenotype_label))
            
            # Check edges for gene-phenotype relationships
            for successor in ontology_graph.successors(node):
                edge_data = ontology_graph.get_edge_data(node, successor, {})
                if edge_data.get('relationship') == 'has_gene' or 'gene' in str(edge_data.get('relationship', '')).lower():
                    gene_id = successor
                    phenotype_id = node
                    phenotype_label = ontology_graph.nodes[node].get('label', node)
                    gene_phenotype_associations.append((gene_id, phenotype_id, phenotype_label))
        
        if verbose:
            print(f"  Found {len(gene_phenotype_associations):,} direct gene-phenotype associations")
    
    elif ontology_dict is not None:
        if verbose:
            print("Extracting gene-phenotype associations from dictionary...")
        
        for phenotype_id, gene_list in ontology_dict.items():
            if not isinstance(gene_list, list):
                gene_list = [gene_list] if gene_list else []
            
            for gene_id in gene_list:
                gene_phenotype_associations.append((gene_id, phenotype_id, phenotype_id))
        
        if verbose:
            print(f"  Found {len(gene_phenotype_associations):,} gene-phenotype associations")
    
    else:
        raise ValueError(
            "Either ontology_owl, ontology_graph, or ontology_dict must be provided"
        )
    
    if len(gene_phenotype_associations) == 0:
        raise ValueError("No gene-phenotype associations found in the ontology")
    
    # Expand associations to include ancestors/descendants if requested
    if (include_ancestors or include_descendants) and ontology_graph is not None:
        if verbose:
            print("  Expanding associations to include ancestors/descendants...")
        
        expanded_associations = []
        phenotype_id_to_label = {}
        
        # Build phenotype ID to label mapping
        for gene_id, phenotype_id, phenotype_label in gene_phenotype_associations:
            phenotype_id_to_label[phenotype_id] = phenotype_label
        
        # For each association, find ancestors/descendants
        for gene_id, phenotype_id, phenotype_label in gene_phenotype_associations:
            expanded_associations.append((gene_id, phenotype_id, phenotype_label))
            
            if phenotype_id not in ontology_graph:
                continue
            
            # Find ancestors (broader terms)
            if include_ancestors:
                try:
                    ancestors = list(nx.ancestors(ontology_graph, phenotype_id))
                    for ancestor_id in ancestors:
                        ancestor_label = ontology_graph.nodes[ancestor_id].get('label', ancestor_id)
                        score = 1.0
                        if score_by_depth:
                            try:
                                depth = nx.shortest_path_length(ontology_graph, ancestor_id, phenotype_id)
                                score = 1.0 / (depth + 1)  # Deeper = lower score
                            except:
                                pass
                        expanded_associations.append((gene_id, ancestor_id, ancestor_label))
                        phenotype_id_to_label[ancestor_id] = ancestor_label
                except:
                    pass
            
            # Find descendants (more specific terms)
            if include_descendants:
                try:
                    descendants = list(nx.descendants(ontology_graph, phenotype_id))
                    for descendant_id in descendants:
                        descendant_label = ontology_graph.nodes[descendant_id].get('label', descendant_id)
                        score = 1.0
                        if score_by_depth:
                            try:
                                depth = nx.shortest_path_length(ontology_graph, phenotype_id, descendant_id)
                                score = 1.0 / (depth + 1)  # Deeper = lower score
                            except:
                                pass
                        expanded_associations.append((gene_id, descendant_id, descendant_label))
                        phenotype_id_to_label[descendant_id] = descendant_label
                except:
                    pass
        
        gene_phenotype_associations = expanded_associations
        
        if verbose:
            print(f"  Expanded to {len(gene_phenotype_associations):,} associations")
    
    # Build unique lists and mappings
    unique_genes = sorted(set(gene_id for gene_id, _, _ in gene_phenotype_associations))
    unique_phenotypes = sorted(set(phenotype_id for _, phenotype_id, _ in gene_phenotype_associations))
    
    gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(unique_genes)}
    phenotype_id_to_idx = {phenotype_id: idx for idx, phenotype_id in enumerate(unique_phenotypes)}
    
    # Build phenotype labels list
    phenotype_labels = []
    for phenotype_id in unique_phenotypes:
        # Find label for this phenotype
        label = None
        for _, pid, plabel in gene_phenotype_associations:
            if pid == phenotype_id:
                label = plabel
                break
        if label is None:
            label = phenotype_id
        phenotype_labels.append(label)
    
    # Build sparse matrix
    n_genes = len(unique_genes)
    n_phenotypes = len(unique_phenotypes)
    
    if verbose:
        print(f"\nBuilding gene-phenotype matrix:")
        print(f"  Genes: {n_genes:,}")
        print(f"  Phenotypes: {n_phenotypes:,}")
        print(f"  Associations: {len(gene_phenotype_associations):,}")
    
    # Create matrix data
    row_indices = []
    col_indices = []
    data = []
    
    for gene_id, phenotype_id, _ in gene_phenotype_associations:
        row_idx = gene_id_to_idx[gene_id]
        col_idx = phenotype_id_to_idx[phenotype_id]
        row_indices.append(row_idx)
        col_indices.append(col_idx)
        data.append(1.0)  # Binary association
    
    # Build sparse matrix
    if return_sparse and SCIPY_AVAILABLE:
        matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_genes, n_phenotypes),
            dtype=np.float32
        )
    else:
        # Dense matrix
        matrix = np.zeros((n_genes, n_phenotypes), dtype=np.float32)
        for i, j, val in zip(row_indices, col_indices, data):
            matrix[i, j] = val
    
    # Compute sparsity
    if return_sparse and SCIPY_AVAILABLE:
        sparsity = 1.0 - (matrix.nnz / (n_genes * n_phenotypes))
    else:
        sparsity = 1.0 - (np.count_nonzero(matrix) / (n_genes * n_phenotypes))
    
    # Build metadata
    metadata = {
        'gene_ids': unique_genes,
        'phenotype_ids': unique_phenotypes,
        'phenotype_labels': phenotype_labels,
        'gene_id_to_idx': gene_id_to_idx,
        'phenotype_id_to_idx': phenotype_id_to_idx,
        'n_genes': n_genes,
        'n_phenotypes': n_phenotypes,
        'n_associations': len(gene_phenotype_associations),
        'sparsity': sparsity,
    }
    
    if verbose:
        print(f"  Matrix shape: {matrix.shape}")
        print(f"  Sparsity: {sparsity:.2%}")
        if return_sparse and SCIPY_AVAILABLE:
            print(f"  Non-zero elements: {matrix.nnz:,}")
    
    return matrix, metadata


def get_ontology_id_label_mapping(
    ontology_owl: Optional[Any] = None,
    ontology_graph: Optional[nx.DiGraph] = None,
) -> Dict[str, str]:
    """
    Create a bidirectional mapping between ontology IDs and labels.
    
    For OWL ontologies like MONDO, terms can be referenced by:
    - IDs: 'MONDO:0000208'
    - Labels: 'diabetes mellitus'
    
    This function creates mappings in both directions:
    - id_to_label: Maps 'MONDO:0000208' -> 'diabetes mellitus'
    - label_to_id: Maps 'diabetes mellitus' -> 'MONDO:0000208'
    
    Parameters
    ----------
    ontology_owl : owlready2.Ontology, optional
        OWL ontology object. If provided, extracts IDs and labels from OWL.
    ontology_graph : nx.DiGraph, optional
        NetworkX graph. If provided and has node attributes, extracts from graph.
        If only graph is provided, returns empty dict (graph nodes are labels, not IDs).
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping IDs to labels: {'MONDO:0000208': 'diabetes mellitus', ...}
        Also includes reverse mapping (labels to IDs) as separate entries.
        To get reverse mapping, use: {v: k for k, v in mapping.items() if k.startswith('MONDO:')}
    """
    id_to_label = {}
    
    if ontology_owl is not None:
        if not OWLREADY2_AVAILABLE:
            raise ImportError(
                "owlready2 is required for extracting ID-label mappings. "
                "Please install it: pip install owlready2"
            )
        
        # Extract from OWL ontology
        for cls in ontology_owl.classes():
            # Get ID from IRI
            try:
                iri = str(cls.iri)
                # Extract ID from IRI (e.g., "http://purl.obolibrary.org/obo/MONDO_0000208" -> "MONDO:0000208")
                term_id = None
                if 'MONDO_' in iri:
                    mondo_num = iri.split('MONDO_')[-1].split('/')[0].split('#')[0]
                    term_id = f"MONDO:{mondo_num}"
                elif 'DOID_' in iri:
                    doid_num = iri.split('DOID_')[-1].split('/')[0].split('#')[0]
                    term_id = f"DOID:{doid_num}"
                elif 'HP_' in iri:
                    hp_num = iri.split('HP_')[-1].split('/')[0].split('#')[0]
                    term_id = f"HP:{hp_num}"
                else:
                    # Try to extract any pattern like "PREFIX_NUMBER"
                    import re
                    match = re.search(r'([A-Z]+)_(\d+)', iri)
                    if match:
                        prefix, number = match.groups()
                        term_id = f"{prefix}:{number}"
                
                if term_id is None:
                    continue
                
                # Get label - handle locstr objects
                if hasattr(cls, 'label') and cls.label:
                    label_obj = cls.label.first()
                    # Handle locstr objects (from owlready2)
                    if hasattr(label_obj, '__str__'):
                        label = str(label_obj)
                    elif hasattr(label_obj, 'first'):
                        label = label_obj.first()
                    else:
                        label = str(label_obj)
                else:
                    label = str(cls).split('.')[-1]
                
                # Clean up label - remove namespace if present, handle locstr format
                if isinstance(label, str):
                    if ':' in label:
                        label = label.split(':')[-1]
                    # Handle locstr('text', 'en') format
                    import re
                    locstr_match = re.match(r"locstr\('([^']+)'", label)
                    if locstr_match:
                        label = locstr_match.group(1)
                
                # Store both directions
                id_to_label[term_id] = label
                id_to_label[label] = term_id  # Reverse mapping
            except Exception as e:
                # Skip on error
                continue
    
    elif ontology_graph is not None:
        # Try to extract from graph node attributes if available
        for node, data in ontology_graph.nodes(data=True):
            if 'id' in data and 'label' in data:
                node_id = data['id']
                node_label = data['label']
                id_to_label[node_id] = node_label
                id_to_label[node_label] = node_id
    
    return id_to_label


def compute_pairwise_ontological_similarity(
    ontology_owl: Optional[Any] = None,
    ontology_graph: Optional[nx.DiGraph] = None,
    term_ids: Optional[List[str]] = None,
    similarity_method: str = "path_based",
    relationship_types: Optional[List[str]] = None,
    max_pairs: Optional[int] = None,
    id_label_mapping: Optional[Dict[str, str]] = None,
    min_distance: Optional[int] = None,
    max_distance: Optional[int] = None,
    as_directed: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise ontological similarity for terms in an OWL ontology.
    
    Computes similarity between all pairs of terms (or a subset) using various
    ontological similarity metrics. This can be used to validate that embedding
    similarities correlate with ontological structure.
    
    Supports both ontology IDs (e.g., 'MONDO:0000208') and labels (e.g., 'diabetes mellitus').
    If term_ids contains IDs, they will be automatically converted to labels using
    id_label_mapping (or auto-extracted from ontology_owl if provided).
    
    Parameters
    ----------
    ontology_owl : owlready2.Ontology, optional
        OWL ontology object (from owlready2). If provided, will extract graph.
    ontology_graph : nx.DiGraph, optional
        NetworkX directed graph representing the ontology.
        If None and ontology_owl is provided, will extract graph from OWL.
    term_ids : List[str], optional
        List of term identifiers to compute similarities for.
        Can be either IDs (e.g., 'MONDO:0000208') or labels (e.g., 'diabetes mellitus').
        If None, computes for all terms in the ontology.
    similarity_method : str, default "shortest_path_length"
        Method for computing ontological similarity:
        - "shortest_path_length": Uses nx.single_source_shortest_path_length - BFS-based, fastest for unweighted graphs
        - "bellman_ford": Uses nx.single_source_bellman_ford_path_length - Works with negative weights
        - "dijkstra_path": Uses nx.single_source_dijkstra_path_length - Best for weighted graphs
        - "floyd_warshall": Uses nx.floyd_warshall_numpy - Matrix-based, efficient for dense graphs or when computing all pairs
        - "lca_depth": Depth of lowest common ancestor (LCA) normalized by max depth
        - "resnik": Information content of LCA (requires IC computation)
        - "lin": Lin similarity: 2 * IC(LCA) / (IC(term1) + IC(term2))
    relationship_types : List[str], optional
        Types of relationships to consider when computing paths.
        If None, uses ["is_a"] by default.
    max_pairs : int, optional
        Maximum number of pairs to compute similarities for.
        If None, computes all pairs (can be very slow for large ontologies).
        If term_ids is provided, limits to max_pairs random pairs.
    id_label_mapping : Dict[str, str], optional
        Mapping between IDs and labels. If None and ontology_owl is provided,
        will be auto-extracted. Format: {'MONDO:0000208': 'diabetes mellitus', ...}
    min_distance : int, optional
        Minimum path distance (number of hops) to consider.
        Pairs with shorter path distances will be excluded.
        If None, no minimum distance filter is applied.
        Example: min_distance=2 excludes self-pairs and direct neighbors.
    max_distance : int, optional
        Maximum path distance (number of hops) to consider.
        Pairs with longer path distances will be excluded.
        If None, no maximum distance filter is applied.
        Example: max_distance=5 only considers pairs within 5 hops.
    as_directed : bool, default False
        If True, keeps the graph as directed (only follows edge directions).
        If False, converts to undirected graph (allows traversing up and down hierarchy).
        For ontologies with is_a relationships, False (undirected) is typically preferred
        to allow finding neighbors both up (parents) and down (children) the hierarchy.
    verbose : bool, default True
        Whether to print progress information.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - term1: First term identifier (preserves original format: ID or label)
        - term2: Second term identifier (preserves original format: ID or label)
        - similarity: Ontological similarity score (0-1, higher = more similar)
        - path_length: Shortest path length (if path_based method)
        - lca: Lowest common ancestor (if applicable)
    
    Examples
    --------
    >>> # From MONDO ontology with IDs
    >>> mondo_graph, mondo_owl = load_mondo_ontology(return_owl=True)
    >>> 
    >>> # Compute similarities for specific terms using IDs
    >>> term_list = ["MONDO:0000208", "MONDO:0005148", "MONDO:0003789"]
    >>> similarities = compute_pairwise_ontological_similarity(
    ...     ontology_graph=mondo_graph,
    ...     ontology_owl=mondo_owl,  # Needed for ID-to-label mapping
    ...     term_ids=term_list,
    ...     similarity_method="path_based"
    ... )
    >>> 
    >>> # Or using labels
    >>> term_list = ["diabetes", "type 2 diabetes", "cancer"]
    >>> similarities = compute_pairwise_ontological_similarity(
    ...     ontology_graph=mondo_graph,
    ...     term_ids=term_list,
    ...     similarity_method="path_based"
    ... )
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "networkx is required for computing ontological similarity. "
            "Please install it: pip install networkx"
        )
    
    # Extract graph if needed
    if ontology_graph is None:
        if ontology_owl is None:
            raise ValueError("Either ontology_graph or ontology_owl must be provided")
        
        if verbose:
            print("Extracting graph from OWL ontology...")
        ontology_graph = extract_graph_from_owl(
            ontology_owl=ontology_owl,
            relationship_types=relationship_types or ["is_a"],
            bidirectional=False
        )
    
    # Get ID-to-label mapping if needed
    if id_label_mapping is None and ontology_owl is not None:
        if verbose:
            print("Extracting ID-to-label mapping from ontology...")
        id_label_mapping = get_ontology_id_label_mapping(ontology_owl=ontology_owl)
        if verbose and len(id_label_mapping) > 0:
            print(f"  Found {len(id_label_mapping) // 2:,} ID-label pairs")
    elif id_label_mapping is None:
        id_label_mapping = {}
    
    # Convert term IDs to labels if needed (graph nodes are labels)
    original_term_ids = term_ids.copy() if term_ids is not None else None
    if term_ids is not None:
        # Check if any terms look like IDs (e.g., 'MONDO:0000208')
        term_ids_converted = []
        conversion_stats = {'converted': 0, 'not_found': 0, 'already_label': 0}
        for term in term_ids:
            # Check if it's an ID (contains colon and looks like ontology ID)
            if ':' in term and (term.startswith('MONDO:') or term.startswith('DOID:') or term.startswith('HP:')):
                # Try to convert ID to label
                if term in id_label_mapping:
                    label = id_label_mapping[term]
                    term_ids_converted.append(label)
                    conversion_stats['converted'] += 1
                    if verbose and conversion_stats['converted'] <= 5:
                        print(f"  Converted {term} -> {label}")
                else:
                    # ID not found in mapping
                    conversion_stats['not_found'] += 1
                    if verbose and conversion_stats['not_found'] <= 5:
                        print(f"  Warning: ID {term} not found in mapping, trying as-is")
                    # Try using as-is (might work if graph has IDs as nodes)
                    term_ids_converted.append(term)
            else:
                # Assume it's already a label
                conversion_stats['already_label'] += 1
                term_ids_converted.append(term)
        
        if verbose:
            print(f"  Conversion stats: {conversion_stats['converted']} converted, "
                  f"{conversion_stats['not_found']} not found, {conversion_stats['already_label']} already labels")
        
        term_ids = term_ids_converted
    
    # Get term list
    if term_ids is None:
        # Use all terms in the graph
        term_ids = list(ontology_graph.nodes())
        if verbose:
            print(f"Computing similarities for all {len(term_ids):,} terms in ontology...")
    else:
        # Filter to terms that exist in the graph
        # Graph nodes might be locstr objects, so we need to convert them to strings for comparison
        import re
        graph_nodes_str = {str(n) if not isinstance(n, str) else n for n in ontology_graph.nodes()}
        # Also handle locstr format in graph nodes - extract clean text
        graph_nodes_clean = {}
        for node in ontology_graph.nodes():
            node_str = str(node) if not isinstance(node, str) else node
            # Extract text from locstr('text', 'en') format
            locstr_match = re.match(r"locstr\('([^']+)'", node_str)
            if locstr_match:
                node_clean = locstr_match.group(1)
            else:
                node_clean = node_str
            graph_nodes_clean[node_clean] = node  # Map clean name to original node
        
        # Now check if converted term_ids are in the graph
        term_ids_in_graph = []
        for term in term_ids:
            # Try direct match
            if term in ontology_graph:
                term_ids_in_graph.append(term)
            # Try with cleaned graph nodes
            elif term in graph_nodes_clean:
                term_ids_in_graph.append(graph_nodes_clean[term])
            # Try string conversion
            elif str(term) in graph_nodes_str:
                # Find the original node
                for orig_node in ontology_graph.nodes():
                    if str(orig_node) == str(term):
                        term_ids_in_graph.append(orig_node)
                        break
        
        if len(term_ids_in_graph) == 0:
            # Provide detailed error message
            sample_original = original_term_ids[:5] if original_term_ids else term_ids[:5]
            sample_converted = term_ids[:5]
            sample_graph_nodes = [str(n) for n in list(ontology_graph.nodes())[:10]]
            
            # Check if we have mapping info
            mapping_info = ""
            if len(id_label_mapping) > 0:
                # Show some example mappings
                sample_mappings = list(id_label_mapping.items())[:5]
                mapping_info = f"\n  Sample ID-label mappings: {sample_mappings}"
            
            raise ValueError(
                f"No terms from term_ids found in ontology graph.\n"
                f"  Original term_ids (first 5): {sample_original}\n"
                f"  Converted to labels (first 5): {sample_converted}\n"
                f"  Graph contains {len(ontology_graph.nodes()):,} nodes.\n"
                f"  First few graph nodes (as strings): {sample_graph_nodes}\n"
                f"  ID-label mapping has {len(id_label_mapping) // 2:,} pairs.{mapping_info}\n"
                f"  If using MONDO IDs, ensure ontology_owl is provided for ID-to-label conversion."
            )
        if len(term_ids_in_graph) < len(term_ids):
            missing = set(term_ids) - set(term_ids_in_graph)
            if verbose:
                print(f"  Warning: {len(missing)} terms not found in graph (first few: {list(missing)[:5]})")
        term_ids = term_ids_in_graph
        if verbose:
            print(f"Computing similarities for {len(term_ids):,} specified terms...")
    
    # Convert to undirected graph for path computation (if using is_a relationships)
    # This allows us to traverse up and down the hierarchy
    use_bidirectional = False
    if as_directed:
        # Keep as directed graph
        G_undirected = ontology_graph
        if verbose:
            print(f"Using directed graph: {G_undirected.number_of_nodes():,} nodes, {G_undirected.number_of_edges():,} edges")
    else:
        # Check if graph is already undirected to avoid expensive conversion
        if isinstance(ontology_graph, nx.Graph) and not isinstance(ontology_graph, nx.DiGraph):
            # Already undirected, use directly
            G_undirected = ontology_graph
            if verbose:
                print(f"Using undirected graph directly: {G_undirected.number_of_nodes():,} nodes, {G_undirected.number_of_edges():,} edges")
        else:
            # OPTIMIZATION: Instead of converting entire graph to undirected (slow for large graphs),
            # compute bidirectional paths on the directed graph. This is much faster.
            use_bidirectional = True
            G_undirected = None  # We'll handle bidirectional traversal differently
            if verbose:
                print(f"Using bidirectional path computation on directed graph (faster than full conversion)")
                print(f"  Directed graph: {ontology_graph.number_of_nodes():,} nodes, {ontology_graph.number_of_edges():,} edges")
    results = []
    
    # Generate all possible pairs first
    n_terms = len(term_ids)
    n_possible_pairs = n_terms * (n_terms - 1) // 2
    
    if max_pairs is not None and n_possible_pairs > max_pairs:
        # Sample random pairs
        if verbose:
            print(f"Sampling {max_pairs:,} random pairs from {n_possible_pairs:,} possible pairs...")
        np.random.seed(42)
        sampled_indices = set()
        attempts = 0
        while len(sampled_indices) < max_pairs and attempts < max_pairs * 10:
            i, j = np.random.choice(n_terms, 2, replace=False)
            pair = tuple(sorted([i, j]))
            if pair not in sampled_indices:
                sampled_indices.add(pair)
            attempts += 1
        pairs = [(term_ids[i], term_ids[j]) for i, j in sampled_indices]
    else:
        # All pairs
        if verbose:
            print(f"Computing similarities for all {n_possible_pairs:,} pairs...")
        pairs = [(term_ids[i], term_ids[j]) 
                for i in range(n_terms) 
                for j in range(i + 1, n_terms)]
    
    # Helper function to extract path lengths and compute similarities using indices
    def extract_similarities_from_path_lengths(path_lengths_dict, pairs, term_ids, original_term_ids, id_label_mapping, verbose):
        """Extract similarities from path lengths dictionary using term indices."""
        # Create term to index mapping
        term_to_idx = {term: idx for idx, term in enumerate(term_ids)}
        n_terms = len(term_ids)
        
        if verbose:
            print(f"Computing similarities for {len(pairs):,} pairs...")
            try:
                from tqdm.auto import tqdm
                pair_iterator = tqdm(pairs, desc="Computing similarities")
            except ImportError:
                pair_iterator = pairs
        else:
            pair_iterator = pairs
        
        # Pre-compute label to ID mapping if needed
        label_to_id = None
        if original_term_ids is not None:
            label_to_id = {v: k for k, v in id_label_mapping.items() if ':' in k}
        
        results = []
        for term1, term2 in pair_iterator:
            if term1 == term2:
                continue
            
            # Use indices for lookup (much faster)
            idx1 = term_to_idx[term1]
            idx2 = term_to_idx[term2]
            path_length = path_lengths_dict.get((idx1, idx2), np.nan)
            
            if np.isnan(path_length):
                similarity = 0.0
            else:
                similarity = 1.0 / (1.0 + path_length)
            
            # Convert labels back to original IDs if original terms were IDs
            term1_original = term1
            term2_original = term2
            if label_to_id is not None:
                if term1 in label_to_id:
                    term1_original = label_to_id[term1]
                elif term1 in original_term_ids:
                    term1_original = term1
                if term2 in label_to_id:
                    term2_original = label_to_id[term2]
                elif term2 in original_term_ids:
                    term2_original = term2
            
            results.append({
                'term1': term1_original,
                'term2': term2_original,
                'similarity': similarity,
                'path_length': path_length if not np.isnan(path_length) else np.nan,
                'lca': None,
            })
        
        return results
    
    # Use NetworkX's built-in all-pairs shortest path functions
    if similarity_method in ["shortest_path_length", "bellman_ford", "dijkstra_path"]:
        # Map method names to single-source NetworkX functions (for bidirectional computation)
        nx_single_source_functions = {
            "shortest_path_length": nx.single_source_shortest_path_length,
            "bellman_ford": lambda G, source, **kwargs: nx.single_source_bellman_ford_path_length(G, source, weight=None),
            "dijkstra_path": lambda G, source, **kwargs: nx.single_source_dijkstra_path_length(G, source, weight=None),
        }
        
        # Map method names to all-pairs NetworkX functions (for undirected graph)
        nx_all_pairs_functions = {
            "shortest_path_length": (nx.all_pairs_shortest_path_length, {}),
            "bellman_ford": (nx.all_pairs_bellman_ford_path_length, {"weight": None}),
            "dijkstra_path": (nx.all_pairs_dijkstra_path_length, {"weight": None}),
        }
        
        method_name = similarity_method
        term_set = set(term_ids)
        term_to_idx = {term: idx for idx, term in enumerate(term_ids)}
        path_lengths_dict = {}
        
        if use_bidirectional:
            # OPTIMIZATION: Compute bidirectional paths on directed graph (much faster than converting)
            if verbose:
                print(f"Computing bidirectional path lengths using {method_name} for {len(term_ids)} terms...")
                print(f"  Computing paths in both directions (forward and reverse)...")
            
            nx_single_func = nx_single_source_functions[similarity_method]
            
            # Create reverse graph once (lazy, doesn't copy edges)
            reverse_graph = ontology_graph.reverse(copy=False)
            
            if verbose:
                try:
                    from tqdm.auto import tqdm
                    term_iterator = tqdm(term_set, desc="Computing bidirectional paths")
                except ImportError:
                    term_iterator = term_set
            else:
                term_iterator = term_set
            
            for source in term_iterator:
                source_idx = term_to_idx[source]
                
                # Forward paths (following edges)
                if similarity_method == "bellman_ford":
                    forward_paths = nx.single_source_bellman_ford_path_length(ontology_graph, source, weight=None)
                elif similarity_method == "dijkstra_path":
                    forward_paths = nx.single_source_dijkstra_path_length(ontology_graph, source, weight=None)
                else:
                    forward_paths = nx.single_source_shortest_path_length(ontology_graph, source)
                
                # Reverse paths (against edges)
                if similarity_method == "bellman_ford":
                    reverse_paths = nx.single_source_bellman_ford_path_length(reverse_graph, source, weight=None)
                elif similarity_method == "dijkstra_path":
                    reverse_paths = nx.single_source_dijkstra_path_length(reverse_graph, source, weight=None)
                else:
                    reverse_paths = nx.single_source_shortest_path_length(reverse_graph, source)
                
                # Combine: take minimum path length from either direction
                for target in term_set:
                    if target == source:
                        continue
                    target_idx = term_to_idx[target]
                    
                    forward_dist = forward_paths.get(target, float('inf'))
                    reverse_dist = reverse_paths.get(target, float('inf'))
                    min_dist = min(forward_dist, reverse_dist)
                    
                    if min_dist != float('inf'):
                        path_lengths_dict[(source_idx, target_idx)] = min_dist
        else:
            # Original approach: use all-pairs on undirected graph
            nx_func, nx_kwargs = nx_all_pairs_functions[similarity_method]
            
            if verbose:
                print(f"Computing path lengths using {method_name} for {len(term_ids)} terms...")
                print(f"  Computing all-pairs shortest paths (this may take a moment)...")
            
            path_iterator = nx_func(G_undirected, **nx_kwargs)
            # Direct dict conversion - same as what works fast outside the function
            all_path_lengths = dict(path_iterator)
            
            # Extract path lengths only for terms we requested
            if verbose:
                try:
                    from tqdm.auto import tqdm
                    term_iterator = tqdm(term_set, desc="Extracting term pairs")
                except ImportError:
                    term_iterator = term_set
            else:
                term_iterator = term_set
            
            for source in term_iterator:
                if source not in all_path_lengths:
                    continue
                source_idx = term_to_idx[source]
                distances = all_path_lengths[source]
                # Find valid targets using set intersection (faster than checking each item)
                valid_targets = term_set.intersection(distances.keys())
                valid_targets.discard(source)  # Remove self if present
                # Bulk update path_lengths_dict
                for target in valid_targets:
                    path_length = distances[target]
                    target_idx = term_to_idx[target]
                    path_lengths_dict[(source_idx, target_idx)] = path_length
                    path_lengths_dict[(target_idx, source_idx)] = path_length
        
        # Filter pairs by distance if requested
        if min_distance is not None or max_distance is not None:
            if verbose:
                print(f"Filtering pairs by distance (min={min_distance}, max={max_distance})...")
            
            filtered_pairs = []
            for term1, term2 in pairs:
                idx1 = term_to_idx[term1]
                idx2 = term_to_idx[term2]
                path_length = path_lengths_dict.get((idx1, idx2), np.nan)
                
                if np.isnan(path_length):
                    continue
                
                # Apply distance filters
                if min_distance is not None and path_length < min_distance:
                    continue
                if max_distance is not None and path_length > max_distance:
                    continue
                
                filtered_pairs.append((term1, term2))
            
            pairs = filtered_pairs
            if verbose:
                print(f"  Found {len(pairs):,} pairs within distance range")
            
            # Apply max_pairs limit if specified
            if max_pairs is not None and len(pairs) > max_pairs:
                if verbose:
                    print(f"  Sampling {max_pairs:,} random pairs from {len(pairs):,} filtered pairs...")
                np.random.seed(42)
                sampled_indices = np.random.choice(len(pairs), 
                                                 size=min(max_pairs, len(pairs)), 
                                                 replace=False)
                pairs = [pairs[idx] for idx in sampled_indices]
        
        print("Extracting similarities from path lengths...")
        results = extract_similarities_from_path_lengths(
            path_lengths_dict, pairs, term_ids, original_term_ids, id_label_mapping, verbose
        )
    
    elif similarity_method == "floyd_warshall":
        # Use NetworkX's floyd_warshall_numpy for matrix-based all-pairs shortest paths
        if verbose:
            print(f"Computing path lengths using nx.floyd_warshall_numpy for {len(term_ids)} terms...")
        
        # Use full graph (don't create subgraph - it messes up the structure)
        term_set = set(term_ids)
        term_to_idx = {term: idx for idx, term in enumerate(term_ids)}
        
        # Get all nodes in graph for nodelist
        nodelist = list(G_undirected.nodes())
        node_to_matrix_idx = {node: idx for idx, node in enumerate(nodelist)}
        
        if verbose:
            print(f"  Computing all-pairs shortest paths using Floyd-Warshall for {len(nodelist):,} nodes...")
        
        try:
            distance_matrix = nx.floyd_warshall_numpy(G_undirected, nodelist=nodelist, weight=None)
            distance_matrix = np.asarray(distance_matrix)
            # Replace inf with nan
            distance_matrix = np.where(np.isinf(distance_matrix), np.nan, distance_matrix)
            
            # Extract all path lengths for our term pairs
            path_lengths_dict = {}
            for term1 in term_ids:
                if term1 not in node_to_matrix_idx:
                    continue
                idx1 = node_to_matrix_idx[term1]
                term1_idx = term_to_idx[term1]
                
                for term2 in term_ids:
                    if term1 == term2 or term2 not in node_to_matrix_idx:
                        continue
                    idx2 = node_to_matrix_idx[term2]
                    term2_idx = term_to_idx[term2]
                    
                    path_length = distance_matrix[idx1, idx2]
                    if not np.isnan(path_length):
                        path_lengths_dict[(term1_idx, term2_idx)] = path_length
                        path_lengths_dict[(term2_idx, term1_idx)] = path_length
            
        except Exception as e:
            if verbose:
                print(f"  Warning: Floyd-Warshall computation failed ({e}), falling back to all_pairs method...")
            # Fallback: use all_pairs_shortest_path_length
            path_lengths_dict = {}
            for source, distances in nx.all_pairs_shortest_path_length(G_undirected):
                if source not in term_set:
                    continue
                source_idx = term_to_idx[source]
                for target, path_length in distances.items():
                    if target in term_set and source != target:
                        target_idx = term_to_idx[target]
                        path_lengths_dict[(source_idx, target_idx)] = path_length
                        path_lengths_dict[(target_idx, source_idx)] = path_length
        
        # Filter pairs by distance if requested
        if min_distance is not None or max_distance is not None:
            if verbose:
                print(f"Filtering pairs by distance (min={min_distance}, max={max_distance})...")
            
            filtered_pairs = []
            for term1, term2 in pairs:
                idx1 = term_to_idx[term1]
                idx2 = term_to_idx[term2]
                path_length = path_lengths_dict.get((idx1, idx2), np.nan)
                
                if np.isnan(path_length):
                    continue
                
                # Apply distance filters
                if min_distance is not None and path_length < min_distance:
                    continue
                if max_distance is not None and path_length > max_distance:
                    continue
                
                filtered_pairs.append((term1, term2))
            
            pairs = filtered_pairs
            if verbose:
                print(f"  Found {len(pairs):,} pairs within distance range")
            
            # Apply max_pairs limit if specified
            if max_pairs is not None and len(pairs) > max_pairs:
                if verbose:
                    print(f"  Sampling {max_pairs:,} random pairs from {len(pairs):,} filtered pairs...")
                np.random.seed(42)
                sampled_indices = np.random.choice(len(pairs), 
                                                 size=min(max_pairs, len(pairs)), 
                                                 replace=False)
                pairs = [pairs[idx] for idx in sampled_indices]
        
        results = extract_similarities_from_path_lengths(
            path_lengths_dict, pairs, term_ids, original_term_ids, id_label_mapping, verbose
        )
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\nComputed {len(results_df):,} pairwise similarities")
        print(f"  Similarity range: [{results_df['similarity'].min():.3f}, {results_df['similarity'].max():.3f}]")
        print(f"  Mean similarity: {results_df['similarity'].mean():.3f}")
    
    return results_df
