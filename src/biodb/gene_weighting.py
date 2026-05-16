"""
Gene Weighting and Attention Mechanisms for Patient Embeddings

This module implements efficient two-stage gene weighting for incorporating
genomic information into clinical event embeddings at scale (400k+ patients).

Key features:
- Fast gene scoring using lightweight LLM embeddings
- Lazy computation of expensive genomic sequence embeddings
- Support for multiple conditions and temporal evolution
- Memory-efficient batch processing
"""

from typing import Optional, Union, List, Dict, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class GeneWeightingConfig:
    """Configuration for gene weighting."""
    top_k: int = 50  # Number of top genes to select
    temperature: float = 0.1  # Temperature for attention softmax
    alpha: float = 0.5  # Weight for mixing direct vs indirect gene scores
    temporal_decay: float = 0.9  # Decay factor for temporal weighting
    aggregation_method: str = "max"  # 'max', 'mean', 'weighted_sum'
    batch_size: int = 1000  # Batch size for processing patients
    use_temporal_weighting: bool = True  # Whether to weight by time
    cache_size: int = 1000  # Number of gene embeddings to cache


def compute_gene_weights_fast(
    event_embeddings: torch.Tensor,  # (batch_size, n_events, embed_dim) or (batch_size, embed_dim)
    gene_name_embeddings: torch.Tensor,  # (n_genes, embed_dim) - pre-computed LLM embeddings
    disease_concept_embeddings: Optional[torch.Tensor] = None,  # (n_diseases, embed_dim)
    disease_to_gene_mapping: Optional[torch.Tensor] = None,  # (n_diseases, n_genes) - binary or weights
    config: Optional[GeneWeightingConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast gene scoring without computing expensive sequence embeddings.
    
    This function computes gene relevance scores using lightweight LLM embeddings,
    enabling efficient selection of top-k genes before computing expensive
    genomic sequence embeddings.
    
    Parameters
    ----------
    event_embeddings : torch.Tensor
        Event embeddings, shape (batch_size, n_events, embed_dim) or (batch_size, embed_dim)
    gene_name_embeddings : torch.Tensor
        Pre-computed gene name embeddings from LLM, shape (n_genes, embed_dim)
    disease_concept_embeddings : torch.Tensor, optional
        Disease concept embeddings, shape (n_diseases, embed_dim)
    disease_to_gene_mapping : torch.Tensor, optional
        Mapping from diseases to genes, shape (n_diseases, n_genes)
        Can be binary (0/1) or continuous weights
    config : GeneWeightingConfig, optional
        Configuration object. If None, uses defaults.
    
    Returns
    -------
    top_k_indices : torch.Tensor
        Indices of top-k genes, shape (batch_size, top_k)
    top_k_values : torch.Tensor
        Scores of top-k genes, shape (batch_size, top_k)
    all_scores : torch.Tensor
        All gene scores, shape (batch_size, n_genes)
    """
    if config is None:
        config = GeneWeightingConfig()
    
    # Normalize embeddings for cosine similarity
    event_embeddings = F.normalize(event_embeddings, p=2, dim=-1)
    gene_name_embeddings = F.normalize(gene_name_embeddings, p=2, dim=1)
    
    # Handle different input shapes
    if event_embeddings.dim() == 2:
        # (batch_size, embed_dim) - single event per patient
        batch_size = event_embeddings.shape[0]
        event_embeddings = event_embeddings.unsqueeze(1)  # (batch_size, 1, embed_dim)
    else:
        # (batch_size, n_events, embed_dim) - multiple events per patient
        batch_size, n_events, embed_dim = event_embeddings.shape
    
    # Option 1: Direct event → gene name similarity (LLM space)
    # (batch_size, n_events, embed_dim) @ (embed_dim, n_genes) = (batch_size, n_events, n_genes)
    gene_scores_direct = torch.matmul(
        event_embeddings,
        gene_name_embeddings.t()
    )  # (batch_size, n_events, n_genes)
    
    # Option 2: Event → Disease → Gene (multi-hop) if provided
    if disease_concept_embeddings is not None:
        disease_concept_embeddings = F.normalize(disease_concept_embeddings, p=2, dim=1)
        
        # Event → Disease similarity
        # (batch_size, n_events, embed_dim) @ (embed_dim, n_diseases) = (batch_size, n_events, n_diseases)
        disease_scores = torch.matmul(
            event_embeddings,
            disease_concept_embeddings.t()
        )
        disease_weights = F.softmax(disease_scores / config.temperature, dim=-1)
        
        # Map diseases to genes
        if disease_to_gene_mapping is not None:
            # (batch_size, n_events, n_diseases) @ (n_diseases, n_genes) = (batch_size, n_events, n_genes)
            gene_scores_indirect = torch.matmul(
                disease_weights,
                disease_to_gene_mapping
            )
        else:
            # Learned mapping: disease_concept_embeddings @ gene_name_embeddings.T
            disease_gene_similarity = torch.matmul(
                disease_concept_embeddings,
                gene_name_embeddings.t()
            )  # (n_diseases, n_genes)
            gene_scores_indirect = torch.matmul(
                disease_weights,
                disease_gene_similarity
            )  # (batch_size, n_events, n_genes)
        
        # Combine direct and indirect scores
        gene_scores = (
            config.alpha * gene_scores_direct + 
            (1 - config.alpha) * gene_scores_indirect
        )
    else:
        gene_scores = gene_scores_direct
    
    # Aggregate across events (patient-level)
    # Options: max, mean, weighted_sum
    if config.aggregation_method == "max":
        patient_gene_scores = gene_scores.max(dim=1)[0]  # (batch_size, n_genes)
    elif config.aggregation_method == "mean":
        patient_gene_scores = gene_scores.mean(dim=1)  # (batch_size, n_genes)
    elif config.aggregation_method == "weighted_sum":
        # Could add event importance weights here
        patient_gene_scores = gene_scores.mean(dim=1)  # Default to mean
    else:
        raise ValueError(f"Unknown aggregation method: {config.aggregation_method}")
    
    # Get top-k
    top_k_values, top_k_indices = torch.topk(
        patient_gene_scores, 
        k=min(config.top_k, patient_gene_scores.shape[1]), 
        dim=1
    )
    
    return top_k_indices, top_k_values, patient_gene_scores


def temporal_gene_weighting(
    patient_timeline: torch.Tensor,  # (n_events, embed_dim) or (batch_size, n_events, embed_dim)
    gene_name_embeddings: torch.Tensor,  # (n_genes, embed_dim)
    disease_concept_embeddings: Optional[torch.Tensor] = None,
    disease_to_gene_mapping: Optional[torch.Tensor] = None,
    config: Optional[GeneWeightingConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weight genes considering temporal evolution of conditions.
    
    Recent events are weighted more heavily than older events, reflecting
    the importance of current conditions for gene relevance.
    
    Parameters
    ----------
    patient_timeline : torch.Tensor
        Sequence of event embeddings over time, shape (n_events, embed_dim) 
        or (batch_size, n_events, embed_dim)
    gene_name_embeddings : torch.Tensor
        Gene name embeddings, shape (n_genes, embed_dim)
    disease_concept_embeddings : torch.Tensor, optional
        Disease concept embeddings, shape (n_diseases, embed_dim)
    disease_to_gene_mapping : torch.Tensor, optional
        Disease to gene mapping, shape (n_diseases, n_genes)
    config : GeneWeightingConfig, optional
        Configuration object
    
    Returns
    -------
    top_k_indices : torch.Tensor
        Top-k gene indices, shape (batch_size, top_k) or (top_k,)
    top_k_values : torch.Tensor
        Top-k gene scores, shape (batch_size, top_k) or (top_k,)
    all_scores : torch.Tensor
        All gene scores, shape (batch_size, n_genes) or (n_genes,)
    """
    if config is None:
        config = GeneWeightingConfig()
    
    # Handle batch vs single patient
    if patient_timeline.dim() == 2:
        # Single patient: (n_events, embed_dim)
        patient_timeline = patient_timeline.unsqueeze(0)  # (1, n_events, embed_dim)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, n_events, embed_dim = patient_timeline.shape
    device = patient_timeline.device
    
    # Compute temporal weights (more recent = higher weight)
    temporal_weights = torch.tensor(
        [config.temporal_decay ** (n_events - i - 1) for i in range(n_events)],
        device=device,
        dtype=patient_timeline.dtype
    )  # (n_events,)
    temporal_weights = temporal_weights / temporal_weights.sum()  # Normalize
    
    # Compute gene scores per event
    event_gene_scores = []
    for event_idx in range(n_events):
        event_emb = patient_timeline[:, event_idx, :]  # (batch_size, embed_dim)
        _, _, scores = compute_gene_weights_fast(
            event_emb.unsqueeze(1),  # (batch_size, 1, embed_dim)
            gene_name_embeddings,
            disease_concept_embeddings,
            disease_to_gene_mapping,
            config,
        )
        event_gene_scores.append(scores)
    
    event_gene_scores = torch.stack(event_gene_scores, dim=1)  # (batch_size, n_events, n_genes)
    
    # Weight by time
    temporal_weights_expanded = temporal_weights.unsqueeze(0).unsqueeze(-1)  # (1, n_events, 1)
    weighted_scores = temporal_weights_expanded * event_gene_scores  # (batch_size, n_events, n_genes)
    patient_gene_scores = weighted_scores.sum(dim=1)  # (batch_size, n_genes)
    
    # Get top-k
    top_k_values, top_k_indices = torch.topk(
        patient_gene_scores,
        k=min(config.top_k, patient_gene_scores.shape[1]),
        dim=1
    )
    
    if squeeze_output:
        top_k_indices = top_k_indices.squeeze(0)
        top_k_values = top_k_values.squeeze(0)
        patient_gene_scores = patient_gene_scores.squeeze(0)
    
    return top_k_indices, top_k_values, patient_gene_scores


def multi_condition_gene_weighting(
    patient_conditions: torch.Tensor,  # (n_conditions, embed_dim) or (batch_size, n_conditions, embed_dim)
    gene_name_embeddings: torch.Tensor,  # (n_genes, embed_dim)
    condition_to_gene_mapping: Optional[torch.Tensor] = None,  # (n_conditions, n_genes) or dict
    config: Optional[GeneWeightingConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate gene relevance across multiple concurrent conditions.
    
    Parameters
    ----------
    patient_conditions : torch.Tensor
        Condition embeddings, shape (n_conditions, embed_dim) or (batch_size, n_conditions, embed_dim)
    gene_name_embeddings : torch.Tensor
        Gene name embeddings, shape (n_genes, embed_dim)
    condition_to_gene_mapping : torch.Tensor or dict, optional
        Mapping from conditions to genes. If tensor, shape (n_conditions, n_genes).
        If dict, maps condition indices to gene indices.
    config : GeneWeightingConfig, optional
        Configuration object
    
    Returns
    -------
    top_k_indices : torch.Tensor
        Top-k gene indices
    top_k_values : torch.Tensor
        Top-k gene scores
    all_scores : torch.Tensor
        All gene scores
    """
    if config is None:
        config = GeneWeightingConfig()
    
    # Handle batch vs single patient
    if patient_conditions.dim() == 2:
        patient_conditions = patient_conditions.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, n_conditions, embed_dim = patient_conditions.shape
    device = patient_conditions.device
    n_genes = gene_name_embeddings.shape[0]
    
    # Normalize
    patient_conditions = F.normalize(patient_conditions, p=2, dim=-1)
    gene_name_embeddings = F.normalize(gene_name_embeddings, p=2, dim=1)
    
    # Compute condition-gene relevance
    condition_gene_scores = []
    
    for condition_idx in range(n_conditions):
        condition_emb = patient_conditions[:, condition_idx, :]  # (batch_size, embed_dim)
        
        if condition_to_gene_mapping is not None:
            if isinstance(condition_to_gene_mapping, dict):
                # Dict mapping: condition_idx -> [gene_indices]
                relevant_genes = condition_to_gene_mapping.get(condition_idx, [])
                if len(relevant_genes) > 0:
                    relevant_gene_embeddings = gene_name_embeddings[relevant_genes]  # (n_relevant, embed_dim)
                    scores = torch.matmul(
                        condition_emb.unsqueeze(1),  # (batch_size, 1, embed_dim)
                        relevant_gene_embeddings.t().unsqueeze(0)  # (1, embed_dim, n_relevant)
                    ).squeeze(1)  # (batch_size, n_relevant)
                    # Map back to full gene space
                    full_scores = torch.zeros(batch_size, n_genes, device=device)
                    full_scores[:, relevant_genes] = scores
                else:
                    full_scores = torch.zeros(batch_size, n_genes, device=device)
            else:
                # Tensor mapping: (n_conditions, n_genes)
                mapping = condition_to_gene_mapping[condition_idx]  # (n_genes,)
                condition_emb_expanded = condition_emb.unsqueeze(1)  # (batch_size, 1, embed_dim)
                gene_emb_expanded = gene_name_embeddings.unsqueeze(0)  # (1, n_genes, embed_dim)
                similarity = torch.sum(
                    condition_emb_expanded * gene_emb_expanded, dim=-1
                )  # (batch_size, n_genes)
                full_scores = similarity * mapping.unsqueeze(0)  # Weight by mapping
        else:
            # No mapping: compute direct similarity
            full_scores = torch.matmul(
                condition_emb,
                gene_name_embeddings.t()
            )  # (batch_size, n_genes)
        
        condition_gene_scores.append(full_scores)
    
    condition_gene_scores = torch.stack(condition_gene_scores, dim=1)  # (batch_size, n_conditions, n_genes)
    
    # Aggregate across conditions
    if config.aggregation_method == "max":
        # Take max relevance (any condition makes gene relevant)
        patient_gene_scores = condition_gene_scores.max(dim=1)[0]  # (batch_size, n_genes)
    elif config.aggregation_method == "mean":
        patient_gene_scores = condition_gene_scores.mean(dim=1)  # (batch_size, n_genes)
    elif config.aggregation_method == "weighted_sum":
        # Could add condition importance weights here
        patient_gene_scores = condition_gene_scores.mean(dim=1)  # Default to mean
    else:
        raise ValueError(f"Unknown aggregation method: {config.aggregation_method}")
    
    # Get top-k
    top_k_values, top_k_indices = torch.topk(
        patient_gene_scores,
        k=min(config.top_k, patient_gene_scores.shape[1]),
        dim=1
    )
    
    if squeeze_output:
        top_k_indices = top_k_indices.squeeze(0)
        top_k_values = top_k_values.squeeze(0)
        patient_gene_scores = patient_gene_scores.squeeze(0)
    
    return top_k_indices, top_k_values, patient_gene_scores


class GeneEmbeddingCache:
    """Cache for frequently used gene sequence embeddings."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.cache: Dict[int, torch.Tensor] = {}
        self.access_counts: Dict[int, int] = {}
    
    def get(self, gene_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached embeddings for gene indices.
        
        Returns
        -------
        cached_embeddings : torch.Tensor
            Cached embeddings, shape (n_cached, embed_dim)
        cached_indices : torch.Tensor
            Indices that were cached, shape (n_cached,)
        """
        cached_indices = []
        cached_embeddings = []
        
        for idx in gene_indices.cpu().numpy():
            if idx in self.cache:
                cached_indices.append(idx)
                cached_embeddings.append(self.cache[idx])
                self.access_counts[idx] = self.access_counts.get(idx, 0) + 1
        
        if len(cached_embeddings) > 0:
            return torch.stack(cached_embeddings), torch.tensor(cached_indices)
        else:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
    
    def put(self, gene_indices: torch.Tensor, embeddings: torch.Tensor):
        """Add embeddings to cache."""
        for idx, emb in zip(gene_indices.cpu().numpy(), embeddings):
            if len(self.cache) >= self.cache_size:
                # Evict least recently used
                lru_idx = min(self.access_counts, key=self.access_counts.get)
                del self.cache[lru_idx]
                del self.access_counts[lru_idx]
            
            self.cache[int(idx)] = emb
            self.access_counts[int(idx)] = self.access_counts.get(int(idx), 0) + 1


def compute_gene_sequence_embeddings_lazy(
    top_k_gene_indices: torch.Tensor,  # (batch_size, top_k) or (top_k,)
    gene_sequences: Union[torch.Tensor, Callable],  # Gene sequence data or loader function
    genomic_model: nn.Module,  # Expensive genomic embedding model
    cache: Optional[GeneEmbeddingCache] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute expensive sequence embeddings only for top-k genes.
    
    This function implements lazy loading and computation of gene sequence
    embeddings, only computing them for genes selected in the fast scoring stage.
    
    Parameters
    ----------
    top_k_gene_indices : torch.Tensor
        Gene indices from fast scoring stage, shape (batch_size, top_k) or (top_k,)
    gene_sequences : torch.Tensor or Callable
        Gene sequence data. If tensor, shape (n_genes, seq_len).
        If callable, function that takes gene indices and returns sequences.
    genomic_model : nn.Module
        Model that computes embeddings from gene sequences
    cache : GeneEmbeddingCache, optional
        Cache for frequently used embeddings
    device : torch.device, optional
        Device to compute on
    
    Returns
    -------
    gene_sequence_embeddings : torch.Tensor
        Sequence embeddings, shape (batch_size, top_k, embed_dim) or (top_k, embed_dim)
    """
    if device is None:
        device = next(genomic_model.parameters()).device
    
    # Handle batch vs single patient
    if top_k_gene_indices.dim() == 1:
        top_k_gene_indices = top_k_gene_indices.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, top_k = top_k_gene_indices.shape
    
    # Collect unique genes across batch
    unique_genes = torch.unique(top_k_gene_indices.flatten())
    unique_genes = unique_genes.to(device)
    
    # Check cache first
    cached_embeddings = None
    cached_indices = None
    uncached_indices = unique_genes
    
    
    if cache is not None:
        cached_embeddings, cached_indices = cache.get(unique_genes)
        if len(cached_indices) > 0:
            cached_indices = cached_indices.to(device)
            # Find uncached indices
            cached_set = set(cached_indices.cpu().numpy())
            uncached_indices = unique_genes[
                ~torch.isin(unique_genes, torch.tensor(list(cached_set), device=device))
            ]
    
    # Load sequences for uncached genes
    if len(uncached_indices) > 0:
        if callable(gene_sequences):
            uncached_sequences = gene_sequences(uncached_indices)
        else:
            uncached_sequences = gene_sequences[uncached_indices.cpu().numpy()]
        
        if isinstance(uncached_sequences, np.ndarray):
            uncached_sequences = torch.from_numpy(uncached_sequences).to(device)
        
        # Compute embeddings
        genomic_model.eval()
        with torch.no_grad():
            uncached_embeddings = genomic_model(uncached_sequences)
        
        # Update cache
        if cache is not None:
            cache.put(uncached_indices, uncached_embeddings)
    else:
        uncached_embeddings = torch.empty(0, device=device)
        uncached_indices = torch.empty(0, dtype=torch.long, device=device)
    
    # Combine cached and uncached embeddings
    if cached_embeddings is not None and len(cached_embeddings) > 0:
        all_embeddings = torch.zeros(
            len(unique_genes), 
            uncached_embeddings.shape[1] if len(uncached_embeddings) > 0 
            else cached_embeddings.shape[1],
            device=device
        )
        all_indices = torch.cat([cached_indices, uncached_indices])
        
        # Map embeddings to unique gene indices
        gene_to_embedding = {}
        for idx, emb in zip(cached_indices, cached_embeddings):
            gene_to_embedding[int(idx)] = emb
        for idx, emb in zip(uncached_indices, uncached_embeddings):
            gene_to_embedding[int(idx)] = emb
        
        # Create mapping for batch
        batch_embeddings = []
        for patient_idx in range(batch_size):
            patient_genes = top_k_gene_indices[patient_idx]
            patient_emb = torch.stack([
                gene_to_embedding[int(gid)] for gid in patient_genes
            ])
            batch_embeddings.append(patient_emb)
        
        result = torch.stack(batch_embeddings)  # (batch_size, top_k, embed_dim)
    else:
        # No cache, simpler case
        if len(uncached_indices) == len(unique_genes):
            # All genes were uncached
            gene_to_embedding = {
                int(gid): emb for gid, emb in zip(uncached_indices, uncached_embeddings)
            }
            batch_embeddings = []
            for patient_idx in range(batch_size):
                patient_genes = top_k_gene_indices[patient_idx]
                patient_emb = torch.stack([
                    gene_to_embedding[int(gid)] for gid in patient_genes
                ])
                batch_embeddings.append(patient_emb)
            result = torch.stack(batch_embeddings)
        else:
            # Should not happen, but handle gracefully
            result = torch.zeros(
                batch_size, top_k, uncached_embeddings.shape[1],
                device=device
            )
    
    if squeeze_output:
        result = result.squeeze(0)
    
    return result


def patient_gene_attention_with_lazy_embeddings(
    patient_events: torch.Tensor,  # (batch_size, n_events, embed_dim) or (batch_size, embed_dim)
    gene_name_embeddings: torch.Tensor,  # (n_genes, embed_dim) - pre-computed LLM
    gene_sequences: Union[torch.Tensor, Callable],  # On-demand, expensive
    genomic_model: nn.Module,  # Expensive to run
    disease_concept_embeddings: Optional[torch.Tensor] = None,
    disease_to_gene_mapping: Optional[torch.Tensor] = None,
    config: Optional[GeneWeightingConfig] = None,
    cache: Optional[GeneEmbeddingCache] = None,
    use_temporal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Complete pipeline: fast gene scoring + lazy embedding computation + attention.
    
    This is the main function for incorporating genomic information into patient
    embeddings efficiently at scale.
    
    Parameters
    ----------
    patient_events : torch.Tensor
        Patient event embeddings, shape (batch_size, n_events, embed_dim) or (batch_size, embed_dim)
    gene_name_embeddings : torch.Tensor
        Pre-computed gene name embeddings from LLM, shape (n_genes, embed_dim)
    gene_sequences : torch.Tensor or Callable
        Gene sequence data or loader function
    genomic_model : nn.Module
        Model that computes embeddings from gene sequences
    disease_concept_embeddings : torch.Tensor, optional
        Disease concept embeddings, shape (n_diseases, embed_dim)
    disease_to_gene_mapping : torch.Tensor, optional
        Disease to gene mapping, shape (n_diseases, n_genes)
    config : GeneWeightingConfig, optional
        Configuration object
    cache : GeneEmbeddingCache, optional
        Cache for gene embeddings
    use_temporal : bool
        Whether to use temporal weighting
    
    Returns
    -------
    enhanced_events : torch.Tensor
        Event embeddings enhanced with gene information, shape (batch_size, embed_dim)
    top_k_indices : torch.Tensor
        Top-k gene indices, shape (batch_size, top_k)
    attention_weights : torch.Tensor
        Attention weights over genes, shape (batch_size, top_k)
    all_scores : torch.Tensor
        All gene scores, shape (batch_size, n_genes)
    """
    if config is None:
        config = GeneWeightingConfig()
    
    device = patient_events.device
    
    # Stage 1: Fast scoring (no expensive embeddings)
    if use_temporal and patient_events.dim() == 3:
        top_k_indices, top_k_values, all_scores = temporal_gene_weighting(
            patient_events,
            gene_name_embeddings,
            disease_concept_embeddings,
            disease_to_gene_mapping,
            config,
        )
    else:
        top_k_indices, top_k_values, all_scores = compute_gene_weights_fast(
            patient_events,
            gene_name_embeddings,
            disease_concept_embeddings,
            disease_to_gene_mapping,
            config,
        )
    
    # Stage 2: Compute expensive embeddings only for top-k
    gene_sequence_embeddings = compute_gene_sequence_embeddings_lazy(
        top_k_indices,
        gene_sequences,
        genomic_model,
        cache,
        device,
    )  # (batch_size, top_k, embed_dim)
    
    # Stage 3: Attention with sequence embeddings
    attention_weights = F.softmax(top_k_values / config.temperature, dim=-1)  # (batch_size, top_k)
    
    # Aggregate: (batch_size, top_k) @ (batch_size, top_k, embed_dim)
    # We need to do element-wise multiplication and sum
    attention_weights_expanded = attention_weights.unsqueeze(-1)  # (batch_size, top_k, 1)
    weighted_embeddings = attention_weights_expanded * gene_sequence_embeddings  # (batch_size, top_k, embed_dim)
    enhanced_events = weighted_embeddings.sum(dim=1)  # (batch_size, embed_dim)
    
    return enhanced_events, top_k_indices, attention_weights, all_scores


def memory_efficient_gene_attention(
    patient_events: torch.Tensor,  # (n_patients, n_events, embed_dim) or (n_patients, embed_dim)
    gene_name_embeddings: torch.Tensor,  # (n_genes, embed_dim)
    gene_sequences: Union[torch.Tensor, Callable],
    genomic_model: nn.Module,
    disease_concept_embeddings: Optional[torch.Tensor] = None,
    disease_to_gene_mapping: Optional[torch.Tensor] = None,
    config: Optional[GeneWeightingConfig] = None,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process 400k+ patients efficiently in batches.
    
    Parameters
    ----------
    patient_events : torch.Tensor
        All patient event embeddings
    gene_name_embeddings : torch.Tensor
        Pre-computed gene name embeddings
    gene_sequences : torch.Tensor or Callable
        Gene sequence data or loader
    genomic_model : nn.Module
        Genomic embedding model
    disease_concept_embeddings : torch.Tensor, optional
        Disease concept embeddings
    disease_to_gene_mapping : torch.Tensor, optional
        Disease to gene mapping
    config : GeneWeightingConfig, optional
        Configuration
    show_progress : bool
        Show progress bar
    
    Returns
    -------
    enhanced_embeddings : torch.Tensor
        Enhanced patient embeddings, shape (n_patients, embed_dim)
    gene_indices_all : torch.Tensor
        Top-k gene indices for all patients, shape (n_patients, top_k)
    """
    if config is None:
        config = GeneWeightingConfig()
    
    n_patients = patient_events.shape[0]
    batch_size = config.batch_size
    
    # Initialize cache
    cache = GeneEmbeddingCache(cache_size=config.cache_size)
    
    enhanced_embeddings_list = []
    gene_indices_list = []
    
    iterator = range(0, n_patients, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing patients")
    
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_patients)
        patient_batch = patient_events[start_idx:end_idx]
        
        enhanced_batch, top_k_indices, _, _ = patient_gene_attention_with_lazy_embeddings(
            patient_batch,
            gene_name_embeddings,
            gene_sequences,
            genomic_model,
            disease_concept_embeddings,
            disease_to_gene_mapping,
            config,
            cache,
            use_temporal=config.use_temporal_weighting,
        )
        
        enhanced_embeddings_list.append(enhanced_batch.cpu())
        gene_indices_list.append(top_k_indices.cpu())
    
    enhanced_embeddings = torch.cat(enhanced_embeddings_list, dim=0)
    gene_indices_all = torch.cat(gene_indices_list, dim=0)
    
    return enhanced_embeddings, gene_indices_all

