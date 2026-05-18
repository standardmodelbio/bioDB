"""
Monarch Initiative data integration for querying gene-disease associations.

This module provides functions to:
1. Download and read causal gene-to-disease association data from Monarch Initiative

References:
- Monarch Initiative: https://monarchinitiative.org/
- Monarch Knowledge Graph: https://data.monarchinitiative.org/
"""

import pandas as pd
import polars as pl
from typing import Optional, Union, Dict
from pathlib import Path
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gzip
import time
from tqdm import tqdm
import re
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# Base URL for Monarch Initiative associations
ASSOCIATIONS_BASE_URL = "https://data.monarchinitiative.org/monarch-kg/latest/tsv/all_associations/"

# Default URL for causal gene-to-disease associations
CAUSAL_GENE_TO_DISEASE_URL = "https://data.monarchinitiative.org/monarch-kg/latest/tsv/all_associations/causal_gene_to_disease_association.all.tsv.gz"
CACHE_DIR = Path("~/.cache/monarch").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def list_datasets(
    base_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    List all available association datasets in the Monarch Initiative all_associations directory.
    
    This function fetches the directory listing from the Monarch Initiative server and
    extracts all TSV.gz files that can be used with read functions.
    
    Parameters
    ----------
    base_url : str, optional
        Base URL of the Monarch Initiative associations directory.
        If None, uses ASSOCIATIONS_BASE_URL (default: latest/tsv/all_associations/).
    
    Returns
    -------
    dict
        Dictionary mapping dataset names to their full URLs.
        Example: {"causal_gene_to_disease_association": "https://...", ...}
    
    Examples
    --------
    >>> import biodb.monarch as monarch
    >>> 
    >>> # List all datasets
    >>> datasets = monarch.list_datasets()
    >>> print(datasets)
    >>> # {'causal_gene_to_disease_association': 'https://...', ...}
    >>> 
    >>> # Use a dataset URL with read function
    >>> datasets = monarch.list_datasets()
    >>> df = monarch.read_causal_gene_to_disease_association(url=datasets["causal_gene_to_disease_association"])
    
    Notes
    -----
    - See https://data.monarchinitiative.org/monarch-kg/latest/tsv/all_associations/ for available datasets
    - All datasets are TSV.gz files
    """
    if base_url is None:
        base_url = ASSOCIATIONS_BASE_URL
    
    # Ensure URL ends with slash for directory listing
    if not base_url.endswith('/'):
        base_url = base_url + '/'
    
    logger.info(f"Fetching directory listing from: {base_url}")
    
    # Fetch the directory listing (HTML page)
    try:
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        logger.error(f"Error fetching directory listing from {base_url}: {e}")
        raise
    
    # Parse HTML to find TSV.gz files
    # Pattern 1: <a href="filename.tsv.gz">filename.tsv.gz</a>
    # Pattern 2: <a href="filename.tsv.gz"> (just the link)
    # Pattern 3: Directory entries in table format
    
    datasets = []
    
    # Helper function to check if a link is a parent directory or non-dataset file
    def is_valid_dataset(filename: str) -> bool:
        """Check if filename is a valid dataset file."""
        if not filename:
            return False
        # Skip parent directory links
        if filename in ['..', '.', 'Parent Directory', '../', './']:
            return False
        if filename.startswith('../') or filename.startswith('./'):
            return False
        # Only include .tsv.gz files
        if not filename.endswith('.tsv.gz'):
            return False
        return True
    
    # Helper function to extract just the filename from a URL or path
    def extract_filename(path_or_url: str) -> str:
        """Extract just the filename from a URL or path."""
        # Remove any URL encoding
        try:
            path_or_url = unquote(path_or_url)
        except Exception:
            pass
        
        # Extract filename (part after last '/')
        if '/' in path_or_url:
            filename = path_or_url.split('/')[-1]
        else:
            filename = path_or_url
        
        return filename
    
    # Try multiple patterns to extract TSV.gz filenames
    # Pattern 1: Links to .tsv.gz files
    link_pattern = r'<a[^>]+href=["\']([^"\']+\.tsv\.gz)["\']'
    matches = re.findall(link_pattern, html_content, re.IGNORECASE)
    
    for match in matches:
        filename = extract_filename(match)
        
        if is_valid_dataset(filename):
            if filename not in datasets:
                datasets.append(filename)
    
    # Pattern 2: Look for files in table format or plain text listings
    if not datasets:
        # Try finding any .tsv.gz references in the HTML
        file_pattern = r'([^/\s<>"]+\.tsv\.gz)'
        matches = re.findall(file_pattern, html_content, re.IGNORECASE)
        
        for match in matches:
            filename = extract_filename(match)
            
            if is_valid_dataset(filename):
                if filename not in datasets:
                    datasets.append(filename)
    
    # Pattern 3: Look for directory entries in table format
    if not datasets:
        # Try finding table cells with links
        table_pattern = r'<td[^>]*>.*?<a[^>]*href=["\']([^"\']+\.tsv\.gz)["\'][^>]*>.*?</a>.*?</td>'
        matches = re.findall(table_pattern, html_content, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            filename = extract_filename(match)
            
            if is_valid_dataset(filename):
                if filename not in datasets:
                    datasets.append(filename)
    
    # Sort datasets alphabetically
    datasets = sorted(set(datasets))
    
    if len(datasets) == 0:
        logger.warning("No TSV.gz datasets found in directory listing")
        return {}
    
    # Build dictionary mapping dataset names to URLs
    # Extract base name from filename (remove .tsv.gz and .all if present)
    result_dict = {}
    for dataset_filename in datasets:
        # Ensure URL is properly formatted (dataset_filename should already be just the filename)
        dataset_url = base_url.rstrip('/') + '/' + dataset_filename
        
        # Extract base name: remove .tsv.gz extension and .all suffix if present
        base_name = dataset_filename
        if base_name.endswith('.tsv.gz'):
            base_name = base_name[:-7]  # Remove .tsv.gz
        if base_name.endswith('.all'):
            base_name = base_name[:-4]  # Remove .all
        
        result_dict[base_name] = dataset_url
    
    logger.info(f"Found {len(datasets)} datasets")
    
    return result_dict


def get_dataset(
    dataset: Optional[str] = None,
    url: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
    output_format: str = "pandas",
    verbose: int = 1,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Download and read a dataset from Monarch Initiative by name.
    
    This function downloads a TSV.gz file from Monarch Initiative, caches it locally,
    and returns it as a DataFrame. The file is only downloaded once unless force=True.
    
    Parameters
    ----------
    dataset : str, optional
        Name of the dataset (e.g., "causal_gene_to_disease_association").
        If provided, the URL will be automatically looked up from list_datasets().
        Use list_datasets() to see all available dataset names.
        If both dataset and url are provided, url takes precedence.
    url : str, optional
        Direct URL of the TSV.gz file to download.
        If not provided and dataset is not provided, defaults to causal_gene_to_disease_association.
    cache_dir : str or Path, optional
        Local directory to cache downloaded files. If None, uses ~/.cache/monarch.
    force : bool, default False
        If True, re-download the file even if it already exists in cache.
    output_format : str, default "polars"
        Output format: "pandas" or "polars".
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        DataFrame containing the dataset.
    
    Examples
    --------
    >>> import biodb.monarch as monarch
    >>> 
    >>> # Get a dataset by name
    >>> df = monarch.get_dataset(dataset="causal_gene_to_disease_association")
    >>> 
    >>> # List available datasets first
    >>> datasets = monarch.list_datasets()
    >>> df = monarch.get_dataset(dataset="gene_to_phenotypic_feature_association")
    >>> 
    >>> # Use a direct URL (takes precedence over dataset)
    >>> df = monarch.get_dataset(url="https://data.monarchinitiative.org/.../file.tsv.gz")
    >>> 
    >>> # Force re-download
    >>> df = monarch.get_dataset(dataset="causal_gene_to_disease_association", force=True)
    """
    # Determine URL from dataset name or use provided URL
    if url is None:
        if dataset is None:
            # Default to causal_gene_to_disease_association
            url = CAUSAL_GENE_TO_DISEASE_URL
        else:
            # Look up URL from list_datasets()
            datasets_dict = list_datasets(base_url=ASSOCIATIONS_BASE_URL)
            if dataset not in datasets_dict:
                available = ", ".join(sorted(datasets_dict.keys())[:10])
                raise ValueError(
                    f"Dataset '{dataset}' not found. "
                    f"Available datasets include: {available}... "
                    f"(use list_datasets() to see all {len(datasets_dict)} datasets)"
                )
            url = datasets_dict[dataset]
    
    if cache_dir is None:
        cache_dir = CACHE_DIR
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL
    filename = url.split('/')[-1]
    local_file_path = cache_dir / filename
    
    # Download file if needed
    if force or not local_file_path.exists():
        if verbose >= 1:
            logger.info(f"Downloading {filename} from Monarch Initiative...")
        
        # Create a session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=5,  # Total number of retries
            backoff_factor=2,  # Exponential backoff: 2, 4, 8, 16, 32 seconds
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET", "HEAD"],  # Only retry safe methods
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Download file with retry logic
        max_retries = 5
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                if verbose >= 2:
                    logger.info(f"Downloading {filename} (attempt {retry_count + 1}/{max_retries})...")
                
                response = session.get(url, stream=True, timeout=120)
                response.raise_for_status()
                
                # Get file size if available
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar if verbose
                with open(local_file_path, "wb") as f:
                    if verbose >= 1 and total_size > 0:
                        # Use tqdm for download progress
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename, leave=False) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        # Simple download without progress bar
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                
                success = True
                if verbose >= 1:
                    logger.info(f"Downloaded {filename}")
            
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, 
                    requests.exceptions.RequestException) as e:
                retry_count += 1
                if retry_count < max_retries:
                    # Exponential backoff: wait 2^retry_count seconds
                    wait_time = 2 ** retry_count
                    if verbose >= 1:
                        logger.warning(f"Error downloading {filename} (attempt {retry_count}/{max_retries}): {e}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error downloading {filename} after {max_retries} attempts: {e}")
                    # Remove partial file if it exists
                    if local_file_path.exists():
                        try:
                            local_file_path.unlink()
                        except Exception:
                            pass
                    raise
            except Exception as e:
                # For non-retryable errors, fail immediately
                logger.error(f"Error downloading {filename}: {e}")
                # Remove partial file if it exists
                if local_file_path.exists():
                    try:
                        local_file_path.unlink()
                    except Exception:
                        pass
                raise
    else:
        if verbose >= 1:
            logger.info(f"Using cached file: {local_file_path}")
    
    # Read the TSV file
    if verbose >= 1:
        logger.info(f"Reading TSV file: {filename}")
    
    try:
        # Read gzipped TSV file using polars
        # Polars can handle gzip compression automatically
        read_kwargs = {
            "separator": "\t",
            "infer_schema_length": 10000,  # Sample more rows for better schema inference
            "ignore_errors": True,  # Handle type mismatches and ragged lines gracefully
            "try_parse_dates": True,  # Try to parse date columns
            "truncate_ragged_lines": True,  # Handle rows with inconsistent column counts
            "quote_char": None,  # Disable quote parsing for TSV
        }
        
        df = pl.read_csv(local_file_path, **read_kwargs)
        
        if verbose >= 1:
            logger.info(f"Loaded {len(df)} rows from {filename}")
        
        # Convert to pandas if requested
        if output_format == "pandas":
            df = df.to_pandas()
        
        # Report DataFrame shape
        if verbose:
            print(f"\nDataFrame shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error reading TSV file {local_file_path}: {e}")
        raise


def read_causal_gene_to_disease_association(
    url: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
    output_format: str = "polars",
    verbose: int = 1,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Download and read the causal gene-to-disease association TSV file from Monarch Initiative.
    
    This function downloads the compressed TSV file from Monarch Initiative, caches it locally,
    and returns it as a DataFrame. The file is only downloaded once unless force=True.
    
    Parameters
    ----------
    url : str, optional
        URL of the TSV.gz file to download.
        If None, uses the default Monarch Initiative URL for causal gene-to-disease associations.
    cache_dir : str or Path, optional
        Local directory to cache downloaded files. If None, uses ~/.cache/monarch.
    force : bool, default False
        If True, re-download the file even if it already exists in cache.
    output_format : str, default "polars"
        Output format: "pandas" or "polars".
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        DataFrame containing the causal gene-to-disease association data.
    
    Examples
    --------
    >>> import biodb.monarch as monarch
    >>> 
    >>> # Read the default causal gene-to-disease associations
    >>> df = monarch.read_causal_gene_to_disease_association()
    >>> 
    >>> # Use pandas format
    >>> df = monarch.read_causal_gene_to_disease_association(output_format="pandas")
    >>> 
    >>>     # Force re-download
    >>> df = monarch.read_causal_gene_to_disease_association(force=True)
    """
    if url is None:
        url = CAUSAL_GENE_TO_DISEASE_URL
    
    if cache_dir is None:
        cache_dir = CACHE_DIR
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from URL
    filename = url.split('/')[-1]
    local_file_path = cache_dir / filename
    
    # Download file if needed
    if force or not local_file_path.exists():
        if verbose >= 1:
            logger.info(f"Downloading {filename} from Monarch Initiative...")
        
        # Create a session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=5,  # Total number of retries
            backoff_factor=2,  # Exponential backoff: 2, 4, 8, 16, 32 seconds
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET", "HEAD"],  # Only retry safe methods
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Download file with retry logic
        max_retries = 5
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                if verbose >= 2:
                    logger.info(f"Downloading {filename} (attempt {retry_count + 1}/{max_retries})...")
                
                response = session.get(url, stream=True, timeout=120)
                response.raise_for_status()
                
                # Get file size if available
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar if verbose
                with open(local_file_path, "wb") as f:
                    if verbose >= 1 and total_size > 0:
                        # Use tqdm for download progress
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename, leave=False) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        # Simple download without progress bar
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                
                success = True
                if verbose >= 1:
                    logger.info(f"Downloaded {filename}")
            
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, 
                    requests.exceptions.RequestException) as e:
                retry_count += 1
                if retry_count < max_retries:
                    # Exponential backoff: wait 2^retry_count seconds
                    wait_time = 2 ** retry_count
                    if verbose >= 1:
                        logger.warning(f"Error downloading {filename} (attempt {retry_count}/{max_retries}): {e}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error downloading {filename} after {max_retries} attempts: {e}")
                    # Remove partial file if it exists
                    if local_file_path.exists():
                        try:
                            local_file_path.unlink()
                        except Exception:
                            pass
                    raise
            except Exception as e:
                # For non-retryable errors, fail immediately
                logger.error(f"Error downloading {filename}: {e}")
                # Remove partial file if it exists
                if local_file_path.exists():
                    try:
                        local_file_path.unlink()
                    except Exception:
                        pass
                raise
    else:
        if verbose >= 1:
            logger.info(f"Using cached file: {local_file_path}")
    
    # Read the TSV file
    if verbose >= 1:
        logger.info(f"Reading TSV file: {filename}")
    
    try:
        # Read gzipped TSV file using polars
        # Polars can handle gzip compression automatically
        read_kwargs = {
            "separator": "\t",
            "infer_schema_length": 10000,  # Sample more rows for better schema inference
            "ignore_errors": True,  # Handle type mismatches and ragged lines gracefully
            "try_parse_dates": True,  # Try to parse date columns
            "truncate_ragged_lines": True,  # Handle rows with inconsistent column counts
            "quote_char": None,  # Disable quote parsing for TSV
        }
        
        df = pl.read_csv(local_file_path, **read_kwargs)
        
        if verbose >= 1:
            logger.info(f"Loaded {len(df)} rows from {filename}")
        
        # Convert to pandas if requested
        if output_format == "pandas":
            df = df.to_pandas()
        
        # Report DataFrame shape
        if verbose:
            print(f"\nDataFrame shape: {df.shape}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error reading TSV file {local_file_path}: {e}")
        raise


def get_gene_associations(
    datasets: Optional[list] = None,
    species: Optional[list] = None,
    default_score: Optional[float] = None,
    output_format: str = "pandas",
    cache_dir: Optional[Union[str, Path]] = None,
    force: bool = False,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare gene association matrix from multiple Monarch Initiative datasets.
    
    This function downloads and processes multiple Monarch datasets, standardizes
    their format, and combines them into a single DataFrame ready for use with
    create_gene_association_matrix().
    
    Parameters
    ----------
    datasets : list of str, optional
        List of dataset names to include. If None, uses default selection:
        - causal_gene_to_disease_association
        - correlated_gene_to_disease_association
        - gene_to_expression_site_association
        - gene_to_pathway_association
        - gene_to_phenotypic_feature_association
    species : list of str, optional
        List of species to filter by (based on subject_taxon_label column).
        If None, defaults to ["Homo sapiens"].
    default_score : float or None, default None
        Default score value to assign if a dataset doesn't have a "score" column.
        If None, fills the score column with NaN/NA values.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    cache_dir : str or Path, optional
        Local directory to cache downloaded files. If None, uses ~/.cache/monarch.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with standardized columns:
        - database: "Monarch" for all rows
        - dataset: Name of the source dataset
        - sourceId: dataset + "." + object (unique identifier for each association source)
        - targetId: subject (gene identifier)
        - score: Association score (from dataset or default_score)
        - All other columns from the original datasets
    
    Examples
    --------
    >>> import biodb.monarch as monarch
    >>> 
    >>> # Use default datasets and species
    >>> associations = monarch.prepare_gene_association_matrix()
    >>> 
    >>> # Custom dataset selection
    >>> datasets = [
    ...     "causal_gene_to_disease_association",
    ...     "gene_to_phenotypic_feature_association"
    ... ]
    >>> associations = monarch.prepare_gene_association_matrix(datasets=datasets)
    >>> 
    >>> # Filter by multiple species
    >>> associations = monarch.prepare_gene_association_matrix(
    ...     species=["Homo sapiens", "Mus musculus"]
    ... )
    >>> 
    >>> # Use with create_gene_association_matrix
    >>> import biodb.utils as utils
    >>> associations = monarch.prepare_gene_association_matrix()
    >>> X, metadata = utils.create_gene_association_matrix(associations)
    """
    # Default datasets if not provided
    if datasets is None:
        datasets = [
            "causal_gene_to_disease_association",
            "correlated_gene_to_disease_association",
            "gene_to_expression_site_association",
            "gene_to_pathway_association",
            "gene_to_phenotypic_feature_association",
        ]
    
    # Default species if not provided
    if species is None:
        species = ["Homo sapiens"]
    
    if verbose >= 1:
        logger.info(f"Preparing gene association matrix from {len(datasets)} Monarch datasets")
        logger.info(f"Filtering by species: {species}")
    
    monarch_associations = []
    
    # Process each dataset
    for ds in tqdm(datasets, desc="Loading datasets", disable=(verbose == 0)):
        if verbose >= 2:
            logger.info(f"Processing dataset: {ds}")
        
        # Download and load dataset
        df = get_dataset(
            dataset=ds,
            cache_dir=cache_dir,
            force=force,
            output_format=output_format,
            verbose=verbose - 1 if verbose > 0 else 0,
        )
        
        # Add database and dataset columns at the beginning
        df.insert(0, "database", "Monarch")
        df.insert(1, "dataset", ds)
        
        # Add score column if it doesn't exist
        if "score" not in df.columns:
            if default_score is None:
                df["score"] = pd.NA
            else:
                df["score"] = default_score
            if verbose >= 2:
                if default_score is None:
                    logger.info("  Added score column with NaN/NA values")
                else:
                    logger.info(f"  Added default score column with value {default_score}")
        
        monarch_associations.append(df)
    
    # Concatenate all datasets
    if verbose >= 1:
        logger.info("Combining datasets...")
    monarch_associations = pd.concat(monarch_associations, ignore_index=True)
    
    # Create sourceId from dataset + "." + object
    if "object" not in monarch_associations.columns:
        raise ValueError(
            f"'object' column not found in Monarch datasets. "
            f"Available columns: {list(monarch_associations.columns)}"
        )
    monarch_associations["sourceId"] = (
        monarch_associations["dataset"] + "." + monarch_associations["object"].astype(str)
    )
    
    # Create targetId from subject
    if "subject" not in monarch_associations.columns:
        raise ValueError(
            f"'subject' column not found in Monarch datasets. "
            f"Available columns: {list(monarch_associations.columns)}"
        )
    monarch_associations["targetId"] = monarch_associations["subject"]
    
    # Filter by species if subject_taxon_label column exists
    if "subject_taxon_label" in monarch_associations.columns:
        initial_count = len(monarch_associations)
        monarch_associations = monarch_associations.loc[
            monarch_associations["subject_taxon_label"].isin(species)
        ]
        filtered_count = len(monarch_associations)
        if verbose >= 1:
            logger.info(
                f"Filtered by species: {initial_count:,} -> {filtered_count:,} rows "
                f"({filtered_count/initial_count*100:.1f}% retained)"
            )
    else:
        if verbose >= 1:
            logger.warning(
                "'subject_taxon_label' column not found. "
                "Skipping species filtering."
            )
    
    if verbose >= 1:
        logger.info(f"Final DataFrame shape: {monarch_associations.shape}")
        logger.info(f"Unique sourceIds: {monarch_associations['sourceId'].nunique():,}")
        logger.info(f"Unique targetIds: {monarch_associations['targetId'].nunique():,}")
    
    return monarch_associations


# ─── Monarch BioLink v3 REST API (targeted lookups) ────────────────────────
# Complements the bulk TSV readers above with one-record-at-a-time
# queries against ``api-v3.monarchinitiative.org``. Use this when you need
# a fresh single-entity payload (gene, disease, phenotype) without
# downloading a whole Monarch KG dump.

MONARCH_API_BASE_URL = "https://api-v3.monarchinitiative.org/v3/api"
"""Monarch v3 BioLink REST API root."""


def _monarch_get(path: str, params: Optional[Dict] = None, timeout: int = 30):
    """GET ``MONARCH_API_BASE_URL/path`` and return the JSON body.

    Raises
    ------
    requests.HTTPError
        on non-2xx response.
    """
    url = f"{MONARCH_API_BASE_URL}/{path.lstrip('/')}"
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def query_entity(entity_id: str, *, timeout: int = 30) -> Dict:
    """Fetch one entity (gene, disease, phenotype, …) by CURIE.

    Parameters
    ----------
    entity_id : str
        Entity identifier in CURIE form, e.g. ``"HGNC:1100"`` (BRCA1),
        ``"MONDO:0007254"`` (breast cancer), ``"HP:0001250"`` (seizure).
    timeout : int, default 30

    Returns
    -------
    dict
        The Monarch BioLink entity payload: ``id``, ``name``, ``category``,
        ``description``, ``xref``, ``synonym``, ``in_taxon``, …

    Examples
    --------
    >>> brca1 = query_entity("HGNC:1100")  # doctest: +SKIP
    >>> brca1["name"]  # doctest: +SKIP
    'BRCA1'
    """
    return _monarch_get(f"entity/{entity_id}", timeout=timeout)


def query_associations(
    subject: Optional[str] = None,
    object: Optional[str] = None,
    predicate: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    timeout: int = 30,
) -> Dict:
    """Fetch BioLink associations from the Monarch v3 REST API.

    Parameters
    ----------
    subject : str, optional
        Subject CURIE filter (e.g. ``"HGNC:1100"``).
    object : str, optional
        Object CURIE filter.
    predicate : str, optional
        BioLink predicate (e.g. ``"biolink:causes"``).
    category : str, optional
        BioLink association category.
    limit, offset : int
        Pagination knobs.
    timeout : int, default 30

    Returns
    -------
    dict
        ``{"limit", "offset", "total", "items": [association, …]}``.
        Each item carries ``subject`` / ``predicate`` / ``object`` triples
        plus rich label/closure/taxon metadata.

    Examples
    --------
    >>> hits = query_associations(subject="HGNC:1100", limit=5)  # doctest: +SKIP
    >>> hits["total"]  # doctest: +SKIP
    2352
    """
    params = {"limit": limit, "offset": offset}
    if subject is not None:
        params["subject"] = subject
    if object is not None:
        params["object"] = object
    if predicate is not None:
        params["predicate"] = predicate
    if category is not None:
        params["category"] = category
    return _monarch_get("association", params=params, timeout=timeout)


def query_gene_associations(gene_id: str, *, limit: int = 100) -> pd.DataFrame:
    """Convenience: every association whose subject is ``gene_id``.

    Parameters
    ----------
    gene_id : str
        Gene CURIE, e.g. ``"HGNC:1100"`` for BRCA1.
    limit : int, default 100
        Max rows per response page; the function paginates internally
        until ``total`` is exhausted (or ``limit * pages_seen`` reaches
        ~1000, whichever comes first — keeps response size bounded).

    Returns
    -------
    pandas.DataFrame
        Columns mirror the BioLink association schema.
    """
    rows: list[Dict] = []
    offset = 0
    for _ in range(10):  # cap at 10 pages = ~1000 rows
        page = query_associations(subject=gene_id, limit=limit, offset=offset)
        items = page.get("items", [])
        rows.extend(items)
        if len(rows) >= page.get("total", 0) or not items:
            break
        offset += limit
    return pd.DataFrame(rows)
