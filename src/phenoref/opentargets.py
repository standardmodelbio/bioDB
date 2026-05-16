"""
OpenTargets Platform API integration for querying gene information.

This module provides functions to:
1. Get all gene names available in OpenTargets Platform
2. Get detailed information for each gene, especially associated phenotypes

This module uses gget (https://www.gget.bio) as the backend for querying OpenTargets Platform.

References:
- gget: https://pachterlab.github.io/gget/en/opentargets.html
- OpenTargets Platform: https://platform-docs.opentargets.org/
"""


import pandas as pd
import polars as pl
from typing import Optional, List, Dict, Any, Union, Set, Tuple
from tqdm import tqdm
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gzip
import json
import re
import io
import os
import time
import hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, spmatrix
import pickle

logger = logging.getLogger(__name__)
DOWNLOADS_BASE_URL = "http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.12/output/"
CACHE_DIR = Path("~/.cache/opentargets").expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SCORE = 0.5

def list_datasets(
    base_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    List all available datasets (top-level directories) in the OpenTargets Platform output directory.
    
    This function fetches the directory listing from the OpenTargets FTP server and
    extracts all top-level dataset directories that can be used with `get_dataset()`.
    
    Parameters
    ----------
    base_url : str, optional
        Base URL of the OpenTargets output directory.
        If None, uses DOWNLOADS_BASE_URL (default: platform 25.12).
    
    Returns
    -------
    dict
        Dictionary mapping dataset names to their full URLs.
        Example: {"association_overall_direct": "http://.../association_overall_direct", ...}
    
    Examples
    --------
    >>> import phenoref.opentargets as opentargets
    >>> 
    >>> # List all datasets
    >>> datasets = opentargets.list_datasets()
    >>> print(datasets)
    >>> # {'association_overall_direct': 'http://...', ...}
    >>> 
    >>> # Use a dataset URL with get_dataset
    >>> datasets = opentargets.list_datasets()
    >>> df = opentargets.get_dataset(remote_url=datasets["association_overall_direct"])
    
    Notes
    -----
    - See https://platform.opentargets.org/downloads for documentation on available datasets
    - Dataset names are URL-encoded in the directory listing (e.g., "association_overall_direct")
    """
    if base_url is None:
        base_url = DOWNLOADS_BASE_URL
    
    # Ensure URL doesn't end with slash for consistency
    base_url = base_url.rstrip('/')
    
    # Extract parent path pattern to filter out parent directory links
    # e.g., from "http://.../platform/25.12/output" extract "/pub/databases/opentargets/platform"
    # This will match parent directory links for any version
    parent_path_pattern = None
    try:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        # Extract path segments up to "platform" (before version number)
        path_parts = parsed.path.strip('/').split('/')
        if 'platform' in path_parts:
            platform_idx = path_parts.index('platform')
            # Get path up to and including "platform"
            parent_path = '/' + '/'.join(path_parts[:platform_idx + 1])
            parent_path_pattern = parent_path
    except Exception:
        pass
    
    logger.info(f"Fetching directory listing from: {base_url}")
    
    # Fetch the directory listing (HTML page)
    try:
        response = requests.get(base_url + '/', timeout=30)
        response.raise_for_status()
        html_content = response.text
    except Exception as e:
        logger.error(f"Error fetching directory listing from {base_url}: {e}")
        raise
    
    # Helper function to check if a directory is a parent directory path
    def is_parent_directory(dir_name: str) -> bool:
        """Check if directory name represents a parent directory path."""
        if not dir_name:
            return False
        # Check standard parent directory patterns
        if dir_name in ['..', '.', 'Parent Directory', '../', './']:
            return True
        if dir_name.startswith('../'):
            return True
        # Check if it matches the parent path pattern (e.g., /pub/databases/opentargets/platform)
        if parent_path_pattern and dir_name.startswith(parent_path_pattern):
            return True
        # Check if it's a path going up (starts with /pub/)
        if dir_name.startswith('/pub/'):
            return True
        return False
    
    # Parse HTML to find directories
    # Directories appear as links ending with "/" or as folder icons
    # Pattern 1: <a href="directory_name/">directory_name/</a>
    # Pattern 2: <a href="directory_name/"> (just the link)
    # Pattern 3: Directory entries in table format
    
    datasets = []
    
    # Try multiple patterns to extract directory names
    # Pattern 1: Links ending with "/" (excluding parent directory)
    link_pattern = r'<a[^>]+href=["\']([^"\']+/)[^"\']*["\'][^>]*>'
    matches = re.findall(link_pattern, html_content, re.IGNORECASE)
    
    for match in matches:
        # Remove trailing slash and decode URL encoding
        dir_name = match.rstrip('/')
        # Skip parent directory and other non-dataset entries
        if is_parent_directory(dir_name):
            continue
        # Decode URL encoding (e.g., %5F -> _)
        try:
            from urllib.parse import unquote
            dir_name = unquote(dir_name)
        except Exception:
            pass
        # Check again after decoding
        if is_parent_directory(dir_name):
            continue
        if dir_name and dir_name not in datasets:
            datasets.append(dir_name)
    
    # Pattern 2: Look for directory entries in table format
    # Sometimes directories are listed as <td> entries
    if not datasets:
        # Try finding folder icons or directory indicators
        folder_pattern = r'<td[^>]*>.*?<a[^>]*href=["\']([^"\']+)[^"\']*["\'][^>]*>.*?</a>.*?</td>'
        matches = re.findall(folder_pattern, html_content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            if match.endswith('/') or '%' in match:
                dir_name = match.rstrip('/')
                try:
                    from urllib.parse import unquote
                    dir_name = unquote(dir_name)
                except Exception:
                    pass
                # Skip parent directory and other non-dataset entries
                if is_parent_directory(dir_name):
                    continue
                if dir_name and dir_name not in datasets:
                    datasets.append(dir_name)
    
    # Sort datasets alphabetically
    datasets = sorted(set(datasets))
    
    if len(datasets) == 0:
        logger.warning("No datasets found in directory listing")
        return {}
    
    # Build dictionary mapping dataset names to URLs
    result_dict = {}
    for dataset in datasets:
        result_dict[dataset] = f"{base_url}/{dataset}"
    
    return result_dict


def get_dataset(
    dataset: Optional[str] = None,
    remote_url: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force: bool = False,
    output_format: str = "pandas",
    verbose: int = 1,
    limit: Optional[int] = None,
    parse: bool = True,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Download all parquet files from a remote OpenTargets directory, cache them locally,
    and concatenate them in memory.

    See here for all datasets: https://platform.opentargets.org/downloads
    
    This function is useful for downloading large association datasets from OpenTargets
    Platform releases. It will:
    1. List all .parquet files in the remote directory
    2. Download them to the cache directory (skipping if already cached)
    3. Load and concatenate all files in memory
    4. Return the concatenated DataFrame
    
    Parameters
    ----------
    dataset : str, optional
        Name of the dataset (e.g., "association_overall_direct", "disease", "target").
        If provided, the remote_url will be automatically constructed from DOWNLOADS_BASE_URL.
        Use list_datasets() to see all available dataset names.
        If both dataset and remote_url are provided, remote_url takes precedence.
    remote_url : str, optional
        Base URL of the remote directory containing parquet files.
        Should end without a trailing slash.
        If not provided and dataset is not provided, defaults to association_overall_direct.
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses DEFAULT_CACHE_DIR.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars"
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress bars and summary
        - 2: Show detailed information
    limit : int, optional
        Limit the number of parquet files to read. Useful for checking content quickly.
        If None, all files will be read. Default None.
    parse : bool, default True
        If True, automatically parse nested structures for datasets that support it
        (e.g., "target_essentiality", "expression"). If False, return raw data.
    
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Concatenated DataFrame containing all data from the parquet files.
    
    Examples
    --------
    >>> import phenoref.opentargets as opentargets
    >>> 
    >>> # Download and concatenate association data using dataset name
    >>> associations_df = opentargets.get_dataset(dataset="association_overall_direct")
    >>> 
    >>> # List available datasets first
    >>> datasets = opentargets.list_datasets()
    >>> df = opentargets.get_dataset(dataset="disease")
    >>> 
    >>> # Use a custom remote_url (takes precedence over dataset)
    >>> associations_df = opentargets.get_dataset(
    ...     remote_url="http://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.09/output/association_overall_direct"
    ... )
    >>> 
    >>> # Force re-download
    >>> associations_df = opentargets.get_dataset(dataset="association_overall_direct", force=True)
    """
    # Determine remote_url from dataset or use provided remote_url
    if remote_url is None:
        if dataset is None:
            # Default to association_overall_direct
            remote_url = f"{DOWNLOADS_BASE_URL}association_overall_direct"
        else:
            # Construct URL from dataset name (DOWNLOADS_BASE_URL already ends with /)
            remote_url = f"{DOWNLOADS_BASE_URL}{dataset}"
    
    if cache_dir is None:
        cache_dir = CACHE_DIR
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure URL doesn't end with slash
    remote_url = remote_url.rstrip('/')
    
    # Create a unique marker file name based on the remote URL
    # Sanitize URL to create a valid filename
    url_hash = hashlib.md5(remote_url.encode()).hexdigest()[:12]
    marker_file = cache_dir / f".download_complete_{url_hash}.json"
    
    # Check if download is complete (unless forcing)
    if not force and marker_file.exists():
        try:
            with open(marker_file, 'r') as f:
                marker_data = json.load(f)
            
            # Verify the marker matches this URL
            if marker_data.get('remote_url') == remote_url:
                if verbose >= 1:
                    logger.info(f"Using cached files (download complete marker found). Use force=True to re-download.")
                
                # Get list of parquet files from marker
                parquet_files = marker_data.get('parquet_files', [])
                
                if not parquet_files:
                    if verbose >= 1:
                        logger.warning("Marker file exists but no parquet files listed. Re-fetching directory listing...")
                    # Fall through to normal download process
                else:
                    # Skip FTP check and use cached files
                    if verbose >= 1:
                        logger.info(f"Found {len(parquet_files)} parquet files from cache marker")
                    
                    # Apply limit if specified
                    if limit is not None and limit > 0:
                        parquet_files = parquet_files[:limit]
                        if verbose >= 1:
                            logger.info(f"Limiting to first {limit} parquet files")
                    
                    # Verify all files exist
                    missing_files = []
                    for filename in parquet_files:
                        if not (cache_dir / filename).exists():
                            missing_files.append(filename)
                    
                    if missing_files:
                        if verbose >= 1:
                            logger.warning(f"Some cached files are missing: {missing_files}. Re-downloading...")
                        # Delete marker and fall through to normal download
                        try:
                            marker_file.unlink()
                        except Exception:
                            pass
                    else:
                        # All files exist, skip to loading phase
                        downloaded_files = [cache_dir / filename for filename in parquet_files]
                        # Extract just filenames for later use
                        parquet_filenames = [f.name for f in downloaded_files]
                        # Skip to the loading section (will be handled below)
                        skip_download = True
            else:
                # URL mismatch, delete old marker
                if verbose >= 1:
                    logger.info(f"Marker file exists but for different URL. Re-downloading...")
                try:
                    marker_file.unlink()
                except Exception:
                    pass
                skip_download = False
        except Exception as e:
            if verbose >= 1:
                logger.warning(f"Error reading marker file: {e}. Re-fetching directory listing...")
            skip_download = False
    else:
        skip_download = False
        parquet_filenames = None
    
    # If forcing, delete marker file
    if force and marker_file.exists():
        try:
            marker_file.unlink()
            if verbose >= 1:
                logger.info("Deleted existing download complete marker (force=True)")
        except Exception:
            pass
    
    if not skip_download:
        if verbose >= 1:
            logger.info(f"Listing parquet files from: {remote_url}")
        
        # Fetch the directory listing (HTML page)
        try:
            response = requests.get(remote_url + '/', timeout=30)
            response.raise_for_status()
            html_content = response.text
        except Exception as e:
            logger.error(f"Error fetching directory listing from {remote_url}: {e}")
            raise
        
        # Parse HTML to find .parquet files
        # Common FTP/HTTP directory listing formats:
        # - Apache: <a href="filename.parquet">filename.parquet</a>
        # - Simple: filename.parquet (plain text)
        parquet_files = []
        
        # Try to find parquet files using regex (works for most directory listings)
        # Pattern: href="filename.parquet" or just "filename.parquet"
        parquet_pattern = r'([^/\s<>"]+\.parquet)'
        matches = re.findall(parquet_pattern, html_content, re.IGNORECASE)
        
        # Filter to get unique filenames
        parquet_files = sorted(set(matches))
        
        if len(parquet_files) == 0:
            # Try alternative: look for links in HTML
            # Pattern for <a href="..."> tags
            link_pattern = r'<a[^>]+href=["\']([^"\']+\.parquet)["\']'
            matches = re.findall(link_pattern, html_content, re.IGNORECASE)
            parquet_files = sorted(set(matches))
        
        if len(parquet_files) == 0:
            raise ValueError(f"No parquet files found in directory listing at {remote_url}")
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            parquet_files = parquet_files[:limit]
            if verbose >= 1:
                logger.info(f"Limiting to first {limit} parquet files")
        
        if verbose >= 1:
            logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Store parquet filenames for marker file
        parquet_filenames = parquet_files.copy()
        
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
        
        # Download files
        downloaded_files = []
        skipped_files = []
        
        iterator = tqdm(parquet_files, desc="Downloading parquet files") if verbose >= 1 else parquet_files
        
        for idx, filename in enumerate(iterator):
            remote_file_url = f"{remote_url}/{filename}"
            local_file_path = cache_dir / filename
            
            # Skip if already exists and not forcing
            if not force and local_file_path.exists():
                if verbose >= 2:
                    logger.info(f"Skipping {filename} (already cached)")
                skipped_files.append(filename)
                downloaded_files.append(local_file_path)
                continue
            
            # Add delay between requests to avoid rate limiting (except for first file)
            if idx > 0:
                delay = 1.0  # 1 second delay between downloads
                if verbose >= 2:
                    logger.debug(f"Waiting {delay}s before next download to avoid rate limits...")
                time.sleep(delay)
            
            # Download file with retry logic
            max_retries = 5
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    if verbose >= 2:
                        logger.info(f"Downloading {filename} (attempt {retry_count + 1}/{max_retries})...")
                    
                    response = session.get(remote_file_url, stream=True, timeout=120)
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
                    
                    downloaded_files.append(local_file_path)
                    success = True
                    if verbose >= 2:
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
        
        # After download loop completes, update parquet_filenames if needed
        if parquet_filenames is None:
            parquet_filenames = [f.name for f in downloaded_files]
    
    # Load and concatenate all parquet files
    if verbose >= 1:
        logger.info(f"Loading and concatenating {len(downloaded_files)} parquet files...")
    
    dataframes = []
    iterator = tqdm(downloaded_files, desc="Loading parquet files") if verbose >= 1 else downloaded_files
    skipped_corrupted = []
    
    for file_path in iterator:
        try:
            if output_format == "pandas":
                df = pd.read_parquet(file_path)
            else:
                df = pl.read_parquet(file_path)
            
            if len(df) > 0:
                dataframes.append(df)
                if verbose >= 2:
                    logger.info(f"Loaded {len(df)} rows from {file_path.name}")
        except Exception as e:
            # Check if it's a parquet corruption error
            error_str = str(e)
            is_corrupted = (
                "Parquet magic bytes not found" in error_str or
                "corrupted" in error_str.lower() or
                "ArrowInvalid" in str(type(e).__name__) or
                "not a parquet file" in error_str.lower()
            )
            
            if is_corrupted:
                logger.warning(f"Corrupted parquet file detected: {file_path.name}. Attempting to re-download...")
                
                # Delete corrupted file
                try:
                    file_path.unlink()
                    if verbose >= 1:
                        logger.info(f"Deleted corrupted file: {file_path.name}")
                except Exception as del_e:
                    logger.warning(f"Could not delete corrupted file {file_path.name}: {del_e}")
                
                # Try to re-download once
                filename = file_path.name
                remote_file_url = f"{remote_url}/{filename}"
                max_retries = 3
                retry_count = 0
                download_success = False
                
                while retry_count < max_retries and not download_success:
                    try:
                        if verbose >= 1:
                            logger.info(f"Re-downloading {filename} (attempt {retry_count + 1}/{max_retries})...")
                        
                        response = session.get(remote_file_url, stream=True, timeout=120)
                        response.raise_for_status()
                        
                        with open(file_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # Try reading again
                        if output_format == "pandas":
                            df = pd.read_parquet(file_path)
                        else:
                            df = pl.read_parquet(file_path)
                        
                        if len(df) > 0:
                            dataframes.append(df)
                            download_success = True
                            if verbose >= 1:
                                logger.info(f"Successfully re-downloaded and loaded {filename}")
                        else:
                            retry_count += 1
                            if retry_count < max_retries:
                                file_path.unlink()  # Delete empty file
                    
                    except Exception as retry_e:
                        retry_count += 1
                        if retry_count < max_retries:
                            if file_path.exists():
                                try:
                                    file_path.unlink()
                                except Exception:
                                    pass
                            wait_time = 2 ** retry_count
                            if verbose >= 1:
                                logger.warning(f"Re-download attempt {retry_count} failed: {retry_e}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to re-download {filename} after {max_retries} attempts. Skipping file.")
                            skipped_corrupted.append(filename)
                
                if not download_success:
                    continue  # Skip to next file
            else:
                # For non-corruption errors, raise as before
                logger.error(f"Error loading {file_path}: {e}")
                raise
    
    if skipped_corrupted and verbose >= 1:
        logger.warning(f"Skipped {len(skipped_corrupted)} corrupted files that could not be recovered: {skipped_corrupted}")
    
    if len(dataframes) == 0:
        raise ValueError("No data loaded from parquet files")
    
    # Concatenate all DataFrames
    # Filter out empty DataFrames to avoid FutureWarning about empty/all-NA columns
    if output_format == "pandas":
        non_empty_dfs = [df for df in dataframes if not df.empty]
        if verbose >= 1:
            logger.info(f"Concatenating {len(non_empty_dfs)} non-empty DataFrames (filtered {len(dataframes) - len(non_empty_dfs)} empty)...")
        
        if non_empty_dfs:
            result_df = pd.concat(non_empty_dfs, ignore_index=True)
        else:
            # All DataFrames were empty, return empty DataFrame with columns from first (if any)
            if dataframes:
                result_df = pd.DataFrame(columns=dataframes[0].columns)
            else:
                result_df = pd.DataFrame()
        
        if verbose >= 1:
            logger.info(f"Concatenated DataFrame shape: {result_df.shape}")
    else:
        non_empty_dfs = [df for df in dataframes if df.height > 0]
        if verbose >= 1:
            logger.info(f"Concatenating {len(non_empty_dfs)} non-empty DataFrames (filtered {len(dataframes) - len(non_empty_dfs)} empty)...")
        
        if non_empty_dfs:
            result_df = pl.concat(non_empty_dfs)
        else:
            # All DataFrames were empty, return empty DataFrame with schema from first (if any)
            if dataframes:
                result_df = pl.DataFrame(schema=dataframes[0].schema)
            else:
                result_df = pl.DataFrame()
        
        if verbose >= 1:
            logger.info(f"Concatenated DataFrame shape: {result_df.shape}")
    
    # Report unique counts for columns ending in "Id" (case sensitive) or equal to "id"
    if output_format == "pandas":
        id_columns = [col for col in result_df.columns if col.endswith("Id") or col == "id"]
    else:
        id_columns = [col for col in result_df.columns if col.endswith("Id") or col == "id"]
    
    if id_columns:
        if verbose:
            print("\nUnique counts for columns ending in 'Id' or equal to 'id':")
            print("-" * 50)
        for col in id_columns:
            if output_format == "pandas":
                unique_count = result_df[col].nunique()
            else:
                unique_count = result_df[col].n_unique()
            if verbose:
                print(f"  {col}: {unique_count:,}")
    
    if verbose:
        print(f"\nDataFrame shape: {result_df.shape}")
    
    # Create download complete marker file (only if we actually downloaded/verified files)
    if not skip_download and parquet_filenames:
        try:
            marker_data = {
                'remote_url': remote_url,
                'parquet_files': parquet_filenames,
                'num_files': len(parquet_filenames),
                'timestamp': time.time(),
                'cache_dir': str(cache_dir),
            }
            with open(marker_file, 'w') as f:
                json.dump(marker_data, f, indent=2)
            if verbose >= 1:
                logger.info(f"Created download complete marker: {marker_file.name}")
        except Exception as e:
            if verbose >= 1:
                logger.warning(f"Could not create download complete marker: {e}")
    
    # Automatically parse target_essentiality dataset if requested and parse=True
    if parse and dataset == "target_essentiality" and 'geneEssentiality' in result_df.columns:
        if verbose >= 1:
            logger.info("Automatically parsing geneEssentiality column...")
            print("Automatically parsing geneEssentiality column...")
        result_df = parse_gene_essentiality(result_df)
        if verbose >= 1:
            logger.info(f"Parsed DataFrame shape: {result_df.shape}")
            print(f"Parsed DataFrame shape: {result_df.shape}")
    
    # Automatically parse expression dataset if requested and parse=True
    if parse and dataset == "expression" and 'tissues' in result_df.columns:
        if verbose >= 1:
            logger.info("Automatically parsing tissues column...")
            print("Automatically parsing tissues column...")
        result_df = parse_expression(result_df)
        if verbose >= 1:
            logger.info(f"Parsed DataFrame shape: {result_df.shape}")
            print(f"Parsed DataFrame shape: {result_df.shape}")
    
    return result_df


def df_to_markdown(
    target_row: Union[pd.Series, Dict[str, Any]],
    include_cols: Optional[List[str]] = None,
    include_fields: Optional[Dict[str, List[str]]] = None,
    disease_to_gene: Optional[pd.DataFrame] = None,
    drug_to_gene: Optional[pd.DataFrame] = None,
    gene_to_pharmacogenomics: Optional[pd.DataFrame] = None,
) -> str:
    """
    Convert a target dataset row to markdown format.
    
    Parameters
    ----------
    target_row : pd.Series or dict
        A single row from the target dataset (from get_dataset(dataset="target"))
    include_cols : list of str, optional
        List of column/section names to include. Available sections:
        - "functionDescriptions" or "function_descriptions"
        - "subcellularLocations" or "subcellular_locations"
        - "go" or "go_terms"
        - "pathways"
        - "tractability"
        - "constraint"
        If None (default), includes all sections.
        Basic gene info (approvedSymbol, approvedName, id, biotype, genomicLocation)
        is always included.
    include_fields : dict, optional
        Dictionary mapping column names to field name(s) to extract from nested structures.
        For columns in this dict, only the specified fields will be extracted and shown.
        Can be a single string or a list of strings per column.
        Example: {"symbolSynonyms": "label", "pathways": ["pathway"]} will extract:
        - "label" field from items in symbolSynonyms list
        - "pathway" field from items in pathways list
        Multiple fields can be specified per column: {"symbolSynonyms": ["label", "source"]}
    disease_to_gene : pd.DataFrame, optional
        DataFrame with disease-to-gene associations. If provided, will be filtered by the target ID
        from target_row, aggregated, and appended to the markdown output. Expected columns:
        targetId, name, diseaseId, score, evidenceCount, description, synonyms.
    drug_to_gene : pd.DataFrame, optional
        DataFrame with drug-to-gene associations. If provided, will be filtered by the target ID
        from target_row and appended to the markdown output. Expected columns:
        targetId, drugId, prefName, tradeNames, synonyms, drugType, mechanismOfAction, phase, status, etc.
    
    Returns
    -------
    str
        Markdown formatted string with gene information
    
    Examples
    --------
    >>> import phenoref.opentargets as opentargets
    >>> target_df = opentargets.get_dataset(dataset="target")
    >>> # Get markdown for first gene with all sections
    >>> markdown = opentargets.target_to_markdown(target_df.iloc[0])
    >>> print(markdown)
    >>> 
    >>> # Get markdown with only GO terms and tractability
    >>> markdown = opentargets.target_to_markdown(
    ...     target_df.iloc[0],
    ...     include_cols=["go", "tractability"]
    ... )
    >>> 
    >>> # Extract only label values from symbolSynonyms and pathway from pathways
    >>> markdown = opentargets.target_to_markdown(
    ...     target_df.iloc[0],
    ...     include_fields={"symbolSynonyms": ["label"], "pathways": ["pathway"]}
    ... )
    """
    import json
    
    # Handle Series, DataFrame, and dict
    if isinstance(target_row, pd.DataFrame):
        # If DataFrame, take the first row
        if len(target_row) == 0:
            raise ValueError("target_row DataFrame is empty")
        row_dict = target_row.iloc[0].to_dict()
    elif isinstance(target_row, pd.Series):
        row_dict = target_row.to_dict()
    else:
        row_dict = target_row
    
    # Helper function to safely get value
    def get_value(key, default=""):
        val = row_dict.get(key, default)
        if val is None:
            return default
        # Check for NaN/None - handle arrays separately
        try:
            if pd.isna(val):
                return default
        except (ValueError, TypeError):
            # pd.isna might fail for arrays or other types, check if it's an empty array
            if isinstance(val, (list, np.ndarray)) and len(val) == 0:
                return default
            # If it's not NaN and not empty, return the value
            pass
        return val
    
    # Helper to check if value is valid (not None, NaN, or empty)
    def is_valid_value(val):
        if val is None:
            return False
        try:
            if isinstance(val, float) and pd.isna(val):
                return False
        except (ValueError, TypeError):
            pass
        if isinstance(val, (list, np.ndarray)):
            return len(val) > 0
        # For other types, check if truthy (but avoid array truthiness check)
        if isinstance(val, (dict, str)):
            return bool(val)
        try:
            return bool(val)
        except (ValueError, TypeError):
            return False
    
    # Generic formatters by data type
    def format_scalar(val):
        """Format scalar values (str, int, float, bool)."""
        if not is_valid_value(val):
            return ""
        if isinstance(val, bool):
            return "Yes" if val else "No"
        if isinstance(val, (int, float)):
            # Format numbers with thousand separators
            if isinstance(val, float) and val.is_integer():
                return f"{int(val):,}"
            return f"{val:,}"
        return str(val)
    
    def format_dict(val, preferred_keys=None):
        """Format dictionary values, extracting preferred keys or all key-value pairs."""
        if not isinstance(val, dict) or not is_valid_value(val):
            return None
        
        # Try preferred keys first (for extracting specific fields)
        if preferred_keys:
            for key in preferred_keys:
                if key in val and is_valid_value(val[key]):
                    formatted = format_value(val[key])
                    # Ensure formatted is a string (format_value can return lists)
                    if formatted:
                        if isinstance(formatted, (list, np.ndarray)):
                            if isinstance(formatted, np.ndarray):
                                formatted = formatted.tolist()
                            formatted = ", ".join(str(x) for x in formatted)
                        elif not isinstance(formatted, str):
                            formatted = str(formatted)
                    return formatted
        
        # If no preferred keys or none found, format as key-value pairs
        formatted_items = []
        for k, v in val.items():
            if is_valid_value(v):
                formatted_v = format_value(v)
                if formatted_v:
                    # Ensure formatted_v is a string (format_value can return lists)
                    if isinstance(formatted_v, (list, np.ndarray)):
                        if isinstance(formatted_v, np.ndarray):
                            formatted_v = formatted_v.tolist()
                        formatted_v = ", ".join(str(x) for x in formatted_v)
                    elif not isinstance(formatted_v, str):
                        formatted_v = str(formatted_v)
                    formatted_items.append(f"{k}: {formatted_v}")
        return ", ".join(formatted_items) if formatted_items else None
    
    def format_list(val, item_formatter=None, deduplicate=True):
        """Format list/array values."""
        if not is_valid_value(val):
            return []
        
        # Parse JSON string if needed
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return [val] if val else []
        
        # Convert numpy array to list
        if isinstance(val, np.ndarray):
            val = val.tolist()
        
        if not isinstance(val, list):
            formatted_item = format_value(val, item_formatter)
            # Ensure formatted_item is a string (format_value can return lists)
            if formatted_item:
                if isinstance(formatted_item, (list, np.ndarray)):
                    if isinstance(formatted_item, np.ndarray):
                        formatted_item = formatted_item.tolist()
                    # Recursively ensure all items are strings
                    str_items = []
                    for x in formatted_item:
                        if isinstance(x, (list, np.ndarray)):
                            if isinstance(x, np.ndarray):
                                x = x.tolist()
                            str_items.append(", ".join(str(y) for y in x))
                        else:
                            str_items.append(str(x))
                    formatted_item = ", ".join(str_items)
                elif isinstance(formatted_item, dict):
                    formatted_item = format_dict(formatted_item) or str(formatted_item)
                elif not isinstance(formatted_item, str):
                    formatted_item = str(formatted_item)
                return [formatted_item] if formatted_item else []
            return []
        
        # Format each item
        formatted = []
        for item in val:
            if item_formatter:
                formatted_item = item_formatter(item)
            else:
                formatted_item = format_value(item)
            if formatted_item:
                # Ensure formatted_item is a string (handle lists, dicts, etc.)
                if not isinstance(formatted_item, str):
                    if isinstance(formatted_item, (list, np.ndarray)):
                        if isinstance(formatted_item, np.ndarray):
                            formatted_item = formatted_item.tolist()
                        # Recursively ensure all items are strings
                        str_items = []
                        for x in formatted_item:
                            if isinstance(x, (list, np.ndarray)):
                                if isinstance(x, np.ndarray):
                                    x = x.tolist()
                                str_items.append(", ".join(str(y) for y in x))
                            else:
                                str_items.append(str(x))
                        formatted_item = ", ".join(str_items)
                    elif isinstance(formatted_item, dict):
                        formatted_item = format_dict(formatted_item) or str(formatted_item)
                    else:
                        formatted_item = str(formatted_item)
                formatted.append(formatted_item)
        
        # Remove duplicates if requested
        if deduplicate:
            seen = set()
            unique = []
            for item in formatted:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            return unique
        
        return formatted
    
    def format_value(val, custom_formatter=None):
        """Generic formatter that dispatches based on type."""
        if custom_formatter:
            return custom_formatter(val)
        
        if not is_valid_value(val):
            return ""
        
        # Handle strings (including JSON strings)
        if isinstance(val, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(val)
                return format_value(parsed)
            except (json.JSONDecodeError, TypeError):
                return val
        
        # Handle dicts
        if isinstance(val, dict):
            return format_dict(val)
        
        # Handle lists/arrays
        if isinstance(val, (list, np.ndarray)):
            return format_list(val)
        
        # Handle scalars
        return format_scalar(val)
    
    # Helper function to recursively ensure a value is a string
    def ensure_string(val):
        """Recursively convert any value to a string, handling nested lists/dicts."""
        if isinstance(val, str):
            return val
        elif isinstance(val, (list, np.ndarray)):
            if isinstance(val, np.ndarray):
                val = val.tolist()
            # Recursively convert all items to strings
            str_items = []
            for item in val:
                str_items.append(ensure_string(item))
            return ", ".join(str_items)
        elif isinstance(val, dict):
            formatted = format_dict(val)
            return formatted if formatted else str(val)
        else:
            return str(val) if val is not None else ""
    
    # Special formatter for genomic location dict
    def format_genomic_location(gloc):
        """Format genomic location dict with specific formatting."""
        if not is_valid_value(gloc):
            return ""
        
        # Parse JSON string if needed
        if isinstance(gloc, str):
            try:
                gloc = json.loads(gloc)
            except (json.JSONDecodeError, TypeError):
                return gloc
        
        if isinstance(gloc, dict):
            parts = []
            if "chromosome" in gloc and is_valid_value(gloc["chromosome"]):
                parts.append(f"chr{gloc['chromosome']}")
            if "start" in gloc and is_valid_value(gloc["start"]):
                parts.append(f"start: {gloc['start']:,}")
            if "end" in gloc and is_valid_value(gloc["end"]):
                parts.append(f"end: {gloc['end']:,}")
            if "strand" in gloc and is_valid_value(gloc["strand"]):
                parts.append(f"strand: {gloc['strand']}")
            return ", ".join(parts) if parts else ""
        
        return format_value(gloc)
    
    # Normalize include_cols - if None, include all sections
    if include_cols is None:
        include_cols = ["functionDescriptions", "subcellularLocations", "go", "pathways", "tractability", "constraint"]
    else:
        # Normalize column names (allow both camelCase and snake_case)
        normalized_cols = []
        col_mapping = {
            "function_descriptions": "functionDescriptions",
            "subcellular_locations": "subcellularLocations",
            "go_terms": "go",
        }
        for col in include_cols:
            normalized_col = col_mapping.get(col, col)
            normalized_cols.append(normalized_col)
        include_cols = normalized_cols
    
    # Helper function to convert camelCase to Title Case
    def camel_to_title(name):
        """Convert camelCase to Title Case (e.g., 'approvedSymbol' -> 'Approved Symbol')."""
        import re
        # Insert space before uppercase letters (but not at the start)
        spaced = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
        # Capitalize first letter of each word
        return spaced.title()
    
    # Build markdown
    lines = []
    
    # Parse include_fields - it's now a dict mapping column_name -> list of fields to extract
    # Supports both {"column": "field"} and {"column": ["field1", "field2"]}
    field_mapping = {}
    if include_fields:
        if isinstance(include_fields, dict):
            # Direct dictionary format: {"column": ["field1", "field2"]} or {"column": "field"}
            for col_name, fields in include_fields.items():
                # Normalize: convert string to list, keep list as-is
                if isinstance(fields, str):
                    field_mapping[col_name] = [fields]
                elif isinstance(fields, list):
                    field_mapping[col_name] = fields
                else:
                    # Convert other types to string and then to list
                    field_mapping[col_name] = [str(fields)]
        else:
            # Backward compatibility: if it's a list, treat as old format and parse
            # This allows for graceful migration
            for field_path in include_fields:
                # Support format: "column.field" or "column.field1,field2"
                if '.' in field_path:
                    parts = field_path.split('.', 1)
                    col_name = parts[0]
                    fields_str = parts[1]
                    # Support multiple fields: "column.field1,field2"
                    fields = [f.strip() for f in fields_str.split(',')]
                    if col_name not in field_mapping:
                        field_mapping[col_name] = []
                    field_mapping[col_name].extend(fields)
                else:
                    # If no dot, treat as column name with default "label" field
                    if field_path not in field_mapping:
                        field_mapping[field_path] = []
                    field_mapping[field_path].append("label")
    
    # Helper function to extract specified fields from a list of dicts
    def extract_fields_from_list(val, column_name):
        """Extract specified fields from a list of dicts based on include_fields."""
        if not is_valid_value(val):
            return []
        
        # Check if this column has field specifications
        fields_to_extract = field_mapping.get(column_name, [])
        
        # Parse JSON string if needed
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return []
        
        # Convert numpy array to list
        if isinstance(val, np.ndarray):
            val = val.tolist()
        
        if not isinstance(val, list):
            return []
        
        extracted_values = []
        for item in val:
            if isinstance(item, dict):
                if fields_to_extract:
                    # Extract specified fields
                    field_values = []
                    for field in fields_to_extract:
                        field_val = item.get(field)
                        if is_valid_value(field_val):
                            # Format the field value properly (handle lists, dicts, etc.)
                            if isinstance(field_val, (list, np.ndarray)):
                                # If it's a list, format it
                                if isinstance(field_val, np.ndarray):
                                    field_val = field_val.tolist()
                                formatted_list = format_list(field_val, deduplicate=False)
                                if formatted_list:
                                    # Ensure all items are strings (handle nested structures)
                                    str_items = []
                                    for v in formatted_list:
                                        if isinstance(v, (list, np.ndarray)):
                                            if isinstance(v, np.ndarray):
                                                v = v.tolist()
                                            str_items.append(", ".join(str(x) for x in v))
                                        else:
                                            str_items.append(str(v))
                                    field_values.append(", ".join(str_items))
                            elif isinstance(field_val, dict):
                                # If it's a dict, format it
                                formatted_dict = format_dict(field_val)
                                if formatted_dict:
                                    field_values.append(formatted_dict)
                            else:
                                # Simple value, convert to string
                                field_values.append(str(field_val))
                    if field_values:
                        # Join multiple fields with comma
                        extracted_values.append(", ".join(field_values))
                else:
                    # No field specification, format as key-value pairs
                    formatted = format_dict(item)
                    if formatted:
                        extracted_values.append(formatted)
            else:
                # If it's not a dict, use it as-is
                if is_valid_value(item):
                    # Handle lists and dicts in non-dict items too
                    if isinstance(item, (list, np.ndarray)):
                        if isinstance(item, np.ndarray):
                            item = item.tolist()
                        formatted_list = format_list(item, deduplicate=False)
                        if formatted_list:
                            # Ensure all items are strings (handle nested structures)
                            str_items = []
                            for v in formatted_list:
                                if isinstance(v, (list, np.ndarray)):
                                    if isinstance(v, np.ndarray):
                                        v = v.tolist()
                                    str_items.append(", ".join(str(x) for x in v))
                                else:
                                    str_items.append(str(v))
                            extracted_values.append(", ".join(str_items))
                    elif isinstance(item, dict):
                        formatted_dict = format_dict(item)
                        if formatted_dict:
                            extracted_values.append(formatted_dict)
                    else:
                        extracted_values.append(str(item))
        
        # Remove duplicates while preserving order
        # Ensure all values are strings (handle any nested structures)
        seen = set()
        unique_values = []
        for val in extracted_values:
            # Ensure val is a string using the recursive helper
            val_str = ensure_string(val)
            
            if val_str and val_str not in seen:
                seen.add(val_str)
                unique_values.append(val_str)
        
        return unique_values
    
    # Header (always included, but skip None values)
    lines.append("# Gene Info")
    approved_symbol = get_value('approvedSymbol')
    if is_valid_value(approved_symbol):
        lines.append(f"{camel_to_title('approvedSymbol')}: {approved_symbol}")
    
    # Symbol synonyms (only include if available) - placed right after approvedSymbol
    symbol_synonyms_raw = get_value('symbolSynonyms')
    if is_valid_value(symbol_synonyms_raw):
        # Extract fields based on include_fields, or default to "label" if not specified
        if "symbolSynonyms" not in field_mapping:
            # Default: extract "label" field for symbolSynonyms
            field_mapping["symbolSynonyms"] = ["label"]
        
        unique_values = extract_fields_from_list(symbol_synonyms_raw, "symbolSynonyms")
        
        # Exclude approved symbol from synonyms (case-insensitive)
        if unique_values and is_valid_value(approved_symbol):
            approved_symbol_lower = str(approved_symbol).lower()
            unique_values = [
                val for val in unique_values 
                if str(val).lower() != approved_symbol_lower
            ]
        
        if unique_values:
            synonyms_str = ", ".join(unique_values)
            lines.append(f"{camel_to_title('symbolSynonyms')}: {synonyms_str}")
    
    approved_name = get_value('approvedName')
    if is_valid_value(approved_name):
        lines.append(f"{camel_to_title('approvedName')}: {approved_name}")
    
    gene_id = get_value('id')
    if is_valid_value(gene_id):
        lines.append(f"{camel_to_title('id')}: {gene_id}")
    
    biotype = get_value('biotype')
    if is_valid_value(biotype):
        lines.append(f"{camel_to_title('biotype')}: {biotype}")
    
    # Genomic location (only include if available)
    gloc = format_genomic_location(get_value('genomicLocation'))
    if gloc:
        # Ensure gloc is a string
        gloc_str = ensure_string(gloc)
        lines.append(f"{camel_to_title('genomicLocation')}: {gloc_str}")
    
    lines.append("")
    
    # Generic formatter for list items that are dicts - extract preferred keys
    def format_dict_item(item, preferred_keys=None):
        """Format a dict item by extracting preferred keys or formatting as key-value pairs."""
        if not isinstance(item, dict):
            return format_value(item)
        
        # Try preferred keys first
        if preferred_keys:
            for key in preferred_keys:
                if key in item and is_valid_value(item[key]):
                    return format_value(item[key])
        
        # Fallback: format as key-value pairs
        return format_dict(item)
    
    # Function Descriptions - list of strings
    if "functionDescriptions" in include_cols:
        func_descs = format_list(get_value('functionDescriptions'), deduplicate=True)
        if func_descs:
            lines.append("## Function Descriptions")
            for desc in func_descs:
                # Ensure desc is a string and clean up trailing periods and spaces
                desc_str = ensure_string(desc)
                cleaned_desc = desc_str.rstrip('. ')
                lines.append(f"- {cleaned_desc}")
            lines.append("")
    
    # Subcellular Locations - list of dicts, extract 'location' or 'label'
    if "subcellularLocations" in include_cols:
        subcell_locs = format_list(
            get_value('subcellularLocations'),
            item_formatter=lambda x: format_dict_item(x, preferred_keys=['location', 'label', 'name', 'labelSL']),
            deduplicate=True
        )
        if subcell_locs:
            lines.append("## Subcellular Locations")
            for loc in subcell_locs:
                # Ensure loc is a string
                loc_str = ensure_string(loc)
                lines.append(f"- {loc_str}")
            lines.append("")
    
    # Gene Ontology Terms - list of dicts, format as "ID - Name [Aspect]"
    if "go" in include_cols:
        def format_go_term(x):
            """Format a single GO term item."""
            if isinstance(x, dict):
                parts = []
                go_id = x.get('id', '')
                if go_id:
                    # Ensure go_id is a string
                    if not isinstance(go_id, str):
                        go_id = str(go_id)
                    parts.append(go_id)
                
                name = x.get('name', '') or x.get('label', '')
                if name:
                    # Ensure name is a string (handle lists)
                    if isinstance(name, (list, np.ndarray)):
                        if isinstance(name, np.ndarray):
                            name = name.tolist()
                        name = ", ".join(str(n) for n in name)
                    elif not isinstance(name, str):
                        name = str(name)
                    parts.append(name)
                
                aspect = x.get('aspect', '') or x.get('category', '')
                if aspect:
                    # Ensure aspect is a string
                    if not isinstance(aspect, str):
                        aspect = str(aspect)
                    parts.append(f"[{aspect}]")
                
                return " - ".join(parts) if parts else None
            else:
                # Ensure format_value result is a string
                formatted = format_value(x)
                if isinstance(formatted, (list, np.ndarray)):
                    if isinstance(formatted, np.ndarray):
                        formatted = formatted.tolist()
                    # Recursively ensure all items are strings
                    str_items = []
                    for item in formatted:
                        if isinstance(item, (list, np.ndarray)):
                            if isinstance(item, np.ndarray):
                                item = item.tolist()
                            str_items.append(", ".join(str(y) for y in item))
                        else:
                            str_items.append(str(item))
                    return ", ".join(str_items) if str_items else None
                elif isinstance(formatted, dict):
                    return format_dict(formatted) or str(formatted)
                elif not isinstance(formatted, str):
                    return str(formatted) if formatted else None
                return formatted
        
        go_terms = format_list(
            get_value('go'),
            item_formatter=format_go_term,
            deduplicate=True
        )
        if go_terms:
            lines.append("## Gene Ontology Terms")
            for term in go_terms:
                # Ensure term is a string
                term_str = ensure_string(term)
                lines.append(f"- {term_str}")
            lines.append("")
    
    # Pathways - list (could be strings or dicts)
    if "pathways" in include_cols:
        pathways_raw = get_value('pathways')
        if is_valid_value(pathways_raw):
            # Check if fields are specified for pathways
            if "pathways" in field_mapping:
                pathways = extract_fields_from_list(pathways_raw, "pathways")
            else:
                # Default: format as before
                pathways = format_list(
                    pathways_raw,
                    item_formatter=lambda x: format_dict_item(x, preferred_keys=['name', 'label', 'id']),
                    deduplicate=True
                )
            
            if pathways:
                lines.append("## Pathways")
                for pathway in pathways:
                    # Ensure pathway is a string
                    pathway_str = ensure_string(pathway)
                    lines.append(f"- {pathway_str}")
                lines.append("")
    
    # Tractability - list of dicts, group by modality
    if "tractability" in include_cols:
        tractability = get_value('tractability')
        
        if is_valid_value(tractability):
            # Parse JSON string if needed
            if isinstance(tractability, str):
                try:
                    tractability = json.loads(tractability)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            has_content = False
            if isinstance(tractability, list):
                # Group by modality
                modalities = {}
                for entry in tractability:
                    if isinstance(entry, dict):
                        modality = entry.get('modality', 'Unknown')
                        entry_id = entry.get('id', '')
                        value = entry.get('value', False)
                        
                        if modality not in modalities:
                            modalities[modality] = []
                        
                        # Only show entries with value=True or important categories
                        if value or entry_id in ['Approved Drug', 'Advanced Clinical', 'Phase 1 Clinical']:
                            modalities[modality].append((entry_id, value))
                
                # Format by modality
                modality_names = {
                    'SM': 'Small Molecule',
                    'AB': 'Antibody',
                    'PR': 'PROTAC',
                    'OC': 'Other Modalities',
                }
                
                for modality, entries in modalities.items():
                    if entries:
                        has_content = True
                        if not lines or lines[-1] != "## Tractability":
                            lines.append("## Tractability")
                        modality_display = modality_names.get(modality, modality)
                        lines.append(f"### {modality_display}")
                        for entry_id, value in entries:
                            status = "✓" if value else "✗"
                            lines.append(f"- {status} {entry_id}")
                        lines.append("")
            elif isinstance(tractability, dict):
                # Format as key-value pairs
                formatted_items = []
                for key, value in tractability.items():
                    if is_valid_value(value):
                        formatted_value = format_value(value)
                        if formatted_value:
                            # Ensure formatted_value is a string
                            formatted_value_str = ensure_string(formatted_value)
                            formatted_items.append(f"- **{key}**: {formatted_value_str}")
                
                if formatted_items:
                    has_content = True
                    lines.append("## Tractability")
                    lines.extend(formatted_items)
                    lines.append("")
            else:
                formatted = format_value(tractability)
                if formatted:
                    # Ensure formatted is a string
                    formatted_str = ensure_string(formatted)
                    has_content = True
                    lines.append("## Tractability")
                    lines.append(formatted_str)
                    lines.append("")
    
    # Constraint - dict or list structure
    if "constraint" in include_cols:
        constraint = get_value('constraint')
        
        if is_valid_value(constraint):
            # Parse JSON string if needed
            if isinstance(constraint, str):
                try:
                    constraint = json.loads(constraint)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            has_content = False
            if isinstance(constraint, dict):
                # Format as key-value pairs
                formatted_items = []
                for key, value in constraint.items():
                    if is_valid_value(value):
                        formatted_value = format_value(value)
                        if formatted_value:
                            # Ensure formatted_value is a string
                            formatted_value_str = ensure_string(formatted_value)
                            formatted_items.append(f"- **{camel_to_title(key)}**: {formatted_value_str}")
                
                if formatted_items:
                    has_content = True
                    lines.append("## Constraint")
                    lines.extend(formatted_items)
                    lines.append("")
            elif isinstance(constraint, list):
                # Format list items
                constraint_items = format_list(constraint, deduplicate=True)
                if constraint_items:
                    has_content = True
                    lines.append("## Constraint")
                    for item in constraint_items:
                        # Ensure item is a string
                        item_str = ensure_string(item)
                        lines.append(f"- {item_str}")
                    lines.append("")
            else:
                formatted = format_value(constraint)
                if formatted:
                    # Ensure formatted is a string
                    formatted_str = ensure_string(formatted)
                    has_content = True
                    lines.append("## Constraint")
                    lines.append(formatted_str)
                    lines.append("")
    
    # Handle any remaining custom columns that aren't in the predefined list
    predefined_cols = {
        "functionDescriptions", "subcellularLocations", "go", "pathways", 
        "tractability", "constraint", "id", "approvedSymbol", "approvedName", 
        "biotype", "genomicLocation", "symbolSynonyms", "proteinIds"
    }
    
    for col in include_cols:
        # Skip if already handled or if it's a predefined column
        if col in predefined_cols:
            continue
        
        # Get the value for this column
        col_value = get_value(col)
        if is_valid_value(col_value):
            # Check if this column has field specifications
            if col in field_mapping:
                # Extract specified fields
                if isinstance(col_value, (list, np.ndarray)):
                    if isinstance(col_value, np.ndarray):
                        col_value = col_value.tolist()
                    unique_values = extract_fields_from_list(col_value, col)
                    if unique_values:
                        values_str = ", ".join(unique_values)
                        lines.append(f"{camel_to_title(col)}: {values_str}")
                else:
                    # For non-list values, try to extract fields if it's a dict
                    if isinstance(col_value, dict):
                        field_values = []
                        for field in field_mapping[col]:
                            field_val = col_value.get(field)
                            if is_valid_value(field_val):
                                field_values.append(ensure_string(field_val))
                        if field_values:
                            values_str = ", ".join(field_values)
                            lines.append(f"{camel_to_title(col)}: {values_str}")
                    else:
                        # Simple value
                        lines.append(f"{camel_to_title(col)}: {ensure_string(col_value)}")
            else:
                # No field specification, format normally
                formatted_value = format_value(col_value)
                if formatted_value:
                    formatted_str = ensure_string(formatted_value)
                    # If it's a list, format as bullet points
                    if isinstance(col_value, (list, np.ndarray)) and len(str(formatted_str).split('\n')) == 1:
                        # Single line, might be better as a section
                        lines.append(f"## {camel_to_title(col)}")
                        if isinstance(col_value, list):
                            for item in col_value:
                                item_str = ensure_string(item)
                                lines.append(f"- {item_str}")
                        else:
                            lines.append(formatted_str)
                        lines.append("")
                    else:
                        # Simple value or multi-line, add as key-value or section
                        if '\n' in formatted_str or len(formatted_str) > 100:
                            # Long value, make it a section
                            lines.append(f"## {camel_to_title(col)}")
                            lines.append(formatted_str)
                            lines.append("")
                        else:
                            # Short value, add as key-value
                            lines.append(f"{camel_to_title(col)}: {formatted_str}")
    
    # Process disease_to_gene if provided
    if disease_to_gene is not None and len(disease_to_gene) > 0:
        # Check if this is already a pre-processed DataFrame (from batch processing)
        # Pre-processed DataFrames have already been aggregated, so they won't need groupby
        # We can detect this by checking if diseaseId contains arrays (from aggregation)
        is_preprocessed = False
        if len(disease_to_gene) > 0:
            # Check if diseaseId column exists and contains arrays (indicating pre-aggregation)
            if "diseaseId" in disease_to_gene.columns:
                sample_val = disease_to_gene["diseaseId"].iloc[0]
                is_preprocessed = isinstance(sample_val, (list, np.ndarray))
        
        if is_preprocessed:
            # Already pre-processed (already filtered and aggregated)
            diseases_markdown = globals()['diseases_to_markdown'](disease_to_gene)
            lines.append(diseases_markdown)
        else:
            # Need to filter by targetId and aggregate (original behavior)
            target_id = get_value('id')
            if target_id and is_valid_value(target_id):
                disease_to_gene_filtered = disease_to_gene.loc[disease_to_gene["targetId"] == target_id]
                
                if len(disease_to_gene_filtered) > 0:
                    # Group by name and targetId, aggregate
                    disease_to_gene_row = disease_to_gene_filtered.groupby(["name", "targetId"]).agg({
                        "diseaseId": "unique",
                        "score": "mean",
                        "evidenceCount": "sum",
                        "description": "first",
                        "synonyms": (lambda srs: list({v for d in srs.dropna() for k in d for v in d[k]})),
                    }).reset_index().sort_values("score", ascending=False)
                    
                    # Convert to markdown and append
                    diseases_markdown = globals()['diseases_to_markdown'](disease_to_gene_row)
                    lines.append(diseases_markdown)
    
    # Process drug_to_gene if provided
    if drug_to_gene is not None and len(drug_to_gene) > 0:
        # Check if this is already a pre-processed DataFrame (from batch processing)
        # Pre-processed DataFrames have already been deduplicated
        # We can detect this by checking if there are multiple rows with same drugId (not preprocessed)
        # or if targetId column is missing (definitely preprocessed)
        is_preprocessed = "targetId" not in drug_to_gene.columns
        
        if is_preprocessed:
            # Already pre-processed (already filtered and deduplicated)
            drugs_markdown = globals()['drugs_to_markdown'](drug_to_gene)
            lines.append(drugs_markdown)
        else:
            # Need to filter by targetId (original behavior)
            target_id = get_value('id')
            if target_id and is_valid_value(target_id):
                drug_to_gene_filtered = drug_to_gene.loc[drug_to_gene["targetId"] == target_id]
                
                if len(drug_to_gene_filtered) > 0:
                    # Drop duplicates and take the latest entry
                    if "drugId" in drug_to_gene_filtered.columns:
                        # Sort by phase (descending, with NaN last) to get latest phase first
                        if "phase" in drug_to_gene_filtered.columns:
                            drug_to_gene_filtered = drug_to_gene_filtered.sort_values(
                                "phase", 
                                ascending=False, 
                                na_position='last'
                            )
                        # Take first entry for each drugId (which will be the one with highest phase after sorting)
                        drug_to_gene_filtered = drug_to_gene_filtered.drop_duplicates(
                            subset=["drugId"], 
                            keep="first"
                        )
                    
                    # Convert to markdown and append
                    drugs_markdown = globals()['drugs_to_markdown'](drug_to_gene_filtered)
                    lines.append(drugs_markdown)
    
    # Process gene_to_pharmacogenomics if provided
    if gene_to_pharmacogenomics is not None and len(gene_to_pharmacogenomics) > 0:
        # Check if this is already a pre-processed DataFrame (from batch processing)
        # Pre-processed DataFrames have already been filtered by targetId
        # We can detect this by checking if targetId columns are missing
        target_id_col = None
        for col in ["targetFromSourceId", "targetId", "id"]:
            if col in gene_to_pharmacogenomics.columns:
                target_id_col = col
                break
        
        is_preprocessed = target_id_col is None
        
        if is_preprocessed:
            # Already pre-processed (already filtered)
            pgx_markdown = globals()['pharmacogenomics_to_markdown'](gene_to_pharmacogenomics)
            lines.append(pgx_markdown)
        else:
            # Need to filter by targetId (original behavior)
            target_id = get_value('id')
            if target_id and is_valid_value(target_id):
                pgx_filtered = gene_to_pharmacogenomics.loc[gene_to_pharmacogenomics[target_id_col] == target_id]
                
                if len(pgx_filtered) > 0:
                    # Convert to markdown and append
                    pgx_markdown = globals()['pharmacogenomics_to_markdown'](pgx_filtered)
                    lines.append(pgx_markdown)
    
    return "\n".join(lines)


def _preprocess_disease_to_gene(disease_to_gene: pd.DataFrame, target_ids: Optional[Set[str]] = None, limit: Optional[int] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Pre-process disease_to_gene DataFrame by grouping and aggregating per targetId.
    Returns a dictionary mapping targetId to pre-processed DataFrame.
    
    Parameters
    ----------
    disease_to_gene : pd.DataFrame
        DataFrame with disease-to-gene associations
    target_ids : set of str, optional
        If provided, only process these target IDs. Otherwise, process all unique target IDs.
    limit : int, optional
        Maximum number of associations per target. If provided, takes top N by score.
    verbose : bool, default True
        Whether to show progress
    """
    if disease_to_gene is None or len(disease_to_gene) == 0:
        return {}
    
    # Filter upfront if target_ids provided (much faster than filtering per target_id)
    if verbose:
        print(f"  Filtering disease associations for {len(target_ids) if target_ids else 'all'} target IDs...")
    if target_ids is not None:
        disease_to_gene = disease_to_gene[disease_to_gene["targetId"].isin(target_ids)].copy()
        if len(disease_to_gene) == 0:
            return {}
        if verbose:
            print(f"  Filtered to {len(disease_to_gene)} disease associations")
    
    # Pre-process synonyms column to flatten it before groupby (much faster)
    # This avoids calling the aggregation function for every group
    if "synonyms" in disease_to_gene.columns and verbose:
        print(f"  Pre-processing synonyms column...")
    
    def flatten_synonyms(val):
        """Flatten synonyms dict/list to a set of strings."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return set()
        result = set()
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, (list, np.ndarray)):
                    result.update(str(x) for x in v if x is not None)
                elif v is not None:
                    result.add(str(v))
        elif isinstance(val, (list, np.ndarray)):
            result.update(str(x) for x in val if x is not None)
        else:
            result.add(str(val))
        return result
    
    # Pre-process synonyms to sets (much faster than doing it in groupby)
    if "synonyms" in disease_to_gene.columns:
        disease_to_gene = disease_to_gene.copy()  # Make sure we have a copy
        disease_to_gene["_synonyms_set"] = disease_to_gene["synonyms"].apply(flatten_synonyms)
    else:
        disease_to_gene["_synonyms_set"] = pd.Series([set()] * len(disease_to_gene), index=disease_to_gene.index)
    
    # Optimize groupby by using categorical dtypes for groupby columns (if not too many unique values)
    if verbose:
        print(f"  Grouping and aggregating disease associations...")
        print(f"    DataFrame size: {len(disease_to_gene):,} rows")
    
    # Use more efficient aggregation - combine sets using union
    def combine_synonym_sets(srs):
        """Combine sets of synonyms."""
        result = set()
        for s in srs:
            if isinstance(s, set):
                result.update(s)
        return list(result) if result else []
    
    # Group by targetId and name, aggregate all at once
    # For very large DataFrames, chunk the groupby operation
    groupby_cols = ["targetId", "name"]
    chunk_size = 1_000_000  # Process in chunks of 1M rows if larger
    
    if len(disease_to_gene) > chunk_size:
        if verbose:
            print(f"    Large DataFrame detected ({len(disease_to_gene):,} rows), processing in chunks...")
        # Process in chunks
        chunks = []
        num_chunks = (len(disease_to_gene) + chunk_size - 1) // chunk_size
        
        if verbose:
            from tqdm.auto import tqdm
            chunk_iterator = tqdm(range(0, len(disease_to_gene), chunk_size), 
                                 desc="      Processing chunks", unit="chunk", total=num_chunks)
        else:
            chunk_iterator = range(0, len(disease_to_gene), chunk_size)
        
        for i in chunk_iterator:
            chunk = disease_to_gene.iloc[i:i+chunk_size]
            chunk_grouped = chunk.groupby(groupby_cols, sort=False).agg({
                "diseaseId": "unique",
                "score": "mean",
                "evidenceCount": "sum",
                "description": "first",
                "_synonyms_set": combine_synonym_sets,
            })
            chunks.append(chunk_grouped)
        
        # Combine chunks and re-aggregate
        if verbose:
            print(f"    Combining {len(chunks)} chunks...")
        combined = pd.concat(chunks)
        grouped = combined.groupby(groupby_cols, sort=False).agg({
            "diseaseId": lambda x: np.unique(np.concatenate([arr if isinstance(arr, np.ndarray) else [arr] for arr in x])),
            "score": "mean",
            "evidenceCount": "sum",
            "description": "first",
            "_synonyms_set": combine_synonym_sets,
        }).reset_index()
    else:
        # Standard groupby for smaller DataFrames
        grouped = disease_to_gene.groupby(groupby_cols, sort=False).agg({
            "diseaseId": "unique",
            "score": "mean",
            "evidenceCount": "sum",
            "description": "first",
            "_synonyms_set": combine_synonym_sets,
        }).reset_index()
    
    # Rename the synonyms column back
    grouped = grouped.rename(columns={"_synonyms_set": "synonyms"})
    
    # Sort by score descending
    if verbose:
        print(f"  Sorting by score...")
    grouped = grouped.sort_values("score", ascending=False)
    
    # Split into dict by targetId (vectorized using groupby) and apply per-target limit
    if verbose:
        print(f"  Creating per-target dictionaries...")
        from tqdm import tqdm
        target_id_groups = list(grouped.groupby("targetId", sort=False))
        iterator = tqdm(target_id_groups, desc="  Processing targets", unit="target", disable=not verbose)
    else:
        iterator = grouped.groupby("targetId", sort=False)
    
    result = {}
    for target_id, group in iterator:
        # Apply per-target limit if specified
        if limit is not None:
            group = group.head(limit)
        # Drop targetId column to mark as preprocessed
        group_clean = group.drop(columns=["targetId"]).copy()
        result[target_id] = group_clean
    
    if verbose:
        print(f"  Pre-processed {len(result)} target IDs for diseases" + (f" (limited to {limit} per target)" if limit else ""))
    
    return result


def _preprocess_drug_to_gene(drug_to_gene: pd.DataFrame, target_ids: Optional[Set[str]] = None, limit: Optional[int] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Pre-process drug_to_gene DataFrame by deduplicating per targetId.
    Returns a dictionary mapping targetId to pre-processed DataFrame.
    
    Parameters
    ----------
    drug_to_gene : pd.DataFrame
        DataFrame with drug-to-gene associations
    target_ids : set of str, optional
        If provided, only process these target IDs. Otherwise, process all unique target IDs.
    limit : int, optional
        Maximum number of drugs per target. If provided, takes top N by phase.
    verbose : bool, default True
        Whether to show progress
    """
    if drug_to_gene is None or len(drug_to_gene) == 0:
        return {}
    
    # Filter upfront if target_ids provided (much faster than filtering per target_id)
    if verbose:
        print(f"  Filtering drug associations for {len(target_ids) if target_ids else 'all'} target IDs...")
    if target_ids is not None:
        drug_to_gene = drug_to_gene[drug_to_gene["targetId"].isin(target_ids)].copy()
        if len(drug_to_gene) == 0:
            return {}
        if verbose:
            print(f"  Filtered to {len(drug_to_gene)} drug associations")
    
    # Vectorized: sort and deduplicate all at once, then split by targetId
    if "drugId" in drug_to_gene.columns:
        if verbose:
            print(f"  Sorting and deduplicating drugs...")
        # Sort by phase (descending, with NaN last) to get latest phase first
        if "phase" in drug_to_gene.columns:
            drug_to_gene = drug_to_gene.sort_values("phase", ascending=False, na_position='last')
        # Drop duplicates per (targetId, drugId) combination, keeping first (highest phase)
        drug_to_gene = drug_to_gene.drop_duplicates(subset=["targetId", "drugId"], keep="first")
    
    # Split into dict by targetId (vectorized using groupby) and apply per-target limit
    if verbose:
        print(f"  Creating per-target dictionaries...")
        from tqdm import tqdm
        target_id_groups = list(drug_to_gene.groupby("targetId", sort=False))
        iterator = tqdm(target_id_groups, desc="  Processing targets", unit="target", disable=not verbose)
    else:
        iterator = drug_to_gene.groupby("targetId", sort=False)
    
    result = {}
    for target_id, group in iterator:
        # Apply per-target limit if specified
        if limit is not None:
            group = group.head(limit)
        # Drop targetId column to mark as preprocessed
        if "targetId" in group.columns:
            group_clean = group.drop(columns=["targetId"]).copy()
        else:
            group_clean = group.copy()
        result[target_id] = group_clean
    
    if verbose:
        print(f"  Pre-processed {len(result)} target IDs for drugs" + (f" (limited to {limit} per target)" if limit else ""))
    
    return result


def _preprocess_pharmacogenomics(gene_to_pgx: pd.DataFrame, target_ids: Optional[Set[str]] = None, limit: Optional[int] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Pre-process gene_to_pharmacogenomics DataFrame by grouping per targetId.
    Returns a dictionary mapping targetId to pre-processed DataFrame.
    
    Parameters
    ----------
    gene_to_pgx : pd.DataFrame
        DataFrame with pharmacogenomics data
    target_ids : set of str, optional
        If provided, only process these target IDs. Otherwise, process all unique target IDs.
    limit : int, optional
        Maximum number of pharmacogenomics records per target. If provided, takes top N by evidenceLevel and isDirectTarget.
    verbose : bool, default True
        Whether to show progress
    """
    if gene_to_pgx is None or len(gene_to_pgx) == 0:
        return {}
    
    # Check which column name is used for target ID
    target_id_col = None
    for col in ["targetFromSourceId", "targetId", "id"]:
        if col in gene_to_pgx.columns:
            target_id_col = col
            break
    
    if target_id_col is None:
        return {}
    
    # Filter upfront if target_ids provided (much faster than filtering per target_id)
    if verbose:
        print(f"  Filtering pharmacogenomics for {len(target_ids) if target_ids else 'all'} target IDs...")
    if target_ids is not None:
        gene_to_pgx = gene_to_pgx[gene_to_pgx[target_id_col].isin(target_ids)].copy()
        if len(gene_to_pgx) == 0:
            return {}
        if verbose:
            print(f"  Filtered to {len(gene_to_pgx)} pharmacogenomics records")
    
    # Sort by evidenceLevel and isDirectTarget if limit is specified (for per-target limiting)
    if limit is not None:
        if verbose:
            print(f"  Sorting pharmacogenomics records...")
        sort_cols = []
        if "evidenceLevel" in gene_to_pgx.columns:
            sort_cols.append("evidenceLevel")
        if "isDirectTarget" in gene_to_pgx.columns:
            sort_cols.append("isDirectTarget")
        if sort_cols:
            gene_to_pgx = gene_to_pgx.sort_values(sort_cols, ascending=False, na_position='last')
    
    # Split into dict by targetId (vectorized using groupby) and apply per-target limit
    if verbose:
        print(f"  Creating per-target dictionaries...")
        from tqdm import tqdm
        target_id_groups = list(gene_to_pgx.groupby(target_id_col, sort=False))
        iterator = tqdm(target_id_groups, desc="  Processing targets", unit="target", disable=not verbose)
    else:
        iterator = gene_to_pgx.groupby(target_id_col, sort=False)
    
    result = {}
    for target_id, group in iterator:
        # Apply per-target limit if specified
        if limit is not None:
            group = group.head(limit)
        # Drop targetId column to mark as preprocessed
        if target_id_col in group.columns:
            group_clean = group.drop(columns=[target_id_col]).copy()
        else:
            group_clean = group.copy()
        result[target_id] = group_clean
    
    if verbose:
        print(f"  Pre-processed {len(result)} target IDs for pharmacogenomics" + (f" (limited to {limit} per target)" if limit else ""))
    
    return result


def df_to_markdown_batch(
    target_df: pd.DataFrame,
    include_cols: Optional[List[str]] = None,
    include_fields: Optional[Dict[str, List[str]]] = None,
    disease_to_gene: Optional[pd.DataFrame] = None,
    drug_to_gene: Optional[pd.DataFrame] = None,
    gene_to_pharmacogenomics: Optional[pd.DataFrame] = None,
    n_jobs: int = 1,
    verbose: bool = True,
    save_path: Optional[str] = None,
    force: bool = False,
    limit_associations: Optional[int] = None,
    limit_drugs: Optional[int] = None,
    limit_pharmacogenomics: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> pd.Series:
    """
    Batch process multiple target rows to markdown format (optimized version).
    
    This function is much faster than iterating with iterrows() because it:
    1. Pre-processes disease_to_gene, drug_to_gene, and gene_to_pharmacogenomics DataFrames once
    2. Uses apply() instead of iterrows()
    3. Optionally uses parallel processing
    
    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame with target rows to process
    include_cols : list of str, optional
        List of column/section names to include (see df_to_markdown)
    include_fields : dict, optional
        Dictionary mapping column names to field name(s) to extract (see df_to_markdown)
    disease_to_gene : pd.DataFrame, optional
        DataFrame with disease-to-gene associations (will be pre-processed)
    drug_to_gene : pd.DataFrame, optional
        DataFrame with drug-to-gene associations (will be pre-processed)
    gene_to_pharmacogenomics : pd.DataFrame, optional
        DataFrame with pharmacogenomics data (will be pre-processed)
    n_jobs : int, default 1
        Number of parallel jobs. If 1, uses single-threaded processing.
        If > 1, uses multiprocessing (requires joblib).
    verbose : bool, default True
        Whether to show progress
    save_path : str, optional
        If provided, save the resulting DataFrame (with markdown column added) to this parquet file path.
        The DataFrame will include all original columns plus the new "markdown" column.
    force : bool, default False
        If True, force regeneration even if save_path already exists.
        If False and save_path exists, load the existing parquet file instead of regenerating.
    limit_associations : int, optional
        Maximum number of disease associations per target. If provided, takes top N by score.
    limit_drugs : int, optional
        Maximum number of drugs per target. If provided, takes top N by phase.
    limit_pharmacogenomics : int, optional
        Maximum number of pharmacogenomics records per target. If provided, takes top N by evidenceLevel and isDirectTarget.
    
    Returns
    -------
    pd.Series
        Series with markdown strings, indexed by target_df index
    """
    from tqdm import tqdm
    
    # Check if file exists and force is False
    if save_path is not None and not force:
        save_path_obj = Path(save_path)
        if save_path_obj.exists():
            if verbose:
                print(f"Loading existing parquet file: {save_path}")
            try:
                cached_df = pd.read_parquet(save_path)
                # Check if markdown column exists
                if "markdown" in cached_df.columns:
                    # Match rows by index if target_df has an index, or by id column
                    if target_df.index.name is not None or len(target_df.index.names) > 1:
                        # Try to match by index
                        if cached_df.index.equals(target_df.index):
                            if verbose:
                                print(f"  Loaded {len(cached_df)} rows with markdown column")
                            return cached_df["markdown"]
                    # Otherwise, try to match by id column
                    if "id" in cached_df.columns and "id" in target_df.columns:
                        # Create a mapping from id to markdown
                        id_to_markdown = dict(zip(cached_df["id"], cached_df["markdown"]))
                        # Get markdown for each id in target_df
                        markdown_series = target_df["id"].map(id_to_markdown)
                        if verbose:
                            matched_count = markdown_series.notna().sum()
                            print(f"  Matched {matched_count}/{len(target_df)} rows from cached file")
                        return markdown_series
                    # If no good matching strategy, just return the cached markdown
                    if verbose:
                        print(f"  Warning: Could not match cached data to target_df, returning cached markdown")
                    return cached_df["markdown"]
                else:
                    if verbose:
                        print(f"  Warning: Cached file exists but has no 'markdown' column, regenerating...")
            except Exception as e:
                if verbose:
                    print(f"  Warning: Error loading cached file ({e}), regenerating...")
    
    # Extract target IDs from target_df to only preprocess those
    target_ids = set(target_df["id"].unique()) if "id" in target_df.columns else None
    
    if verbose:
        print(f"\nPre-processing data for {len(target_ids) if target_ids else 'all'} target IDs...")
    
    # Pre-process disease, drug, and pharmacogenomics DataFrames once (only for target IDs in target_df)
    if disease_to_gene is not None:
        if verbose:
            print("Pre-processing disease associations...")
        disease_dict = _preprocess_disease_to_gene(disease_to_gene, target_ids=target_ids, limit=limit_associations, verbose=verbose)
    else:
        disease_dict = {}
    
    if drug_to_gene is not None:
        if verbose:
            print("Pre-processing drug associations...")
        drug_dict = _preprocess_drug_to_gene(drug_to_gene, target_ids=target_ids, limit=limit_drugs, verbose=verbose)
    else:
        drug_dict = {}
    
    if gene_to_pharmacogenomics is not None:
        if verbose:
            print("Pre-processing pharmacogenomics...")
        pgx_dict = _preprocess_pharmacogenomics(gene_to_pharmacogenomics, target_ids=target_ids, limit=limit_pharmacogenomics, verbose=verbose)
    else:
        pgx_dict = {}
    
    if verbose:
        print(f"\nPre-processing complete: {len(disease_dict)} targets with diseases, {len(drug_dict)} with drugs, {len(pgx_dict)} with pharmacogenomics")
    
    # NOTE: The process_row function below will be pickled by joblib along with the entire
    # closure, including disease_dict, drug_dict, and pgx_dict. For large datasets, this
    # means each worker process receives a copy of these large dictionaries, which is expensive.
    # However, joblib reuses workers, so the dictionaries are only pickled once per worker,
    # not once per row. Using smaller batches helps reduce memory pressure.
    def process_row(row):
        """Process a single row with pre-processed dictionaries"""
        target_id = row.get('id') if isinstance(row, dict) else row['id']
        
        # Get pre-processed DataFrames for this targetId
        disease_df = disease_dict.get(target_id) if target_id else None
        drug_df = drug_dict.get(target_id) if target_id else None
        pgx_df = pgx_dict.get(target_id) if target_id else None
        
        # Call df_to_markdown with pre-processed DataFrames
        # Let exceptions propagate - don't silently catch them
        return df_to_markdown(
            target_row=row,
            include_cols=include_cols,
            include_fields=include_fields,
            disease_to_gene=disease_df,
            drug_to_gene=drug_df,
            gene_to_pharmacogenomics=pgx_df,
        )
    
    if n_jobs == 1:
        # Single-threaded with progress bar
        # Use to_dict('records') + list comprehension which is faster than apply()
        rows = target_df.to_dict('records')
        if verbose:
            # Use tqdm with explicit total and better formatting
            results = []
            with tqdm(total=len(rows), desc="Generating markdown", unit="target") as pbar:
                for row in rows:
                    results.append(process_row(row))
                    pbar.update(1)
            markdown_series = pd.Series(results, index=target_df.index)
        else:
            results = [process_row(row) for row in rows]
            markdown_series = pd.Series(results, index=target_df.index)
    else:
        # Parallel processing
        try:
            from joblib import Parallel, delayed
            if verbose:
                print(f"Using {n_jobs} parallel workers...")
            
            # Convert DataFrame to list of dicts more efficiently
            # Using to_dict('records') is much faster than iterrows()
            rows = target_df.to_dict('records')
            
            # Limit n_jobs if max_workers is specified
            if max_workers is not None and n_jobs > max_workers:
                if verbose:
                    print(f"  Limiting workers from {n_jobs} to {max_workers}")
                n_jobs = max_workers
            
            # For parallel processing, use smaller batches and process with timeouts
            # The key issue: joblib pickles the entire closure (disease_dict, drug_dict, pgx_dict)
            # for each worker. For large datasets, this is extremely expensive.
            # Solution: Use smaller batches and let joblib reuse workers efficiently
            
            if verbose:
                # Use much smaller batches to avoid memory issues and allow progress tracking
                # With 78k rows, we want batches of ~100-500 rows to see progress
                batch_size = max(50, min(len(rows) // (n_jobs * 20), 500))  # 50-500 rows per batch
                if verbose:
                    print(f"  Processing in batches of {batch_size} rows...")
                
                pbar = tqdm(total=len(rows), desc="Generating markdown", unit="target")
                
                results = []
                failed_batches = 0
                try:
                    # Process rows in batches
                    for batch_idx, i in enumerate(range(0, len(rows), batch_size)):
                        batch = rows[i:i + batch_size]
                        batch_start_time = time.time()
                        
                        try:
                            # Use loky backend with timeout per batch
                            # Set a reasonable timeout: 5 minutes per batch
                            batch_timeout = max(60, batch_size * 2)  # At least 1 min, or 2s per row
                            
                            batch_results = Parallel(
                                n_jobs=n_jobs, 
                                verbose=0,
                                backend='loky',
                                timeout=batch_timeout,  # Timeout for entire batch
                                batch_size=max(1, batch_size // n_jobs),  # Smaller per-worker batches
                                prefer='processes',  # Use processes, not threads
                            )(
                                delayed(process_row)(row) for row in batch
                            )
                            results.extend(batch_results)
                            
                            batch_time = time.time() - batch_start_time
                            pbar.update(len(batch))
                            
                            # Warn if batch is taking too long
                            if batch_time > 60 and verbose:
                                avg_time = batch_time / len(batch)
                                print(f"\n  Warning: Batch {batch_idx + 1} took {batch_time:.1f}s ({avg_time:.2f}s per row)")
                                
                        except Exception as e:
                            failed_batches += 1
                            if verbose:
                                print(f"\n  Error in batch {batch_idx + 1}: {e}")
                                print(f"  Processing batch sequentially as fallback...")
                            
                            # Process batch sequentially as fallback
                            # Let exceptions propagate - don't silently catch them
                            batch_results = []
                            for row_idx, row in enumerate(batch):
                                row_start = time.time()
                                result = process_row(row)
                                batch_results.append(result)
                                
                                # Warn about slow rows
                                row_time = time.time() - row_start
                                if row_time > 10 and verbose and row_idx < 3:
                                    target_id = row.get('id', f'row_{i+row_idx}') if isinstance(row, dict) else getattr(row, 'id', f'row_{i+row_idx}')
                                    print(f"    Warning: Row {row_idx + 1} (target {target_id}) took {row_time:.1f}s")
                            
                            results.extend(batch_results)
                            pbar.update(len(batch))
                            
                finally:
                    pbar.close()
                    if failed_batches > 0 and verbose:
                        print(f"\n  Completed with {failed_batches} batch(es) requiring fallback processing")
            else:
                # Non-verbose mode: process all at once but with error handling
                try:
                    results = Parallel(
                        n_jobs=n_jobs, 
                        verbose=0,
                        backend='loky',
                        timeout=None,
                        batch_size='auto'
                    )(
                        delayed(process_row)(row) for row in rows
                    )
                except Exception as e:
                    # Fallback to sequential processing if parallel fails
                    import warnings
                    warnings.warn(f"Parallel processing failed: {e}. Falling back to sequential processing.")
                    # Let exceptions propagate - don't silently catch them
                    results = [process_row(row) for row in rows]
            
            markdown_series = pd.Series(results, index=target_df.index)
        except ImportError:
            import warnings
            warnings.warn("joblib not available, falling back to single-threaded processing")
            rows = target_df.to_dict('records')
            if verbose:
                results = []
                with tqdm(total=len(rows), desc="Generating markdown", unit="target") as pbar:
                    for row in rows:
                        results.append(process_row(row))
                        pbar.update(1)
            else:
                results = [process_row(row) for row in rows]
            markdown_series = pd.Series(results, index=target_df.index)
    
    # Save to parquet if save_path is provided
    if save_path is not None:
        # Create a copy of target_df with markdown column added
        result_df = target_df.copy()
        result_df["markdown"] = markdown_series
        
        # Ensure the directory exists
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        result_df.to_parquet(save_path, index=False)
        
        if verbose:
            print(f"\nSaved DataFrame with markdown to: {save_path}")
            print(f"  Shape: {result_df.shape}")
            print(f"  Columns: {list(result_df.columns)}")
    
    return markdown_series


def get_targets(
    add_markdown: bool = True,
    include_cols: Optional[List[str]] = None,
    include_fields: Optional[Dict[str, List[str]]] = None,
    n_jobs: int = 1,
    save_path: Optional[str] = None,
    force: bool = False,
    limit: Optional[Union[int, Dict[str, Optional[int]]]] = None,
    exclude_fields: Optional[Union[str, List[str], Dict[str, List[str]]]] = None,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Get target (gene) dataset from OpenTargets, optionally with markdown generation.
    
    This function loads the target dataset and optionally enriches it with markdown
    descriptions that include disease associations, drug information, and pharmacogenomics data.
    
    Parameters
    ----------
    add_markdown : bool, default True
        If True, generate markdown descriptions for each target by loading and merging
        disease associations, known drugs, and pharmacogenomics data.
        If False, just return the target dataset without markdown.
    include_cols : list of str, optional
        List of column/section names to include in markdown (see df_to_markdown_batch).
        If None, uses default columns: ["id", "approvedSymbol", "approvedName", "biotype",
        "genomicLocation", "symbolSynonyms", "proteinIds", "functionDescriptions",
        "subcellularLocations", "pathways"].
    include_fields : dict, optional
        Dictionary mapping column names to field name(s) to extract from nested structures
        (see df_to_markdown_batch). If None, uses default: {"symbolSynonyms": ["label"]}.
    n_jobs : int, default 1
        Number of parallel jobs for markdown generation (see df_to_markdown_batch).
    save_path : str, optional
        If provided, save the resulting DataFrame with markdown to this parquet file path.
    limit : int or dict, optional
        Maximum number of entries to return. Can be:
        - A single integer: limits only the number of targets
        - A dict with keys: "targets", "drugs", "associations", "pharmacogenomics"
          Each value is an int (limit) or None (no limit).
          Example: {"targets": 100, "drugs": 10, "associations": 20, "pharmacogenomics": None}
          For "drugs", "associations", and "pharmacogenomics", limits are applied PER TARGET
          (not globally). Top-ranking entries per target are taken based on:
          - associations: ranked by score
          - drugs: ranked by phase
          - pharmacogenomics: ranked by evidenceLevel and isDirectTarget
    verbose : bool, default True
        If True, print progress messages for each step.
    exclude_fields : str, list of str, or dict, optional
        Fields to exclude from DataFrames. Can be:
        - A single string: exclude that field from all tables
        - A list of strings: exclude all those fields from all tables
        - A dict with keys: "targets", "drugs", "associations", "pharmacogenomics", "disease"
          Each value is a list of field names to exclude from that table.
          Example: {"targets": ["field1"], "associations": ["field2", "field3"]}
    **kwargs
        Additional keyword arguments passed to get_dataset() for loading the target dataset.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with target information. If add_markdown=True, includes a "markdown" column.
    
    Examples
    --------
    >>> import phenoref.opentargets as ot
    >>> 
    >>> # Get targets without markdown
    >>> targets = ot.get_targets(add_markdown=False)
    >>> 
    >>> # Get targets with markdown (default)
    >>> targets = ot.get_targets()
    >>> 
    >>> # Get targets with custom markdown options
    >>> targets = ot.get_targets(
    ...     include_cols=["id", "approvedSymbol", "pathways"],
    ...     n_jobs=4
    ... )
    >>> 
    >>> # Get only first 100 targets
    >>> targets = ot.get_targets(limit=100)
    >>> 
    >>> # Limit different tables separately
    >>> targets = ot.get_targets(limit={"targets": 100, "drugs": 10, "associations": 20, "pharmacogenomics": 5})
    >>> 
    >>> # Exclude fields from all tables
    >>> targets = ot.get_targets(exclude_fields="field1")
    >>> targets = ot.get_targets(exclude_fields=["field1", "field2"])
    >>> 
    >>> # Exclude table-specific fields
    >>> targets = ot.get_targets(exclude_fields={"targets": ["field1"], "associations": ["field2"]})
    """
    # Check if file exists and force is False - if so, just load and return it
    if save_path is not None and not force:
        save_path_obj = Path(save_path)
        if save_path_obj.exists():
            if verbose:
                print(f"Loading existing targets DataFrame from: {save_path}")
            try:
                target = pd.read_parquet(save_path)
                if verbose:
                    print(f"  Loaded {len(target)} targets from existing file")
                    print("=" * 60)
                return target
            except Exception as e:
                if verbose:
                    print(f"  Error loading existing file: {e}")
                    print(f"  Will regenerate targets DataFrame...")
    
    if verbose:
        print("=" * 60)
        print("get_targets: Loading target dataset")
        print("=" * 60)
    
    # Parse exclude_fields parameter
    exclude_targets = []
    exclude_drugs = []
    exclude_associations = []
    exclude_pharmacogenomics = []
    exclude_disease = []
    
    if exclude_fields is not None:
        if isinstance(exclude_fields, str):
            # Single string: exclude from all tables
            exclude_targets = [exclude_fields]
            exclude_drugs = [exclude_fields]
            exclude_associations = [exclude_fields]
            exclude_pharmacogenomics = [exclude_fields]
            exclude_disease = [exclude_fields]
        elif isinstance(exclude_fields, list):
            # List of strings: exclude from all tables
            exclude_targets = exclude_fields
            exclude_drugs = exclude_fields
            exclude_associations = exclude_fields
            exclude_pharmacogenomics = exclude_fields
            exclude_disease = exclude_fields
        elif isinstance(exclude_fields, dict):
            # Dict: table-specific exclusions
            exclude_targets = exclude_fields.get("targets", [])
            exclude_drugs = exclude_fields.get("drugs", [])
            exclude_associations = exclude_fields.get("associations", [])
            exclude_pharmacogenomics = exclude_fields.get("pharmacogenomics", [])
            exclude_disease = exclude_fields.get("disease", [])
        else:
            raise ValueError(f"exclude_fields must be str, list, or dict, got {type(exclude_fields)}")
    
    # Helper function to apply exclusions to a DataFrame
    def apply_exclusions(df: pd.DataFrame, fields_to_exclude: List[str], table_name: str) -> pd.DataFrame:
        """Apply field exclusions to a DataFrame."""
        if not fields_to_exclude or len(df) == 0:
            return df
        
        # Get fields that actually exist in the DataFrame
        existing_fields = [f for f in fields_to_exclude if f in df.columns]
        if existing_fields:
            if verbose:
                print(f"    Excluding fields from {table_name}: {existing_fields}")
            df = df.drop(columns=existing_fields)
        return df
    
    # Parse limit parameter
    limit_targets = None
    limit_drugs = None
    limit_associations = None
    limit_pharmacogenomics = None
    
    if limit is not None:
        if isinstance(limit, int):
            limit_targets = limit
        elif isinstance(limit, dict):
            limit_targets = limit.get("targets")
            limit_drugs = limit.get("drugs")
            limit_associations = limit.get("associations")
            limit_pharmacogenomics = limit.get("pharmacogenomics")
        else:
            raise ValueError(f"limit must be int or dict, got {type(limit)}")
    
    # Get target dataset
    # If limit_targets is set, pass a small limit to get_dataset to avoid reading all parquet files
    dataset_kwargs = kwargs.copy()
    if limit_targets is not None:
        # Estimate number of parquet files needed (conservative: assume ~5k-10k targets per file)
        # Read at least 1 file, but cap at reasonable number to avoid reading too many
        estimated_files = max(1, min(10, (limit_targets // 5000) + 1))
        dataset_kwargs['limit'] = estimated_files
        if verbose:
            print(f"  Limiting to {limit_targets} targets (will read ~{estimated_files} parquet file(s))")
    
    if verbose:
        print("  Loading target dataset from OpenTargets...")
    target = get_dataset(dataset="target", **dataset_kwargs)
    
    if verbose:
        print(f"  Loaded {len(target)} targets")
    
    # Apply exclusions to targets
    target = apply_exclusions(target, exclude_targets, "targets")
    
    # Limit number of targets if limit_targets is specified
    if limit_targets is not None:
        if verbose:
            print(f"  Truncating to {limit_targets} targets")
        target = target.head(limit_targets)
        if verbose:
            print(f"  Final target count: {len(target)}")
    
    # If add_markdown is False, just return the target dataset
    if not add_markdown:
        if verbose:
            print("  Skipping markdown generation (add_markdown=False)")
            print("=" * 60)
        return target
    
    if verbose:
        print("\n" + "=" * 60)
        print("get_targets: Generating markdown descriptions")
        print("=" * 60)
    
    # Set default include_cols if not provided
    if include_cols is None:
        include_cols = [
            "id", "approvedSymbol", "approvedName", "biotype", "genomicLocation",
            "symbolSynonyms", "proteinIds", "functionDescriptions", "subcellularLocations",
            "pathways"
        ]
        if verbose:
            print("  Using default include_cols")
    else:
        if verbose:
            print(f"  Using custom include_cols: {include_cols}")
    
    # Set default include_fields if not provided
    if include_fields is None:
        include_fields = {"symbolSynonyms": ["label"]}
        if verbose:
            print("  Using default include_fields")
    else:
        if verbose:
            print(f"  Using custom include_fields: {include_fields}")
    
    # Import disease-gene associations
    if verbose:
        print("\n  Step 1/5: Loading disease-gene associations...")
    association_overall_direct = get_dataset(dataset="association_overall_direct", **kwargs)
    if verbose:
        print(f"    Loaded {len(association_overall_direct)} disease-gene associations")
    
    # Apply exclusions to associations
    association_overall_direct = apply_exclusions(association_overall_direct, exclude_associations, "associations")
    
    # Note: Limits are now applied per-target in preprocessing, not globally
    if limit_associations is not None and verbose:
        print(f"    Will limit to top {limit_associations} associations per target (ranked by score)")
    
    # Import disease metadata
    if verbose:
        print("\n  Step 2/5: Loading disease metadata...")
    disease = get_dataset(dataset="disease", **kwargs)
    if verbose:
        print(f"    Loaded {len(disease)} diseases")
    
    # Apply exclusions to disease metadata
    disease = apply_exclusions(disease, exclude_disease, "disease")
    
    # Merge with disease data
    if verbose:
        print("    Merging disease associations with disease metadata...")
    disease_to_gene = association_overall_direct.merge(
        disease[["id", "name", "description", "synonyms"]],
        left_on="diseaseId",
        right_on="id",
        how="left"
    )
    if verbose:
        print(f"    Merged dataset shape: {disease_to_gene.shape}")
    
    # Import known drugs
    if verbose:
        print("\n  Step 3/5: Loading known drugs...")
    known_drug = get_dataset(dataset="known_drug", **kwargs)
    if verbose:
        print(f"    Loaded {len(known_drug)} known drug associations")
    
    # Apply exclusions to drugs
    known_drug = apply_exclusions(known_drug, exclude_drugs, "drugs")
    
    # Note: Limits are now applied per-target in preprocessing, not globally
    if limit_drugs is not None and verbose:
        print(f"    Will limit to top {limit_drugs} drugs per target (ranked by phase)")
    
    known_drug["data_id"] = known_drug["drugId"] + "." + known_drug["diseaseId"]
    
    # Import pharmacogenomics
    if verbose:
        print("\n  Step 4/5: Loading and processing pharmacogenomics data...")
    pharmacogenomics = get_dataset(dataset="pharmacogenomics", **kwargs)
    if verbose:
        print(f"    Loaded {len(pharmacogenomics)} pharmacogenomics records")
    
    # Apply exclusions to pharmacogenomics
    pharmacogenomics = apply_exclusions(pharmacogenomics, exclude_pharmacogenomics, "pharmacogenomics")
    
    # Coerce non-numeric 'evidenceLevel' to NaN to avoid ValueError
    if verbose:
        print("    Processing pharmacogenomics data...")
    pharmacogenomics["evidenceLevel"] = pd.to_numeric(
        pharmacogenomics["evidenceLevel"], errors="coerce"
    )
    
    # Apply per-target limit BEFORE aggregation if specified
    if limit_pharmacogenomics is not None:
        if verbose:
            print(f"    Limiting to top {limit_pharmacogenomics} pharmacogenomics records per target (ranked by evidenceLevel, isDirectTarget)...")
        # Sort by evidenceLevel and isDirectTarget (higher is better)
        sort_cols = ["evidenceLevel"]
        if "isDirectTarget" in pharmacogenomics.columns:
            sort_cols.append("isDirectTarget")
        pharmacogenomics = pharmacogenomics.sort_values(sort_cols, ascending=False, na_position='last')
        # Group by target and take top N per target
        pharmacogenomics = pharmacogenomics.groupby("targetFromSourceId").head(limit_pharmacogenomics).reset_index(drop=True)
        if verbose:
            print(f"    After per-target limiting: {len(pharmacogenomics)} pharmacogenomics records")
    
    # Process pharmacogenomics data (aggregate per target)
    gene_to_pharmacogenomics = pharmacogenomics.groupby("targetFromSourceId").agg({
        "variantId": "nunique",
        "isDirectTarget": "mean",
        "evidenceLevel": "median",
        "drugs": (lambda srs: list({
            drug["drugFromSource"]
            for arr in srs.dropna()
            for drug in (arr if isinstance(arr, (list, np.ndarray)) else [arr])
            if isinstance(drug, dict) and "drugFromSource" in drug
        })),
    }).reset_index().sort_values(
        ["targetFromSourceId", "evidenceLevel", "isDirectTarget"], ascending=False
    )
    if verbose:
        print(f"    Processed {len(gene_to_pharmacogenomics)} gene-pharmacogenomics associations")
    
    # Generate markdown
    if verbose:
        print(f"\n  Generating markdown for {len(target)} targets (n_jobs={n_jobs})...")
    # Don't pass save_path to df_to_markdown_batch - we'll save the final result with tokens
    target["markdown"] = df_to_markdown_batch(
        target_df=target,
        include_cols=include_cols,
        include_fields=include_fields,
        disease_to_gene=disease_to_gene,
        drug_to_gene=known_drug,
        gene_to_pharmacogenomics=gene_to_pharmacogenomics,
        n_jobs=n_jobs,
        save_path=None,  # Don't save intermediate result
        force=force,
        verbose=verbose,
        limit_associations=limit_associations,
        limit_drugs=limit_drugs,
        limit_pharmacogenomics=limit_pharmacogenomics,
    )
    
    # Count tokens in markdown
    if verbose:
        print(f"\n  Step 5/5: Counting tokens in markdown...")
    from . import utils
    target["tokens"] = target["markdown"].apply(lambda x: utils.count_tokens(x, approximate=True))
    if verbose:
        total_tokens = target["tokens"].sum()
        avg_tokens = target["tokens"].mean()
        print(f"    Total tokens: {total_tokens:,}")
        print(f"    Average tokens per target: {avg_tokens:.1f}")
    
    # Save to parquet if save_path is provided (save after adding tokens)
    if save_path is not None:
        # Ensure the directory exists
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        target.to_parquet(save_path, index=False)
        
        if verbose:
            print(f"\nSaved targets DataFrame to: {save_path}")
            print(f"  Shape: {target.shape}")
            print(f"  Columns: {list(target.columns)}")
    
    if verbose:
        print(f"\n  Completed! Returning {len(target)} targets with markdown and token counts")
        print("=" * 60)
    
    return target


def diseases_to_markdown(
    diseases_df: pd.DataFrame,
    disease_id_col: str = "diseaseId",
    name_col: str = "name",
    score_col: str = "score",
    evidence_count_col: str = "evidenceCount",
    description_col: str = "description",
    synonyms_col: str = "synonyms",
    omit_duplicate_synonyms: bool = True,
    sep: str = "; ",
    add_association_rank: bool = True,
    limit: Optional[int] = None,
    header: str = "## Trait Associations",
) -> str:
    """
    Convert a diseases DataFrame to markdown format.
    
    Each disease row becomes a bullet point with sub-bullets for additional info.
    
    Parameters
    ----------
    diseases_df : pd.DataFrame
        DataFrame with disease information. Must have columns for name, diseaseId,
        score, evidenceCount, description, and synonyms.
    disease_id_col : str, default "diseaseId"
        Column name for disease ID
    name_col : str, default "name"
        Column name for disease name
    score_col : str, default "score"
        Column name for association score
    evidence_count_col : str, default "evidenceCount"
        Column name for evidence count
    description_col : str, default "description"
        Column name for disease description
    synonyms_col : str, default "synonyms"
        Column name for disease synonyms
    omit_duplicate_synonyms : bool, default True
        If True, omit synonyms that are already in the disease name (case insensitive)
    sep : str, default ";"
        Separator to use when concatenating list columns (e.g., diseaseId, synonyms)
    add_association_rank : bool, default True
        If True, add an "Association Rank" field based on score (rank 1 = highest score)
    limit : int, optional
        Maximum number of rows to include in the output. Applied after sorting by score.
        If None (default), all rows are included.
    header : str, default "## Disease Associations"
        Markdown header for the diseases section
    
    Returns
    -------
    str
        Markdown formatted string with disease information
    
    Examples
    --------
    >>> import phenoref.opentargets as opentargets
    >>> diseases_df = gene2disease.loc[gene2disease["targetId"]==row["id"]]
    >>> markdown = opentargets.diseases_to_markdown(diseases_df)
    >>> print(markdown)
    """
    lines = []
    lines.append(header)
    lines.append("")
    
    if len(diseases_df) == 0:
        lines.append("(No diseases found)")
        return "\n".join(lines)
    
    # Calculate ranks based on score if requested
    if add_association_rank and score_col in diseases_df.columns:
        # Rank by score in descending order (highest score = rank 1)
        diseases_df = diseases_df.copy()
        diseases_df['_association_rank'] = diseases_df[score_col].rank(method='min', ascending=False).astype(int)
    
    # Sort by score (descending) if score column exists
    if score_col in diseases_df.columns:
        diseases_df = diseases_df.sort_values(score_col, ascending=False)
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        diseases_df = diseases_df.head(limit)
    
    # Helper function to safely check if a value is valid (not None, NaN, or empty)
    def is_valid_disease_value(val):
        """Check if a value is valid for disease markdown formatting."""
        if val is None:
            return False
        if isinstance(val, (list, np.ndarray)):
            if isinstance(val, np.ndarray):
                val = val.tolist()
            return len(val) > 0
        try:
            if pd.isna(val):
                return False
        except (ValueError, TypeError):
            # pd.isna might fail for arrays, but we already handled that above
            pass
        if val == "":
            return False
        return True
    
    for idx, row in diseases_df.iterrows():
        # Get disease name
        name = row.get(name_col, "")
        if not is_valid_disease_value(name):
            name = "Unknown"
        else:
            name = str(name)
        
        # Get disease ID (handle arrays/lists)
        disease_id = row.get(disease_id_col, "")
        disease_id_str = ""
        if is_valid_disease_value(disease_id):
            # Handle array/list format
            if isinstance(disease_id, (list, np.ndarray)):
                if isinstance(disease_id, np.ndarray):
                    disease_id = disease_id.tolist()
                # Filter out empty values
                disease_id = [d for d in disease_id if is_valid_disease_value(d)]
                if disease_id:
                    # Format using sep parameter
                    if len(disease_id) == 1:
                        disease_id_str = str(disease_id[0])
                    else:
                        disease_id_str = sep.join(str(d) for d in disease_id)
            else:
                disease_id_str = str(disease_id)
        
        # Format disease header: "name (id)" or just "name" if no id
        if disease_id_str:
            lines.append(f"- {name} ({disease_id_str})")
        else:
            lines.append(f"- {name}")
        
        # Synonyms (handle lists/arrays) - FIRST sub-bullet
        synonyms = row.get(synonyms_col, "")
        if is_valid_disease_value(synonyms):
            # Get the disease name in lowercase for comparison
            name_lower = name.lower() if name else ""
            
            if isinstance(synonyms, (list, np.ndarray)):
                if isinstance(synonyms, np.ndarray):
                    synonyms = synonyms.tolist()
                # Filter out empty strings
                synonyms_list = [str(s) for s in synonyms if s and str(s).strip()]
                
                # Filter out synonyms that are already in the name (case insensitive) if requested
                if omit_duplicate_synonyms and name_lower:
                    synonyms_list = [
                        s for s in synonyms_list 
                        if s.lower() not in name_lower
                    ]
                
                if synonyms_list:
                    synonyms_str = sep.join(synonyms_list)
                    lines.append(f"   * Synonyms: {synonyms_str}")
            elif isinstance(synonyms, str) and synonyms.strip():
                # Check if synonym is already in the name (case insensitive)
                if not (omit_duplicate_synonyms and name_lower and synonyms.lower() in name_lower):
                    lines.append(f"   * Synonyms: {synonyms}")
            else:
                synonyms_str = str(synonyms).strip()
                if synonyms_str:
                    # Check if synonym is already in the name (case insensitive)
                    if not (omit_duplicate_synonyms and name_lower and synonyms_str.lower() in name_lower):
                        lines.append(f"   * Synonyms: {synonyms_str}")
        
        # Score
        score = row.get(score_col, None)
        if is_valid_disease_value(score):
            if isinstance(score, (int, float)):
                lines.append(f"   * Association Score: {score:.6f}")
            else:
                lines.append(f"   * Association Score: {score}")
        
        # Association Rank (if requested)
        if add_association_rank:
            rank = row.get('_association_rank', None)
            if rank is not None and not pd.isna(rank):
                lines.append(f"   * Association Rank: {int(rank)}")
        
        # Evidence Count
        evidence_count = row.get(evidence_count_col, None)
        if is_valid_disease_value(evidence_count):
            if isinstance(evidence_count, (int, float)):
                lines.append(f"   * Evidence Count: {int(evidence_count)}")
            else:
                lines.append(f"   * Evidence Count: {evidence_count}")
        
        # Description
        description = row.get(description_col, "")
        if is_valid_disease_value(description):
            desc_str = str(description).strip()
            if desc_str:
                lines.append(f"   * Description: {desc_str}")
        
        lines.append("")  # Empty line between diseases
    
    return "\n".join(lines)


def drugs_to_markdown(
    drugs_df: pd.DataFrame,
    drug_id_col: str = "drugId",
    pref_name_col: str = "prefName",
    trade_names_col: str = "tradeNames",
    synonyms_col: str = "synonyms",
    drug_type_col: str = "drugType",
    mechanism_of_action_col: str = "mechanismOfAction",
    phase_col: str = "phase",
    status_col: str = "status",
    target_name_col: str = "targetName",
    approved_symbol_col: str = "approvedSymbol",
    disease_id_col: str = "diseaseId",
    disease_label_col: str = "label",
    omit_duplicate_synonyms: bool = True,
    sep: str = "; ",
    limit: Optional[int] = None,
    header: str = "## Drug Associations",
) -> str:
    """
    Convert a drugs DataFrame to markdown format.
    
    Each drug row becomes a bullet point with sub-bullets for additional info.
    
    Parameters
    ----------
    drugs_df : pd.DataFrame
        DataFrame with drug information. Expected columns include drugId, prefName,
        tradeNames, synonyms, drugType, mechanismOfAction, phase, status, etc.
    drug_id_col : str, default "drugId"
        Column name for drug ID
    pref_name_col : str, default "prefName"
        Column name for preferred drug name
    trade_names_col : str, default "tradeNames"
        Column name for trade names
    synonyms_col : str, default "synonyms"
        Column name for drug synonyms
    drug_type_col : str, default "drugType"
        Column name for drug type
    mechanism_of_action_col : str, default "mechanismOfAction"
        Column name for mechanism of action
    phase_col : str, default "phase"
        Column name for clinical phase
    status_col : str, default "status"
        Column name for status
    target_name_col : str, default "targetName"
        Column name for target name
    approved_symbol_col : str, default "approvedSymbol"
        Column name for approved symbol
    disease_id_col : str, default "diseaseId"
        Column name for disease ID
    disease_label_col : str, default "label"
        Column name for disease label
    omit_duplicate_synonyms : bool, default True
        If True, omit synonyms that are already in the drug name (case insensitive)
    sep : str, default ";"
        Separator to use when concatenating list columns (e.g., tradeNames, synonyms)
    limit : int, optional
        Maximum number of rows to include in the output.
        If None (default), all rows are included.
    header : str, default "## Drug Associations"
        Markdown header for the drugs section
    
    Returns
    -------
    str
        Markdown formatted string with drug information
    
    Examples
    --------
    >>> import phenoref.opentargets as opentargets
    >>> drugs_df = known_drug.loc[known_drug["targetId"]==row["id"]]
    >>> markdown = opentargets.drugs_to_markdown(drugs_df)
    >>> print(markdown)
    """
    lines = []
    lines.append(header)
    lines.append("")
    
    if len(drugs_df) == 0:
        lines.append("(No drugs found)")
        return "\n".join(lines)
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        drugs_df = drugs_df.head(limit)
    
    # Helper function to safely check if a value is valid (not None, NaN, or empty)
    def is_valid_drug_value(val):
        """Check if a value is valid for drug markdown formatting."""
        if val is None:
            return False
        if isinstance(val, (list, np.ndarray)):
            if isinstance(val, np.ndarray):
                val = val.tolist()
            return len(val) > 0
        try:
            if pd.isna(val):
                return False
        except (ValueError, TypeError):
            # pd.isna might fail for arrays, but we already handled that above
            pass
        if val == "":
            return False
        return True
    
    for idx, row in drugs_df.iterrows():
        # Get preferred name
        pref_name = row.get(pref_name_col, "")
        if not is_valid_drug_value(pref_name):
            pref_name = "Unknown"
        else:
            pref_name = str(pref_name)
        
        # Get drug ID
        drug_id = row.get(drug_id_col, "")
        drug_id_str = ""
        if is_valid_drug_value(drug_id):
            drug_id_str = str(drug_id)
        
        # Format drug header: "name (id)" or just "name" if no id
        if drug_id_str:
            lines.append(f"- {pref_name} ({drug_id_str})")
        else:
            lines.append(f"- {pref_name}")
        
        # Synonyms (handle lists/arrays) - FIRST sub-bullet
        synonyms = row.get(synonyms_col, "")
        if is_valid_drug_value(synonyms):
            # Get the preferred name in lowercase for comparison
            pref_name_lower = pref_name.lower() if pref_name else ""
            
            if isinstance(synonyms, (list, np.ndarray)):
                if isinstance(synonyms, np.ndarray):
                    synonyms = synonyms.tolist()
                # Filter out empty strings
                synonyms_list = [str(s) for s in synonyms if s and str(s).strip()]
                
                # Filter out synonyms that match the preferred name (case insensitive) if requested
                if omit_duplicate_synonyms and pref_name_lower:
                    synonyms_list = [
                        s for s in synonyms_list 
                        if s.lower() not in pref_name_lower
                    ]
                
                if synonyms_list:
                    synonyms_str = sep.join(synonyms_list)
                    lines.append(f"   * Synonyms: {synonyms_str}")
            elif isinstance(synonyms, str) and synonyms.strip():
                # Check if synonym matches preferred name (case insensitive)
                if not (omit_duplicate_synonyms and pref_name_lower and synonyms.lower() == pref_name_lower):
                    lines.append(f"   * Synonyms: {synonyms}")
            else:
                synonyms_str = str(synonyms).strip()
                if synonyms_str:
                    # Check if synonym matches preferred name (case insensitive)
                    if not (omit_duplicate_synonyms and pref_name_lower and synonyms_str.lower() == pref_name_lower):
                        lines.append(f"   * Synonyms: {synonyms_str}")
        
        # Drug Type
        drug_type = row.get(drug_type_col, None)
        if is_valid_drug_value(drug_type):
            lines.append(f"   * Drug Type: {drug_type}")
        
        # Mechanism of Action
        mechanism = row.get(mechanism_of_action_col, None)
        if is_valid_drug_value(mechanism):
            lines.append(f"   * Mechanism of Action: {mechanism}")
        
        # Phase
        phase = row.get(phase_col, None)
        if is_valid_drug_value(phase):
            if isinstance(phase, (int, float)):
                lines.append(f"   * Phase: {int(phase)}")
            else:
                lines.append(f"   * Phase: {phase}")
        
        # Status
        status = row.get(status_col, None)
        if is_valid_drug_value(status):
            lines.append(f"   * Status: {status}")
        
        # Target Name
        target_name = row.get(target_name_col, None)
        if is_valid_drug_value(target_name):
            lines.append(f"   * Target: {target_name}")
        
        # Approved Symbol
        approved_symbol = row.get(approved_symbol_col, None)
        if is_valid_drug_value(approved_symbol):
            lines.append(f"   * Target Symbol: {approved_symbol}")
        
        # Disease ID (handle arrays/lists)
        disease_id = row.get(disease_id_col, "")
        disease_id_str = ""
        if is_valid_drug_value(disease_id):
            # Handle array/list format
            if isinstance(disease_id, (list, np.ndarray)):
                if isinstance(disease_id, np.ndarray):
                    disease_id = disease_id.tolist()
                # Filter out empty values
                disease_id = [d for d in disease_id if is_valid_drug_value(d)]
                if disease_id:
                    # Format using sep parameter
                    if len(disease_id) == 1:
                        disease_id_str = str(disease_id[0])
                    else:
                        disease_id_str = sep.join(str(d) for d in disease_id)
            else:
                disease_id_str = str(disease_id)
        
        if disease_id_str:
            lines.append(f"   * Disease ID: {disease_id_str}")
        
        # Disease Label
        disease_label = row.get(disease_label_col, None)
        if is_valid_drug_value(disease_label):
            lines.append(f"   * Disease: {disease_label}")
        
        # Trade Names (handle lists/arrays)
        trade_names = row.get(trade_names_col, "")
        if is_valid_drug_value(trade_names):
            # Get the preferred name in lowercase for comparison
            pref_name_lower = pref_name.lower() if pref_name else ""
            
            if isinstance(trade_names, (list, np.ndarray)):
                if isinstance(trade_names, np.ndarray):
                    trade_names = trade_names.tolist()
                # Filter out empty strings
                trade_names_list = [str(t) for t in trade_names if t and str(t).strip()]
                
                # Filter out trade names that match the preferred name (case insensitive) if requested
                if omit_duplicate_synonyms and pref_name_lower:
                    trade_names_list = [
                        t for t in trade_names_list 
                        if t.lower() not in pref_name_lower
                    ]
                
                if trade_names_list:
                    trade_names_str = sep.join(trade_names_list)
                    lines.append(f"   * Trade Names: {trade_names_str}")
            elif isinstance(trade_names, str) and trade_names.strip():
                # Check if trade name matches preferred name (case insensitive)
                if not (omit_duplicate_synonyms and pref_name_lower and trade_names.lower() == pref_name_lower):
                    lines.append(f"   * Trade Names: {trade_names}")
            else:
                trade_names_str = str(trade_names).strip()
                if trade_names_str:
                    # Check if trade name matches preferred name (case insensitive)
                    if not (omit_duplicate_synonyms and pref_name_lower and trade_names_str.lower() == pref_name_lower):
                        lines.append(f"   * Trade Names: {trade_names_str}")
        
        lines.append("")  # Empty line between drugs
    
    return "\n".join(lines)


def pharmacogenomics_to_markdown(
    pgx_df: pd.DataFrame,
    target_id_col: str = "targetFromSourceId",
    variant_id_col: str = "variantId",
    is_direct_target_col: str = "isDirectTarget",
    evidence_level_col: str = "evidenceLevel",
    drugs_col: str = "drugs",
    sep: str = "; ",
    limit: Optional[int] = None,
    header: str = "## Pharmacogenomic Interactions",
) -> str:
    """
    Convert a pharmacogenomics DataFrame to markdown format.
    
    Each row becomes a bullet point with sub-bullets for variant count, direct target status,
    evidence level, and associated drugs.
    
    Parameters
    ----------
    pgx_df : pd.DataFrame
        DataFrame with pharmacogenomics information. Expected columns include:
        targetFromSourceId, variantId, isDirectTarget, evidenceLevel, drugs.
    target_id_col : str, default "targetFromSourceId"
        Column name for target/gene ID
    variant_id_col : str, default "variantId"
        Column name for number of unique variants
    is_direct_target_col : str, default "isDirectTarget"
        Column name for direct target status (mean value, 0-1)
    evidence_level_col : str, default "evidenceLevel"
        Column name for evidence level (median value, numeric)
    drugs_col : str, default "drugs"
        Column name for list of associated drugs
    sep : str, default "; "
        Separator to use when concatenating list columns (e.g., drugs)
    limit : int, optional
        Maximum number of rows to include in the output. Applied after sorting.
        If None (default), all rows are included.
    header : str, default "## Pharmacogenomics"
        Markdown header for the pharmacogenomics section
    
    Returns
    -------
    str
        Markdown formatted string with pharmacogenomics information
    
    Examples
    --------
    >>> import phenoref.opentargets as opentargets
    >>> pgx_df = gene_to_pgx.loc[gene_to_pgx["targetFromSourceId"]==row["id"]]
    >>> markdown = opentargets.pharmacogenomics_to_markdown(pgx_df)
    >>> print(markdown)
    """
    lines = []
    lines.append(header)
    lines.append("")
    
    if len(pgx_df) == 0:
        lines.append("(No pharmacogenomics data found)")
        return "\n".join(lines)
    
    # Sort by evidence level (descending), then by isDirectTarget (descending), then by variant count (descending)
    sort_cols = []
    if evidence_level_col in pgx_df.columns:
        sort_cols.append(evidence_level_col)
    if is_direct_target_col in pgx_df.columns:
        sort_cols.append(is_direct_target_col)
    if variant_id_col in pgx_df.columns:
        sort_cols.append(variant_id_col)
    
    if sort_cols:
        pgx_df = pgx_df.sort_values(sort_cols, ascending=False)
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        pgx_df = pgx_df.head(limit)
    
    # Helper function to safely check if a value is valid (not None, NaN, or empty)
    def is_valid_pgx_value(val):
        """Check if a value is valid for pharmacogenomics markdown formatting."""
        if val is None:
            return False
        if isinstance(val, (list, np.ndarray)):
            if isinstance(val, np.ndarray):
                val = val.tolist()
            return len(val) > 0
        try:
            if pd.isna(val):
                return False
        except (ValueError, TypeError):
            pass
        if val == "":
            return False
        return True
    
    for idx, row in pgx_df.iterrows():
        # Get drugs first (handle lists/arrays) - this will be the main bullet
        drugs = row.get(drugs_col, "")
        drugs_str = ""
        if is_valid_pgx_value(drugs):
            if isinstance(drugs, (list, np.ndarray)):
                if isinstance(drugs, np.ndarray):
                    drugs = drugs.tolist()
                # Filter out empty strings
                drugs_list = [str(d) for d in drugs if d and str(d).strip()]
                if drugs_list:
                    drugs_str = sep.join(drugs_list)
            elif isinstance(drugs, str) and drugs.strip():
                drugs_str = drugs.strip()
            else:
                drugs_str = str(drugs).strip()
        
        # If no drugs, skip this entry
        if not drugs_str:
            continue
        
        # Format main bullet with drugs
        lines.append(f"- Drugs: {drugs_str}")
        
        # Variant Count
        variant_count = row.get(variant_id_col, None)
        if is_valid_pgx_value(variant_count):
            if isinstance(variant_count, (int, float)):
                lines.append(f"   * Variant Count: {int(variant_count)}")
            else:
                lines.append(f"   * Variant Count: {variant_count}")
        
        # Is Direct Target (as percentage)
        is_direct = row.get(is_direct_target_col, None)
        if is_valid_pgx_value(is_direct):
            if isinstance(is_direct, (int, float)):
                direct_pct = is_direct * 100
                lines.append(f"   * Direct Target: {direct_pct:.1f}%")
            else:
                lines.append(f"   * Direct Target: {is_direct}")
        
        # Evidence Level
        evidence_level = row.get(evidence_level_col, None)
        if is_valid_pgx_value(evidence_level):
            if isinstance(evidence_level, (int, float)):
                lines.append(f"   * Evidence Level: {evidence_level:.1f}")
            else:
                lines.append(f"   * Evidence Level: {evidence_level}")
        
        lines.append("")  # Empty line between entries
    
    return "\n".join(lines)


# Import create_gene_association_matrix and filter_adaptive from utils
from .utils import create_gene_association_matrix, filter_adaptive


def _prepare_disease_to_gene_associations(
    association_dataset: str = "association_by_datasource_direct",
    cache_dir: Optional[str] = None,
    force: bool = False,
    output_format: str = "pandas",
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare disease-to-gene associations from OpenTargets association dataset.
    
    This function downloads the association dataset and disease metadata, merges them,
    and creates standardized columns for use with create_gene_association_matrix().
    
    Parameters
    ----------
    association_dataset : str, default "association_by_datasource_direct"
        Name of the association dataset to use.
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: Name of the association dataset
        - sourceId: diseaseId.datatypeId.datasourceId (unique identifier for each association source)
        - targetId: targetId (gene identifier)
        - score: Association score
        - Additional columns from disease metadata (name, description, synonyms)
    """
    if verbose >= 1:
        logger.info(f"Preparing disease-to-gene associations from {association_dataset}")
    
    # Import disease-gene associations
    association_by_datasource_direct = get_dataset(
        dataset=association_dataset,
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )
    
    # Import disease metadata
    disease = get_dataset(
        dataset="disease",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )
    
    # Merge with disease data
    disease_to_gene = association_by_datasource_direct.merge(
        disease[["id", "name", "description", "synonyms"]],
        left_on="diseaseId",
        right_on="id",
        how="left"
    )
    
    # Create sourceId column
    disease_to_gene["sourceId"] = (
        disease_to_gene["diseaseId"] + "." + 
        disease_to_gene["datatypeId"] + "." + 
        disease_to_gene["datasourceId"]
    )
    
    disease_to_gene["dataset"] = association_dataset
    disease_to_gene["database"] = "OpenTargets"
    
    # Add label column (from disease name)
    disease_to_gene["label"] = disease_to_gene["name"]
    
    if verbose >= 1:
        logger.info(f"Unique sourceIds: {disease_to_gene['sourceId'].nunique():,}")
        logger.info(f"DataFrame shape: {disease_to_gene.shape}")
    
    return disease_to_gene


def _prepare_known_drug_associations(
    cache_dir: Optional[str] = None,
    force: bool = False,
    output_format: str = "pandas",
    default_score: Optional[float] = None,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare known drug associations from OpenTargets.
    
    This function downloads the known_drug dataset and creates standardized columns
    for use with create_gene_association_matrix().
    
    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value to assign (known_drug dataset doesn't have scores).
        If None, fills the score column with NaN/NA values.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "known_drug"
        - sourceId: drugId.diseaseId (unique identifier for each association source)
        - targetId: targetId (gene identifier)
        - score: Default score (0.5)
    """
    if verbose >= 1:
        logger.info("Preparing known drug associations")
    
    known_drug = get_dataset(
        dataset="known_drug",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )
    
    known_drug["sourceId"] = known_drug["drugId"] + "." + known_drug["diseaseId"]
    if default_score is None:
        known_drug["score"] = pd.NA
    else:
        known_drug["score"] = default_score
    known_drug["dataset"] = "known_drug"
    known_drug["database"] = "OpenTargets"
    
    # Add label column (from prefName - drug preferred name)
    known_drug["label"] = known_drug["prefName"]
    
    if verbose >= 1:
        logger.info(f"DataFrame shape: {known_drug.shape}")
    
    return known_drug


def _prepare_pharmacogenomics_associations(
    cache_dir: Optional[str] = None,
    force: bool = False,
    output_format: str = "pandas",
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare pharmacogenomics associations from OpenTargets.
    
    This function downloads the pharmacogenomics dataset, aggregates by target and datasource,
    extracts drug information, and creates standardized columns for use with 
    create_gene_association_matrix().
    
    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "pharmacogenomics"
        - sourceId: drugFromSource.datasourceId (unique identifier for each association source)
        - targetId: targetId (gene identifier, renamed from targetFromSourceId)
        - score: Normalized evidenceLevel (0-1 scale)
        - Additional columns: variantId (count), isDirectTarget, evidenceLevel
    """
    if verbose >= 1:
        logger.info("Preparing pharmacogenomics associations")
    
    pharmacogenomics = get_dataset(
        dataset="pharmacogenomics",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )
    
    # First, coerce non-numeric 'evidenceLevel' to NaN to avoid ValueError
    pharmacogenomics["evidenceLevel"] = pd.to_numeric(pharmacogenomics["evidenceLevel"], errors="coerce")
    
    # Custom aggregation functions for each drug column separately
    def extract_drug_from_source(srs):
        drug_from_source_set = set()
        for arr in srs.dropna():
            if isinstance(arr, (list, np.ndarray)):
                drugs = arr
            else:
                drugs = [arr]
            for drug in drugs:
                if isinstance(drug, dict) and "drugFromSource" in drug and drug["drugFromSource"] is not None:
                    drug_from_source_set.add(drug["drugFromSource"])
        return list(drug_from_source_set)
    
    def extract_drug_id(srs):
        drug_id_set = set()
        for arr in srs.dropna():
            if isinstance(arr, (list, np.ndarray)):
                drugs = arr
            else:
                drugs = [arr]
            for drug in drugs:
                if isinstance(drug, dict) and "drugId" in drug and drug["drugId"] is not None:
                    drug_id_set.add(drug["drugId"])
        return list(drug_id_set)
    
    # First do the standard aggregations
    gene_to_pharmacogenomics = pharmacogenomics.groupby(["targetFromSourceId", "datasourceId"]).agg({
        "variantId": "nunique",
        "isDirectTarget": "mean",
        "evidenceLevel": "mean",
    }).reset_index()
    
    # Extract drug columns separately using apply with reset_index(name="...")
    drug_from_source_col = pharmacogenomics.groupby(["targetFromSourceId", "datasourceId"])["drugs"].apply(extract_drug_from_source).reset_index(name="drugFromSource")
    drug_id_col = pharmacogenomics.groupby(["targetFromSourceId", "datasourceId"])["drugs"].apply(extract_drug_id).reset_index(name="drugId")
    
    # Merge both drug columns
    gene_to_pharmacogenomics = gene_to_pharmacogenomics.merge(
        drug_from_source_col,
        on=["targetFromSourceId", "datasourceId"],
        how="left"
    ).merge(
        drug_id_col,
        on=["targetFromSourceId", "datasourceId"],
        how="left"
    )
    
    # Rename
    gene_to_pharmacogenomics = gene_to_pharmacogenomics.rename(columns={"targetFromSourceId": "targetId"})
    
    # Sort first
    gene_to_pharmacogenomics = gene_to_pharmacogenomics.sort_values(["targetId", "evidenceLevel", "isDirectTarget"], ascending=False)
    
    # Explode drugFromSource column - this will create one row per drug
    gene_to_pharmacogenomics = gene_to_pharmacogenomics.explode("drugFromSource")
    
    gene_to_pharmacogenomics["sourceId"] = gene_to_pharmacogenomics["drugFromSource"] + "." + gene_to_pharmacogenomics["datasourceId"]
    
    # Normalize score from evidenceLevel (0-1 scale)
    max_evidence = gene_to_pharmacogenomics["evidenceLevel"].max()
    if pd.notna(max_evidence) and max_evidence > 0:
        gene_to_pharmacogenomics["score"] = gene_to_pharmacogenomics["evidenceLevel"] / max_evidence
    else:
        gene_to_pharmacogenomics["score"] = 0.0
    
    gene_to_pharmacogenomics["dataset"] = "pharmacogenomics"
    gene_to_pharmacogenomics["database"] = "OpenTargets"
    
    # Add label column (from drugFromSource - drug name from source)
    gene_to_pharmacogenomics["label"] = gene_to_pharmacogenomics["drugFromSource"]
    
    if verbose >= 1:
        logger.info(f"DataFrame shape: {gene_to_pharmacogenomics.shape}")
    
    return gene_to_pharmacogenomics


def _prepare_mouse_phenotype_associations(
    cache_dir: Optional[str] = None,
    force: bool = False,
    output_format: str = "pandas",
    default_score: Optional[float] = None,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare mouse phenotype associations from OpenTargets.
    
    This function downloads the mouse_phenotype dataset and creates standardized columns
    for use with create_gene_association_matrix().
    
    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value to assign (mouse_phenotype dataset doesn't have scores).
        If None, fills the score column with NaN/NA values.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "mouse_phenotype"
        - sourceId: modelPhenotypeId (unique identifier for each phenotype)
        - targetId: targetFromSourceId (human gene identifier)
        - score: DEFAULT_SCORE (0.5) for all rows
        - label: modelPhenotypeLabel (phenotype label)
    """
    if verbose >= 1:
        logger.info("Preparing mouse phenotype associations")
    
    mouse_phenotype = get_dataset(
        dataset="mouse_phenotype",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )
    
    # Rename targetFromSourceId to targetId (human gene ID)
    mouse_phenotype = mouse_phenotype.rename(columns={"targetFromSourceId": "targetId"})
    
    # Use modelPhenotypeId as sourceId
    mouse_phenotype["sourceId"] = mouse_phenotype["modelPhenotypeId"]
    
    # Add score column - always use DEFAULT_SCORE (0.5) for mouse_phenotype
    mouse_phenotype["score"] = DEFAULT_SCORE
    
    # Add dataset and database columns
    mouse_phenotype["dataset"] = "mouse_phenotype"
    mouse_phenotype["database"] = "OpenTargets"
    
    # Add label column (from modelPhenotypeLabel)
    mouse_phenotype["label"] = mouse_phenotype["modelPhenotypeLabel"]
    
    if verbose >= 1:
        logger.info(f"DataFrame shape: {mouse_phenotype.shape}")
    
    return mouse_phenotype


def _prepare_expression_associations(
    cache_dir: Optional[str] = None,
    force: bool = False,
    output_format: str = "pandas",
    default_score: Optional[float] = None,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare expression associations from OpenTargets.
    
    This function downloads the expression dataset (which is automatically
    parsed by get_dataset), and creates standardized columns for use with 
    create_gene_association_matrix().
    
    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value to assign if rna_value is not available.
        If None, uses rna_value as score, or NaN if rna_value is missing.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "expression"
        - sourceId: efo_code (tissue identifier)
        - targetId: id (gene identifier)
        - score: rna_value (expression value)
        - label: tissueLabel (tissue name)
    """
    if verbose >= 1:
        logger.info("Preparing expression associations")
    
    # get_dataset automatically parses expression
    expression = get_dataset(
        dataset="expression",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )
    
    # Rename id to targetId
    if 'id' in expression.columns:
        expression = expression.rename(columns={'id': 'targetId'})
    
    # Map efo_code to sourceId
    if 'efo_code' in expression.columns:
        expression['sourceId'] = expression['efo_code']
    
    # Add score column (use rna_value, or default_score, or NaN)
    if 'rna_value' in expression.columns:
        # Use rna_value directly as score
        expression['score'] = expression['rna_value']
        # Replace NaN with default_score if provided
        if default_score is not None:
            expression['score'] = expression['score'].fillna(default_score)
    else:
        # No rna_value column, use default_score or NaN
        if default_score is None:
            expression['score'] = pd.NA
        else:
            expression['score'] = default_score
    
    # Add dataset and database columns
    expression['dataset'] = "expression"
    expression['database'] = "OpenTargets"
    
    # Map tissueLabel to label
    if 'tissueLabel' in expression.columns:
        expression['label'] = expression['tissueLabel']
    else:
        expression['label'] = pd.NA
    
    if verbose >= 1:
        logger.info(f"DataFrame shape: {expression.shape}")
    
    return expression


def _prepare_target_essentiality_associations(
    cache_dir: Optional[str] = None,
    force: bool = False,
    output_format: str = "pandas",
    default_score: Optional[float] = None,
    verbose: int = 1,
) -> pd.DataFrame:
    """
    Prepare target essentiality associations from OpenTargets.
    
    This function downloads the target_essentiality dataset (which is automatically
    parsed by get_dataset), and creates standardized columns for use with 
    create_gene_association_matrix().
    
    Parameters
    ----------
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value to assign if geneEffect is not available.
        If None, uses geneEffect (absolute value) as score, or NaN if geneEffect is missing.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - database: "OpenTargets"
        - dataset: "target_essentiality"
        - sourceId: tissueId.depmapId.diseaseCellLineId.mutation (unique identifier)
        - targetId: geneId (gene identifier)
        - score: geneEffect or default_score
        - label: tissueName + cellLineName + diseaseFromSource + mutation (if not None)
    """
    if verbose >= 1:
        logger.info("Preparing target essentiality associations")
    
    # get_dataset automatically parses target_essentiality
    target_essentiality = get_dataset(
        dataset="target_essentiality",
        cache_dir=cache_dir,
        force=force,
        output_format=output_format,
        verbose=verbose - 1 if verbose > 0 else 0,
    )
    
    # Rename geneId to targetId
    if 'geneId' in target_essentiality.columns:
        target_essentiality = target_essentiality.rename(columns={'geneId': 'targetId'})
    
    # Add score column (use geneEffect directly, or default_score, or NaN)
    if 'geneEffect' in target_essentiality.columns:
        # Use geneEffect directly as score
        target_essentiality['score'] = target_essentiality['geneEffect']
        # Replace NaN with default_score if provided
        if default_score is not None:
            target_essentiality['score'] = target_essentiality['score'].fillna(default_score)
    else:
        # No geneEffect column, use default_score or NaN
        if default_score is None:
            target_essentiality['score'] = pd.NA
        else:
            target_essentiality['score'] = default_score
    
    # Add dataset and database columns
    target_essentiality['dataset'] = "target_essentiality"
    target_essentiality['database'] = "OpenTargets"
    
    # Add label column: tissueName + cellLineName + diseaseFromSource + mutation (if not None)
    def combine_label(row):
        parts = []
        if 'tissueName' in row.index and pd.notna(row['tissueName']) and row['tissueName']:
            parts.append(str(row['tissueName']))
        if 'cellLineName' in row.index and pd.notna(row['cellLineName']) and row['cellLineName']:
            parts.append(str(row['cellLineName']))
        if 'diseaseFromSource' in row.index and pd.notna(row['diseaseFromSource']) and row['diseaseFromSource']:
            parts.append(str(row['diseaseFromSource']))
        if 'mutation' in row.index and pd.notna(row['mutation']) and row['mutation']:
            parts.append(str(row['mutation']))
        return ' '.join(parts) if parts else pd.NA
    
    target_essentiality['label'] = target_essentiality.apply(combine_label, axis=1)
    
    if verbose >= 1:
        logger.info(f"DataFrame shape: {target_essentiality.shape}")
    
    return target_essentiality


def get_gene_associations(
    datasets: Optional[list] = None,
    association_dataset: str = "association_by_datasource_direct",
    cache_dir: Optional[str] = None,
    force: int = 0,
    output_format: str = "pandas",
    default_score: Optional[float] = None,
    verbose: int = 1,
    save_path: Optional[Union[str, Path]] = None,
    filter_adaptive_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """
    Prepare gene association matrix from multiple OpenTargets datasets.
    
    This function downloads and processes multiple OpenTargets datasets, standardizes
    their format, and combines them into a single DataFrame ready for use with
    create_gene_association_matrix().
    
    Parameters
    ----------
    datasets : list of str, optional
        List of dataset names to include. If None, uses default selection:
        - "disease-to-gene"
        - "known_drug"
        - "pharmacogenomics"
        - "mouse_phenotype"
        - "target_essentiality"
        - "expression"
    association_dataset : str, default "association_by_datasource_direct"
        Name of the association dataset to use for disease-to-gene associations.
    cache_dir : str, optional
        Local directory to cache downloaded files. If None, uses default cache directory.
    force : bool, default False
        If True, re-download files even if they already exist in cache.
    output_format : str, default "pandas"
        Output format: "pandas" or "polars".
    default_score : float or None, default None
        Default score value for known_drug dataset (which doesn't have scores).
        If None, fills the score column with NaN/NA values.
    verbose : int, default 1
        Verbosity level:
        - 0: Minimal output
        - 1: Show progress information
        - 2: Show detailed information
    save_path : str or Path, optional
        Path to save the final combined DataFrame as a parquet file.
        Can be a string or Path object (including PosixPath).
        If provided, the DataFrame will be saved before returning.
        If None, the DataFrame is not saved.
    force : int, default 0
        Controls caching behavior:
        - 0: Use cached merged file if save_path exists, otherwise process from cached individual chunks
        - 1: Recreate merged DataFrame from cached individual chunks and resave (don't re-download chunks)
        - 2: Re-download individual parquet chunks and recreate merged DataFrame
    filter_adaptive_kwargs : dict, optional
        Keyword arguments to pass to filter_adaptive for each dataset.
        If a single dict, applies the same arguments to all datasets.
        If a dict of dicts, keys are dataset names and values are kwargs for each dataset.
        E.g., {"target_essentiality": {"min_genes": 3, "percentile": 0.99}, 
               "expression": {"min_genes": 3, "percentile": 0.99}}
        Common kwargs: percentile, min_genes, max_genes, sort_by, etc.
        If None, no filtering is applied.
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with standardized columns:
        - database: "OpenTargets" for all rows
        - dataset: Name of the source dataset
        - sourceId: Unique identifier for each association source
        - targetId: Gene identifier
        - score: Association score
        - label: Human-readable label (disease name, drug name, etc. depending on dataset)
        - All other columns from the original datasets
    
    Examples
    --------
    >>> import phenoref.opentargets as opentargets
    >>> 
    >>> # Use all datasets (default)
    >>> associations = opentargets.get_gene_associations()
    >>> 
    >>> # Include only disease-to-gene and known_drug
    >>> associations = opentargets.get_gene_associations(
    ...     datasets=["disease-to-gene", "known_drug"]
    ... )
    >>> 
    >>> # Use with create_gene_association_matrix
    >>> import phenoref.utils as utils
    >>> associations = opentargets.get_gene_associations()
    >>> X, metadata = utils.create_gene_association_matrix(associations)
    """
    # Check save_path at the beginning - if it exists and force is False, load and return it
    if save_path is not None:
        # Convert to Path if it's a string, otherwise use as-is (already a Path/PosixPath)
        if isinstance(save_path, str):
            save_path_obj = Path(save_path)
        else:
            save_path_obj = save_path
        
        # If file exists and force is 0, load and return it
        if save_path_obj.exists() and force == 0:
            if verbose >= 1:
                logger.info(f"Loading existing file from: {save_path_obj}")
                print(f"Loading existing file from: {save_path_obj}")
            return pd.read_parquet(save_path_obj)
    
    # Default datasets if not provided
    if datasets is None:
        datasets = ["disease-to-gene", "known_drug", "pharmacogenomics", "mouse_phenotype", "target_essentiality", "expression"]
    
    if verbose >= 1:
        msg1 = "Preparing gene associations from OpenTargets datasets"
        msg2 = f"Including {len(datasets)} datasets: {', '.join(datasets)}"
        logger.info(msg1)
        logger.info(msg2)
        # Also print to stdout for interactive usage (e.g., Jupyter)
        print(msg1)
        print(msg2)
    
    associations_list = []
    
    # Helper function to apply filter_adaptive if specified
    def apply_filter_adaptive(df, dataset_name):
        """Apply filter_adaptive to a dataset if filter_adaptive_kwargs is specified."""
        if filter_adaptive_kwargs is None:
            return df
        
        # Determine the kwargs for this dataset
        if isinstance(filter_adaptive_kwargs, dict):
            # Check if it's a dict of dicts (dataset-specific) or a single dict (apply to all)
            if dataset_name in filter_adaptive_kwargs and isinstance(filter_adaptive_kwargs[dataset_name], dict):
                # Dataset-specific kwargs
                kwargs = filter_adaptive_kwargs[dataset_name].copy()
            elif all(isinstance(v, dict) for v in filter_adaptive_kwargs.values() if v is not None):
                # It's a dict of dicts but this dataset not specified
                kwargs = None
            else:
                # Single dict to apply to all datasets
                kwargs = filter_adaptive_kwargs.copy()
        else:
            kwargs = None
        
        # Apply filtering if kwargs are specified for this dataset
        if kwargs is not None:
            if verbose >= 1:
                kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                logger.info(f"Applying filter_adaptive to {dataset_name} with: {kwargs_str}")
                print(f"Applying filter_adaptive to {dataset_name} with: {kwargs_str}")
            
            # Set default sort_by column based on dataset if not specified
            if "sort_by" not in kwargs:
                if dataset_name == "target_essentiality" and "geneEffect" in df.columns:
                    kwargs["sort_by"] = "geneEffect"
                else:
                    kwargs["sort_by"] = "score"
            
            # Set default verbose if not specified
            if "verbose" not in kwargs:
                kwargs["verbose"] = verbose >= 2
            
            df = filter_adaptive(
                df=df,
                source_id_col="sourceId",
                target_id_col="targetId",
                **kwargs
            )
            
            if verbose >= 1:
                logger.info(f"After filtering: {len(df):,} rows")
                print(f"After filtering: {len(df):,} rows")
        
        return df
    
    # Prepare disease-to-gene associations
    if "disease-to-gene" in datasets:
        disease_to_gene = _prepare_disease_to_gene_associations(
            association_dataset=association_dataset,
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            verbose=verbose,
        )
        disease_to_gene = apply_filter_adaptive(disease_to_gene, "disease-to-gene")
        associations_list.append(disease_to_gene)
    
    # Prepare known drug associations
    if "known_drug" in datasets:
        known_drug = _prepare_known_drug_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            default_score=default_score,
            verbose=verbose,
        )
        known_drug = apply_filter_adaptive(known_drug, "known_drug")
        associations_list.append(known_drug)
    
    # Prepare pharmacogenomics associations
    if "pharmacogenomics" in datasets:
        gene_to_pharmacogenomics = _prepare_pharmacogenomics_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            verbose=verbose,
        )
        gene_to_pharmacogenomics = apply_filter_adaptive(gene_to_pharmacogenomics, "pharmacogenomics")
        associations_list.append(gene_to_pharmacogenomics)
    
    # Prepare mouse phenotype associations
    if "mouse_phenotype" in datasets:
        mouse_phenotype = _prepare_mouse_phenotype_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            default_score=default_score,
            verbose=verbose,
        )
        mouse_phenotype = apply_filter_adaptive(mouse_phenotype, "mouse_phenotype")
        associations_list.append(mouse_phenotype)
    
    # Prepare target essentiality associations
    if "target_essentiality" in datasets:
        target_essentiality = _prepare_target_essentiality_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            default_score=default_score,
            verbose=verbose,
        )
        target_essentiality = apply_filter_adaptive(target_essentiality, "target_essentiality")
        associations_list.append(target_essentiality)
    
    # Prepare expression associations
    if "expression" in datasets:
        expression = _prepare_expression_associations(
            cache_dir=cache_dir,
            force=(force >= 2),  # Only re-download if force >= 2
            output_format=output_format,
            default_score=default_score,
            verbose=verbose,
        )
        expression = apply_filter_adaptive(expression, "expression")
        associations_list.append(expression)
    
    if len(associations_list) == 0:
        raise ValueError("At least one dataset must be included")
    
    # Select common columns and concatenate
    select_cols = ["database", "dataset", "sourceId", "targetId", "score", "label"]
    
    # Get all available columns from all dataframes
    all_cols = set()
    for df in associations_list:
        all_cols.update(df.columns)
    
    # Select columns that exist in all dataframes
    common_cols = [col for col in select_cols if all(col in df.columns for df in associations_list)]
    
    # Also include any additional columns that are in all dataframes
    for col in all_cols:
        if col not in common_cols and all(col in df.columns for df in associations_list):
            common_cols.append(col)
    
    if verbose >= 1:
        logger.info(f"Combining {len(associations_list)} datasets...")
        logger.info(f"Using columns: {common_cols}")
    
    # Concatenate all associations
    # Filter out empty DataFrames and ensure all have the same columns to avoid FutureWarning
    dfs_to_concat = []
    for df in associations_list:
        if not df.empty:
            # Select only common columns that exist in this DataFrame
            df_subset = df[[col for col in common_cols if col in df.columns]].copy()
            # Ensure all common_cols are present (fill missing with NaN)
            for col in common_cols:
                if col not in df_subset.columns:
                    df_subset[col] = pd.NA
            # Reorder columns to match common_cols
            df_subset = df_subset[common_cols]
            dfs_to_concat.append(df_subset)
    
    if dfs_to_concat:
        opentargets_associations = pd.concat(dfs_to_concat, ignore_index=True)
    else:
        # Return empty DataFrame with correct columns if all are empty
        opentargets_associations = pd.DataFrame(columns=common_cols)
    
    if verbose >= 1:
        logger.info(f"Final DataFrame shape: {opentargets_associations.shape}")
        logger.info("Unique counts for categorical/string columns:")
        categorical_cols = ['database', 'dataset', 'sourceId', 'targetId', 'label']
        for col in categorical_cols:
            if col in opentargets_associations.columns:
                n_unique = opentargets_associations[col].nunique()
                logger.info(f"  {col:15s}: {n_unique:,} unique values")
        # Also print to stdout for interactive usage (e.g., Jupyter)
        print(f"Final DataFrame shape: {opentargets_associations.shape}")
        print("Unique counts for categorical/string columns:")
        for col in categorical_cols:
            if col in opentargets_associations.columns:
                n_unique = opentargets_associations[col].nunique()
                print(f"  {col:15s}: {n_unique:,} unique values")
    
    # Save to parquet if save_path is provided
    if save_path is not None:
        # save_path_obj was already created and checked at the beginning of the function
        if isinstance(save_path, str):
            save_path_obj = Path(save_path)
        else:
            save_path_obj = save_path
        
        if verbose >= 1:
            logger.info(f"Saving DataFrame to: {save_path_obj}")
            print(f"Saving DataFrame to: {save_path_obj}")
        
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        opentargets_associations.to_parquet(save_path_obj, index=False)
        
        if verbose >= 1:
            logger.info(f"Saved {len(opentargets_associations):,} rows to {save_path_obj}")
            print(f"Saved {len(opentargets_associations):,} rows to {save_path_obj}")
    
    return opentargets_associations


# Parse nested geneEssentiality column into flattened rows
def parse_gene_essentiality(df):
    """
    Flatten the nested geneEssentiality column into multiple rows.
    Each row will represent one screen (cell line) with all parent information.
    
    Structure:
    - geneEssentiality: list of dicts
      - isEssential: bool
      - depMapEssentiality: list of dicts (tissues)
        - tissueId: str
        - tissueName: str
        - screens: list of dicts (cell lines)
          - depmapId, cellLineName, diseaseFromSource, etc.
    """
    rows = []
    
    # Add progress bar for parsing
    iterator = tqdm(df.iterrows(), total=len(df), desc="Parsing geneEssentiality")
    for idx, row in iterator:
        gene_id = row['id']
        gene_essentiality = row['geneEssentiality']
        
        # Handle case where geneEssentiality might be None or empty
        if pd.isna(gene_essentiality):
            continue
        
        # Convert numpy array to list if needed
        if isinstance(gene_essentiality, np.ndarray):
            gene_essentiality = gene_essentiality.tolist()
        
        if not gene_essentiality:
            continue
        
        # Iterate through each essentiality entry
        for ess_entry in gene_essentiality:
            # Handle dict access - convert numpy types if needed
            if isinstance(ess_entry, np.ndarray):
                ess_entry = ess_entry.item() if ess_entry.size == 1 else ess_entry.tolist()
            
            if not isinstance(ess_entry, dict):
                continue
                
            is_essential = ess_entry.get('isEssential', None)
            depmap_essentiality = ess_entry.get('depMapEssentiality', [])
            
            # Convert numpy array to list if needed
            if isinstance(depmap_essentiality, np.ndarray):
                depmap_essentiality = depmap_essentiality.tolist()
            
            # Iterate through each tissue
            for tissue_entry in depmap_essentiality:
                # Handle dict access
                if isinstance(tissue_entry, np.ndarray):
                    tissue_entry = tissue_entry.item() if tissue_entry.size == 1 else tissue_entry.tolist()
                
                if not isinstance(tissue_entry, dict):
                    continue
                
                tissue_id = tissue_entry.get('tissueId', None)
                tissue_name = tissue_entry.get('tissueName', None)
                screens = tissue_entry.get('screens', [])
                
                # Convert numpy array to list if needed
                if isinstance(screens, np.ndarray):
                    screens = screens.tolist()
                
                # Iterate through each screen (cell line)
                for screen in screens:
                    # Handle dict access
                    if isinstance(screen, np.ndarray):
                        screen = screen.item() if screen.size == 1 else screen.tolist()
                    
                    if not isinstance(screen, dict):
                        continue
                    
                    row_data = {
                        'geneId': gene_id,
                        'isEssential': is_essential,
                        'tissueId': tissue_id,
                        'tissueName': tissue_name,
                        'depmapId': screen.get('depmapId', None),
                        'cellLineName': screen.get('cellLineName', None),
                        'diseaseFromSource': screen.get('diseaseFromSource', None),
                        'diseaseCellLineId': screen.get('diseaseCellLineId', None),
                        'mutation': screen.get('mutation', None),
                        'geneEffect': screen.get('geneEffect', None),
                        'expression': screen.get('expression', None),
                    }
                    rows.append(row_data)
    df = pd.DataFrame(rows)
    cols = ["tissueId", "depmapId","diseaseCellLineId","mutation"]
    df['sourceId'] = df[cols].astype(str).agg('.'.join, axis=1)
    return df


def parse_expression(df):
    """
    Flatten the nested expression column into multiple rows.
    Each row will represent one gene-tissue combination.
    
    Structure:
    - id: gene ID
    - tissues: list of dicts
      - efo_code: tissue identifier
      - label: tissue name
      - organs: array of organ names
      - anatomical_systems: array of anatomical system names
      - rna: dict with value, zscore, level, unit
      - protein: dict with reliability, level, cell_type
    
    Returns DataFrame with columns:
    - geneId (renamed to id): gene identifier
    - efo_code: tissue identifier
    - tissueLabel: tissue name
    - organs: list of organ names
    - anatomical_systems: list of anatomical system names
    - rna_value: RNA expression value
    - rna_zscore: RNA z-score
    - rna_level: RNA expression level
    - rna_unit: RNA unit
    - protein_reliability: protein reliability flag
    - protein_level: protein level
    - protein_cell_type: list of cell types
    """
    rows = []
    
    # Add progress bar for parsing
    iterator = tqdm(df.iterrows(), total=len(df), desc="Parsing expression")
    for idx, row in iterator:
        gene_id = row['id']
        tissues = row['tissues']
        
        # Handle case where tissues might be None or empty
        # Check for numpy array first before using pd.isna()
        if isinstance(tissues, np.ndarray):
            if tissues.size == 0:
                continue
            tissues = tissues.tolist()
        elif tissues is None:
            continue
        elif pd.isna(tissues):
            continue
        
        # Check if empty after conversion (avoid boolean check on numpy arrays)
        try:
            if len(tissues) == 0:
                continue
        except (TypeError, ValueError):
            # If it's not a sequence, skip
            continue
        
        # Iterate through each tissue
        for tissue_entry in tissues:
            # Convert numpy array to dict if needed
            if isinstance(tissue_entry, np.ndarray):
                if tissue_entry.dtype == object and tissue_entry.size > 0:
                    tissue_entry = tissue_entry.item() if tissue_entry.size == 1 else tissue_entry.tolist()[0]
                else:
                    continue
            
            if not isinstance(tissue_entry, dict):
                continue
            
            efo_code = tissue_entry.get('efo_code', None)
            label = tissue_entry.get('label', None)
            
            # Extract organs (convert numpy array to list)
            organs = tissue_entry.get('organs', [])
            if isinstance(organs, np.ndarray):
                organs = organs.tolist()
            
            # Extract anatomical_systems (convert numpy array to list)
            anatomical_systems = tissue_entry.get('anatomical_systems', [])
            if isinstance(anatomical_systems, np.ndarray):
                anatomical_systems = anatomical_systems.tolist()
            
            # Extract RNA data
            rna = tissue_entry.get('rna', {})
            rna_value = rna.get('value', None) if isinstance(rna, dict) else None
            rna_zscore = rna.get('zscore', None) if isinstance(rna, dict) else None
            rna_level = rna.get('level', None) if isinstance(rna, dict) else None
            rna_unit = rna.get('unit', None) if isinstance(rna, dict) else None
            
            # Extract protein data
            protein = tissue_entry.get('protein', {})
            protein_reliability = protein.get('reliability', None) if isinstance(protein, dict) else None
            protein_level = protein.get('level', None) if isinstance(protein, dict) else None
            protein_cell_type = protein.get('cell_type', [])
            if isinstance(protein_cell_type, np.ndarray):
                protein_cell_type = protein_cell_type.tolist()
            
            row_data = {
                'geneId': gene_id,
                'efo_code': efo_code,
                'tissueLabel': label,
                'organs': organs,
                'anatomical_systems': anatomical_systems,
                'rna_value': rna_value,
                'rna_zscore': rna_zscore,
                'rna_level': rna_level,
                'rna_unit': rna_unit,
                'protein_reliability': protein_reliability,
                'protein_level': protein_level,
                'protein_cell_type': protein_cell_type,
            }
            rows.append(row_data)
    
    df = pd.DataFrame(rows)
    # Rename geneId to id to match original structure
    if 'geneId' in df.columns:
        df = df.rename(columns={'geneId': 'id'})
    
    return df

# import gget
# logger = logging.getLogger(__name__)
# class JSONEncoder(json.JSONEncoder):
#     """Custom JSON encoder that handles numpy arrays and pandas types."""
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, (np.integer, np.floating)):
#             return obj.item()
#         elif isinstance(obj, np.bool_):
#             return bool(obj)
#         elif pd.isna(obj):
#             return None
#         elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
#             return str(obj)
#         elif isinstance(obj, (pd.Series, pd.Index)):
#             return obj.tolist()
#         return super().default(obj)

# # Default cache directory
# CACHE_DIR = Path.home() / ".cache" / "aou" / "opentargets"
# CACHE_DIR.mkdir(parents=True, exist_ok=True)

# # Available resource types in gene info results
# GENE_INFO_RESOURCE_TYPES = {
#     "associated_diseases": "Associated diseases/phenotypes",
#     "associated_drugs": "Associated drugs",
#     "tractability": "Druggability assessment data",
#     "pharmacogenetics": "Pharmacogenetic response data",
#     "expression": "Gene expression by tissues, organs, and anatomical systems",
#     "depmap": "DepMap gene→disease-effect data",
#     "interactions": "Protein⇄protein interactions",
# }

# # List of resource type keys (for convenience)
# GENE_INFO_RESOURCE_KEYS = list(GENE_INFO_RESOURCE_TYPES.keys())


# def get_all_genes(
#     output_format: str = "pandas",
#     cache_file: Optional[str] = None,
#     force: int = 0,
#     biotype_filter: Optional[str] = None,
#     species: str = "homo_sapiens",
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get all gene names available from Ensembl (used by OpenTargets Platform).

#     This function uses gget.ref() to fetch the Ensembl GTF file and extracts
#     all gene information. This provides a comprehensive list of all genes.

#     Parameters
#     ----------
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"
#     cache_file : str, optional
#         Path to cache file to save/load results. If None, uses default cache
#         location: ~/.cache/aou/opentargets/all_genes_{species}_{biotype}.parquet
#         If file exists and force=0, will load from cache instead of querying API.
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached parquet if exists, otherwise fetch and process
#         - 1: Use cached raw GTF data if exists, regenerate parquet from it
#         - 2: Force fresh download of raw GTF data, then process
#     biotype_filter : str, optional, default "protein_coding"
#         Filter genes by biotype. Set to None to get all genes.
#         Common biotypes: "protein_coding", "lncRNA", "pseudogene", etc.
#     species : str, default "homo_sapiens"
#         Species name in Ensembl format (e.g., "homo_sapiens", "mus_musculus")

#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with columns:
#         - ensembl_id: Ensembl gene ID (e.g., "ENSG00000157764")
#         - approved_symbol: Official gene symbol (e.g., "BRAF")
#         - approved_name: Full gene name
#         - biotype: Gene biotype (e.g., "protein_coding")
#     """
#     # Set default cache files if not provided
#     if cache_file is None:
#         biotype_str = biotype_filter if biotype_filter else "all"
#         cache_file = str(DEFAULT_CACHE_DIR / f"all_genes_{species}_{biotype_str}.parquet")
    
#     raw_cache_file = str(DEFAULT_CACHE_DIR / f"raw_gtf_{species}.gz")
    
#     # Check parquet cache (force=0)
#     if force == 0 and cache_file:
#         try:
#             if output_format == "pandas":
#                 df = pd.read_parquet(cache_file)
#                 logger.info(f"Loaded {len(df)} genes from cache: {cache_file}")
#                 return df
#             else:
#                 df = pl.read_parquet(cache_file)
#                 logger.info(f"Loaded {len(df)} genes from cache: {cache_file}")
#                 return df
#         except FileNotFoundError:
#             pass
    
#     # Check raw cache (force=1) or download fresh (force=2)
#     download_raw = force == 2 or not os.path.exists(raw_cache_file)
    
#     if download_raw:
#         logger.info(f"Fetching GTF file from Ensembl for {species}...")
#         try:
#             # Get reference data from Ensembl using gget
#             ref_data = gget.ref(species)
            
#             # gget.ref() returns a dict with file URLs
#             # The structure may vary, so we need to find the GTF URL
#             gtf_url = None
#             if isinstance(ref_data, dict):
#                 # Try common keys for GTF file
#                 for key in ["gtf", "annotation_gtf", "gtf_ftp"]:
#                     if key in ref_data:
#                         if isinstance(ref_data[key], dict):
#                             gtf_url = ref_data[key].get("ftp") or ref_data[key].get("url")
#                         elif isinstance(ref_data[key], str):
#                             gtf_url = ref_data[key]
#                         if gtf_url:
#                             break
                
#                 # If not found, try nested structure
#                 if not gtf_url and species in ref_data:
#                     species_data = ref_data[species]
#                     if isinstance(species_data, dict):
#                         for key in ["annotation_gtf", "gtf"]:
#                             if key in species_data:
#                                 if isinstance(species_data[key], dict):
#                                     gtf_url = species_data[key].get("ftp") or species_data[key].get("url")
#                                 elif isinstance(species_data[key], str):
#                                     gtf_url = species_data[key]
#                                 if gtf_url:
#                                     break
            
#             if not gtf_url:
#                 raise ValueError(f"Could not retrieve GTF file URL from gget.ref() for {species}. Returned: {ref_data}")
            
#             logger.info(f"Downloading GTF file from: {gtf_url}")
            
#             # Download and save raw GTF file
#             response = requests.get(gtf_url, stream=True)
#             response.raise_for_status()
            
#             # Save raw data to cache
#             with open(raw_cache_file, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
            
#             logger.info(f"Cached raw GTF file to: {raw_cache_file}")
            
#         except Exception as e:
#             logger.error(f"Error fetching GTF file from Ensembl: {e}")
#             raise
#     else:
#         logger.info(f"Using cached raw GTF file: {raw_cache_file}")
    
#     # Parse GTF file from cache
#     logger.info("Reading and parsing GTF file...")
#     try:
#         # Read cached raw GTF file
#         if raw_cache_file.endswith(".gz"):
#             gtf_file = gzip.open(raw_cache_file, "rt")
#         else:
#             gtf_file = open(raw_cache_file, "rt")
        
#         # Read GTF file with pandas (skip comment lines)
#         gtf_df = pd.read_csv(
#             gtf_file,
#             sep="\t",
#             comment="#",
#             header=None,
#             names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
#             low_memory=False,
#         )
        
#         gtf_file.close()
        
#     except Exception as e:
#         logger.error(f"Error reading cached GTF file: {e}")
#         raise
    
#     # Filter for gene entries only
#     logger.info("Extracting gene information...")
#     genes_df = gtf_df[gtf_df["feature"] == "gene"].copy()
    
#     # Extract gene_id, gene_name, gene_biotype, and description from attributes column
#     # Attributes format: gene_id "ENSG00000157764"; gene_name "BRAF"; gene_biotype "protein_coding"; description "...";
#     def extract_attr(attr_str, attr_name):
#         """Extract attribute value from GTF attribute string."""
#         pattern = f'{attr_name} "([^"]+)"'
#         match = re.search(pattern, attr_str)
#         return match.group(1) if match else ""
    
#     genes_df["ensembl_id"] = genes_df["attribute"].apply(lambda x: extract_attr(x, "gene_id"))
#     genes_df["approved_symbol"] = genes_df["attribute"].apply(lambda x: extract_attr(x, "gene_name"))
#     genes_df["biotype"] = genes_df["attribute"].apply(lambda x: extract_attr(x, "gene_biotype"))
#     genes_df["approved_name"] = genes_df["attribute"].apply(lambda x: extract_attr(x, "description"))
    
#     # Apply biotype filter if specified
#     if biotype_filter:
#         genes_df = genes_df[genes_df["biotype"] == biotype_filter].copy()
    
#     # Select and rename columns
#     gene_details = genes_df[["ensembl_id", "approved_symbol", "approved_name", "biotype"]].copy()
    
#     # Remove duplicates (keep first occurrence)
#     gene_details = gene_details.drop_duplicates(subset=["ensembl_id"], keep="first")
    
#     logger.info(f"Found {len(gene_details)} genes")
    
#     # Create DataFrame
#     if output_format == "pandas":
#         df = pd.DataFrame(gene_details)
#     else:
#         df = pl.DataFrame(gene_details)
    
#     # Sort by gene symbol for easier browsing
#     if output_format == "pandas":
#         df = df.sort_values("approved_symbol").reset_index(drop=True)
#     else:
#         df = df.sort("approved_symbol")
    
#     # Save to cache as parquet
#     if cache_file:
#         if output_format == "pandas":
#             df.to_parquet(cache_file, index=False)
#         else:
#             df.write_parquet(cache_file)
#         logger.info(f"Cached {len(df)} genes to: {cache_file}")
    
#     return df


# def _convert_results_to_dataframes(
#     results: List[Dict[str, Any]],
#     include_phenotypes: bool,
#     include_drugs: bool,
#     include_tractability: bool,
#     include_pharmacogenetics: bool,
#     include_expression: bool,
#     include_depmap: bool,
#     include_interactions: bool,
# ) -> Dict[str, pd.DataFrame]:
#     """
#     Convert a list of gene info dictionaries into a dictionary of DataFrames.
    
#     Each resource type becomes its own DataFrame with a gene_id column.
    
#     Parameters
#     ----------
#     results : list of dict
#         List of gene info dictionaries (one per gene)
#     include_phenotypes, include_drugs, etc. : bool
#         Flags indicating which resources were included
    
#     Returns
#     -------
#     dict of pandas.DataFrame
#         Dictionary with keys:
#         - "gene_info": Basic gene information (one row per gene)
#         - "associated_diseases": All diseases (one row per disease-gene pair)
#         - "associated_drugs": All drugs (one row per drug-gene pair)
#         - "tractability": Tractability data (one row per gene)
#         - "pharmacogenetics": Pharmacogenetics data (one row per entry)
#         - "expression": Expression data (one row per entry)
#         - "depmap": DepMap data (one row per entry)
#         - "interactions": Interactions data (one row per interaction)
#     """
#     output = {}
    
#     # Extract basic gene info
#     gene_info_rows = []
#     for result in results:
#         gene_info_rows.append({
#             "gene_id": result.get("ensembl_id", ""),
#             "ensembl_id": result.get("ensembl_id", ""),
#             "approved_symbol": result.get("approved_symbol", ""),
#             "approved_name": result.get("approved_name", ""),
#             "biotype": result.get("biotype", ""),
#         })
#     output["gene_info"] = pd.DataFrame(gene_info_rows)
    
#     # Extract associated_diseases
#     if include_phenotypes:
#         diseases_rows = []
#         for result in results:
#             gene_id = result.get("ensembl_id", "")
#             diseases = result.get("associated_diseases", [])
#             if isinstance(diseases, list) and len(diseases) > 0:
#                 for disease in diseases:
#                     if isinstance(disease, dict):
#                         disease_row = disease.copy()
#                         disease_row["gene_id"] = gene_id
#                         diseases_rows.append(disease_row)
#         if diseases_rows:
#             output["associated_diseases"] = pd.DataFrame(diseases_rows)
#         else:
#             output["associated_diseases"] = pd.DataFrame(columns=["gene_id"])
    
#     # Extract associated_drugs
#     if include_drugs:
#         drugs_rows = []
#         for result in results:
#             gene_id = result.get("ensembl_id", "")
#             drugs = result.get("associated_drugs", [])
#             if isinstance(drugs, list) and len(drugs) > 0:
#                 for drug in drugs:
#                     if isinstance(drug, dict):
#                         drug_row = drug.copy()
#                         drug_row["gene_id"] = gene_id
#                         drugs_rows.append(drug_row)
#         if drugs_rows:
#             output["associated_drugs"] = pd.DataFrame(drugs_rows)
#         else:
#             output["associated_drugs"] = pd.DataFrame(columns=["gene_id"])
    
#     # Extract tractability
#     if include_tractability:
#         tractability_rows = []
#         for result in results:
#             gene_id = result.get("ensembl_id", "")
#             tractability = result.get("tractability", [])
#             if isinstance(tractability, list) and len(tractability) > 0:
#                 for entry in tractability:
#                     if isinstance(entry, dict):
#                         entry_row = entry.copy()
#                         entry_row["gene_id"] = gene_id
#                         tractability_rows.append(entry_row)
#         if tractability_rows:
#             output["tractability"] = pd.DataFrame(tractability_rows)
#         else:
#             output["tractability"] = pd.DataFrame(columns=["gene_id"])
    
#     # Extract pharmacogenetics
#     if include_pharmacogenetics:
#         pharmacogenetics_rows = []
#         for result in results:
#             gene_id = result.get("ensembl_id", "")
#             pharmacogenetics = result.get("pharmacogenetics", [])
#             if isinstance(pharmacogenetics, list) and len(pharmacogenetics) > 0:
#                 for entry in pharmacogenetics:
#                     if isinstance(entry, dict):
#                         entry_row = entry.copy()
#                         entry_row["gene_id"] = gene_id
#                         pharmacogenetics_rows.append(entry_row)
#         if pharmacogenetics_rows:
#             output["pharmacogenetics"] = pd.DataFrame(pharmacogenetics_rows)
#         else:
#             output["pharmacogenetics"] = pd.DataFrame(columns=["gene_id"])
    
#     # Extract expression
#     if include_expression:
#         expression_rows = []
#         for result in results:
#             gene_id = result.get("ensembl_id", "")
#             expression = result.get("expression", [])
#             if isinstance(expression, list) and len(expression) > 0:
#                 for entry in expression:
#                     if isinstance(entry, dict):
#                         entry_row = entry.copy()
#                         entry_row["gene_id"] = gene_id
#                         expression_rows.append(entry_row)
#         if expression_rows:
#             output["expression"] = pd.DataFrame(expression_rows)
#         else:
#             output["expression"] = pd.DataFrame(columns=["gene_id"])
    
#     # Extract depmap
#     if include_depmap:
#         depmap_rows = []
#         for result in results:
#             gene_id = result.get("ensembl_id", "")
#             depmap = result.get("depmap", [])
#             if isinstance(depmap, list) and len(depmap) > 0:
#                 for entry in depmap:
#                     if isinstance(entry, dict):
#                         entry_row = entry.copy()
#                         entry_row["gene_id"] = gene_id
#                         depmap_rows.append(entry_row)
#         if depmap_rows:
#             output["depmap"] = pd.DataFrame(depmap_rows)
#         else:
#             output["depmap"] = pd.DataFrame(columns=["gene_id"])
    
#     # Extract interactions
#     if include_interactions:
#         interactions_rows = []
#         for result in results:
#             gene_id = result.get("ensembl_id", "")
#             interactions = result.get("interactions", [])
#             if isinstance(interactions, list) and len(interactions) > 0:
#                 for entry in interactions:
#                     if isinstance(entry, dict):
#                         entry_row = entry.copy()
#                         entry_row["gene_id"] = gene_id
#                         interactions_rows.append(entry_row)
#         if interactions_rows:
#             output["interactions"] = pd.DataFrame(interactions_rows)
#         else:
#             output["interactions"] = pd.DataFrame(columns=["gene_id"])
    
#     return output


# def get_gene_info(
#     gene_id: Union[str, List[str]],
#     include_phenotypes: bool = True,
#     include_drugs: bool = True,
#     include_tractability: bool = True,
#     include_pharmacogenetics: bool = True,
#     include_expression: bool = True,
#     include_depmap: bool = True,
#     include_interactions: bool = True,
#     phenotype_size: int = None,
#     phenotype_from: int = 0,
#     force: int = 0,
#     verbose: int = 0,
#     cache_as: str = "json",
#     return_as: str = "pandas",
# ) -> Union[Dict[str, Any], List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
#     """
#     Get detailed information for a specific gene or multiple genes, especially associated phenotypes.

#     This function queries the OpenTargets Platform via gget to retrieve comprehensive
#     gene information including basic gene metadata and associated diseases/phenotypes.

#     Parameters
#     ----------
#     gene_id : str or list of str
#         Gene identifier(s). Can be either:
#         - Ensembl ID (e.g., "ENSG00000157764" for BRAF)
#         - Gene symbol (e.g., "BRAF", "TP53", "EGFR")
#         - List of gene identifiers (mix of Ensembl IDs and symbols is allowed)
#           Gene symbols will be automatically resolved to Ensembl IDs using gget.search()
    
#     include_phenotypes : bool, default True
#         Whether to include associated diseases/phenotypes from OpenTargets Platform.
    
#     include_drugs : bool, default True
#         Whether to include associated drugs from OpenTargets Platform.
    
#     include_tractability : bool, default True
#         Whether to include tractability data (druggability assessment).
    
#     include_pharmacogenetics : bool, default True
#         Whether to include pharmacogenetic response data.
    
#     include_expression : bool, default True
#         Whether to include gene expression data by tissues, organs, and anatomical systems.
    
#     include_depmap : bool, default True
#         Whether to include DepMap gene→disease-effect data.
    
#     include_interactions : bool, default True
#         Whether to include protein⇄protein interactions.
    
#     phenotype_size : int, default 1000
#         Maximum number of phenotypes/diseases to retrieve. The OpenTargets Platform
#         returns results sorted by association score, so this limits the top N results.
#         Set to None to retrieve all available associations (may be slow for genes
#         with many associations).
    
#     phenotype_from : int, default 0
#         Offset for phenotype pagination. Use this to skip the first N results.
#         For example, phenotype_from=10 will skip the top 10 associations and return
#         results starting from the 11th.
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached file if exists, otherwise fetch and cache
#         - 1: Force new query, bypass cache
    
#     verbose : int, default 0
#         Verbosity level for logging:
#         - 0: Minimal logging (default for multiple genes: progress bar only)
#         - 1: Standard logging (default for single gene)
#         - 2: Full verbose logging (prints details for each query/resource)
    
#     cache_as : str, default "json"
#         Cache format:
#         - "json": Store as JSON.gz (default, preserves nested structures)
#         - "parquet": Store as parquet (more efficient for large datasets, but nested DataFrames are converted to JSON strings)
    
#     return_as : str, default "pandas"
#         Return format:
#         - "pandas": Return as dictionary of pandas DataFrames (default)
#           Each resource type becomes its own DataFrame:
#           - "gene_info": Basic gene information (ensembl_id, approved_symbol, approved_name, biotype)
#           - "associated_diseases": All diseases (one row per disease, includes gene_id column)
#           - "associated_drugs": All drugs (one row per drug, includes gene_id column)
#           - "tractability": Tractability data (includes gene_id column)
#           - "pharmacogenetics": Pharmacogenetics data (includes gene_id column)
#           - "expression": Expression data (includes gene_id column)
#           - "depmap": DepMap data (includes gene_id column)
#           - "interactions": Interactions data (includes gene_id column)
#         - "json": Return as original dict/list format

#     Returns
#     -------
#     dict, list of dict, or dict of pandas.DataFrame
#         If return_as="json":
#             - If gene_id is a string, returns a dictionary containing gene information.
#             - If gene_id is a list, returns a list of dictionaries (one per gene).
#         If return_as="pandas":
#             - Returns a dictionary of pandas DataFrames, where each key is a resource type
#               and the value is a DataFrame containing that resource's data.
#               All resource DataFrames include a "gene_id" column linking back to the gene.
        
#         Each dictionary contains the following keys:
#         - ensembl_id (str): Ensembl gene ID (e.g., "ENSG00000157764")
#         - approved_symbol (str): Official gene symbol from HGNC (e.g., "BRAF")
#         - approved_name (str): Full gene name/description
#         - biotype (str): Gene biotype (e.g., "protein_coding", "lncRNA")
#         - associated_diseases (list, optional): List of associated diseases/phenotypes
#           (only present if include_phenotypes=True). Each entry is a dict with all fields
#           from the OpenTargets Platform API.
#         - associated_drugs (list, optional): List of associated drugs
#           (only present if include_drugs=True). Each entry is a dict with all fields
#           from the OpenTargets Platform API.
#         - tractability (list, optional): Tractability data indicating how "druggable" the gene is
#           (only present if include_tractability=True). Each entry is a dict with all fields
#           from the OpenTargets Platform API.
#         - pharmacogenetics (list, optional): Pharmacogenetic response data
#           (only present if include_pharmacogenetics=True). Each entry is a dict with all fields
#           from the OpenTargets Platform API.
#         - expression (list, optional): Gene expression data by tissues, organs, and anatomical systems
#           (only present if include_expression=True). Each entry is a dict with all fields
#           from the OpenTargets Platform API.
#         - depmap (list, optional): DepMap gene→disease-effect data
#           (only present if include_depmap=True). Each entry is a dict with all fields
#           from the OpenTargets Platform API.
#         - interactions (list, optional): Protein⇄protein interactions
#           (only present if include_interactions=True). Each entry is a dict with all fields
#           from the OpenTargets Platform API.

#     Examples
#     --------
#     >>> import phenoref.opentargets as opentargets
#     >>> 
#     >>> # Get gene info with phenotypes using Ensembl ID (single gene)
#     >>> gene_info = opentargets.get_gene_info("ENSG00000157764", include_phenotypes=True)
#     >>> print(gene_info["approved_symbol"])  # "BRAF"
#     >>> print(len(gene_info["associated_diseases"]))  # Number of associated diseases
#     >>> 
#     >>> # Get gene info using gene symbol (auto-resolved, single gene)
#     >>> gene_info = opentargets.get_gene_info("BRAF", include_phenotypes=True)
#     >>> 
#     >>> # Get info for multiple genes (returns list)
#     >>> genes_info = opentargets.get_gene_info(["BRAF", "TP53", "EGFR"], include_phenotypes=True)
#     >>> print(len(genes_info))  # 3
#     >>> print(genes_info[0]["approved_symbol"])  # "BRAF"
#     >>> 
#     >>> # Get top 10 diseases only
#     >>> gene_info = opentargets.get_gene_info("TP53", phenotype_size=10)
#     >>> 
#     >>> # Get diseases starting from rank 11 (skip top 10)
#     >>> gene_info = opentargets.get_gene_info("EGFR", phenotype_from=10, phenotype_size=20)
#     >>> 
#     >>> # Get basic gene info without phenotypes (faster)
#     >>> gene_info = opentargets.get_gene_info("BRAF", include_phenotypes=False)
#     >>> 
#     >>> # Get gene info as dictionary of pandas DataFrames (default)
#     >>> gene_data = opentargets.get_gene_info("BRAF", return_as="pandas")
#     >>> print(gene_data["gene_info"]["approved_symbol"].iloc[0])  # "BRAF"
#     >>> print(len(gene_data["associated_diseases"]))  # Number of diseases
#     >>> 
#     >>> # Get gene info as original JSON format
#     >>> gene_dict = opentargets.get_gene_info("BRAF", return_as="json")
#     >>> print(gene_dict["approved_symbol"])  # "BRAF"

#     Notes
#     -----
#     - This function uses the gget library to query OpenTargets Platform
#     - Gene symbols are resolved to Ensembl IDs automatically
#     - Each resource type can be individually controlled via its include_* parameter
#     - Available resources:
#       * diseases (include_phenotypes): Associated diseases/phenotypes (sorted by association score)
#       * drugs (include_drugs): Associated drugs
#       * tractability (include_tractability): Druggability assessment data
#       * pharmacogenetics (include_pharmacogenetics): Pharmacogenetic response data
#       * expression (include_expression): Gene expression by tissues, organs, and anatomical systems
#       * depmap (include_depmap): DepMap gene→disease-effect data
#       * interactions (include_interactions): Protein⇄protein interactions
#     - All resource data is cached as parquet files for performance
#     - Complete gene info results are cached as JSON.gz (default) or parquet files per gene ID
#     - When processing multiple genes, errors for individual genes are logged but don't stop processing

#     References
#     ----------
#     - OpenTargets Platform: https://platform.opentargets.org/
#     - gget documentation: https://pachterlab.github.io/gget/en/opentargets.html
#     """
#     # Handle list of gene IDs
#     if isinstance(gene_id, list):
#         results = []
#         # Use progress bar for multiple genes (unless verbose=2)
#         iterator = tqdm(gene_id, desc="Fetching gene info") if verbose < 2 else gene_id
#         for gid in iterator:
#             try:
#                 result = _get_gene_info_single(
#                     gid,
#                     include_phenotypes=include_phenotypes,
#                     include_drugs=include_drugs,
#                     include_tractability=include_tractability,
#                     include_pharmacogenetics=include_pharmacogenetics,
#                     include_expression=include_expression,
#                     include_depmap=include_depmap,
#                     include_interactions=include_interactions,
#                     phenotype_size=phenotype_size,
#                     phenotype_from=phenotype_from,
#                     force=force,
#                     verbose=verbose,
#                     cache_as=cache_as,
#                 )
#                 results.append(result)
#             except Exception as e:
#                 if verbose >= 1:
#                     logger.warning(f"Error fetching info for {gid}: {e}")
#                 # Add error entry
#                 error_entry = {
#                     "ensembl_id": gid,
#                     "approved_symbol": "",
#                     "approved_name": "",
#                     "biotype": "",
#                 }
#                 if include_phenotypes:
#                     error_entry["associated_diseases"] = []
#                 if include_drugs:
#                     error_entry["associated_drugs"] = []
#                 if include_tractability:
#                     error_entry["tractability"] = []
#                 if include_pharmacogenetics:
#                     error_entry["pharmacogenetics"] = []
#                 if include_expression:
#                     error_entry["expression"] = []
#                 if include_depmap:
#                     error_entry["depmap"] = []
#                 if include_interactions:
#                     error_entry["interactions"] = []
#                 results.append(error_entry)
        
#         # Convert to pandas if requested
#         if return_as == "pandas":
#             return _convert_results_to_dataframes(results, include_phenotypes, include_drugs, 
#                                                    include_tractability, include_pharmacogenetics,
#                                                    include_expression, include_depmap, include_interactions)
#         else:
#             return results
    
#     # Single gene ID - delegate to helper function
#     # Default verbose=1 for single gene (show standard logging)
#     if verbose == 0:
#         verbose = 1
#     result = _get_gene_info_single(
#         gene_id,
#         include_phenotypes=include_phenotypes,
#         include_drugs=include_drugs,
#         include_tractability=include_tractability,
#         include_pharmacogenetics=include_pharmacogenetics,
#         include_expression=include_expression,
#         include_depmap=include_depmap,
#         include_interactions=include_interactions,
#         phenotype_size=phenotype_size,
#         phenotype_from=phenotype_from,
#         force=force,
#         verbose=verbose,
#         cache_as=cache_as,
#     )
    
#     # Convert to pandas if requested
#     if return_as == "pandas":
#         return _convert_results_to_dataframes([result], include_phenotypes, include_drugs,
#                                                include_tractability, include_pharmacogenetics,
#                                                include_expression, include_depmap, include_interactions)
#     else:
#         return result


# def _get_gene_info_single(
#     gene_id: str,
#     include_phenotypes: bool = True,
#     include_drugs: bool = True,
#     include_tractability: bool = True,
#     include_pharmacogenetics: bool = True,
#     include_expression: bool = True,
#     include_depmap: bool = True,
#     include_interactions: bool = True,
#     phenotype_size: int = None,
#     phenotype_from: int = 0,
#     force: int = 0,
#     verbose: int = 1,
#     cache_as: str = "json",
# ) -> Dict[str, Any]:
#     """
#     Internal helper function to get info for a single gene.
    
#     This contains the core logic that was previously in get_gene_info.
#     Results are cached as JSON.gz (default) or parquet files per gene ID.
#     """
#     # Suppress gget INFO messages when verbose < 2
#     gget_logger = logging.getLogger("gget")
#     original_level = gget_logger.level
#     if verbose < 2:
#         gget_logger.setLevel(logging.WARNING)
    
#     # Resolve gene symbol to Ensembl ID if needed
#     resolved_symbol = None
#     if not gene_id.startswith("ENSG"):
#         try:
#             # Use gget.search to resolve gene symbol to Ensembl ID
#             search_results = gget.search(gene_id, species="human")
#             if search_results is not None and len(search_results) > 0:
#                 # Find the first result that looks like our gene
#                 for result_item in search_results:
#                     if isinstance(result_item, dict):
#                         ens_id = result_item.get("ensembl_id", "")
#                         if ens_id.startswith("ENSG"):
#                             gene_id = ens_id
#                             # Also try to get the symbol from search results
#                             resolved_symbol = (
#                                 result_item.get("gene_name") or
#                                 result_item.get("name") or
#                                 result_item.get("external_name") or
#                                 gene_id  # Use original input as fallback
#                             )
#                             break
#                     elif isinstance(result_item, str) and result_item.startswith("ENSG"):
#                         gene_id = result_item
#                         resolved_symbol = gene_id  # Use original input as fallback
#                         break
#                 else:
#                     raise ValueError(f"Could not resolve gene symbol to Ensembl ID: {gene_id}")
#             else:
#                 raise ValueError(f"Gene symbol not found: {gene_id}")
#         except Exception as e:
#             logger.error(f"Error resolving gene ID {gene_id}: {e}")
#             raise ValueError(f"Could not resolve gene identifier: {gene_id}")

#     ensembl_id = gene_id

#     # Build cache file path - separate file for each gene ID
#     # Include resource flags in cache filename to ensure different combinations get different cache files
#     cache_parts = ["gene_info", ensembl_id]
#     resource_flags = []
#     if include_phenotypes:
#         resource_flags.append("phenotypes")
#     if include_drugs:
#         resource_flags.append("drugs")
#     if include_tractability:
#         resource_flags.append("tractability")
#     if include_pharmacogenetics:
#         resource_flags.append("pharmacogenetics")
#     if include_expression:
#         resource_flags.append("expression")
#     if include_depmap:
#         resource_flags.append("depmap")
#     if include_interactions:
#         resource_flags.append("interactions")
    
#     if resource_flags:
#         cache_parts.append("_".join(resource_flags))
#     else:
#         cache_parts.append("basic_only")
    
#     if include_phenotypes and phenotype_size is not None:
#         cache_parts.append(f"size{phenotype_size}")
#     if include_phenotypes and phenotype_from > 0:
#         cache_parts.append(f"from{phenotype_from}")
    
#     # Determine cache file extension based on cache_as
#     if cache_as == "parquet":
#         cache_file = str(DEFAULT_CACHE_DIR / f"{'_'.join(cache_parts)}.parquet")
#     else:
#         cache_file = str(DEFAULT_CACHE_DIR / f"{'_'.join(cache_parts)}.json.gz")
    
#     # Check cache (force=0)
#     if force == 0:
#         try:
#             if os.path.exists(cache_file):
#                 if cache_as == "parquet":
#                     # Load from parquet
#                     df = pd.read_parquet(cache_file)
#                     # Convert DataFrame back to dict (assuming single row)
#                     if len(df) > 0:
#                         result = df.iloc[0].to_dict()
#                         # Parse JSON strings back to objects if needed
#                         # Only parse strings that look like JSON (start with [ or { and are longer than 1 char)
#                         for key, val in result.items():
#                             if isinstance(val, str) and len(val) > 1 and (val.strip().startswith('[') or val.strip().startswith('{')):
#                                 try:
#                                     parsed = json.loads(val)
#                                     result[key] = parsed
#                                 except (json.JSONDecodeError, ValueError):
#                                     pass  # Keep as string if not valid JSON
#                     else:
#                         result = {}
#                     if verbose >= 2:
#                         logger.info(f"Loaded gene info from cache: {cache_file}")
#                     return result
#                 else:
#                     # Load from JSON.gz
#                     with gzip.open(cache_file, "rt", encoding="utf-8") as f:
#                         result = json.load(f)
#                     if verbose >= 2:
#                         logger.info(f"Loaded gene info from cache: {cache_file}")
#                     return result
#         except (json.JSONDecodeError, ValueError, OSError) as e:
#             logger.warning(f"Error loading cache from {cache_file}: {e}. Cache file may be corrupted. Deleting and re-fetching.")
#             # Delete corrupted cache file
#             try:
#                 os.remove(cache_file)
#             except Exception:
#                 pass
#             # Continue to fetch fresh data
#         except Exception as e:
#             logger.warning(f"Unexpected error loading cache from {cache_file}: {e}. Re-fetching.")
#             # Continue to fetch fresh data

#     # Get gene information using gget
#     result = {
#         "ensembl_id": ensembl_id,
#         "approved_symbol": "",
#         "approved_name": "",
#         "biotype": "",
#     }
    
#     # If we resolved from a symbol, use that as the approved_symbol (fallback if gget.info doesn't provide it)
#     if resolved_symbol and resolved_symbol != ensembl_id:
#         result["approved_symbol"] = resolved_symbol

#     # Get all resources if requested
#     if include_phenotypes:
#         # Fetch all resources using our helper functions
#         # Diseases
#         try:
#             diseases_df = _get_gene_diseases_single(ensembl_id, limit=phenotype_size, output_format="pandas", force=0)
#             if diseases_df is not None and len(diseases_df) > 0:
#                 # Apply pagination offset
#                 if phenotype_from > 0:
#                     diseases_df = diseases_df.iloc[phenotype_from:]
#                 result["associated_diseases"] = diseases_df.to_dict("records")
#                 if verbose >= 2:
#                     logger.info(f"Fetched {len(result['associated_diseases'])} diseases for {ensembl_id}")
#             else:
#                 result["associated_diseases"] = []
#         except Exception as e:
#             logger.warning(f"Error fetching diseases for {ensembl_id}: {e}")
#             import traceback
#             logger.debug(traceback.format_exc())
#             result["associated_diseases"] = []
        
#         # Drugs
#         if include_drugs:
#             try:
#                 drugs_df = _get_gene_drugs_single(ensembl_id, output_format="pandas", force=0)
#                 if drugs_df is not None and len(drugs_df) > 0:
#                     result["associated_drugs"] = drugs_df.to_dict("records")
#                     if verbose >= 2:
#                         logger.info(f"Fetched {len(result['associated_drugs'])} drugs for {ensembl_id}")
#                 else:
#                     result["associated_drugs"] = []
#             except Exception as e:
#                 logger.warning(f"Error fetching drugs for {ensembl_id}: {e}")
#                 import traceback
#                 logger.debug(traceback.format_exc())
#                 result["associated_drugs"] = []
        
#         # Tractability
#         if include_tractability:
#             try:
#                 tractability_df = _get_gene_resource_single(ensembl_id, resource="tractability", output_format="pandas", force=0)
#                 if tractability_df is not None and len(tractability_df) > 0:
#                     result["tractability"] = tractability_df.to_dict("records")
#                     if verbose >= 2:
#                         logger.info(f"Fetched tractability data for {ensembl_id}")
#                 else:
#                     result["tractability"] = []
#             except Exception as e:
#                 logger.warning(f"Error fetching tractability for {ensembl_id}: {e}")
#                 import traceback
#                 logger.debug(traceback.format_exc())
#                 result["tractability"] = []
        
#         # Pharmacogenetics
#         if include_pharmacogenetics:
#             try:
#                 pharmacogenetics_df = _get_gene_resource_single(ensembl_id, resource="pharmacogenetics", output_format="pandas", force=0)
#                 if pharmacogenetics_df is not None and len(pharmacogenetics_df) > 0:
#                     result["pharmacogenetics"] = pharmacogenetics_df.to_dict("records")
#                     if verbose >= 2:
#                         logger.info(f"Fetched {len(result['pharmacogenetics'])} pharmacogenetics entries for {ensembl_id}")
#                 else:
#                     result["pharmacogenetics"] = []
#             except Exception as e:
#                 logger.warning(f"Error fetching pharmacogenetics for {ensembl_id}: {e}")
#                 import traceback
#                 logger.debug(traceback.format_exc())
#                 result["pharmacogenetics"] = []
        
#         # Expression
#         if include_expression:
#             try:
#                 expression_df = _get_gene_resource_single(ensembl_id, resource="expression", output_format="pandas", force=0)
#                 if expression_df is not None and len(expression_df) > 0:
#                     result["expression"] = expression_df.to_dict("records")
#                     if verbose >= 2:
#                         logger.info(f"Fetched {len(result['expression'])} expression entries for {ensembl_id}")
#                 else:
#                     result["expression"] = []
#             except Exception as e:
#                 logger.warning(f"Error fetching expression for {ensembl_id}: {e}")
#                 import traceback
#                 logger.debug(traceback.format_exc())
#                 result["expression"] = []
        
#         # DepMap
#         if include_depmap:
#             try:
#                 depmap_df = _get_gene_resource_single(ensembl_id, resource="depmap", output_format="pandas", force=0)
#                 if depmap_df is not None and len(depmap_df) > 0:
#                     result["depmap"] = depmap_df.to_dict("records")
#                     if verbose >= 2:
#                         logger.info(f"Fetched {len(result['depmap'])} depmap entries for {ensembl_id}")
#                 else:
#                     result["depmap"] = []
#             except Exception as e:
#                 logger.warning(f"Error fetching depmap for {ensembl_id}: {e}")
#                 import traceback
#                 logger.debug(traceback.format_exc())
#                 result["depmap"] = []
        
#         # Interactions
#         if include_interactions:
#             try:
#                 interactions_df = _get_gene_resource_single(ensembl_id, resource="interactions", output_format="pandas", force=0)
#                 if interactions_df is not None and len(interactions_df) > 0:
#                     result["interactions"] = interactions_df.to_dict("records")
#                     if verbose >= 2:
#                         logger.info(f"Fetched {len(result['interactions'])} interactions for {ensembl_id}")
#                 else:
#                     result["interactions"] = []
#             except Exception as e:
#                 logger.warning(f"Error fetching interactions for {ensembl_id}: {e}")
#                 import traceback
#                 logger.debug(traceback.format_exc())
#                 result["interactions"] = []

#     # Try to get basic gene info using gget.info
#     try:
#         gene_info = gget.info(ensembl_id)
#         if gene_info is not None:
#             # Log what we received for debugging (only at verbose=2)
#             if verbose >= 2:
#                 logger.info(f"gget.info returned type: {type(gene_info)}")
            
#             # gget.info can return DataFrame or dict
#             if isinstance(gene_info, pd.DataFrame) and len(gene_info) > 0:
#                 # Log columns for debugging (only at verbose=2)
#                 if verbose >= 2:
#                     logger.info(f"gget.info DataFrame columns: {list(gene_info.columns)}")
#                     logger.info(f"gget.info DataFrame shape: {gene_info.shape}")
#                     logger.info(f"gget.info first row:\n{gene_info.iloc[0]}")
                
#                 # Extract from first row - try both .get() and direct access
#                 first_row = gene_info.iloc[0]
                
#                 # Try multiple ways to access the data
#                 def get_from_series(series, *keys):
#                     """Get value from pandas Series trying multiple keys."""
#                     for key in keys:
#                         try:
#                             if key in series.index:
#                                 val = series[key]
#                                 if pd.notna(val) and val != "":
#                                     return str(val)
#                         except (KeyError, AttributeError):
#                             continue
#                     return ""
                
#                 extracted_symbol = get_from_series(first_row, "gene_name", "name", "gene_name_ensembl", "external_name", "hgnc_symbol", "symbol", "gene_symbol")
#                 if extracted_symbol:
#                     result["approved_symbol"] = extracted_symbol
#                 # Otherwise keep the resolved_symbol if we have it
                
#                 # Try many more variations for gene name/description
#                 extracted_name = get_from_series(
#                     first_row, 
#                     "description", 
#                     "gene_description", 
#                     "description_ensembl",
#                     "full_name",
#                     "gene_full_name",
#                     "name",  # Sometimes name is the full name
#                     "gene_name",  # Sometimes gene_name is the full name
#                     "external_name",
#                     "hgnc_name",
#                     "hgnc_full_name",
#                     "long_name",
#                     "gene_long_name"
#                 )
#                 result["approved_name"] = extracted_name or ""
                
#                 result["biotype"] = (
#                     get_from_series(first_row, "gene_biotype", "biotype", "biotype_ensembl") or
#                     ""
#                 )
                
#                 # Log all available columns and their values for debugging (only at verbose=2)
#                 if verbose >= 2:
#                     logger.info(f"All columns in gget.info result: {list(gene_info.columns)}")
#                     logger.info(f"All values in first row: {first_row.to_dict()}")
#                     logger.info(f"Extracted: symbol={result['approved_symbol']}, name={result['approved_name']}, biotype={result['biotype']}")
                
#             elif isinstance(gene_info, dict):
#                 # Log keys for debugging (only at verbose=2)
#                 if verbose >= 2:
#                     logger.info(f"gget.info dict keys: {list(gene_info.keys())}")
#                     logger.info(f"gget.info dict full content: {gene_info}")
                
#                 # Try multiple key variations for symbol
#                 extracted_symbol = (
#                     gene_info.get("gene_name") or 
#                     gene_info.get("name") or 
#                     gene_info.get("gene_name_ensembl") or
#                     gene_info.get("external_name") or
#                     gene_info.get("hgnc_symbol") or
#                     gene_info.get("symbol") or
#                     gene_info.get("gene_symbol") or
#                     ""
#                 )
#                 if extracted_symbol:
#                     result["approved_symbol"] = extracted_symbol
#                 # Otherwise keep the resolved_symbol if we have it
                
#                 # Try many more variations for gene name/description
#                 extracted_name = (
#                     gene_info.get("description") or 
#                     gene_info.get("gene_description") or
#                     gene_info.get("description_ensembl") or
#                     gene_info.get("full_name") or
#                     gene_info.get("gene_full_name") or
#                     gene_info.get("external_name") or
#                     gene_info.get("hgnc_name") or
#                     gene_info.get("hgnc_full_name") or
#                     gene_info.get("long_name") or
#                     gene_info.get("gene_long_name") or
#                     # Sometimes "name" or "gene_name" is actually the full name
#                     (gene_info.get("name") if not extracted_symbol else None) or
#                     (gene_info.get("gene_name") if not extracted_symbol else None) or
#                     ""
#                 )
#                 result["approved_name"] = extracted_name or ""
                
#                 result["biotype"] = (
#                     gene_info.get("gene_biotype") or 
#                     gene_info.get("biotype") or
#                     gene_info.get("biotype_ensembl") or
#                     ""
#                 )
                
#                 if verbose >= 2:
#                     logger.info(f"Extracted from dict: symbol={result['approved_symbol']}, name={result['approved_name']}, biotype={result['biotype']}")
#             else:
#                 logger.warning(f"Unexpected type from gget.info: {type(gene_info)}")
                
#     except Exception as e:
#         logger.warning(f"Error fetching gene info for {ensembl_id}: {e}")
#         import traceback
#         logger.debug(traceback.format_exc())

#     # Save to cache
#     try:
#         if cache_as == "parquet":
#             # Save as parquet
#             # Convert dict to DataFrame (single row)
#             df = pd.DataFrame([result])
            
#             # Flatten nested DataFrames/Series in columns before saving to parquet
#             for col in df.columns:
#                 if len(df) > 0:
#                     non_null_mask = df[col].notna()
#                     if non_null_mask.any():
#                         sample_val = df[col][non_null_mask].iloc[0]
#                         # Check if column contains DataFrames or Series
#                         if isinstance(sample_val, (pd.DataFrame, pd.Series)):
#                             # Convert to JSON string
#                             def convert_to_json(val):
#                                 # Check for DataFrame/Series first (before pd.isna which doesn't work on them)
#                                 if isinstance(val, pd.DataFrame):
#                                     return json.dumps(val.to_dict("records"), default=str)
#                                 elif isinstance(val, pd.Series):
#                                     return json.dumps(val.to_dict(), default=str)
#                                 # Check for NaN/None after DataFrame/Series checks
#                                 try:
#                                     if pd.isna(val):
#                                         return None
#                                 except (ValueError, TypeError):
#                                     pass
#                                 return val
#                             df[col] = df[col].apply(convert_to_json)
#                         elif isinstance(sample_val, list) and len(sample_val) > 0:
#                             # Check if list contains DataFrames
#                             if any(isinstance(v, (pd.DataFrame, pd.Series)) for v in sample_val):
#                                 def convert_list_to_json(val):
#                                     # Check for DataFrame/Series first
#                                     if isinstance(val, pd.DataFrame):
#                                         return json.dumps(val.to_dict("records"), default=str)
#                                     elif isinstance(val, pd.Series):
#                                         return json.dumps(val.to_dict(), default=str)
#                                     # Check for NaN/None or wrong type
#                                     try:
#                                         if pd.isna(val):
#                                             return val
#                                     except (ValueError, TypeError):
#                                         pass
#                                     if not isinstance(val, list):
#                                         return val
#                                     converted = []
#                                     for item in val:
#                                         if isinstance(item, pd.DataFrame):
#                                             converted.append(item.to_dict("records"))
#                                         elif isinstance(item, pd.Series):
#                                             converted.append(item.to_dict())
#                                         else:
#                                             converted.append(item)
#                                     return json.dumps(converted, default=str)
#                                 df[col] = df[col].apply(convert_list_to_json)
#                         elif isinstance(sample_val, dict) and len(sample_val) > 0:
#                             # Check if dict contains DataFrames
#                             if any(isinstance(v, (pd.DataFrame, pd.Series)) for v in sample_val.values()):
#                                 def convert_dict_to_json(val):
#                                     # Check for DataFrame/Series first
#                                     if isinstance(val, pd.DataFrame):
#                                         return json.dumps(val.to_dict("records"), default=str)
#                                     elif isinstance(val, pd.Series):
#                                         return json.dumps(val.to_dict(), default=str)
#                                     # Check for NaN/None or wrong type
#                                     try:
#                                         if pd.isna(val):
#                                             return val
#                                     except (ValueError, TypeError):
#                                         pass
#                                     if not isinstance(val, dict):
#                                         return val
#                                     converted = {}
#                                     for k, v in val.items():
#                                         if isinstance(v, pd.DataFrame):
#                                             converted[k] = v.to_dict("records")
#                                         elif isinstance(v, pd.Series):
#                                             converted[k] = v.to_dict()
#                                         else:
#                                             converted[k] = v
#                                     return json.dumps(converted, default=str)
#                                 df[col] = df[col].apply(convert_dict_to_json)
            
#             # Use atomic write: write to temp file first, then rename
#             import tempfile
#             temp_file = cache_file + ".tmp"
#             try:
#                 df.to_parquet(temp_file, index=False)
#                 # Atomic rename (works on same filesystem)
#                 os.replace(temp_file, cache_file)
#                 if verbose >= 2:
#                     logger.info(f"Cached gene info to: {cache_file}")
#             except Exception as write_error:
#                 # Clean up temp file if it exists
#                 try:
#                     if os.path.exists(temp_file):
#                         os.remove(temp_file)
#                 except Exception:
#                     pass
#                 # Also remove corrupted cache file if it exists
#                 try:
#                     if os.path.exists(cache_file):
#                         os.remove(cache_file)
#                 except Exception:
#                     pass
#                 raise write_error
#         else:
#             # Save as JSON.gz
#             # First, convert all numpy arrays and pandas types to JSON-serializable format
#             def make_json_serializable(obj):
#                 """Recursively convert numpy/pandas types to JSON-serializable types."""
#                 # Check for DataFrame first (before pd.isna which doesn't work on DataFrames)
#                 if isinstance(obj, pd.DataFrame):
#                     return obj.to_dict("records")
#                 elif isinstance(obj, dict):
#                     return {k: make_json_serializable(v) for k, v in obj.items()}
#                 elif isinstance(obj, list):
#                     return [make_json_serializable(item) for item in obj]
#                 elif isinstance(obj, np.ndarray):
#                     return obj.tolist()
#                 elif isinstance(obj, (np.integer, np.floating)):
#                     return obj.item()
#                 elif isinstance(obj, np.bool_):
#                     return bool(obj)
#                 elif isinstance(obj, (pd.Series, pd.Index)):
#                     return obj.tolist()
#                 elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
#                     return str(obj)
#                 # Check for NaN/None after DataFrame/Series checks (pd.isna works on scalars)
#                 try:
#                     if pd.isna(obj):
#                         return None
#                 except (ValueError, TypeError):
#                     # pd.isna might fail for some types, just continue
#                     pass
#                 return obj
            
#             serializable_result = make_json_serializable(result)
            
#             # Use atomic write: write to temp file first, then rename
#             import tempfile
#             temp_file = cache_file + ".tmp"
#             try:
#                 with gzip.open(temp_file, "wt", encoding="utf-8") as f:
#                     json.dump(serializable_result, f, indent=2, ensure_ascii=False, cls=JSONEncoder)
#                 # Atomic rename (works on same filesystem)
#                 os.replace(temp_file, cache_file)
#                 if verbose >= 2:
#                     logger.info(f"Cached gene info to: {cache_file}")
#             except Exception as write_error:
#                 # Clean up temp file if it exists
#                 try:
#                     if os.path.exists(temp_file):
#                         os.remove(temp_file)
#                 except Exception:
#                     pass
#                 # Also remove corrupted cache file if it exists
#                 try:
#                     if os.path.exists(cache_file):
#                         os.remove(cache_file)
#                 except Exception:
#                     pass
#                 raise write_error
#     except Exception as e:
#         logger.warning(f"Error caching gene info to {cache_file}: {e}")
#         import traceback
#         logger.debug(traceback.format_exc())
#         # Ensure corrupted file is deleted
#         try:
#             if os.path.exists(cache_file):
#                 os.remove(cache_file)
#         except Exception:
#             pass
    
#     # Restore original gget logger level
#     gget_logger.setLevel(original_level)
    
#     return result


# def get_genes_info_batch(
#     gene_ids: List[str],
#     include_phenotypes: bool = True,
#     phenotype_size: int = 1000,
#     delay: float = 0.1,
#     output_format: str = "pandas",
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get detailed information for multiple genes in batch.

#     Parameters
#     ----------
#     gene_ids : list of str
#         List of gene identifiers (Ensembl IDs or gene symbols)
#     include_phenotypes : bool, default True
#         Whether to include associated phenotypes/diseases
#     phenotype_size : int, default 1000
#         Maximum number of phenotypes to retrieve per gene
#     delay : float, default 0.1
#         Delay between API requests in seconds (for rate limiting)
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"

#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with gene information. Each row contains:
#         - ensembl_id: Ensembl gene ID
#         - approved_symbol: Official gene symbol
#         - approved_name: Full gene name
#         - biotype: Gene biotype
#         - associated_diseases: JSON string or list of associated diseases
#             (depending on output_format)
#     """
#     results = []

#     for gene_id in tqdm(gene_ids, desc="Fetching gene info"):
#         try:
#             info = get_gene_info(
#                 gene_id,
#                 include_phenotypes=include_phenotypes,
#                 phenotype_size=phenotype_size,
#             )

#             # Flatten diseases for DataFrame
#             if include_phenotypes and "associated_diseases" in info:
#                 diseases = info.pop("associated_diseases")
#                 if output_format == "pandas":
#                     info["associated_diseases"] = str(diseases)  # JSON string for pandas
#                 else:
#                     info["associated_diseases"] = diseases  # Keep as list for polars

#             results.append(info)

#         except Exception as e:
#             logger.warning(f"Error fetching info for {gene_id}: {e}")
#             results.append(
#                 {
#                     "ensembl_id": gene_id,
#                     "approved_symbol": "",
#                     "approved_name": "",
#                     "biotype": "",
#                     "associated_diseases": [] if include_phenotypes else None,
#                 }
#             )

#         # Rate limiting
#         import time
#         time.sleep(delay)

#     # Create DataFrame
#     if output_format == "pandas":
#         df = pd.DataFrame(results)
#     else:
#         df = pl.DataFrame(results)

#     return df


# def get_gene_diseases(
#     gene_id: Union[str, List[str]],
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get diseases associated with a specific gene or multiple genes using gget.

#     This is a convenience wrapper around gget.opentargets() for diseases.
#     Results are cached as parquet files to avoid repeated API calls.

#     Parameters
#     ----------
#     gene_id : str or list of str
#         Gene identifier(s). Can be either:
#         - Ensembl gene ID (e.g., "ENSG00000157764")
#         - Gene symbol (e.g., "BRAF", "TP53", "EGFR")
#         - List of gene identifiers (mix of Ensembl IDs and symbols is allowed)
#           Gene symbols will be automatically resolved to Ensembl IDs
#     limit : int, optional
#         Maximum number of diseases to return per gene
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached parquet if exists, otherwise fetch and cache
#         - 1: Force new query, bypass cache

#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with disease associations. If gene_id is a list, results are
#         concatenated with a 'gene_id' column indicating which gene each disease
#         is associated with.
#     """
#     # Handle list of gene IDs
#     if isinstance(gene_id, list):
#         all_dfs = []
#         for gid in tqdm(gene_id, desc="Fetching diseases"):
#             try:
#                 df = _get_gene_diseases_single(
#                     gid,
#                     limit=limit,
#                     output_format=output_format,
#                     force=force,
#                 )
#                 # Add gene_id column to identify which gene this belongs to
#                 if len(df) > 0:
#                     if output_format == "pandas":
#                         df["gene_id"] = gid
#                     else:
#                         df = df.with_columns(pl.lit(gid).alias("gene_id"))
#                 all_dfs.append(df)
#             except Exception as e:
#                 logger.warning(f"Error fetching diseases for {gid}: {e}")
#                 # Add empty DataFrame with gene_id column
#                 if output_format == "pandas":
#                     empty_df = pd.DataFrame()
#                     empty_df["gene_id"] = []
#                 else:
#                     empty_df = pl.DataFrame({"gene_id": []})
#                 all_dfs.append(empty_df)
        
#         # Concatenate all results
#         if len(all_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame()
#             else:
#                 return pl.DataFrame()
        
#         # Filter out empty DataFrames before concatenating
#         non_empty_dfs = [df for df in all_dfs if len(df) > 0]
#         if len(non_empty_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame(columns=["gene_id"])
#             else:
#                 return pl.DataFrame({"gene_id": []})
        
#         if output_format == "pandas":
#             result_df = pd.concat(non_empty_dfs, ignore_index=True)
#         else:
#             result_df = pl.concat(non_empty_dfs)
        
#         return result_df
    
#     # Single gene ID - delegate to helper function
#     return _get_gene_diseases_single(
#         gene_id,
#         limit=limit,
#         output_format=output_format,
#         force=force,
#     )


# def _get_gene_diseases_single(
#     gene_id: str,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Internal helper function to get diseases for a single gene.
    
#     This contains the core logic that was previously in get_gene_diseases.
#     """
#     # Resolve gene symbol to Ensembl ID if needed
#     ensembl_id = gene_id
#     if not gene_id.startswith("ENSG"):
#         try:
#             # Use gget.search to resolve gene symbol to Ensembl ID
#             search_results = gget.search(gene_id, species="human")
#             if search_results is not None and len(search_results) > 0:
#                 # Find the first result that looks like our gene
#                 for result_item in search_results:
#                     if isinstance(result_item, dict):
#                         ens_id = result_item.get("ensembl_id", "")
#                         if ens_id.startswith("ENSG"):
#                             ensembl_id = ens_id
#                             break
#                     elif isinstance(result_item, str) and result_item.startswith("ENSG"):
#                         ensembl_id = result_item
#                         break
#                 else:
#                     raise ValueError(f"Could not resolve gene symbol to Ensembl ID: {gene_id}")
#             else:
#                 raise ValueError(f"Gene symbol not found: {gene_id}")
#         except Exception as e:
#             logger.error(f"Error resolving gene ID {gene_id}: {e}")
#             raise ValueError(f"Could not resolve gene identifier: {gene_id}")
    
#     # Set up cache file path
#     cache_file = str(DEFAULT_CACHE_DIR / f"diseases_{ensembl_id}.parquet")
#     if limit is not None:
#         cache_file = str(DEFAULT_CACHE_DIR / f"diseases_{ensembl_id}_limit{limit}.parquet")
    
#     # Check cache (force=0)
#     if force == 0:
#         try:
#             if output_format == "pandas":
#                 df = pd.read_parquet(cache_file)
#                 logger.info(f"Loaded {len(df)} diseases from cache: {cache_file}")
#                 return df
#             else:
#                 df = pl.read_parquet(cache_file)
#                 logger.info(f"Loaded {len(df)} diseases from cache: {cache_file}")
#                 return df
#         except FileNotFoundError:
#             pass
    
#     # Fetch from API
#     logger.info(f"Fetching diseases for {ensembl_id} from OpenTargets Platform...")
#     diseases_data = gget.opentargets(ensembl_id, resource="diseases", limit=limit)
    
#     if diseases_data is None or len(diseases_data) == 0:
#         # Return empty DataFrame with expected structure
#         logger.warning(f"No diseases found for {ensembl_id}")
#         if output_format == "pandas":
#             df = pd.DataFrame()
#         else:
#             df = pl.DataFrame()
#     else:
#         # Log raw data structure for debugging
#         if isinstance(diseases_data, list) and len(diseases_data) > 0:
#             logger.debug(f"Raw JSON structure - first item keys: {list(diseases_data[0].keys()) if isinstance(diseases_data[0], dict) else 'Not a dict'}")
#             logger.debug(f"Raw JSON structure - first item sample: {diseases_data[0] if isinstance(diseases_data[0], dict) else 'Not a dict'}")
#         elif isinstance(diseases_data, pd.DataFrame) and len(diseases_data) > 0:
#             logger.debug(f"Raw DataFrame columns: {list(diseases_data.columns)}")
#             logger.debug(f"Raw DataFrame first row: {diseases_data.iloc[0].to_dict()}")
        
#         # Convert list of dicts to DataFrame - this preserves ALL fields from raw JSON
#         df = pd.DataFrame(diseases_data)
        
#         # Log what columns we have after conversion
#         logger.info(f"DataFrame created with {len(df)} rows and columns: {list(df.columns)}")
        
#         if output_format == "polars":
#             df = pl.from_pandas(df)
    
#     # Save to cache
#     if len(df) > 0:
#         try:
#             if output_format == "pandas":
#                 df.to_parquet(cache_file, index=False)
#             else:
#                 # For polars, convert to pandas temporarily to save
#                 df_pd = df.to_pandas()
#                 df_pd.to_parquet(cache_file, index=False)
#                 # Recreate polars DataFrame
#                 df = pl.read_parquet(cache_file)
#             logger.info(f"Cached {len(df)} diseases to: {cache_file}")
#         except Exception as e:
#             logger.warning(f"Error caching diseases to {cache_file}: {e}")
    
#     return df


# def get_gene_drugs(
#     gene_id: Union[str, List[str]],
#     disease_id: Optional[str] = None,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get drugs associated with a specific gene or multiple genes using gget.

#     This is a convenience wrapper around gget.opentargets() for drugs.
#     Results are cached as parquet files to avoid repeated API calls.

#     Parameters
#     ----------
#     gene_id : str or list of str
#         Gene identifier(s). Can be either:
#         - Ensembl gene ID (e.g., "ENSG00000157764")
#         - Gene symbol (e.g., "BRAF", "TP53", "EGFR")
#         - List of gene identifiers (mix of Ensembl IDs and symbols is allowed)
#           Gene symbols will be automatically resolved to Ensembl IDs
#     disease_id : str, optional
#         Filter drugs by disease ID (EFO ID, e.g., 'EFO_0000274')
#     limit : int, optional
#         Maximum number of drugs to return per gene
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached parquet if exists, otherwise fetch and cache
#         - 1: Force new query, bypass cache

#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with drug associations. If gene_id is a list, results are
#         concatenated with a 'gene_id' column indicating which gene each drug
#         is associated with.
#     """
#     # Handle list of gene IDs
#     if isinstance(gene_id, list):
#         all_dfs = []
#         for gid in tqdm(gene_id, desc="Fetching drugs"):
#             try:
#                 df = _get_gene_drugs_single(
#                     gid,
#                     disease_id=disease_id,
#                     limit=limit,
#                     output_format=output_format,
#                     force=force,
#                 )
#                 # Add gene_id column to identify which gene this belongs to
#                 if len(df) > 0:
#                     if output_format == "pandas":
#                         df["gene_id"] = gid
#                     else:
#                         df = df.with_columns(pl.lit(gid).alias("gene_id"))
#                 all_dfs.append(df)
#             except Exception as e:
#                 logger.warning(f"Error fetching drugs for {gid}: {e}")
#                 # Add empty DataFrame with gene_id column
#                 if output_format == "pandas":
#                     empty_df = pd.DataFrame()
#                     empty_df["gene_id"] = []
#                 else:
#                     empty_df = pl.DataFrame({"gene_id": []})
#                 all_dfs.append(empty_df)
        
#         # Concatenate all results
#         if len(all_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame()
#             else:
#                 return pl.DataFrame()
        
#         # Filter out empty DataFrames before concatenating
#         non_empty_dfs = [df for df in all_dfs if len(df) > 0]
#         if len(non_empty_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame(columns=["gene_id"])
#             else:
#                 return pl.DataFrame({"gene_id": []})
        
#         if output_format == "pandas":
#             result_df = pd.concat(non_empty_dfs, ignore_index=True)
#         else:
#             result_df = pl.concat(non_empty_dfs)
        
#         return result_df
    
#     # Single gene ID - delegate to helper function
#     return _get_gene_drugs_single(
#         gene_id,
#         disease_id=disease_id,
#         limit=limit,
#         output_format=output_format,
#         force=force,
#     )


# def _get_gene_drugs_single(
#     gene_id: str,
#     disease_id: Optional[str] = None,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Internal helper function to get drugs for a single gene.
#     """
#     # Resolve gene symbol to Ensembl ID if needed
#     ensembl_id = gene_id
#     if not gene_id.startswith("ENSG"):
#         try:
#             search_results = gget.search(gene_id, species="human")
#             if search_results is not None and len(search_results) > 0:
#                 for result_item in search_results:
#                     if isinstance(result_item, dict):
#                         ens_id = result_item.get("ensembl_id", "")
#                         if ens_id.startswith("ENSG"):
#                             ensembl_id = ens_id
#                             break
#                     elif isinstance(result_item, str) and result_item.startswith("ENSG"):
#                         ensembl_id = result_item
#                         break
#                 else:
#                     raise ValueError(f"Could not resolve gene symbol to Ensembl ID: {gene_id}")
#             else:
#                 raise ValueError(f"Gene symbol not found: {gene_id}")
#         except Exception as e:
#             logger.error(f"Error resolving gene ID {gene_id}: {e}")
#             raise ValueError(f"Could not resolve gene identifier: {gene_id}")
    
#     # Set up cache file path
#     cache_file = str(DEFAULT_CACHE_DIR / f"drugs_{ensembl_id}.parquet")
#     if disease_id:
#         cache_file = str(DEFAULT_CACHE_DIR / f"drugs_{ensembl_id}_disease_{disease_id}.parquet")
#     if limit is not None:
#         cache_file = str(DEFAULT_CACHE_DIR / f"drugs_{ensembl_id}_limit{limit}.parquet")
#         if disease_id:
#             cache_file = str(DEFAULT_CACHE_DIR / f"drugs_{ensembl_id}_disease_{disease_id}_limit{limit}.parquet")
    
#     # Check cache (force=0)
#     if force == 0:
#         try:
#             if output_format == "pandas":
#                 df = pd.read_parquet(cache_file)
#                 logger.info(f"Loaded {len(df)} drugs from cache: {cache_file}")
#                 return df
#             else:
#                 df = pl.read_parquet(cache_file)
#                 logger.info(f"Loaded {len(df)} drugs from cache: {cache_file}")
#                 return df
#         except FileNotFoundError:
#             pass
    
#     # Fetch from API - use filters parameter for disease_id
#     logger.info(f"Fetching drugs for {ensembl_id} from OpenTargets Platform...")
#     filters = {}
#     if disease_id:
#         filters["disease_id"] = [disease_id]
    
#     drugs_data = gget.opentargets(
#         ensembl_id,
#         resource="drugs",
#         limit=limit,
#         filters=filters if filters else None,
#     )
    
#     if drugs_data is None or len(drugs_data) == 0:
#         logger.warning(f"No drugs found for {ensembl_id}")
#         if output_format == "pandas":
#             df = pd.DataFrame()
#         else:
#             df = pl.DataFrame()
#     else:
#         # Convert list of dicts to DataFrame
#         df = pd.DataFrame(drugs_data)
        
#         if output_format == "polars":
#             df = pl.from_pandas(df)
    
#     # Save to cache
#     if len(df) > 0:
#         try:
#             if output_format == "pandas":
#                 df.to_parquet(cache_file, index=False)
#             else:
#                 df_pd = df.to_pandas()
#                 df_pd.to_parquet(cache_file, index=False)
#                 df = pl.read_parquet(cache_file)
#             logger.info(f"Cached {len(df)} drugs to: {cache_file}")
#         except Exception as e:
#             logger.warning(f"Error caching drugs to {cache_file}: {e}")
    
#     return df


# def get_gene_tractability(
#     gene_id: Union[str, List[str]],
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get tractability data for a specific gene or multiple genes using gget.

#     Tractability data indicates how "druggable" a gene is based on various criteria.

#     Parameters
#     ----------
#     gene_id : str or list of str
#         Gene identifier(s). Can be either:
#         - Ensembl gene ID (e.g., "ENSG00000157764")
#         - Gene symbol (e.g., "BRAF", "TP53", "EGFR")
#         - List of gene identifiers (mix of Ensembl IDs and symbols is allowed)
#           Gene symbols will be automatically resolved to Ensembl IDs
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached parquet if exists, otherwise fetch and cache
#         - 1: Force new query, bypass cache

#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with tractability data. If gene_id is a list, results are
#         concatenated with a 'gene_id' column.
#     """
#     return _get_gene_resource_single(
#         gene_id,
#         resource="tractability",
#         output_format=output_format,
#         force=force,
#     )


# def get_gene_pharmacogenetics(
#     gene_id: Union[str, List[str]],
#     drug_id: Optional[str] = None,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get pharmacogenetic responses for a specific gene or multiple genes using gget.

#     Parameters
#     ----------
#     gene_id : str or list of str
#         Gene identifier(s). Can be either:
#         - Ensembl gene ID (e.g., "ENSG00000157764")
#         - Gene symbol (e.g., "BRAF", "TP53", "EGFR")
#         - List of gene identifiers (mix of Ensembl IDs and symbols is allowed)
#           Gene symbols will be automatically resolved to Ensembl IDs
#     drug_id : str, optional
#         Filter by drug ID (e.g., 'CHEMBL1743081'). Only valid for pharmacogenetics resource.
#     limit : int, optional
#         Maximum number of results to return per gene
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached parquet if exists, otherwise fetch and cache
#         - 1: Force new query, bypass cache

#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with pharmacogenetic data. If gene_id is a list, results are
#         concatenated with a 'gene_id' column.
#     """
#     return _get_gene_resource_single(
#         gene_id,
#         resource="pharmacogenetics",
#         drug_id=drug_id,
#         limit=limit,
#         output_format=output_format,
#         force=force,
#     )


# def get_gene_expression(
#     gene_id: Union[str, List[str]],
#     tissue_id: Optional[str] = None,
#     anatomical_system: Optional[str] = None,
#     organ: Optional[str] = None,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get gene expression data (by tissues, organs, and anatomical systems) using gget.

#     Parameters
#     ----------
#     gene_id : str or list of str
#         Gene identifier(s). Can be either:
#         - Ensembl gene ID (e.g., "ENSG00000157764")
#         - Gene symbol (e.g., "BRAF", "TP53", "EGFR")
#         - List of gene identifiers (mix of Ensembl IDs and symbols is allowed)
#           Gene symbols will be automatically resolved to Ensembl IDs
#     tissue_id : str, optional
#         Filter by tissue ID (e.g., 'UBERON_0000473'). Only valid for expression resource.
#     anatomical_system : str, optional
#         Filter by anatomical system (e.g., 'nervous system'). Only valid for expression resource.
#     organ : str, optional
#         Filter by organ (e.g., 'brain'). Only valid for expression resource.
#     limit : int, optional
#         Maximum number of results to return per gene
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached parquet if exists, otherwise fetch and cache
#         - 1: Force new query, bypass cache

#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with expression data. If gene_id is a list, results are
#         concatenated with a 'gene_id' column.
#     """
#     return _get_gene_resource_single(
#         gene_id,
#         resource="expression",
#         tissue_id=tissue_id,
#         anatomical_system=anatomical_system,
#         organ=organ,
#         limit=limit,
#         output_format=output_format,
#         force=force,
#     )


# def get_gene_depmap(
#     gene_id: Union[str, List[str]],
#     tissue_id: Optional[str] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get DepMap gene→disease-effect data using gget.

#     Parameters
#     ----------
#     gene_id : str or list of str
#         Gene identifier(s). Can be either:
#         - Ensembl gene ID (e.g., "ENSG00000157764")
#         - Gene symbol (e.g., "BRAF", "TP53", "EGFR")
#         - List of gene identifiers (mix of Ensembl IDs and symbols is allowed)
#           Gene symbols will be automatically resolved to Ensembl IDs
#     tissue_id : str, optional
#         Filter by tissue ID (e.g., 'UBERON_0000473'). Only valid for depmap resource.
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached parquet if exists, otherwise fetch and cache
#         - 1: Force new query, bypass cache

#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with DepMap data. If gene_id is a list, results are
#         concatenated with a 'gene_id' column.
#     """
#     return _get_gene_resource_single(
#         gene_id,
#         resource="depmap",
#         tissue_id=tissue_id,
#         output_format=output_format,
#         force=force,
#     )


# def get_gene_interactions(
#     gene_id: Union[str, List[str]],
#     protein_a_id: Optional[str] = None,
#     protein_b_id: Optional[str] = None,
#     gene_b_id: Optional[str] = None,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Get protein⇄protein interactions using gget.

#     Parameters
#     ----------
#     gene_id : str or list of str
#         Gene identifier(s). Can be either:
#         - Ensembl gene ID (e.g., "ENSG00000157764")
#         - Gene symbol (e.g., "BRAF", "TP53", "EGFR")
#         - List of gene identifiers (mix of Ensembl IDs and symbols is allowed)
#           Gene symbols will be automatically resolved to Ensembl IDs
#     protein_a_id : str, optional
#         Filter by the protein ID of the first protein (e.g., 'ENSP00000304915').
#         Only valid for interactions resource.
#     protein_b_id : str, optional
#         Filter by the protein ID of the second protein (e.g., 'ENSP00000379111').
#         Only valid for interactions resource.
#     gene_b_id : str, optional
#         Filter by the gene ID of the second protein (e.g., 'ENSG00000077238').
#         Only valid for interactions resource.
#     limit : int, optional
#         Maximum number of results to return per gene
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"
#     force : int, default 0
#         Force refresh level:
#         - 0: Use cached parquet if exists, otherwise fetch and cache
#         - 1: Force new query, bypass cache

#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with interaction data. If gene_id is a list, results are
#         concatenated with a 'gene_id' column.
#     """
#     return _get_gene_resource_single(
#         gene_id,
#         resource="interactions",
#         protein_a_id=protein_a_id,
#         protein_b_id=protein_b_id,
#         gene_b_id=gene_b_id,
#         limit=limit,
#         output_format=output_format,
#         force=force,
#     )


# def _get_gene_resource_single(
#     gene_id: Union[str, List[str]],
#     resource: str,
#     disease_id: Optional[str] = None,
#     drug_id: Optional[str] = None,
#     tissue_id: Optional[str] = None,
#     anatomical_system: Optional[str] = None,
#     organ: Optional[str] = None,
#     protein_a_id: Optional[str] = None,
#     protein_b_id: Optional[str] = None,
#     gene_b_id: Optional[str] = None,
#     limit: Optional[int] = None,
#     output_format: str = "pandas",
#     force: int = 0,
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     Internal helper function to get any resource for a single gene or list of genes.
    
#     This centralizes the logic for fetching, caching, and processing gget.opentargets results.
#     """
#     # Suppress gget INFO messages (default behavior - can be overridden by caller)
#     gget_logger = logging.getLogger("gget")
#     original_level = gget_logger.level
#     gget_logger.setLevel(logging.WARNING)
    
#     # Handle list of gene IDs
#     if isinstance(gene_id, list):
#         all_dfs = []
#         for gid in tqdm(gene_id, desc=f"Fetching {resource}"):
#             try:
#                 df = _get_gene_resource_single(
#                     gid,
#                     resource=resource,
#                     disease_id=disease_id,
#                     drug_id=drug_id,
#                     tissue_id=tissue_id,
#                     anatomical_system=anatomical_system,
#                     organ=organ,
#                     protein_a_id=protein_a_id,
#                     protein_b_id=protein_b_id,
#                     gene_b_id=gene_b_id,
#                     limit=limit,
#                     output_format=output_format,
#                     force=force,
#                 )
#                 if len(df) > 0:
#                     if output_format == "pandas":
#                         df["gene_id"] = gid
#                     else:
#                         df = df.with_columns(pl.lit(gid).alias("gene_id"))
#                 all_dfs.append(df)
#             except Exception as e:
#                 logger.warning(f"Error fetching {resource} for {gid}: {e}")
#                 if output_format == "pandas":
#                     empty_df = pd.DataFrame()
#                     empty_df["gene_id"] = []
#                 else:
#                     empty_df = pl.DataFrame({"gene_id": []})
#                 all_dfs.append(empty_df)
        
#         if len(all_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame()
#             else:
#                 return pl.DataFrame()
        
#         non_empty_dfs = [df for df in all_dfs if len(df) > 0]
#         if len(non_empty_dfs) == 0:
#             if output_format == "pandas":
#                 return pd.DataFrame(columns=["gene_id"])
#             else:
#                 return pl.DataFrame({"gene_id": []})
        
#         if output_format == "pandas":
#             result_df = pd.concat(non_empty_dfs, ignore_index=True)
#         else:
#             result_df = pl.concat(non_empty_dfs)
#         # Restore original gget logger level before returning
#         gget_logger.setLevel(original_level)
#         return result_df
    
#     # Single gene ID - resolve symbol to Ensembl ID if needed
#     ensembl_id = gene_id
#     if not gene_id.startswith("ENSG"):
#         try:
#             search_results = gget.search(gene_id, species="human")
#             if search_results is not None and len(search_results) > 0:
#                 for result_item in search_results:
#                     if isinstance(result_item, dict):
#                         ens_id = result_item.get("ensembl_id", "")
#                         if ens_id.startswith("ENSG"):
#                             ensembl_id = ens_id
#                             break
#                     elif isinstance(result_item, str) and result_item.startswith("ENSG"):
#                         ensembl_id = result_item
#                         break
#                 else:
#                     raise ValueError(f"Could not resolve gene symbol to Ensembl ID: {gene_id}")
#             else:
#                 raise ValueError(f"Gene symbol not found: {gene_id}")
#         except Exception as e:
#             logger.error(f"Error resolving gene ID {gene_id}: {e}")
#             raise ValueError(f"Could not resolve gene identifier: {gene_id}")
    
#     # Build cache file path
#     cache_parts = [resource, ensembl_id]
#     if disease_id:
#         cache_parts.append(f"disease_{disease_id}")
#     if drug_id:
#         cache_parts.append(f"drug_{drug_id}")
#     if tissue_id:
#         cache_parts.append(f"tissue_{tissue_id}")
#     if anatomical_system:
#         cache_parts.append(f"anat_{anatomical_system.replace(' ', '_')}")
#     if organ:
#         cache_parts.append(f"organ_{organ.replace(' ', '_')}")
#     if protein_a_id:
#         cache_parts.append(f"prot_a_{protein_a_id}")
#     if protein_b_id:
#         cache_parts.append(f"prot_b_{protein_b_id}")
#     if gene_b_id:
#         cache_parts.append(f"gene_b_{gene_b_id}")
#     if limit is not None:
#         cache_parts.append(f"limit{limit}")
    
#     cache_file = str(DEFAULT_CACHE_DIR / f"{'_'.join(cache_parts)}.parquet")
    
#     # Check cache (force=0)
#     if force == 0:
#         try:
#             if output_format == "pandas":
#                 df = pd.read_parquet(cache_file)
#                 logger.info(f"Loaded {len(df)} {resource} results from cache: {cache_file}")
#                 gget_logger.setLevel(original_level)
#                 return df
#             else:
#                 df = pl.read_parquet(cache_file)
#                 logger.info(f"Loaded {len(df)} {resource} results from cache: {cache_file}")
#                 gget_logger.setLevel(original_level)
#                 return df
#         except FileNotFoundError:
#             pass
    
#     # Build filters dict for gget
#     filters = {}
#     if disease_id:
#         filters["disease_id"] = [disease_id]
#     if drug_id:
#         filters["drug_id"] = [drug_id]
#     if tissue_id:
#         filters["tissue_id"] = [tissue_id]
#     if anatomical_system:
#         filters["anatomical_system"] = [anatomical_system]
#     if organ:
#         filters["organ"] = [organ]
#     if protein_a_id:
#         filters["protein_a_id"] = [protein_a_id]
#     if protein_b_id:
#         filters["protein_b_id"] = [protein_b_id]
#     if gene_b_id:
#         filters["gene_b_id"] = [gene_b_id]
    
#     # Fetch from API
#     logger.info(f"Fetching {resource} for {ensembl_id} from OpenTargets Platform...")
#     try:
#         data = gget.opentargets(
#             ensembl_id,
#             resource=resource,
#             limit=limit,
#             filters=filters if filters else None,
#         )
#     except Exception as e:
#         logger.warning(f"Error calling gget.opentargets for {resource} on {ensembl_id}: {e}")
#         import traceback
#         logger.debug(traceback.format_exc())
#         gget_logger.setLevel(original_level)
#         if output_format == "pandas":
#             return pd.DataFrame()
#         else:
#             return pl.DataFrame()
    
#     # Handle different return types from gget
#     if data is None:
#         logger.warning(f"No {resource} found for {ensembl_id} (returned None)")
#         if output_format == "pandas":
#             df = pd.DataFrame()
#         else:
#             df = pl.DataFrame()
#     elif isinstance(data, (list, pd.DataFrame)):
#         # Check if empty
#         if len(data) == 0:
#             logger.warning(f"No {resource} found for {ensembl_id} (empty result)")
#             if output_format == "pandas":
#                 df = pd.DataFrame()
#             else:
#                 df = pl.DataFrame()
#         else:
#             # Convert to DataFrame - preserve all fields
#             df = pd.DataFrame(data)
            
#             if output_format == "polars":
#                 df = pl.from_pandas(df)
#     else:
#         # Unexpected type - try to convert anyway
#         logger.warning(f"Unexpected type from gget.opentargets for {resource}: {type(data)}")
#         try:
#             df = pd.DataFrame(data)
#             if output_format == "polars":
#                 df = pl.from_pandas(df)
#         except Exception as e:
#             logger.warning(f"Could not convert {resource} data to DataFrame: {e}")
#             if output_format == "pandas":
#                 df = pd.DataFrame()
#             else:
#                 df = pl.DataFrame()
    
#     # Save to cache
#     if len(df) > 0:
#         try:
#             # Flatten nested DataFrames/Series in columns before saving to parquet
#             # Parquet doesn't support nested DataFrames, so convert them to JSON strings
#             def flatten_column_for_parquet(series):
#                 """Convert nested DataFrames/Series to JSON strings for parquet compatibility."""
#                 def convert_value(val):
#                     # Check for DataFrame/Series first (before pd.isna which doesn't work on them)
#                     if isinstance(val, pd.DataFrame):
#                         # Convert DataFrame to JSON string
#                         return json.dumps(val.to_dict("records"), default=str)
#                     elif isinstance(val, pd.Series):
#                         # Convert Series to JSON string
#                         return json.dumps(val.to_dict(), default=str)
#                     # Check for NaN/None after DataFrame/Series checks
#                     try:
#                         if pd.isna(val):
#                             return None
#                     except (ValueError, TypeError):
#                         # pd.isna might fail for some types (e.g., DataFrames), just continue
#                         pass
#                     if isinstance(val, list):
#                         # Check if list contains DataFrames
#                         converted = []
#                         for item in val:
#                             if isinstance(item, pd.DataFrame):
#                                 converted.append(item.to_dict("records"))
#                             elif isinstance(item, pd.Series):
#                                 converted.append(item.to_dict())
#                             else:
#                                 converted.append(item)
#                         return json.dumps(converted, default=str)
#                     elif isinstance(val, dict):
#                         # Check if dict contains DataFrames
#                         converted = {}
#                         for k, v in val.items():
#                             if isinstance(v, pd.DataFrame):
#                                 converted[k] = v.to_dict("records")
#                             elif isinstance(v, pd.Series):
#                                 converted[k] = v.to_dict()
#                             else:
#                                 converted[k] = v
#                         return json.dumps(converted, default=str)
#                     else:
#                         return val
                
#                 return series.apply(convert_value)
            
#             if output_format == "pandas":
#                 df_to_save = df.copy()
#                 # Check each column for nested DataFrames/Series
#                 for col in df_to_save.columns:
#                     if len(df_to_save) > 0:
#                         # Find first non-null value
#                         non_null_mask = df_to_save[col].notna()
#                         if non_null_mask.any():
#                             sample_val = df_to_save[col][non_null_mask].iloc[0]
#                             # Check if column contains DataFrames or Series
#                             if isinstance(sample_val, (pd.DataFrame, pd.Series)):
#                                 df_to_save[col] = flatten_column_for_parquet(df_to_save[col])
#                             elif isinstance(sample_val, list) and len(sample_val) > 0:
#                                 # Check if list contains DataFrames
#                                 if any(isinstance(v, (pd.DataFrame, pd.Series)) for v in sample_val):
#                                     df_to_save[col] = flatten_column_for_parquet(df_to_save[col])
#                             elif isinstance(sample_val, dict) and len(sample_val) > 0:
#                                 # Check if dict contains DataFrames
#                                 if any(isinstance(v, (pd.DataFrame, pd.Series)) for v in sample_val.values()):
#                                     df_to_save[col] = flatten_column_for_parquet(df_to_save[col])
                
#                 df_to_save.to_parquet(cache_file, index=False)
#             else:
#                 df_pd = df.to_pandas()
#                 # Apply same flattening logic
#                 for col in df_pd.columns:
#                     if len(df_pd) > 0:
#                         non_null_mask = df_pd[col].notna()
#                         if non_null_mask.any():
#                             sample_val = df_pd[col][non_null_mask].iloc[0]
#                             if isinstance(sample_val, (pd.DataFrame, pd.Series)):
#                                 df_pd[col] = flatten_column_for_parquet(df_pd[col])
#                             elif isinstance(sample_val, list) and len(sample_val) > 0:
#                                 if any(isinstance(v, (pd.DataFrame, pd.Series)) for v in sample_val):
#                                     df_pd[col] = flatten_column_for_parquet(df_pd[col])
#                             elif isinstance(sample_val, dict) and len(sample_val) > 0:
#                                 if any(isinstance(v, (pd.DataFrame, pd.Series)) for v in sample_val.values()):
#                                     df_pd[col] = flatten_column_for_parquet(df_pd[col])
                
#                 df_pd.to_parquet(cache_file, index=False)
#                 df = pl.read_parquet(cache_file)
#             logger.info(f"Cached {len(df)} {resource} results to: {cache_file}")
#         except Exception as e:
#             logger.warning(f"Error caching {resource} to {cache_file}: {e}")
#             import traceback
#             logger.debug(traceback.format_exc())
    
#     # Restore original gget logger level
#     gget_logger.setLevel(original_level)
    
#     return df


# def list_cache(
#     pattern: Optional[str] = None,
#     cache_dir: Optional[str] = None,
#     output_format: str = "pandas",
# ) -> Union[pd.DataFrame, pl.DataFrame]:
#     """
#     List all cached files in the OpenTargets cache directory.
    
#     Parameters
#     ----------
#     pattern : str, optional
#         Pattern to filter files (e.g., "gene_info", "pharmacogenetics", "ENSG00000157764").
#         If None, returns all cached files. The pattern is matched against the filename
#         (case-insensitive substring match).
#     cache_dir : str, optional
#         Cache directory to list. If None, uses DEFAULT_CACHE_DIR.
#     output_format : str, default "pandas"
#         Output format: "pandas" or "polars"
    
#     Returns
#     -------
#     pd.DataFrame or pl.DataFrame
#         DataFrame with columns:
#         - filename: Name of the cached file
#         - filepath: Full path to the file
#         - size_bytes: File size in bytes
#         - size_mb: File size in megabytes (rounded to 2 decimals)
#         - modified_time: Last modification time (datetime)
#         - file_type: Type of file ("json.gz", "parquet", or "other")
#         - resource_type: Inferred resource type (e.g., "gene_info", "pharmacogenetics", "diseases", etc.)
#         - gene_id: Extracted gene ID if present (e.g., "ENSG00000157764")
    
#     Examples
#     --------
#     >>> import phenoref.opentargets as opentargets
#     >>> 
#     >>> # List all cached files
#     >>> all_files = opentargets.list_cache()
#     >>> 
#     >>> # List only gene_info files
#     >>> gene_info_files = opentargets.list_cache(pattern="gene_info")
#     >>> 
#     >>> # List files for a specific gene
#     >>> braf_files = opentargets.list_cache(pattern="ENSG00000157764")
#     >>> 
#     >>> # List pharmacogenetics files
#     >>> pgx_files = opentargets.list_cache(pattern="pharmacogenetics")
#     """
#     if cache_dir is None:
#         cache_dir = DEFAULT_CACHE_DIR
#     else:
#         cache_dir = Path(cache_dir)
    
#     cache_dir = Path(cache_dir)
    
#     if not cache_dir.exists():
#         if output_format == "pandas":
#             return pd.DataFrame(columns=["filename", "filepath", "size_bytes", "size_mb", "modified_time", "file_type", "resource_type", "gene_id"])
#         else:
#             return pl.DataFrame({
#                 "filename": [],
#                 "filepath": [],
#                 "size_bytes": [],
#                 "size_mb": [],
#                 "modified_time": [],
#                 "file_type": [],
#                 "resource_type": [],
#                 "gene_id": [],
#             })
    
#     # Get all files in cache directory
#     all_files = []
#     for file_path in cache_dir.iterdir():
#         if file_path.is_file():
#             filename = file_path.name
            
#             # Apply pattern filter if provided
#             if pattern is not None and pattern.lower() not in filename.lower():
#                 continue
            
#             # Get file stats
#             stat = file_path.stat()
#             size_bytes = stat.st_size
#             size_mb = round(size_bytes / (1024 * 1024), 2)
#             modified_time = pd.Timestamp.fromtimestamp(stat.st_mtime)
            
#             # Determine file type
#             if filename.endswith('.json.gz'):
#                 file_type = "json.gz"
#             elif filename.endswith('.parquet'):
#                 file_type = "parquet"
#             elif filename.endswith('.gz'):
#                 file_type = "gz"
#             else:
#                 file_type = "other"
            
#             # Infer resource type from filename
#             resource_type = "unknown"
#             gene_id = None
            
#             # Extract gene ID (ENSG pattern)
#             import re
#             gene_match = re.search(r'ENSG\d+', filename)
#             if gene_match:
#                 gene_id = gene_match.group(0)
            
#             # Determine resource type based on filename patterns
#             if filename.startswith('gene_info_'):
#                 resource_type = "gene_info"
#             elif filename.startswith('pharmacogenetics_'):
#                 resource_type = "pharmacogenetics"
#             elif filename.startswith('diseases_') or filename.startswith('drugs_'):
#                 resource_type = filename.split('_')[0]  # "diseases" or "drugs"
#             elif filename.startswith('tractability_'):
#                 resource_type = "tractability"
#             elif filename.startswith('expression_'):
#                 resource_type = "expression"
#             elif filename.startswith('depmap_'):
#                 resource_type = "depmap"
#             elif filename.startswith('interactions_'):
#                 resource_type = "interactions"
#             elif filename.startswith('all_genes_'):
#                 resource_type = "all_genes"
#             elif filename.startswith('raw_gtf_'):
#                 resource_type = "raw_gtf"
            
#             all_files.append({
#                 "filename": filename,
#                 "filepath": str(file_path),
#                 "size_bytes": size_bytes,
#                 "size_mb": size_mb,
#                 "modified_time": modified_time,
#                 "file_type": file_type,
#                 "resource_type": resource_type,
#                 "gene_id": gene_id,
#             })
    
#     # Convert to DataFrame
#     if len(all_files) == 0:
#         if output_format == "pandas":
#             return pd.DataFrame(columns=["filename", "filepath", "size_bytes", "size_mb", "modified_time", "file_type", "resource_type", "gene_id"])
#         else:
#             return pl.DataFrame({
#                 "filename": [],
#                 "filepath": [],
#                 "size_bytes": [],
#                 "size_mb": [],
#                 "modified_time": [],
#                 "file_type": [],
#                 "resource_type": [],
#                 "gene_id": [],
#             })
    
#     if output_format == "pandas":
#         df = pd.DataFrame(all_files)
#         print(df.shape)
#         # Sort by modified_time (newest first)
#         df = df.sort_values("modified_time", ascending=False).reset_index(drop=True)
#         return df
#     else:
#         df = pl.DataFrame(all_files)
#         print(df.shape)
#         # Sort by modified_time (newest first)
#         df = df.sort("modified_time", descending=True)
#         return df


def get_pathways(targets: pd.DataFrame, gene_symbol_col: str = "approvedSymbol", gene_id_col: str = "id", save_path: Optional[str] = None, force: bool = False) -> pd.DataFrame:
    """
    Extract pathways from the nested pathways column and create a DataFrame with one row per pathway.
    
    The pathways column contains a list of dictionaries, where each dictionary
    has keys like 'pathway', 'pathwayId', 'name', 'label', 'id'. This function
    extracts pathway information and groups genes by pathway.
    
    Parameters
    ----------
    targets : pd.DataFrame
        DataFrame with a "pathways" column containing nested pathway data and
        a gene symbol column (default: "approvedSymbol").
        Each row's pathways column should be either:
        - None/NaN: no pathways
        - A list of dictionaries with pathway information
        - A list of strings (less common)
    gene_symbol_col : str, default "approvedSymbol"
        Column name containing gene symbols to associate with each pathway.
    gene_id_col : str, default "id"
        Column name containing gene IDs to associate with each pathway.
    save_path : str, optional
        If provided, save the resulting DataFrame to this parquet file path.
        The DataFrame will be saved with all columns including nested structures (lists).
        If None (default), the DataFrame is not saved.
    force : bool, default False
        If True, force regeneration even if save_path already exists.
        If False and save_path exists, load the existing parquet file instead of regenerating.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per unique pathway. Columns:
        - pathway: pathway name/identifier (str)
        - pathwayId: pathway ID if available (str, may be NaN)
        - gene_symbols: list of unique gene symbols associated with this pathway (list)
        - gene_ids: list of unique gene IDs associated with this pathway (list)
        - gene_count: number of unique genes associated with this pathway (int)
        - markdown: pathway-centric markdown description (str)
        - tokens: approximate token count in markdown (int)
    
    Examples
    --------
    >>> import phenoref.opentargets as ot
    >>> targets = ot.get_targets(limit=100)
    >>> pathways_df = ot.get_pathways(targets)
    >>> # Get pathways with most genes
    >>> pathways_df.sort_values('gene_count', ascending=False).head()
    >>> # Get all gene symbols for a specific pathway
    >>> pathway_genes = pathways_df[pathways_df['pathway'] == 'Some Pathway']['gene_symbols'].iloc[0]
    >>> # Save to file
    >>> pathways_df = ot.get_pathways(targets, save_path="pathways.parquet")
    >>> # Force regeneration even if file exists
    >>> pathways_df = ot.get_pathways(targets, save_path="pathways.parquet", force=True)
    """
    # Check if file exists and force is False - if so, just load and return it
    if save_path is not None and not force:
        save_path_obj = Path(save_path)
        if save_path_obj.exists():
            print(f"Loading existing pathways DataFrame from: {save_path}")
            try:
                pathways_df = pd.read_parquet(save_path)
                print(f"  Loaded {len(pathways_df)} pathways from existing file")
                print(f"\nNumber of unique pathways: {len(pathways_df)}")
                return pathways_df
            except Exception as e:
                print(f"  Error loading existing file: {e}")
                print(f"  Will regenerate pathways DataFrame...")
    
    if "pathways" not in targets.columns:
        raise ValueError("DataFrame must have a 'pathways' column")
    
    if gene_symbol_col not in targets.columns:
        raise ValueError(f"DataFrame must have a '{gene_symbol_col}' column")
    
    include_gene_ids = gene_id_col in targets.columns
    
    def extract_pathway_info(item):
        """Extract pathway information from a single item (dict or string)."""
        if item is None:
            return None, None
        
        # If it's already a string, use it as pathway name
        if isinstance(item, str):
            return item, None
        
        # If it's a dict, extract pathway name and ID
        if isinstance(item, dict):
            pathway_name = None
            pathway_id = None
            
            # Try to extract pathway name (preferred keys in order)
            for key in ['pathway', 'name', 'label']:
                if key in item and item[key] is not None:
                    val = item[key]
                    pathway_name = str(val) if not isinstance(val, str) else val
                    break
            
            # Try to extract pathway ID
            for key in ['pathwayId', 'id']:
                if key in item and item[key] is not None:
                    val = item[key]
                    pathway_id = str(val) if not isinstance(val, str) else val
                    break
            
            # If no pathway name found, use ID or string representation
            if pathway_name is None:
                if pathway_id is not None:
                    pathway_name = pathway_id
                else:
                    pathway_name = str(item)
            
            return pathway_name, pathway_id
        
        # For other types, convert to string
        return str(item), None
    
    def extract_pathways_from_row(pathways_value):
        """Extract all pathways from a single row's pathways value."""
        if pathways_value is None or (isinstance(pathways_value, float) and pd.isna(pathways_value)):
            return []
        
        # Handle string representation of list/dict (JSON)
        if isinstance(pathways_value, str):
            try:
                pathways_value = json.loads(pathways_value)
            except (json.JSONDecodeError, TypeError):
                # If it's not JSON, treat as a single pathway string
                pathway_name, pathway_id = extract_pathway_info(pathways_value)
                return [(pathway_name, pathway_id)] if pathway_name is not None else []
        
        # Convert numpy array to list if needed
        if isinstance(pathways_value, np.ndarray):
            pathways_value = pathways_value.tolist()
        
        # If it's not a list, wrap it
        if not isinstance(pathways_value, list):
            pathway_name, pathway_id = extract_pathway_info(pathways_value)
            return [(pathway_name, pathway_id)] if pathway_name is not None else []
        
        # Extract pathways from list
        extracted = []
        for item in pathways_value:
            pathway_name, pathway_id = extract_pathway_info(item)
            if pathway_name is not None:
                extracted.append((pathway_name, pathway_id))
        
        return extracted
    
    # Build pathway -> genes mapping
    pathway_to_genes = {}
    
    for idx, row in targets.iterrows():
        gene_symbol = row[gene_symbol_col]
        
        # Skip if gene symbol is missing
        if pd.isna(gene_symbol) or gene_symbol is None:
            continue
        
        # Convert to string
        gene_symbol = str(gene_symbol)
        
        # Get gene ID if available
        gene_id = None
        if include_gene_ids:
            gene_id_val = row[gene_id_col]
            if not (pd.isna(gene_id_val) or gene_id_val is None):
                gene_id = str(gene_id_val)
        
        # Extract pathways for this gene
        pathways_info = extract_pathways_from_row(row["pathways"])
        
        # Add gene to each pathway
        for pathway_name, pathway_id in pathways_info:
            if pathway_name not in pathway_to_genes:
                pathway_to_genes[pathway_name] = {
                    'pathwayId': pathway_id,
                    'gene_symbols': set(),
                    'gene_ids': set()
                }
            pathway_to_genes[pathway_name]['gene_symbols'].add(gene_symbol)
            if gene_id is not None:
                pathway_to_genes[pathway_name]['gene_ids'].add(gene_id)
    
    # Convert to DataFrame
    rows = []
    for pathway_name, pathway_data in pathway_to_genes.items():
        gene_symbols_list = sorted(list(pathway_data['gene_symbols']))  # Sort for consistency
        gene_ids_list = sorted(list(pathway_data['gene_ids'])) if include_gene_ids else []
        
        row_dict = {
            'pathway': pathway_name,
            'pathwayId': pathway_data['pathwayId'],
            'gene_symbols': gene_symbols_list,
            'gene_count': len(gene_symbols_list)
        }
        
        if include_gene_ids:
            row_dict['gene_ids'] = gene_ids_list
        
        rows.append(row_dict)
    
    pathways_df = pd.DataFrame(rows)
    
    # Sort by gene count (descending) then by pathway name for consistency
    pathways_df = pathways_df.sort_values(['gene_count', 'pathway'], ascending=[False, True]).reset_index(drop=True)
    
    # Create pathway-centric markdown
    def pathway_to_markdown(row):
        """Create markdown for a pathway row."""
        lines = []
        
        # Pathway info
        lines.append("# Pathway Info")
        lines.append(f"Pathway: {row['pathway']}")
        if pd.notna(row['pathwayId']):
            lines.append(f"Pathway ID: {row['pathwayId']}")
        lines.append(f"Number of genes: {row['gene_count']}")
        lines.append("")
        
        # Associated genes
        if row['gene_symbols']:
            lines.append("## Associated Genes")
            lines.append(f"Gene symbols: {', '.join(row['gene_symbols'])}")
            if 'gene_ids' in row and row['gene_ids']:
                lines.append(f"Gene IDs: {', '.join(row['gene_ids'])}")
            lines.append("")
        
        return "\n".join(lines)
    
    # Add markdown column
    pathways_df['markdown'] = pathways_df.apply(pathway_to_markdown, axis=1)
    
    # Add tokens column
    try:
        from .utils import count_tokens
        pathways_df['tokens'] = pathways_df['markdown'].apply(lambda x: count_tokens(x, approximate=True))
    except (ImportError, AttributeError):
        # Fallback if count_tokens is not available
        pathways_df['tokens'] = pathways_df['markdown'].apply(lambda x: len(x) // 4)
    
    pathways_df.index = pathways_df.pathwayId.tolist()

    # Save to parquet if save_path is provided
    if save_path is not None:
        # Ensure the directory exists
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet (parquet format handles nested structures like lists automatically)
        pathways_df.to_parquet(save_path, index=True)  # Keep index=True since we set pathwayId as index
        
        print(f"\nSaved pathways DataFrame to: {save_path}")
        print(f"  Shape: {pathways_df.shape}")
        print(f"  Columns: {list(pathways_df.columns)}")
    
    print(f"\nNumber of unique pathways: {len(pathways_df)}")
    
    return pathways_df

