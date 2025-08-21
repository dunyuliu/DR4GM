#!/usr/bin/env python3

"""
Cloud Data Handler for Ground Motion Explorer

Handles loading datasets from various cloud storage providers.
Supports AWS S3, Google Cloud Storage, and direct URL downloads.
"""

import streamlit as st
import numpy as np
import os
import requests
from pathlib import Path
import boto3
from typing import List, Optional
import tempfile

class CloudDataHandler:
    """Handle cloud-based dataset access"""
    
    def __init__(self):
        self.cache_dir = Path(tempfile.gettempdir()) / "gm_data_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    @st.cache_data
    def download_from_url(_self, url: str, filename: str) -> str:
        """Download NPZ file from direct URL"""
        cache_path = _self.cache_dir / filename
        
        if cache_path.exists():
            return str(cache_path)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return str(cache_path)
        except Exception as e:
            st.error(f"Failed to download {url}: {e}")
            return ""
    
    def list_s3_files(self, bucket: str, prefix: str = "") -> List[str]:
        """List NPZ files in S3 bucket"""
        try:
            s3 = boto3.client('s3')
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            
            npz_files = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith('.npz'):
                    npz_files.append(f"s3://{bucket}/{obj['Key']}")
            
            return npz_files
        except Exception as e:
            st.error(f"Failed to list S3 files: {e}")
            return []
    
    @st.cache_data
    def download_from_s3(_self, s3_path: str) -> str:
        """Download NPZ file from S3"""
        # Parse S3 path
        parts = s3_path.replace('s3://', '').split('/')
        bucket = parts[0]
        key = '/'.join(parts[1:])
        filename = os.path.basename(key)
        
        cache_path = _self.cache_dir / filename
        
        if cache_path.exists():
            return str(cache_path)
        
        try:
            s3 = boto3.client('s3')
            s3.download_file(bucket, key, str(cache_path))
            return str(cache_path)
        except Exception as e:
            st.error(f"Failed to download from S3: {e}")
            return ""

# Example dataset configurations
EXAMPLE_DATASETS = {
    "EQDyna Sample": {
        "url": "https://your-domain.com/data/eqdyna_sample.npz",
        "description": "Sample EQDyna ground motion data"
    },
    "Waveqlab3D Sample": {
        "url": "https://your-domain.com/data/waveqlab3d_sample.npz", 
        "description": "Sample Waveqlab3D ground motion data"
    },
    "FD3D Sample": {
        "url": "https://your-domain.com/data/fd3d_sample.npz",
        "description": "Sample FD3D ground motion data"
    }
}