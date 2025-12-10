import io
from typing import Optional

import boto3
import pandas as pd
from botocore.config import Config
from loguru import logger

from src.config.settings import get_settings


class MassiveClient:
    def __init__(self):
        self.settings = get_settings()
        if not self.settings.massive_access_key or not self.settings.massive_secret_key:
            logger.warning("Massive S3 credentials not found. Skipping Massive client init.")
            self.s3 = None
            return

        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.settings.massive_endpoint_url,
            aws_access_key_id=self.settings.massive_access_key,
            aws_secret_access_key=self.settings.massive_secret_key,
            config=Config(signature_version="s3v4"),
        )
        self.bucket = "flatfiles"

    def list_files(self, prefix: str = "equity/minute/"):
        """List available files in the bucket.
        Example prefix: 'equity/minute/' or 'crypto/trades/'
        """
        if not self.s3:
            return []

        try:
            # Handle pagination for large lists (Massive has thousands of files)
            files = []
            paginator = self.s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" in page:
                    files.extend([content["Key"] for content in page["Contents"]])
            return files
        except Exception as e:
            logger.error(f"Failed to list Massive files: {e}")
            return []

    def fetch_file(self, key: str) -> Optional[pd.DataFrame]:
        """Download and parse a CSV file from Massive S3.
        """
        if not self.s3:
            return None

        try:
            logger.info(f"Downloading {key} from Massive S3...")
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))

            # Massive CSV format standardizer
            # ticker,volume,open,close,high,low,window_start,transactions
            if "window_start" in df.columns:
                # Convert nanoseconds to datetime
                df["timestamp"] = pd.to_datetime(df["window_start"], unit="ns")

            df["source"] = "massive"
            return df

        except Exception as e:
            logger.error(f"Failed to download/parse {key}: {e}")
            return None

# Global instance for import
massive_client = MassiveClient()

