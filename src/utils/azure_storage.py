import os
import json
import logging
import pickle
from typing import Any, Dict, Optional, List
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

logger = logging.getLogger("ALC.AzureStorage")

class AzureBlobStorage:
    """
    Azure Blob Storage wrapper for agent persistence.
    """
    def __init__(self):
        try:
            from config import secrets
            connection_string = getattr(secrets, 'AZURE_STORAGE_CONNECTION_STRING', None)
            account_name = getattr(secrets, 'AZURE_STORAGE_ACCOUNT', None)
            account_key = getattr(secrets, 'AZURE_STORAGE_KEY', None)

            if connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            elif account_name and account_key:
                self.blob_service_client = BlobServiceClient(
                    account_url=f"https://{account_name}.blob.core.windows.net",
                    credential=account_key
                )
            else:
                self.blob_service_client = None
                logger.warning("Azure Storage credentials not found. Persistence disabled.")
        except ImportError:
            self.blob_service_client = None
            logger.warning("Config module not found. Persistence disabled.")
        except Exception as e:
            self.blob_service_client = None
            logger.error(f"Failed to initialize Azure Storage: {e}")

    def save_object(self, container_name: str, blob_name: str, data: Any) -> bool:
        """Save a Python object to Azure Blob Storage using pickle."""
        if not self.blob_service_client:
            return False
        
        try:
            # Create container if it doesn't exist
            try:
                container_client = self.blob_service_client.create_container(container_name)
            except ResourceExistsError:
                container_client = self.blob_service_client.get_container_client(container_name)

            blob_client = container_client.get_blob_client(blob_name)
            
            # Pickle the data
            pickled_data = pickle.dumps(data)
            
            blob_client.upload_blob(pickled_data, overwrite=True)
            return True
        except Exception as e:
            logger.error(f"Failed to save object {blob_name} to {container_name}: {e}")
            return False

    def load_object(self, container_name: str, blob_name: str) -> Optional[Any]:
        """Load a Python object from Azure Blob Storage."""
        if not self.blob_service_client:
            return None
            
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                return None
                
            blob_client = container_client.get_blob_client(blob_name)
            if not blob_client.exists():
                return None
                
            download_stream = blob_client.download_blob()
            data = pickle.loads(download_stream.readall())
            return data
        except Exception as e:
            logger.error(f"Failed to load object {blob_name} from {container_name}: {e}")
            return None

    def list_blobs(self, container_name: str) -> List[str]:
        """List blobs in a container."""
        if not self.blob_service_client:
            return []
            
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                return []
            return [b.name for b in container_client.list_blobs()]
        except Exception as e:
            logger.error(f"Failed to list blobs in {container_name}: {e}")
            return []

# Singleton instance
azure_storage = AzureBlobStorage()
