"""================================================================================
AZURE STORAGE CLIENT - Agent Memory & Model Persistence
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Handles persistent storage of agent state and models to Azure Blob Storage.
Enables cross-machine training (Lenovo + MacBook Pro) without data loss.

CONTAINERS:
- agent-memory: Agent persistent state (learning outcomes, beliefs, etc.)
- models: Trained ML models (.pkl files)
- data-cache: Cached market data
- logs: Training and execution logs

PHILOSOPHY:
"Parse and combine data, not overwrite."
Each machine writes to its own namespace, then we merge on load.
================================================================================
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import Azure SDK
try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    logger.warning("Azure SDK not installed. Run: pip install azure-storage-blob")


class AzureStorageClient:
    """Azure Blob Storage client for Alpha Loop.

    Handles all persistent storage needs:
    - Agent state persistence
    - Model storage
    - Data caching
    - Log archival

    USAGE:
        from src.utils.azure_storage import azure_storage

        # Save an object
        azure_storage.save_object("agent-memory", "my_agent_state.pkl", state_dict)

        # Load an object
        state = azure_storage.load_object("agent-memory", "my_agent_state.pkl")

        # List blobs
        blobs = azure_storage.list_blobs("agent-memory")
    """

    # Default containers
    CONTAINERS = [
        "agent-memory",  # Agent persistent state
        "models",        # Trained ML models
        "data-cache",    # Cached market data
        "logs",          # Training/execution logs
        "checkpoints",   # Training checkpoints
    ]

    def __init__(self, connection_string: str = None):
        """Initialize Azure Storage client.

        Args:
        ----
            connection_string: Azure Storage connection string.
                              If None, tries environment variable.
        """
        self.logger = logging.getLogger(__name__)
        self._connected = False
        self._service_client = None

        # Try to get connection string
        if connection_string:
            self._connection_string = connection_string
        else:
            self._connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

        if not self._connection_string:
            self.logger.warning(
                "Azure Storage connection string not found. "
                "Set AZURE_STORAGE_CONNECTION_STRING environment variable.",
            )
            return

        if not AZURE_SDK_AVAILABLE:
            self.logger.warning("Azure SDK not available. Storage operations will be no-ops.")
            return

        try:
            self._service_client = BlobServiceClient.from_connection_string(
                self._connection_string,
            )
            self._connected = True
            self.logger.info("Azure Storage client connected successfully")

            # Ensure containers exist
            self._ensure_containers()

        except Exception as e:
            self.logger.error(f"Failed to connect to Azure Storage: {e}")

    def _ensure_containers(self):
        """Create default containers if they don't exist."""
        if not self._connected:
            return

        for container_name in self.CONTAINERS:
            try:
                container_client = self._service_client.get_container_client(container_name)
                if not container_client.exists():
                    container_client.create_container()
                    self.logger.info(f"Created container: {container_name}")
            except Exception as e:
                self.logger.warning(f"Could not create container {container_name}: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if connected to Azure."""
        return self._connected

    def save_object(
        self,
        container: str,
        blob_name: str,
        obj: Any,
        overwrite: bool = True,
    ) -> bool:
        """Save a Python object to Azure Blob Storage.

        Args:
        ----
            container: Container name
            blob_name: Blob name (filename)
            obj: Python object to save (must be pickle-able)
            overwrite: Whether to overwrite existing blob

        Returns:
        -------
            True if successful, False otherwise
        """
        if not self._connected:
            self.logger.debug(f"Not connected, skipping save: {blob_name}")
            return False

        try:
            container_client = self._service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)

            # Serialize object
            data = pickle.dumps(obj)

            # Upload
            blob_client.upload_blob(data, overwrite=overwrite)
            self.logger.debug(f"Saved {blob_name} to {container}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save {blob_name}: {e}")
            return False

    def load_object(
        self,
        container: str,
        blob_name: str,
    ) -> Optional[Any]:
        """Load a Python object from Azure Blob Storage.

        Args:
        ----
            container: Container name
            blob_name: Blob name (filename)

        Returns:
        -------
            Deserialized Python object, or None if not found/error
        """
        if not self._connected:
            self.logger.debug(f"Not connected, skipping load: {blob_name}")
            return None

        try:
            container_client = self._service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)

            if not blob_client.exists():
                return None

            # Download
            data = blob_client.download_blob().readall()

            # Deserialize
            obj = pickle.loads(data)
            self.logger.debug(f"Loaded {blob_name} from {container}")
            return obj

        except Exception as e:
            self.logger.error(f"Failed to load {blob_name}: {e}")
            return None

    def save_json(
        self,
        container: str,
        blob_name: str,
        data: Union[Dict, List],
        overwrite: bool = True,
    ) -> bool:
        """Save JSON data to Azure Blob Storage.

        Args:
        ----
            container: Container name
            blob_name: Blob name (should end in .json)
            data: JSON-serializable data
            overwrite: Whether to overwrite existing blob

        Returns:
        -------
            True if successful, False otherwise
        """
        if not self._connected:
            return False

        try:
            container_client = self._service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)

            json_data = json.dumps(data, indent=2, default=str)
            blob_client.upload_blob(json_data.encode("utf-8"), overwrite=overwrite)
            return True

        except Exception as e:
            self.logger.error(f"Failed to save JSON {blob_name}: {e}")
            return False

    def load_json(
        self,
        container: str,
        blob_name: str,
    ) -> Optional[Union[Dict, List]]:
        """Load JSON data from Azure Blob Storage.

        Args:
        ----
            container: Container name
            blob_name: Blob name

        Returns:
        -------
            Parsed JSON data, or None if not found/error
        """
        if not self._connected:
            return None

        try:
            container_client = self._service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)

            if not blob_client.exists():
                return None

            data = blob_client.download_blob().readall()
            return json.loads(data.decode("utf-8"))

        except Exception as e:
            self.logger.error(f"Failed to load JSON {blob_name}: {e}")
            return None

    def list_blobs(
        self,
        container: str,
        prefix: str = None,
    ) -> List[str]:
        """List blobs in a container.

        Args:
        ----
            container: Container name
            prefix: Optional prefix filter

        Returns:
        -------
            List of blob names
        """
        if not self._connected:
            return []

        try:
            container_client = self._service_client.get_container_client(container)

            if prefix:
                blobs = container_client.list_blobs(name_starts_with=prefix)
            else:
                blobs = container_client.list_blobs()

            return [blob.name for blob in blobs]

        except Exception as e:
            self.logger.error(f"Failed to list blobs in {container}: {e}")
            return []

    def delete_blob(
        self,
        container: str,
        blob_name: str,
    ) -> bool:
        """Delete a blob.

        Args:
        ----
            container: Container name
            blob_name: Blob name

        Returns:
        -------
            True if successful, False otherwise
        """
        if not self._connected:
            return False

        try:
            container_client = self._service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete {blob_name}: {e}")
            return False

    def blob_exists(
        self,
        container: str,
        blob_name: str,
    ) -> bool:
        """Check if a blob exists.

        Args:
        ----
            container: Container name
            blob_name: Blob name

        Returns:
        -------
            True if exists, False otherwise
        """
        if not self._connected:
            return False

        try:
            container_client = self._service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)
            return blob_client.exists()

        except Exception:
            return False

    def get_blob_url(
        self,
        container: str,
        blob_name: str,
    ) -> Optional[str]:
        """Get the URL for a blob.

        Args:
        ----
            container: Container name
            blob_name: Blob name

        Returns:
        -------
            Blob URL or None
        """
        if not self._connected:
            return None

        try:
            container_client = self._service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)
            return blob_client.url

        except Exception:
            return None


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Create singleton instance
azure_storage = AzureStorageClient()


def get_azure_storage() -> AzureStorageClient:
    """Get Azure storage client singleton."""
    return azure_storage


