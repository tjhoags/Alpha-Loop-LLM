"""================================================================================
UTILITIES MODULE
================================================================================
Author: Tom Hogan | Alpha Loop Capital, LLC

Common utilities used across the Alpha Loop system.
================================================================================
"""

from .azure_storage import AzureStorageClient, azure_storage

__all__ = ["azure_storage", "AzureStorageClient"]


