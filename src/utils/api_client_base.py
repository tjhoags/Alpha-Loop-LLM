"""
Standardized API Client Base Class
==================================
Provides consistent patterns for API calls across all data sources.

Author: Tom Hogan | Alpha Loop Capital, LLC

This base class standardizes:
- Error handling
- Retry logic
- Rate limiting
- Logging
- Timeout handling
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class APIClientBase(ABC):
    """Base class for all API clients.
    
    Provides standardized:
    - Retry logic with exponential backoff
    - Rate limiting
    - Error handling
    - Logging
    - Timeout management
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit_delay: float = 0.25,
    ):
        """Initialize API client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum retry attempts (default: 3)
            rate_limit_delay: Delay between requests in seconds (default: 0.25)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time: float = 0.0
        self.request_count: int = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        reraise=True,
    )
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """Make HTTP request with standardized error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            headers: Request headers
            json_data: JSON body for POST requests
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If request fails after retries
        """
        # Rate limiting
        self._rate_limit()

        # Build URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # Add API key to params if not in headers
        if params is None:
            params = {}
        if headers is None:
            headers = {}

        # Standardized headers
        headers.setdefault("User-Agent", "Alpha-Loop-Capital/1.0")
        headers.setdefault("Accept", "application/json")

        # Add API key (standardized pattern)
        if "Authorization" not in headers and "apiKey" not in params:
            params["apiKey"] = self.api_key

        # Log request
        logger.debug(f"API Request: {method} {url}")

        try:
            # Make request
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(
                    url, params=params, headers=headers, json=json_data, timeout=self.timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Check for API-specific errors
            self._check_api_errors(response)

            # Increment request count
            self.request_count += 1

            return response

        except requests.Timeout as e:
            logger.error(f"API request timeout: {url}")
            raise
        except requests.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            raise

    def _check_api_errors(self, response: requests.Response) -> None:
        """Check for API-specific error responses.
        
        Override in subclasses for API-specific error handling.
        
        Args:
            response: Response object
            
        Raises:
            ValueError: If API returns an error
        """
        response.raise_for_status()

        # Check for JSON error responses
        try:
            data = response.json()
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "error" in data:
                error_msg = data["error"]
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get("message", str(error_msg))
                raise ValueError(f"API Error: {error_msg}")
        except ValueError:
            raise
        except Exception:
            # Not JSON or no error field, continue
            pass

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        """Make GET request.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response object
        """
        return self._make_request("GET", endpoint, params=params, headers=headers)

    def post(self, endpoint: str, json_data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        """Make POST request.
        
        Args:
            endpoint: API endpoint
            json_data: JSON body
            params: Query parameters
            headers: Request headers
            
        Returns:
            Response object
        """
        return self._make_request("POST", endpoint, params=params, headers=headers, json_data=json_data)

    @abstractmethod
    def test_connection(self) -> bool:
        """Test API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get API client statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "rate_limit_delay": self.rate_limit_delay,
            "request_count": self.request_count,
            "api_key_present": bool(self.api_key),
        }

