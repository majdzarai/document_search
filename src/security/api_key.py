"""
API Key Validation Module
=========================

Simple API key storage and validation for multi-tenant authentication.

HOW IT WORKS:
-------------
1. API keys are stored in the API_KEYS environment variable
2. Format: "sk-key1:tenant_id1,sk-key2:tenant_id2"
3. When a request comes in, we extract the key from the Authorization header
4. We look up the tenant_id for that key
5. If valid, the request proceeds with the tenant context

WHY ENV VARS?
-------------
- No database needed
- Easy to manage via Railway dashboard
- Simple to rotate keys (just update env var and restart)
- Secure (not in code)

EXAMPLE:
--------
Set environment variable:
    API_KEYS=sk-acme123:acme_corp,sk-beta456:beta_inc

Then requests with "Authorization: Bearer sk-acme123" will be
authenticated as tenant "acme_corp".
"""

import os
from typing import Optional, Dict


def _load_api_keys() -> Dict[str, str]:
    """
    Load API keys from the API_KEYS environment variable.

    Format: "key1:tenant1,key2:tenant2"

    Returns:
        Dictionary mapping API key -> tenant_id
    """
    keys_str = os.getenv("API_KEYS", "")
    if not keys_str:
        return {}

    keys = {}
    for pair in keys_str.split(","):
        pair = pair.strip()
        if ":" in pair:
            key, tenant_id = pair.split(":", 1)
            key = key.strip()
            tenant_id = tenant_id.strip()
            if key and tenant_id:
                keys[key] = tenant_id

    return keys


# Load keys at module import (can be reloaded by calling _load_api_keys())
_api_keys: Dict[str, str] = {}


def get_api_keys() -> Dict[str, str]:
    """
    Get the current API keys mapping.

    This function lazy-loads the keys on first call.

    Returns:
        Dictionary mapping API key -> tenant_id
    """
    global _api_keys
    if not _api_keys:
        _api_keys = _load_api_keys()
        print(f"[APIKey] Loaded {len(_api_keys)} API keys from environment")
    return _api_keys


def reload_api_keys() -> int:
    """
    Reload API keys from environment variable.

    Call this after updating the API_KEYS env var to refresh.

    Returns:
        Number of keys loaded
    """
    global _api_keys
    _api_keys = _load_api_keys()
    print(f"[APIKey] Reloaded {len(_api_keys)} API keys")
    return len(_api_keys)


def validate_api_key(api_key: str) -> Optional[str]:
    """
    Validate an API key and return the associated tenant ID.

    Args:
        api_key: The API key to validate (just the key, not "Bearer ...")

    Returns:
        The tenant_id if the key is valid, None otherwise.

    Example:
        tenant_id = validate_api_key("sk-acme123")
        if tenant_id:
            print(f"Authenticated as {tenant_id}")
        else:
            print("Invalid key")
    """
    keys = get_api_keys()
    return keys.get(api_key)


def is_auth_enabled() -> bool:
    """
    Check if API key authentication is enabled.

    Auth is enabled if API_KEYS environment variable is set and not empty.

    Returns:
        True if auth is enabled, False otherwise.
    """
    keys = get_api_keys()
    return len(keys) > 0
