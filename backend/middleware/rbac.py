# =====================================================
# middleware/rbac.py
# UPDATED for ES256 (Elliptic Curve JWT signing)
# Supabase newer projects use ES256 instead of HS256.
#
# HOW ES256 WORKS (vs HS256):
# HS256 = one shared secret (symmetric)
#   → same key signs AND verifies
# ES256 = public/private key pair (asymmetric)
#   → Supabase signs with PRIVATE key (we never see it)
#   → We verify with PUBLIC key (fetched from Supabase)
#
# We fetch the public key from Supabase's JWKS endpoint:
# https://your-project.supabase.co/auth/v1/.well-known/jwks.json
# =====================================================

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from jose.backends import ECKey
from loguru import logger
from typing import Optional
import httpx
import json

from ..core.config import settings
from ..core.supabase import supabase_admin
from ..models.schemas import RoleType, UserProfile


bearer_scheme = HTTPBearer(auto_error=False)

ROLE_PRIORITY: dict[RoleType, int] = {
    RoleType.intern:   1,
    RoleType.employee: 2,
    RoleType.manager:  3,
    RoleType.admin:    4,
}

# Cache the public keys so we don't fetch on every request
_jwks_cache: dict = {}


def get_jwks() -> dict:
    """
    Fetch Supabase's public JSON Web Key Set (JWKS).
    This contains the public key used to verify ES256 tokens.

    WHY cache it?
    The JWKS endpoint rarely changes. Fetching it on every
    request would be slow and unnecessary.
    We cache it in memory for the lifetime of the server.
    """
    global _jwks_cache

    if _jwks_cache:
        return _jwks_cache

    jwks_url = f"{settings.supabase_auth_url}/.well-known/jwks.json"

    try:
        response = httpx.get(jwks_url, timeout=10.0)
        response.raise_for_status()
        _jwks_cache = response.json()
        logger.info(f"JWKS fetched successfully from {jwks_url}")
        return _jwks_cache

    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not fetch authentication keys. Try again."
        )


def get_public_key(kid: str) -> str:
    """
    Find the public key matching the token's key ID (kid).

    WHY kid?
    Supabase can have multiple keys (for key rotation).
    The token header contains a 'kid' (key ID) that tells
    us WHICH public key to use for verification.
    """
    jwks = get_jwks()
    keys = jwks.get("keys", [])

    for key in keys:
        if key.get("kid") == kid:
            # Convert JWK format to PEM format that jose can use
            return key

    # Key not found — clear cache and retry once (key rotation)
    global _jwks_cache
    _jwks_cache = {}
    jwks = get_jwks()
    keys = jwks.get("keys", [])

    for key in keys:
        if key.get("kid") == kid:
            return key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token signing key not found."
    )


def decode_supabase_token(token: str) -> dict:
    """
    Decode and verify a Supabase JWT token.
    Handles both ES256 (new) and HS256 (old) Supabase projects.

    Returns the decoded payload dict.
    """
    # Step 1: Peek at the header without verifying
    # This tells us which algorithm and key to use
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token."
        )

    alg = unverified_header.get("alg", "HS256")
    kid = unverified_header.get("kid")

    # Step 2: Decode based on algorithm
    try:
        if alg == "ES256":
            # NEW Supabase: use public key from JWKS
            public_key_data = get_public_key(kid)

            # Convert JWK dict to a key object jose understands
            from cryptography.hazmat.primitives.asymmetric.ec import (
                EllipticCurvePublicKey
            )
            from jwt.algorithms import ECAlgorithm  # PyJWT helper
            import json as _json

            # Use python-jose's built-in JWK support
            payload = jwt.decode(
                token,
                public_key_data,          # pass the JWK dict directly
                algorithms=["ES256"],
                options={"verify_aud": False}
            )

        else:
            # OLD Supabase: use JWT secret (HS256)
            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256", "HS384", "HS512"],
                options={"verify_aud": False}
            )

        return payload

    except JWTError as e:
        logger.warning(f"Token decode failed ({alg}): {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---- Core JWT Verification ----

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> UserProfile:
    """
    Dependency: Verifies JWT and returns the current user's profile.
    Now supports both ES256 (new Supabase) and HS256 (old Supabase).
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Decode and verify the token
    payload = decode_supabase_token(token)

    # Extract user ID
    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user identity.",
        )

    # Fetch fresh profile from DB
    try:
        result = supabase_admin.table("profiles")\
            .select("*")\
            .eq("id", user_id)\
            .single()\
            .execute()

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User profile not found.",
            )

        profile_data = result.data

        if not profile_data.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated. Contact admin.",
            )

        return UserProfile(**profile_data)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error fetching profile for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not verify user profile.",
        )


# ---- Role-Based Guards ----

def require_role(minimum_role: RoleType):
    """
    Dependency factory: requires minimum role level.
    Hierarchical — higher roles include lower ones.
    """
    async def role_checker(
        current_user: UserProfile = Depends(get_current_user)
    ) -> UserProfile:
        user_priority = ROLE_PRIORITY.get(current_user.role, 0)
        required_priority = ROLE_PRIORITY.get(minimum_role, 0)

        if user_priority < required_priority:
            logger.warning(
                f"Access denied | user={current_user.email} "
                f"role={current_user.role} required={minimum_role}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {minimum_role} role or higher. "
                       f"Your role: {current_user.role}",
            )

        return current_user
    return role_checker


require_employee = require_role(RoleType.employee)
require_manager  = require_role(RoleType.manager)
require_admin    = require_role(RoleType.admin)


def check_data_access(
    user: UserProfile,
    data_role_access: list[str],
    data_department: str
) -> bool:
    """
    Application-level RBAC check for retrieved data chunks.
    Second security layer — RLS in the DB is the first.
    """
    user_priority = ROLE_PRIORITY.get(user.role, 0)

    role_allowed = any(
        ROLE_PRIORITY.get(RoleType(r), 0) <= user_priority
        for r in data_role_access
    )

    dept_allowed = (
        data_department == "general"
        or data_department == user.department
    )

    return role_allowed and dept_allowed