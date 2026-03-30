# =====================================================
# routers/auth.py
# Authentication endpoints.
#
# Endpoints:
#   POST /auth/signup  → create account
#   POST /auth/login   → get JWT token
#   GET  /auth/me      → get current user profile
#   POST /auth/logout  → invalidate session
# =====================================================

from fastapi import APIRouter, HTTPException, status, Depends
from loguru import logger

from ..core.supabase import supabase_client, supabase_admin
from ..models.schemas import (
    SignUpRequest, LoginRequest,
    TokenResponse, UserProfile, RoleType
)
from ..middleware.rbac import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/signup",
    response_model=UserProfile,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account"
)
async def signup(request: SignUpRequest):
    """
    Register a new user.
    - Default role is 'intern' (least privilege)
    - Admin must manually promote roles via Supabase dashboard
    - Profile row is auto-created by the DB trigger

    SECURITY: New users always start as 'intern'.
    No self-promotion is possible.
    """
    try:
        # Create user in Supabase Auth
        # The handle_new_user trigger will create the profile row
        response = supabase_client.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    # These go into raw_app_meta_data
                    # The trigger reads them to set role/department
                    "role": "intern",           # Always start as intern
                    "department": request.department,
                }
            }
        })

        if not response.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signup failed. Email may already be registered."
            )

        # Fetch the created profile
        profile = supabase_admin.table("profiles")\
            .select("*")\
            .eq("id", response.user.id)\
            .single()\
            .execute()

        logger.info(f"New user signed up: {request.email}")
        return UserProfile(**profile.data)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Signup error for {request.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signup failed. Please try again."
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive JWT token"
)
async def login(request: LoginRequest):
    """
    Login with email and password.
    Returns a JWT access token to use in subsequent requests.

    The token contains role + department in app_metadata
    (injected by our custom_access_token_hook).

    HOW TO USE THE TOKEN:
    Add to every request header:
        Authorization: Bearer <access_token>
    """
    try:
        response = supabase_client.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password,
        })

        if not response.session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password."
            )

        session = response.session
        logger.info(f"User logged in: {request.email}")

        return TokenResponse(
            access_token=session.access_token,
            token_type="bearer",
            expires_in=session.expires_in or 3600
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Login error for {request.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login failed. Check your credentials."
        )


@router.get(
    "/me",
    response_model=UserProfile,
    summary="Get current user's profile"
)
async def get_my_profile(
    current_user: UserProfile = Depends(get_current_user)
):
    """
    Returns the authenticated user's profile including their role.
    Frontend uses this to show the role badge and control UI.

    Requires: Valid JWT in Authorization header.
    """
    return current_user


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="Logout and invalidate session"
)
async def logout(
    current_user: UserProfile = Depends(get_current_user)
):
    """
    Invalidates the user's Supabase session server-side.
    Frontend should also clear the stored token.
    """
    try:
        supabase_client.auth.sign_out()
        logger.info(f"User logged out: {current_user.email}")
        return {"message": "Logged out successfully."}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Return success anyway — frontend clears token regardless
        return {"message": "Logged out."}