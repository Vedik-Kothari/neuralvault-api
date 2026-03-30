# =====================================================
# core/supabase.py
# Supabase client setup.
#
# TWO clients — this is critical to understand:
#
# 1. ANON CLIENT (supabase_client):
#    Uses the public anon key.
#    Respects RLS policies — users can only see their data.
#    Used for: login, signup, user-facing queries.
#
# 2. SERVICE CLIENT (supabase_admin):
#    Uses the secret service_role key.
#    BYPASSES RLS — can read/write everything.
#    Used for: inserting embeddings, writing audit logs,
#              admin operations.
#    ⚠️ NEVER expose this key to the frontend.
#    ⚠️ NEVER use this for user queries.
# =====================================================

from supabase import create_client, Client
from loguru import logger
from .config import settings


def get_supabase_client() -> Client:
    """
    Anon client — respects RLS.
    Use this for all user-facing operations.
    """
    client = create_client(
        settings.supabase_url,
        settings.supabase_anon_key
    )
    return client


def get_supabase_admin() -> Client:
    """
    Service role client — bypasses RLS.
    Use ONLY for backend operations (ingestion, audit logging).
    This is the 'god mode' client — handle with care.
    """
    client = create_client(
        settings.supabase_url,
        settings.supabase_service_key
    )
    return client


# Singleton instances (created once at import time)
# FastAPI routes can import these directly
supabase_client: Client = get_supabase_client()
supabase_admin: Client = get_supabase_admin()

logger.info("Supabase clients initialized")