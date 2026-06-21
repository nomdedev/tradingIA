"""
Dashboard Authentication Module

Simple password-based authentication for the Streamlit dashboard.
For production use, consider implementing OAuth or API key authentication.

Usage in dashboard/app.py:
    from dashboard.auth import check_password
    
    if not check_password():
        st.stop()
"""

import streamlit as st
import hashlib
import os
from typing import Optional


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def check_password() -> bool:
    """
    Returns True if the user has entered the correct password.
    
    Environment variables:
        DASHBOARD_PASSWORD: The password required to access the dashboard
        DASHBOARD_AUTH_ENABLED: Set to 'false' to disable authentication
    """
    # Check if auth is disabled
    auth_enabled = os.getenv('DASHBOARD_AUTH_ENABLED', 'true').lower()
    if auth_enabled == 'false':
        return True
    
    # Get password from environment
    correct_password = os.getenv('DASHBOARD_PASSWORD')
    
    if not correct_password:
        # No password set - show warning and allow access
        st.warning("⚠️ No DASHBOARD_PASSWORD set in environment. Set it for production use.")
        return True
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        entered = st.session_state.get("password", "")
        if hash_password(entered) == hash_password(correct_password):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.markdown("## 🔐 Login Required")
    st.text_input(
        "Password",
        type="password",
        on_change=password_entered,
        key="password"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Incorrect password. Please try again.")
    
    return False


def logout():
    """Clear authentication state"""
    if "password_correct" in st.session_state:
        del st.session_state["password_correct"]
    st.rerun()


def show_logout_button():
    """Show logout button in sidebar"""
    if st.session_state.get("password_correct", False):
        if st.sidebar.button("🚪 Logout"):
            logout()


# Session timeout functionality
def check_session_timeout(timeout_minutes: int = 30) -> bool:
    """
    Check if session has timed out.
    
    Args:
        timeout_minutes: Minutes of inactivity before timeout
        
    Returns:
        True if session is still valid
    """
    import time
    
    current_time = time.time()
    last_activity = st.session_state.get("last_activity", current_time)
    
    if current_time - last_activity > timeout_minutes * 60:
        # Session timed out
        st.session_state.clear()
        st.warning("Session timed out. Please login again.")
        return False
    
    # Update last activity
    st.session_state["last_activity"] = current_time
    return True


# API Key authentication (for API access)
def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate API key for programmatic access.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        True if valid
    """
    valid_key = os.getenv('DASHBOARD_API_KEY')
    
    if not valid_key:
        return False
    
    return api_key == valid_key


if __name__ == "__main__":
    # Test the module
    print("Auth module loaded successfully")
    print(f"Password hash example: {hash_password('test123')[:16]}...")
