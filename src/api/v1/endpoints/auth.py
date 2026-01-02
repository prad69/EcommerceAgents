from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional

from src.core.database import get_db
from src.schemas.user import UserCreate, UserLogin, UserResponse, Token
from src.services.auth import AuthService

router = APIRouter()
security = HTTPBearer()
auth_service = AuthService()


@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user
    """
    try:
        new_user = await auth_service.register_user(user, db)
        return UserResponse(
            id=str(new_user.id),
            email=new_user.email,
            username=new_user.username,
            full_name=new_user.full_name,
            is_active=new_user.is_active,
            preferences=new_user.preferences,
            created_at=new_user.created_at,
            last_login=new_user.last_login
        )
    except Exception as e:
        if "already" in str(e):
            raise e  # Re-raise validation errors from auth service
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login(
    user_credentials: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    User login and JWT token generation
    """
    try:
        # Authenticate user
        user = await auth_service.authenticate_user(user_credentials, db)
        
        # Get client info
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Create session and token
        token = await auth_service.create_user_session(
            user, ip_address, user_agent, db
        )
        
        return token
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions from auth service
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    User logout and token invalidation
    """
    try:
        result = await auth_service.logout_user(credentials.credentials, db)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Get current user information
    """
    try:
        user = await auth_service.get_current_user(credentials.credentials, db)
        return UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            preferences=user.preferences,
            created_at=user.created_at,
            last_login=user.last_login
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions from auth service
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Refresh access token
    """
    try:
        token = await auth_service.refresh_token(credentials.credentials, db)
        return token
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.get("/sessions")
async def get_user_sessions(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    include_inactive: bool = False
):
    """
    Get user's active sessions
    """
    try:
        user = await auth_service.get_current_user(credentials.credentials, db)
        sessions = await auth_service.get_user_sessions(
            str(user.id), db, include_inactive
        )
        
        return {
            "sessions": [
                {
                    "id": str(session.id),
                    "ip_address": session.ip_address,
                    "user_agent": session.user_agent,
                    "created_at": session.created_at,
                    "expires_at": session.expires_at,
                    "is_active": session.is_active,
                    "is_current": session.session_token == credentials.credentials
                }
                for session in sessions
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get sessions"
        )


@router.post("/revoke-all-sessions")
async def revoke_all_sessions(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    except_current: bool = True
):
    """
    Revoke all user sessions (except current one by default)
    """
    try:
        user = await auth_service.get_current_user(credentials.credentials, db)
        
        revoked_count = await auth_service.revoke_all_sessions(
            str(user.id),
            except_token=credentials.credentials if except_current else None,
            db=db
        )
        
        return {
            "message": f"Revoked {revoked_count} sessions",
            "revoked_count": revoked_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke sessions"
        )