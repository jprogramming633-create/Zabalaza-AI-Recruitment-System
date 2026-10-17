import datetime
import secrets
import smtplib
import logging
from email.message import EmailMessage
from email.utils import formataddr

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from ..core.dependencies import get_db
from ..core.security import verify_password, get_password_hash, create_access_token, create_refresh_token, verify_token
from ..models import User, UserRole, Candidate, PasswordResetToken
from ..schemas import UserCreate, Token, User as UserSchema
from ..config import settings

logger = logging.getLogger(__name__)


def send_email(to_email: str, subject: str, body: str) -> None:
    """Send an email using SMTP settings from config.

    Falls back to logging the message if SMTP is not configured,
    so the app can run in development without a mail server.
    """
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.info(
            "SMTP not configured. Reset email for %s:\nSubject: %s\nBody: %s",
            to_email, subject, body,
        )
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr(("AI Job Recruitment", settings.SMTP_USER))
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)
        logger.info("Password reset email sent to %s", to_email)
    except Exception as exc:
        logger.exception("Failed to send reset email to %s: %s", to_email, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send reset email",
        )


router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Check if username already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )

    try:
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            role=user_data.role
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        # If candidate, create candidate profile
        if user.role == UserRole.CANDIDATE:
            candidate = Candidate(user_id=user.id)
            db.add(candidate)
            db.commit()

        return user
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with provided email or username already exists"
        )
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while creating user"
        )


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token."""
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == form_data.username) | (User.email == form_data.username)
    ).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    # Create access and refresh tokens
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.get("/me/id")
def get_my_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()), db: Session = Depends(get_db)):
    """Get current user's numeric ID."""
    token = credentials.credentials
    payload = verify_token(token, "access")
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    username = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"id": user.id, "username": user.username, "role": user.role.value}


@router.post("/request-password-reset")
def request_password_reset(email: str = Body(..., embed=True), db: Session = Depends(get_db)):
    """
    Request a password reset. A single-use, time-limited token is generated and
    emailed to the address on file. Only the account owner (who can receive email
    at that address) can complete the reset.
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Do not reveal whether the email exists — generic message prevents enumeration
        return {"message": "If an account exists for this email, a reset link has been sent"}

    # Invalidate any existing unused reset tokens for this user
    db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id,
        PasswordResetToken.used == False,  # noqa: E712
    ).delete(synchronize_session=False)

    # Generate a random, unique, single-use token
    temp_token = secrets.token_urlsafe(32)
    expiry = datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)

    reset_entry = PasswordResetToken(
        user_id=user.id,
        token=temp_token,
        expires_at=expiry,
        used=False,
    )
    db.add(reset_entry)
    db.commit()

    reset_link = f"{settings.API_V1_STR}/auth/reset-password?token={temp_token}"
    email_body = (
        f"Hello {user.full_name or user.username},\n\n"
        f"You requested a password reset for your AI Job Recruitment account.\n"
        f"Click the link below to reset your password (expires in 1 hour):\n"
        f"{reset_link}\n\n"
        f"If you did not request this, you can safely ignore this email.\n"
    )
    send_email(user.email, "Password Reset Request", email_body)

    return {"message": "If an account exists for this email, a reset link has been sent"}


@router.post("/reset-password")
def reset_password(
        token: str = Body(..., embed=True),
        new_password: str = Body(..., embed=True),
        db: Session = Depends(get_db)
):
    """
    Reset a password using a single-use, time-limited token delivered via email.
    The token is invalidated (marked used) after a successful reset.
    """
    reset_entry = db.query(PasswordResetToken).filter(PasswordResetToken.token == token).first()

    if not reset_entry:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    if reset_entry.used:
        raise HTTPException(status_code=400, detail="Reset token has already been used")

    if reset_entry.expires_at < datetime.now(datetime.timezone.utc):
        raise HTTPException(status_code=400, detail="Reset token has expired")

    # Look up the user by the foreign key stored on the token
    user = db.query(User).filter(User.id == reset_entry.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update password
    user.hashed_password = get_password_hash(new_password)

    # Mark the token as used (single-use) instead of deleting it
    reset_entry.used = True

    # Invalidate any other unused tokens for this user (one active reset at a time)
    db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id,
        PasswordResetToken.id != reset_entry.id,
        PasswordResetToken.used == False,  # noqa: E712
    ).delete(synchronize_session=False)

    db.commit()

    return {"message": "Password reset successful"}
