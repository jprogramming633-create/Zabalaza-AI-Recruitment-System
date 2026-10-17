from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, Enum, Boolean, UniqueConstraint, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import enum
from .database import Base


def utc_now():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    RECRUITER = "recruiter"
    CANDIDATE = "candidate"


class JobStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    CLOSED = "closed"
    ON_HOLD = "on_hold"


class ApplicationStatus(str, enum.Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    SHORTLISTED = "shortlisted"
    INTERVIEW = "interview"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    first_name = Column(String(100))
    last_name = Column(String(100))
    role = Column(Enum(UserRole), default=UserRole.CANDIDATE)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    jobs = relationship("Job", back_populates="recruiter")
    candidate_profile = relationship("Candidate", back_populates="user", uselist=False)
    reset_tokens = relationship("PasswordResetToken", back_populates="user")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)

    user = relationship("User", back_populates="reset_tokens")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    requirements = Column(Text, nullable=False)
    location = Column(String(255))
    salary_min = Column(Float)
    salary_max = Column(Float)
    job_type = Column(String(50))  # full-time, part-time, contract, etc.
    experience_level = Column(String(50))  # entry, mid, senior
    status = Column(Enum(JobStatus), default=JobStatus.DRAFT)
    recruiter_id = Column(Integer, ForeignKey("users.id"))

    # AI/ML fields
    embedding = Column(Text)  # JSON string of job description embedding
    skills = Column(Text)  # JSON array of required skills

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    recruiter = relationship("User", back_populates="jobs")
    applications = relationship("Application", back_populates="job", cascade="all, delete-orphan")


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)

    # Personal Info
    phone = Column(String(20))
    location = Column(String(255))
    title = Column(String(255))  # Professional title
    bio = Column(Text)  # Professional bio
    website = Column(String(500))
    linkedin = Column(String(500))
    github = Column(String(500))

    # Professional Info
    resume_path = Column(String(500))
    resume_text = Column(Text)
    cover_letter = Column(Text)
    skills = Column(Text)  # JSON array of skills
    experience_years = Column(Float)
    education = Column(Text)  # JSON array of education
    work_experience = Column(Text)  # JSON array of work experience
    certifications = Column(Text)  # JSON array of certifications
    projects = Column(Text)  # JSON array of projects
    languages = Column(Text)  # JSON array of languages

    # AI/ML fields
    embedding = Column(Text)  # JSON string of resume embedding
    profile_summary = Column(Text)  # AI-generated summary

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    user = relationship("User", back_populates="candidate_profile")
    applications = relationship("Application", back_populates="candidate", cascade="all, delete-orphan")


class SavedJob(Base):
    __tablename__ = "saved_jobs"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)  # Nullable for external jobs
    source = Column(String(50), default="internal")  # "internal", "adzuna", "usajobs", etc.
    external_job_id = Column(String(255), nullable=True)  # External job identifier
    job_data = Column(JSON, nullable=True)  # Store full job data for external jobs
    saved_at = Column(DateTime, default=utc_now)

    candidate = relationship("Candidate")
    job = relationship("Job")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now)
    read_at = Column(DateTime, nullable=True)

    candidate = relationship("Candidate")
    job = relationship("Job")


class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)

    status = Column(Enum(ApplicationStatus), default=ApplicationStatus.PENDING)
    cover_letter = Column(Text)

    # AI/ML fields
    match_score = Column(Float)  # AI-calculated match score (0-1)
    match_explanation = Column(Text)  # AI explanation of the match

    applied_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    job = relationship("Job", back_populates="applications")
    candidate = relationship("Candidate", back_populates="applications")

