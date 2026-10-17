from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import json

from torch.fx.experimental.unification.unification_tools import first

from ..core.dependencies import get_db, get_current_active_user, get_current_recruiter_user
from ..models import User, Job, JobStatus, Candidate
from ..schemas import Job as JobSchema, JobCreate, JobUpdate, JobWithApplications
from ..services.ai_service import ai_service
from ..services.hybrid_job_service import HybridJobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


# Optional dependency for authenticated user
def get_current_user_optional(db: Session = Depends(get_db)):
    """Get current user if authenticated, else None"""
    # This is a simplified version - in production use proper JWT validation
    return None


@router.post("/", response_model=JobSchema, status_code=status.HTTP_201_CREATED)
def create_job(
    job_data: JobCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new job posting for any authenticated user."""
    # Create job
    job = Job(
        title=job_data.title,
        description=job_data.description,
        requirements=job_data.requirements,
        location=job_data.location,
        salary_min=job_data.salary_min,
        salary_max=job_data.salary_max,
        job_type=job_data.job_type,
        experience_level=job_data.experience_level,
        recruiter_id=current_user.id,
        status=JobStatus.ACTIVE
    )

    # Generate AI embedding for the job
    job_dict = {
        'title': job.title,
        'description': job.description,
        'requirements': job.requirements,
        'experience_level': job.experience_level,
        'location': job.location
    }

    embedding = ai_service.generate_job_embedding(job_dict)
    job.embedding = json.dumps(embedding)

    db.add(job)
    db.commit()
    db.refresh(job)

    return job


@router.get("/", response_model=List[JobSchema])
def get_jobs(
    skip: int = 0,
    limit: int = 100,
    status: str = None,
    db: Session = Depends(get_db)
):
    """Get all job postings."""
    query = db.query(Job)

    if status:
        try:
            job_status = JobStatus(status)
            query = query.filter(Job.status == job_status)
        except ValueError:
            pass

    jobs = query.offset(skip).limit(limit).all()
    return jobs


@router.get("/{job_id}", response_model=JobSchema)
def get_job(job_id: int, db: Session = Depends(get_db)):
    """Get a specific job by ID."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    return job


@router.put("/{job_id}", response_model=JobSchema)
def update_job(
    job_id: int,
    job_data: JobUpdate,
    current_user: User = Depends(get_current_recruiter_user),
    db: Session = Depends(get_db)
):
    """Update a job posting."""
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check if user owns the job or is admin
    if job.recruiter_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this job"
        )

    # Update fields
    update_data = job_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)

    # Regenerate embedding if content changed
    if any(field in update_data for field in ['title', 'description', 'requirements']):
        job_dict = {
            'title': job.title,
            'description': job.description,
            'requirements': job.requirements,
            'experience_level': job.experience_level,
            'location': job.location
        }
        embedding = ai_service.generate_job_embedding(job_dict)
        job.embedding = json.dumps(embedding)

    db.commit()
    db.refresh(job)

    return job


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(
    job_id: int,
    current_user: User = Depends(get_current_recruiter_user),
    db: Session = Depends(get_db)
):
    """Delete a job posting."""
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check if user owns the job or is admin
    if job.recruiter_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this job"
        )

    db.delete(job)
    db.commit()

    return None


@router.get("/my/jobs", response_model=List[JobWithApplications])
def get_my_jobs(
    current_user: User = Depends(get_current_recruiter_user),
    db: Session = Depends(get_db)
):
    """Get jobs posted by current recruiter."""
    jobs = db.query(Job).filter(Job.recruiter_id == current_user.id).all()

    # Add application count
    result = []
    for job in jobs:
        job_dict = job.__dict__.copy()
        job_dict['application_count'] = len(job.applications)
        result.append(job_dict)

    return result

@router.get("/{job_id}/applications")
def get_job_applications(
        job_id: int,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get applications for a specific job (recruiter only)."""
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if job.recruiter_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view these applications"
        )

    from ..models import Application, Candidate
    applications = db.query(Application).filter(Application.job_id == job_id).all()

    result = []
    for app in applications:
        candidate = db.query(Candidate).filter(Candidate.id == app.candidate_id).first()
        candidate_user = db.query(User).filter(User.id == candidate.user_id).first() if candidate else None
        
        # Get the PDF path from candidate's resume_path
        pdf_path = candidate.resume_path if candidate and candidate.resume_path else None
        
        # Generate candidate summary
        summary = "No summary available"
        try:
            from ..services.ai_service import ai_service
            candidate_dict = {
                'resume_text': candidate.resume_text or '',
                'skills': candidate.skills or '[]',
                'experience_years': candidate.experience_years or 0,
                'education': candidate.education or '[]',
                'work_experience': candidate.work_experience or '[]'
            }
            job_dict = {
                'title': job.title,
                'description': job.description,
                'requirements': job.requirements,
                'experience_level': job.experience_level
            }
            summary = ai_service.generate_candidate_summary(candidate_dict, job_dict, app.match_score or 0)
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = f"Candidate with {candidate.experience_years or 0} years experience. Match score: {int((app.match_score or 0) * 100)}%"
        
        result.append({
            "id": app.id,
            "job_id": app.job_id,
            "job_title": job.title,
            "candidate_id": app.candidate_id,
            "candidate_name": f"{candidate_user.first_name or ''} {candidate_user.last_name or ''}".strip() if candidate_user else "Unknown",
            "candidate_email": candidate_user.email if candidate_user else "",
            "status": app.status,
            "match_score": app.match_score or 0,
            "cover_letter": app.cover_letter,
            "applied_at": app.applied_at,
            "resume_path": pdf_path,
            "summary": summary,
        })
    result.sort(key=lambda x: (x.get("match_score") or 0), reverse=True)

    return result

# ============================================================================
# HYBRID JOB SEARCH ENDPOINTS (Internal + External APIs)
# ============================================================================

@router.get("/search/hybrid")
async def search_jobs_hybrid(
    query: Optional[str] = Query(None, description="Search query"),
    location: Optional[str] = Query(None, description="Location"),
    job_type: Optional[str] = Query(None, description="Job type"),
    experience_level: Optional[str] = Query(None, description="Experience level"),
    remote_only: bool = Query(False, description="Remote only"),
    salary_min: Optional[float] = Query(None, description="Minimum salary"),
    salary_max: Optional[float] = Query(None, description="Maximum salary"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    include_external: bool = Query(True, description="Include external jobs"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Search jobs (hybrid - internal + external)

    This endpoint combines:
    - Internal jobs from our platform
    - External jobs from Indeed, USAJobs, Adzuna, GitHub Jobs

    Returns unified, ranked results with AI match scores for authenticated candidates.
    """
    service = HybridJobService(db)

    # Get candidate ID if user is a candidate
    candidate_id = None
    if current_user and current_user.role == 'candidate':
        candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
        if candidate:
            candidate_id = candidate.id

    result = service.search_jobs(
        query=query,
        location=location,
        job_type=job_type,
        experience_level=experience_level,
        remote_only=remote_only,
        salary_min=salary_min,
        salary_max=salary_max,
        page=page,
        limit=limit,
        include_external=include_external,
        candidate_id=candidate_id
    )

    return result


@router.get("/detail/{job_id}")
async def get_job_detail_hybrid(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed job information (supports both internal and external jobs)

    job_id format:
    - Internal: "internal_123"
    - External: "indeed_abc123", "github_xyz789", etc.
    """
    service = HybridJobService(db)

    job_detail = service.get_job_detail(job_id)
    if not job_detail:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_detail


@router.get("/stats/market")
async def get_job_market_stats(
    db: Session = Depends(get_db)
):
    """Get job market statistics"""
    service = HybridJobService(db)
    return service.get_job_statistics()


@router.get("/sources/available")
async def get_available_sources():
    """Get available job sources"""
    return {
        "sources": [
            {
                "id": "internal",
                "name": "Our Platform",
                "description": "Jobs posted directly on our platform",
                "enabled": True
            },
            {
                "id": "indeed",
                "name": "Indeed",
                "description": "World's largest job site",
                "enabled": False  # Set based on API key configuration
            },
            {
                "id": "usajobs",
                "name": "USAJobs",
                "description": "US Government jobs",
                "enabled": False
            },
            {
                "id": "adzuna",
                "name": "Adzuna",
                "description": "Job search engine",
                "enabled": False
            },
            {
                "id": "github",
                "name": "GitHub Jobs",
                "description": "Tech jobs from GitHub",
                "enabled": True  # Always available
            }
        ]
    }

