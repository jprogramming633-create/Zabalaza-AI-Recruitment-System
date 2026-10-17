from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any, cast
import json
import logging
import os
from ..core.dependencies import get_db, get_current_active_user
from ..models import User, Candidate, Application, Job, ApplicationStatus, SavedJob, Notification, JobStatus
from ..services.pdf_service import pdf_service
from ..schemas import (
    Candidate as CandidateSchema,
    CandidateUpdate,
    Application as ApplicationSchema,
    ApplicationCreate,
    MatchesResponse,
    CareerPathResponse,
    ChatRequest,
)
from ..config import settings
from ..services.ai_service import ai_service

router = APIRouter(prefix="/candidates", tags=["candidates"])
logger = logging.getLogger(__name__)


def _parse_json_list(value):
    try:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _candidate_payload(candidate: Any, user: Any) -> Dict[str, Any]:
    data = {
        "id": candidate.id,
        "user_id": candidate.user_id,
        "email": user.email,
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "phone": candidate.phone,
        "location": candidate.location,
        "title": candidate.title,
        "bio": candidate.bio,
        "cover_letter": candidate.cover_letter,
        "website": candidate.website,
        "linkedin": candidate.linkedin,
        "github": candidate.github,
        "experience_years": candidate.experience_years,
        "resume_path": candidate.resume_path,
        "resume_text": candidate.resume_text,
        "profile_summary": candidate.profile_summary,
        "created_at": candidate.created_at,
        "updated_at": candidate.updated_at,
        "skills_list": _parse_json_list(candidate.skills),
        "education_list": _parse_json_list(candidate.education),
        "work_experience_list": _parse_json_list(candidate.work_experience),
        "certifications": _parse_json_list(candidate.certifications),
        "projects": _parse_json_list(candidate.projects),
        "languages": _parse_json_list(candidate.languages),
        "extraction_report": None,
        "skills": candidate.skills,
        "education": candidate.education,
        "work_experience": candidate.work_experience,
    }
    return data


def _ensure_notification(db: Session, candidate_id: Any, type_: str, title: str, message: str,
                         job_id: Optional[int] = None):
    existing = db.query(Notification).filter(
        Notification.candidate_id == candidate_id,
        Notification.type == type_,
        Notification.title == title,
        Notification.job_id == job_id,
    ).first()
    if existing:
        return existing

    notification = Notification(
        candidate_id=candidate_id,
        type=type_,
        title=title,
        message=message,
        job_id=job_id,
    )
    db.add(notification)
    db.commit()
    db.refresh(notification)
    return notification


@router.get("/me", response_model=CandidateSchema)
def get_my_profile(
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get current candidate's profile."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found"
        )

    return _candidate_payload(candidate, current_user)


@router.put("/me", response_model=CandidateSchema)
def update_my_profile(
        candidate_data: CandidateUpdate,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Update current candidate's profile."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found"
        )

    # Update user fields
    user = db.query(User).filter(User.id == current_user.id).first()
    name_updated = False
    if candidate_data.first_name is not None:
        user.first_name = candidate_data.first_name
        name_updated = True
    if candidate_data.last_name is not None:
        user.last_name = candidate_data.last_name
        name_updated = True

    if name_updated:
        user.full_name = f"{user.first_name or ''} {user.last_name or ''}".strip()

    # Update candidate fields
    update_data = candidate_data.model_dump(exclude_unset=True)

    if 'cover_letter' in update_data and update_data['cover_letter'] is not None:
        candidate.cover_letter = update_data['cover_letter']
        del update_data['cover_letter']

    # Handle structured data - convert lists to JSON strings
    if 'skills' in update_data and update_data['skills'] is not None:
        candidate.skills = json.dumps(update_data['skills'])
        del update_data['skills']

    if 'education' in update_data and update_data['education'] is not None:
        candidate.education = json.dumps([edu.model_dump() for edu in update_data['education']])
        del update_data['education']

    if 'work_experience' in update_data and update_data['work_experience'] is not None:
        candidate.work_experience = json.dumps([exp.model_dump() for exp in update_data['work_experience']])
        del update_data['work_experience']

    if 'certifications' in update_data and update_data['certifications'] is not None:
        candidate.certifications = json.dumps([cert.model_dump() for cert in update_data['certifications']])
        del update_data['certifications']

    if 'projects' in update_data and update_data['projects'] is not None:
        candidate.projects = json.dumps(update_data['projects'])
        del update_data['projects']

    if 'languages' in update_data and update_data['languages'] is not None:
        candidate.languages = json.dumps(update_data['languages'])
        del update_data['languages']

    # Update simple fields
    for field in ['title', 'bio', 'phone', 'location', 'website', 'linkedin', 'github', 'experience_years']:
        if field in update_data and update_data[field] is not None:
            setattr(candidate, field, update_data[field])

    # Regenerate embedding and summary if profile changed
    try:
        candidate_dict = {
            'resume_text': candidate.resume_text or '',
            'skills': candidate.skills or '[]',
            'experience_years': candidate.experience_years or 0,
            'education': candidate.education or '[]',
            'work_experience': candidate.work_experience or '[]'
        }

        embedding = ai_service.generate_candidate_embedding(candidate_dict)
        candidate.embedding = json.dumps(embedding)

        summary = ai_service.generate_profile_summary(candidate_dict)
        candidate.profile_summary = summary
    except Exception as e:
        print(f"Error generating embedding/summary: {e}")
        # Continue even if AI service fails

    db.commit()
    db.refresh(candidate)
    db.refresh(user)

    return _candidate_payload(candidate, user)


@router.get("/me/matches", response_model=MatchesResponse)
def get_job_matches(
        top_k: int = 10,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get AI-powered job matches for current candidate.

    Returns a structured response with `items` (list of JobMatch) and `insights` which
    contains a conversational suggestion when matches are missing or low.
    """
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found"
        )



    # Get active jobs
    jobs = db.query(Job).filter(Job.status == JobStatus.ACTIVE).all()

    if not jobs:
        jobs = db.query(Job).filter(Job.status.in_([JobStatus.ACTIVE, JobStatus.ON_HOLD])).all()
        return {
            'items': [],
            'insights': 'No active jobs are available right now. Check back soon for new opportunities.',
            'career_path': None,
        }

    # Convert to dict format
    jobs_dict = []
    for job in jobs:
        job_dict = {
            'id': job.id,
            'title': job.title,
            'description': job.description,
            'requirements': job.requirements,
            'experience_level': job.experience_level,
            'location': job.location,
            'embedding': job.embedding,
            'skills': job.skills
        }
        jobs_dict.append(job_dict)

    # Get candidate data
    candidate_dict = {
        'resume_text': candidate.resume_text or '',
        'skills': candidate.skills or '[]',
        'experience_years': candidate.experience_years or 0,
        'education': candidate.education or '[]',
        'work_experience': candidate.work_experience or '[]'
    }

    matches = []
    
    # Try AI-powered matching if embeddings exist
    if candidate.embedding:
        try:
            candidate_embedding = json.loads(candidate.embedding)
            matches = ai_service.find_matching_jobs(
                candidate_embedding,
                jobs_dict,
                candidate_dict,
                top_k
            )
        except Exception as e:
            print(f"AI matching failed: {e}")
            matches = []
    
    # If no matches from AI, return all jobs with basic scoring
    if not matches:
        matches = []
        for job in jobs_dict:
            # Calculate basic match score based on skills overlap
            candidate_skills = set()
            try:
                skills_list = json.loads(candidate_dict['skills']) if isinstance(candidate_dict['skills'], str) else candidate_dict['skills']
                candidate_skills = {s.lower() for s in skills_list if isinstance(s, str)}
            except:
                pass
            
            job_skills = set()
            try:
                job_skills_list = json.loads(job['skills']) if isinstance(job['skills'], str) else job['skills']
                job_skills = {s.lower() for s in job_skills_list if isinstance(s, str)}
            except:
                # Extract skills from job description
                job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('requirements', '')}".lower()
                job_skills = {s for s in candidate_skills if s in job_text}
            
            # Calculate overlap
            if candidate_skills and job_skills:
                overlap = len(candidate_skills.intersection(job_skills))
                basic_score = min(0.9, 0.3 + (overlap * 0.1))  # 30% base + 10% per matching skill
            else:
                basic_score = 0.25  # Default low score
            
            matches.append({
                'job_id': job['id'],
                'job_title': job['title'],
                'match_score': basic_score,
                'match_explanation': 'Browse available positions' if basic_score < 0.5 else 'Good potential match',
                'job_details': {
                    'description': job.get('description', ''),
                    'location': job.get('location', 'Remote'),
                    'job_type': job.get('job_type'),
                    'salary_min': job.get('salary_min'),
                    'salary_max': job.get('salary_max'),
                }
            })
        
        # Sort by score and limit
        matches.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        matches = matches[:top_k]

    # Enhance matches with LLM-powered explanations
    items = []
    if matches:
        for match in matches:
            # Try to get Gemini-powered match explanation
            match_explanation = match.get('match_explanation', 'Good match for your profile')
            try:
                job_id = match.get('job_id')
                job = next((j for j in jobs if j.id == job_id), None)
                if job:
                    job_dict_full = {
                        'title': job.title,
                        'description': job.description,
                        'requirements': job.requirements,
                        'requirements_list': getattr(job, 'requirements_list', []),
                    }
                    match_explanation = ai_service.generate_match_explanation(
                        job_dict_full,
                        candidate_dict,
                        match.get('match_score', 0)
                    )
            except Exception as e:
                # If LLM fails, use the fallback explanation
                pass

            items.append({
                'job_id': match.get('job_id'),
                'job_title': match.get('job_title', 'Position'),
                'match_score': match.get('match_score', 0),
                'match_explanation': match_explanation,
                'skill_gaps': match.get('skill_gaps', []),
                'strengths': match.get('strengths', []),
                'job_details': match.get('job_details', {})
            })

    # If no strong matches, provide conversational guidance and suggested alternatives
    insights = None
    career_path = None

    # Determine if matches are below similarity threshold
    try:
        threshold = float(getattr(settings, 'SIMILARITY_THRESHOLD', 0.6))
    except Exception:
        threshold = 0.6

    if not items:
        insights = (
            "We couldn't find close matches for your profile. "
            "Consider adding more technical skills, uploading relevant certificates, or broadening your location or role preferences. "
            "Below are some alternative job postings you can review."
        )
        # Get career path suggestions
        try:
            career_path = ai_service.generate_career_path(candidate_dict)
        except Exception:
            career_path = None

        # Suggest some alternatives (first N active jobs) with calculated match scores
        suggested = []
        for job in jobs[: min(10, len(jobs))]:
            # Calculate a basic match score even if embeddings aren't available
            basic_score = 0.3  # Default low score
            
            # Try to calculate actual score if embeddings exist
            if candidate.embedding and job.embedding:
                try:
                    candidate_emb = json.loads(candidate.embedding)
                    job_emb = json.loads(job.embedding)
                    from ..services.matching_engine import MatchingEngine
                    engine = MatchingEngine()
                    basic_score = engine.calculate_similarity(candidate_emb, job_emb)
                except Exception:
                    pass
            
            suggested.append({
                'job_id': job.id,
                'job_title': job.title,
                'match_score': basic_score,
                'match_explanation': 'Browse all available positions' if basic_score < 0.6 else 'Potential match',
                'job_details': {
                    'description': job.description,
                    'location': job.location,
                    'job_type': job.job_type if hasattr(job, 'job_type') else None,
                    'salary_min': getattr(job, 'salary_min', None),
                    'salary_max': getattr(job, 'salary_max', None),
                }
            })
        items = suggested
    else:
        # If matches exist but average score is low, add an insight message and suggest career path development
        try:
            avg_score = sum(m.get('match_score', 0) for m in items) / max(1, len(items))
            if avg_score < threshold:
                insights = (
                    f"The average match score for your top {len(items)} recommendations is {avg_score:.2%}. "
                    "Consider enhancing your profile (skills, certificates, experience details) to improve matches."
                )
                # Get career path suggestions when matches are low
                try:
                    career_path = ai_service.generate_career_path(candidate_dict)
                except Exception:
                    pass
        except Exception:
            insights = None

    return {
        'items': items,
        'insights': insights,
        'career_path': career_path
    }


@router.get("/me/career-path", response_model=CareerPathResponse)
def get_career_path(
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get AI-powered career path recommendations for the current candidate."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found"
        )

    # Get candidate data
    candidate_dict = {
        'skills': candidate.skills,
        'experience_years': candidate.experience_years,
        'education': candidate.education,
        'work_experience': candidate.work_experience,
        'profile_summary': candidate.profile_summary,
    }

    try:
        career_path = ai_service.generate_career_path(candidate_dict)
        return {
            'career_path': career_path,
            'learning_recommendations': [],  # Can be enhanced later
            'next_roles': []  # Can be enhanced later
        }
    except Exception as e:
        logger.error(f"Error generating career path: {e}")
        return {
            'career_path': "We could not generate career recommendations at this time. Please try again later.",
            'learning_recommendations': [],
            'next_roles': []
        }


@router.post("/me/interview-tips")
def post_interview_tips(
        payload: Dict[str, Any] = None,
        job_id: Optional[int] = None,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get interview preparation tips for internal or external jobs.

    For internal jobs: include job_id in query params and checks application status
    For external jobs: include job_data in POST body, no application check
    """
    # Handle both query param and POST payload
    if payload is None:
        payload = {}

    if not isinstance(payload, dict):
        payload = {"job_data": payload}

    # Extract job_id from query params if not in payload
    if job_id and not payload.get("job_id"):
        payload["job_id"] = job_id

    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found"
        )

    job_id_param = payload.get("job_id")
    job_data = payload.get("job_data")

    # Determine if internal or external job
    if job_id_param and isinstance(job_id_param, int):
        # Internal job - fetch from database and check application status
        job = db.query(Job).filter(Job.id == job_id_param).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        application = db.query(Application).filter(
            Application.job_id == job_id_param,
            Application.candidate_id == candidate.id,
        ).first()
        if not application or application.status != ApplicationStatus.INTERVIEW:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Interview tips are available only after you've been selected for interview."
            )

        job_title = job.title
        job_description = job.description
    elif job_data:
        # External job - use provided data, no application check
        job_title = job_data.get("title", "")
        job_description = job_data.get("description", "")
    else:
        raise HTTPException(status_code=400, detail="job_id or job_data is required")

    try:
        from ..services.gemini_service import get_gemini_service
        gemini_service = get_gemini_service()

        work_exp = candidate.work_experience
        if isinstance(work_exp, str):
            try:
                work_exp = json.loads(work_exp)
            except:
                work_exp = []

        exp_summary = "; ".join([str(e)[:50] for e in work_exp[:3]]) if isinstance(work_exp, list) else str(work_exp)[
            :100]

        tips = gemini_service.generate_interview_tips(
            job_title=job_title,
            job_description=job_description,
            candidate_experience=exp_summary
        )

        return {'interview_tips': tips}
    except Exception as e:
        logger.error(f"Error generating interview tips: {e}")
        return {'interview_tips': 'Please research this role and prepare answers to common interview questions.'}


@router.post("/me/cv-optimization")
def get_cv_optimization_tips(
        section: str,  # e.g., "summary", "skills", "experience"
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get AI suggestions to optimize a CV section."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found"
        )

    try:
        from ..services.gemini_service import get_gemini_service
        gemini_service = get_gemini_service()

        section_text = ""
        if section == "summary":
            section_text = candidate.profile_summary or ""
        elif section == "skills":
            section_text = candidate.skills or ""
        elif section == "experience":
            section_text = candidate.work_experience or ""
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown section: {section}. Valid sections: summary, skills, experience"
            )

        if not section_text:
            return {'optimized_text': "", 'message': f'No {section} found in your profile'}

        optimized = gemini_service.optimize_cv_section(
            section_name=section,
            current_text=section_text
        )

        return {'optimized_text': optimized}
    except Exception as e:
        logger.error(f"Error optimizing CV section: {e}")
        return {'optimized_text': "", 'message': 'Could not optimize section at this time'}


@router.post("/me/applications")
def apply_for_job(
        application_data: ApplicationCreate,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Apply for a job."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found"
        )

    # Check if job exists
    job = db.query(Job).filter(Job.id == application_data.job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check if already applied
    existing_application = db.query(Application).filter(
        Application.candidate_id == candidate.id,
        Application.job_id == application_data.job_id
    ).first()

    if existing_application:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already applied for this job"
        )

    # Calculate match score using AI
    match_score = 0.0
    match_explanation = "No matching data available"

    if candidate.embedding and job.embedding:
        candidate_embedding = json.loads(candidate.embedding)
        job_embedding = json.loads(job.embedding)

        candidate_dict = {
            'skills': candidate.skills,
            'experience_years': candidate.experience_years,
            'education': candidate.education
        }

        job_dict = {
            'title': job.title,
            'description': job.description,
            'requirements': job.requirements,
            'experience_level': job.experience_level,
            'skills': job.skills
        }

        match_score, match_explanation = ai_service.match_candidate_to_job(
            candidate_embedding,
            job_embedding,
            candidate_dict,
            job_dict
        )

    # Create application
    application = Application(
        job_id=application_data.job_id,
        candidate_id=candidate.id,
        cover_letter=application_data.cover_letter,
        status=ApplicationStatus.PENDING,
        match_score=match_score,
        match_explanation=match_explanation
    )

    db.add(application)
    db.commit()
    db.refresh(application)

    _ensure_notification(
        db,
        candidate.id,
        "application_follow_up",
        f"Follow up on {job.title}",
        "Your application has been submitted. If you do not hear back soon, send a polite follow-up message to the company.",
        job.id,
    )

    return application


@router.get("/me/applications", response_model=List[ApplicationSchema])
def get_my_applications(
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get all applications submitted by current candidate."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate profile not found"
        )

    applications = db.query(Application).filter(
        Application.candidate_id == candidate.id
    ).all()

    return applications


@router.get("/me/saved-jobs")
def get_saved_jobs(
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    saved_jobs = []
    for saved in db.query(SavedJob).filter(SavedJob.candidate_id == candidate.id).order_by(
            SavedJob.saved_at.desc()).all():
        if saved.source == "internal":
            # Internal job
            job = db.query(Job).filter(Job.id == saved.job_id).first()
            if not job:
                continue
            saved_jobs.append({
                "saved_job_id": saved.id,
                "saved_at": saved.saved_at,
                "source": "internal",
                "job_id": job.id,
                "job": {
                    "id": job.id,
                    "title": job.title,
                    "description": job.description,
                    "requirements": job.requirements,
                    "location": job.location,
                    "salary_min": job.salary_min,
                    "salary_max": job.salary_max,
                    "job_type": job.job_type,
                    "experience_level": job.experience_level,
                    "company": getattr(job, "company", None),
                    "created_at": job.created_at,
                }
            })
        else:
            # External job - use stored job_data
            saved_jobs.append({
                "saved_job_id": saved.id,
                "saved_at": saved.saved_at,
                "source": saved.source,
                "external_job_id": saved.external_job_id,
                "job_id": saved.external_job_id,
                "job": saved.job_data or {}
            })
    return saved_jobs


@router.post("/me/saved-jobs")
def save_job_for_current_candidate(
        payload: Dict[str, Any],
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    job_id = payload.get("job_id")
    source = payload.get("source", "internal")  # Track source: internal, adzuna, usajobs
    job_data = payload.get("job_data")  # Full job object for external jobs
    external_job_id = payload.get("external_job_id")  # External job identifier

    # For internal jobs, job_id must be provided and exist
    if source == "internal" and not job_id:
        raise HTTPException(status_code=400, detail="job_id is required for internal jobs")

    # For external jobs, we need job_data and external_job_id
    if source != "internal" and not (job_data or job_id):
        raise HTTPException(status_code=400, detail="job_data or job_id is required for external jobs")

    # Check if job already saved (for internal jobs)
    if source == "internal":
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        existing = db.query(SavedJob).filter(SavedJob.candidate_id == candidate.id, SavedJob.job_id == job.id).first()
        if existing:
            return {"message": "Job already saved", "saved_job_id": existing.id}
    else:
        # For external jobs, check if already saved using external_job_id
        existing = db.query(SavedJob).filter(
            SavedJob.candidate_id == candidate.id,
            SavedJob.source == source,
            SavedJob.external_job_id == external_job_id
        ).first()
        if existing:
            return {"message": "Job already saved", "saved_job_id": existing.id}

    # Create saved job record
    saved_job = SavedJob(
        candidate_id=candidate.id,
        job_id=job_id if source == "internal" else None,
        source=source,
        external_job_id=external_job_id,
        job_data=job_data if source != "internal" else None
    )
    db.add(saved_job)
    db.commit()
    db.refresh(saved_job)

    _ensure_notification(
        db,
        candidate.id,
        "saved_job_reminder",
        f"Saved job reminder: {job_data.get('title') if job_data else 'Unknown'}",
        "Remember to apply for this saved job before the deadline.",
        job_id if source == "internal" else None,
    )

    return {"message": "Job saved successfully", "saved_job_id": saved_job.id}


@router.delete("/me/saved-jobs/{job_id}")
def remove_saved_job(
        job_id: str,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    # Try to parse as integer for internal jobs
    try:
        internal_job_id = int(job_id)
        saved = db.query(SavedJob).filter(SavedJob.candidate_id == candidate.id,
                                          SavedJob.job_id == internal_job_id).first()
    except ValueError:
        # External job ID - use external_job_id field
        saved = db.query(SavedJob).filter(SavedJob.candidate_id == candidate.id,
                                          SavedJob.external_job_id == job_id).first()

    if not saved:
        raise HTTPException(status_code=404, detail="Saved job not found")

    db.delete(saved)
    db.commit()
    return {"message": "Saved job removed"}


@router.get("/me/notifications")
def get_candidate_notifications(
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    if not (candidate.resume_path or candidate.resume_text):
        _ensure_notification(
            db,
            candidate.id,
            "update_cv",
            "Update your CV",
            "Upload or refresh your resume so we can improve your match quality.",
            None,
        )

    active_jobs = db.query(Job).filter(Job.status == JobStatus.ACTIVE).order_by(Job.created_at.desc()).limit(5).all()
    for job in active_jobs:
        _ensure_notification(
            db,
            candidate.id,
            "new_job",
            f"New job: {job.title}",
            "A new job matching your profile may be worth reviewing.",
            job.id,
        )

    if candidate.embedding:
        try:
            # Get jobs the candidate hasn't applied to yet
            applied_job_ids = [a.job_id for a in db.query(Application).filter(Application.candidate_id == candidate.id).all()]
            candidate_emb = json.loads(candidate.embedding) if isinstance(candidate.embedding, str) else candidate.embedding
            recent_active = db.query(Job).filter(Job.status == JobStatus.ACTIVE).order_by(Job.created_at.desc()).limit(5).all()
            for job in recent_active:
                if job.id in applied_job_ids:
                    continue
                if job.embedding:
                    try:
                        job_emb = json.loads(job.embedding) if isinstance(job.embedding, str) else job.embedding
                        from ..services.matching_engine import MatchingEngine
                        engine = MatchingEngine()
                        score = engine.calculate_similarity(candidate_emb, job_emb)
                        if score >= 0.7:
                            _ensure_notification(
                                db, candidate.id, "good_match",
                                f"Good match: {job.title}",
                                "You look like a strong fit for this job. Consider applying soon.",
                                job.id,
                            )
                    except:
                        pass
        except Exception:
            pass

    for app in db.query(Application).filter(Application.candidate_id == candidate.id).all():
        job = db.query(Job).filter(Job.id == app.job_id).first()
        if app.status == ApplicationStatus.PENDING and job:
            _ensure_notification(
                db,
                candidate.id,
                "follow_up",
                f"Follow up on {job.title}",
                "Your application is still pending. A short follow-up message may help.",
                job.id,
            )

    notifications = db.query(Notification).filter(Notification.candidate_id == candidate.id).order_by(
        Notification.created_at.desc()).limit(50).all()
    return [
        {
            "id": item.id,
            "type": item.type,
            "title": item.title,
            "message": item.message,
            "job_id": item.job_id,
            "is_read": item.is_read,
            "created_at": item.created_at,
            "read_at": item.read_at,
        }
        for item in notifications
    ]


@router.put("/me/notifications/{notification_id}/read")
def mark_notification_read(
        notification_id: int,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.candidate_id == candidate.id,
    ).first()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    notification.is_read = True
    notification.read_at = datetime.now(timezone.utc)
    db.commit()
    return {"message": "Notification marked as read"}


@router.get("/me/match-analysis")
def get_match_analysis(
        job_id: int,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get detailed match analysis for an internal job."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        from ..services.gemini_service import get_gemini_service
        gemini = get_gemini_service()

        candidate_skills = []
        try:
            candidate_skills = json.loads(candidate.skills) if candidate.skills else []
        except:
            candidate_skills = []

        job_requirements = []
        try:
            job_requirements = json.loads(job.requirements) if job.requirements else []
        except:
            job_requirements = [job.requirements] if job.requirements else []

        match_score = 0.0
        if candidate.embedding and job.embedding:
            try:
                from ..services.matching_engine import MatchingEngine
                engine = MatchingEngine()
                candidate_emb = json.loads(candidate.embedding)
                job_emb = json.loads(job.embedding)
                match_score = engine.calculate_similarity(candidate_emb, job_emb)
            except:
                match_score = 0.0

        analysis = gemini.analyze_match_details(
            job_title=job.title,
            job_description=job.description,
            job_requirements=job_requirements,
            candidate_skills=candidate_skills,
            candidate_experience=candidate.resume_text[:300] if candidate.resume_text else "",
            match_score=match_score
        )
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing match: {e}")
        return {
            "summary": "Analysis temporarily unavailable",
            "strengths": [],
            "gaps": [],
            "recommendations": ""
        }


@router.post("/me/match-analysis")
def post_match_analysis(
        payload: Dict[str, Any],
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get detailed match analysis for internal or external jobs.

    For internal jobs: include job_id
    For external jobs: include job_data with title, description, requirements
    """
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    job_id = payload.get("job_id")
    job_data = payload.get("job_data")

    # Determine if internal or external job
    if job_id and isinstance(job_id, int):
        # Internal job - fetch from database
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job_title = job.title
        job_description = job.description
        job_requirements = []
        try:
            job_requirements = json.loads(job.requirements) if job.requirements else []
        except:
            job_requirements = [job.requirements] if job.requirements else []
    elif job_data:
        # External job - use provided data
        job_title = job_data.get("title", "")
        job_description = job_data.get("description", "")
        job_requirements = job_data.get("requirements", [])
        if isinstance(job_requirements, str):
            try:
                job_requirements = json.loads(job_requirements)
            except:
                job_requirements = [job_requirements] if job_requirements else []
    else:
        raise HTTPException(status_code=400, detail="job_id or job_data is required")

    try:
        from ..services.gemini_service import get_gemini_service
        gemini = get_gemini_service()

        candidate_skills = []
        try:
            candidate_skills = json.loads(candidate.skills) if candidate.skills else []
        except:
            candidate_skills = []

        analysis = gemini.analyze_match_details(
            job_title=job_title,
            job_description=job_description,
            job_requirements=job_requirements,
            candidate_skills=candidate_skills,
            candidate_experience=candidate.resume_text[:300] if candidate.resume_text else "",
            match_score=0.0
        )
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing match: {e}")
        return {
            "summary": "Analysis temporarily unavailable",
            "strengths": [],
            "gaps": [],
            "recommendations": ""
        }


@router.get("/me/resume-tailoring")
def get_resume_tailoring(
        job_id: int,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get resume tailoring suggestions for an internal job."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        from ..services.gemini_service import get_gemini_service
        gemini = get_gemini_service()

        # Load resume training instructions
        resume_instructions = ""
        try:
            instructions_path = os.path.join(os.path.dirname(__file__), '..', '..', 'CV', 'resume.txt')
            with open(instructions_path, 'r', encoding='utf-8') as f:
                resume_instructions = f.read()
        except Exception:
            pass  # File not found is okay, we'll use default behavior

        suggestions = gemini.get_resume_tailoring_suggestions(
            job_title=job.title,
            job_description=job.description,
            current_resume=candidate.resume_text[:500] if candidate.resume_text else candidate.profile_summary or "",
            training_instructions=resume_instructions,
        )
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error getting resume tailoring: {e}")
        return {"suggestions": "Tailoring suggestions not available"}


@router.post("/me/resume-tailoring")
def post_resume_tailoring(
        payload: Dict[str, Any],
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get resume tailoring suggestions for internal or external jobs.

    For internal jobs: include job_id
    For external jobs: include job_data with title, description
    """
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    job_id = payload.get("job_id")
    job_data = payload.get("job_data")

    # Determine if internal or external job
    if job_id and isinstance(job_id, int):
        # Internal job - fetch from database
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job_title = job.title
        job_description = job.description
    elif job_data:
        # External job - use provided data
        job_title = job_data.get("title", "")
        job_description = job_data.get("description", "")
    else:
        raise HTTPException(status_code=400, detail="job_id or job_data is required")

    try:
        from ..services.gemini_service import get_gemini_service
        gemini = get_gemini_service()

        suggestions = gemini.get_resume_tailoring_suggestions(
            job_title=job_title,
            job_description=job_description,
            current_resume=candidate.resume_text[:500] if candidate.resume_text else candidate.profile_summary or ""
        )
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error getting resume tailoring: {e}")
        return {"suggestions": "Tailoring suggestions not available"}


@router.get("/me/profile-improvement")
def get_profile_improvement(
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Get tips to improve profile and stand out."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    try:
        from ..services.gemini_service import get_gemini_service
        gemini = get_gemini_service()

        skills = []
        try:
            skills = json.loads(candidate.skills) if candidate.skills else []
        except:
            skills = []

        tips = gemini.get_profile_improvement_tips(
            current_skills=skills,
            experience_years=candidate.experience_years or 0,
            current_title=candidate.title or "Professional",
            job_market_focus="tech"  # Can be parameterized
        )
        return {"improvements": tips}
    except Exception as e:
        logger.error(f"Error getting improvement tips: {e}")
        return {"improvements": "Improvement tips not available"}


@router.post("/me/chat")
def chat_with_gemini(
        request: ChatRequest,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Chat with Gemini career advisor."""
    try:
        logger.info(f"💬 Chat request from user {current_user.id}: '{request.message[:50]}'")

        from ..services.gemini_service import get_gemini_service
        gemini = get_gemini_service()

        # Verify Gemini is enabled
        if not gemini.enabled:
            logger.warning(f"⚠️ Gemini not enabled for user {current_user.id}")
            return {
                "response": "AI features are currently unavailable. Please check your Gemini API key configuration."
            }

        # Call Gemini service with optional context
        response = gemini.chat(request.message, request.history, context=getattr(request, 'context', None))

        if not response or not response.strip():
            logger.warning(f"⚠️ Empty response from Gemini for user {current_user.id}")
            return {
                "response": "I couldn't generate a response. Please try again."
            }

        logger.info(f"✅ Chat response sent to user {current_user.id}")
        return {"response": response}

    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "response": "An error occurred while processing your message. Please try again."
        }


@router.post("/me/generate-cover-letter")
def generate_cover_letter(
        payload: Dict[str, Any],
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """Generate a cover letter for a specific job using training instructions."""
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    job_id = payload.get("job_id")
    job_data = payload.get("job_data")

    # Determine if internal or external job
    if job_id and isinstance(job_id, int):
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        job_title = job.title
        job_description = job.description
        company_name = getattr(job, 'company', 'the company')
    elif job_data:
        job_title = job_data.get("title", "")
        job_description = job_data.get("description", "")
        company_name = job_data.get("company", 'the company')
    else:
        raise HTTPException(status_code=400, detail="job_id or job_data is required")

    try:
        from ..services.gemini_service import get_gemini_service
        gemini = get_gemini_service()

        # Load cover letter training instructions
        cover_letter_instructions = ""
        try:
            instructions_path = os.path.join(os.path.dirname(__file__), '..', '..', 'CV', 'coverletter.txt')
            with open(instructions_path, 'r', encoding='utf-8') as f:
                cover_letter_instructions = f.read()
        except Exception:
            pass  # File not found is okay, we'll use default behavior

        # Build candidate info
        candidate_name = f"{current_user.first_name or ''} {current_user.last_name or ''}".strip() or current_user.username
        candidate_email = current_user.email
        candidate_phone = candidate.phone or ""

        # Build the prompt with training instructions
        prompt = f"""You are an expert cover letter writer. Generate a professional, personalized cover letter for this specific job application.

{cover_letter_instructions if cover_letter_instructions else ''}

JOB DETAILS:
- Position: {job_title}
- Company: {company_name}
- Description: {job_description[:500]}

CANDIDATE INFORMATION:
- Name: {candidate_name}
- Email: {candidate_email}
- Phone: {candidate_phone}
- Professional Title: {candidate.title or 'Professional'}
- Summary: {candidate.profile_summary or ''}
- Skills: {', '.join(json.loads(candidate.skills) if candidate.skills else [])[:200]}
- Experience: {candidate.experience_years or 0} years

INSTRUCTIONS:
1. Write a professional cover letter addressed to the hiring manager
2. DO NOT mention match scores or percentages
3. Focus on the candidate's relevant skills and experience for THIS specific role
4. Be specific about why this candidate is a good fit for THIS job
5. Keep it concise (3-4 paragraphs)
6. Use the candidate's actual name and information
7. Make it sound professional and genuine, not generic

Generate ONLY the cover letter text, no explanations or markdown."""

        response = gemini.client.generate_content(prompt)
        cover_letter = response.text.strip()

        return {
            "cover_letter": cover_letter,
            "job_title": job_title,
            "company": company_name
        }
    except Exception as e:
        logger.error(f"Error generating cover letter: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate cover letter")


@router.post("/me/button-context")
def get_button_context(
        button_request: Dict[str, Any],
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """
    Get orchestrated context for a button click in JobPortal.

    Combines ML model insights + Gemini LLM for intelligent analysis.
    Used when user clicks: Match Details, Interview Tips, Tailor Resume, Help Stand Out
    """
    try:
        button_type = button_request.get('button_type', 'match_details')
        job_id = button_request.get('job_id')

        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")

        # Get candidate and job
        candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate profile not found")

        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get orchestrator
        from ..services.orchestrator_service import get_orchestrator
        orchestrator = get_orchestrator(ai_service)

        # Prepare candidate data
        candidate_data = {
            "id": candidate.id,
            "title": candidate.title or "",
            "skills": [],
            "experience_years": candidate.experience_years or 0,
            "education": candidate.education or "",
            "years_of_experience": candidate.experience_years or 0,
        }

        # Parse skills
        try:
            skills_json = json.loads(candidate.skills) if candidate.skills else []
            candidate_data["skills"] = skills_json if isinstance(skills_json, list) else []
        except:
            candidate_data["skills"] = []

        # Prepare job data
        job_data = {
            "id": job.id,
            "job_id": job.id,
            "title": job.title,
            "description": job.description or "",
            "requirements": [],
            "company": getattr(job, 'company', 'Unknown Company')
        }

        # Parse requirements
        try:
            reqs_json = json.loads(job.requirements) if job.requirements else []
            job_data["requirements"] = reqs_json if isinstance(reqs_json, list) else []
        except:
            job_data["requirements"] = []

        # Get ML insights if available
        ml_insights = {}
        if candidate.embedding and job.embedding:
            try:
                candidate_emb = json.loads(candidate.embedding)
                job_emb = json.loads(job.embedding)
                from ..services.matching_engine import MatchingEngine
                engine = MatchingEngine()
                match_score = engine.calculate_similarity(candidate_emb, job_emb)
                ml_insights['match_score'] = match_score
            except:
                ml_insights['match_score'] = 0.0

        # Get orchestrated context
        context = orchestrator.analyze_button_context(
            button_type=button_type,
            job_data=job_data,
            candidate_data=candidate_data,
            ml_insights=ml_insights
        )

        logger.info(f"✅ Button context generated for {button_type} on job {job_id}")
        return context

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating button context: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "button_type": "error",
            "context": "Unable to generate context. Please try again.",
            "query": "",
            "error": str(e)
        }


@router.post("/me/apply-with-documents")
async def apply_with_documents(
        payload: Dict[str, Any],
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """
    Enhanced application endpoint that generates ATS-friendly PDF documents
    containing both the resume and cover letter in a single PDF.
    
    If resume_text is provided in payload, it uses that (edited by user).
    Otherwise, it uses the candidate's profile resume_text.
    """
    job_id = payload.get("job_id")
    cover_letter = payload.get("cover_letter", "")
    resume_text = payload.get("resume_text")
    use_optimized_resume = payload.get("use_optimized_resume", False)
    
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required")

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if already applied
    existing = db.query(Application).filter(
        Application.candidate_id == candidate.id,
        Application.job_id == job_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Already applied for this job")

    # Calculate match score
    match_score = 0.0
    match_explanation = "No matching data available"
    if candidate.embedding and job.embedding:
        candidate_emb = json.loads(candidate.embedding)
        job_emb = json.loads(job.embedding)
        candidate_dict = {
            'skills': candidate.skills,
            'experience_years': candidate.experience_years,
            'education': candidate.education
        }
        job_dict = {
            'title': job.title,
            'description': job.description,
            'requirements': job.requirements,
            'experience_level': job.experience_level,
            'skills': job.skills
        }
        match_score, match_explanation = ai_service.match_candidate_to_job(
            candidate_emb, job_emb, candidate_dict, job_dict
        )

    # Get user info for the PDF
    user = db.query(User).filter(User.id == current_user.id).first()
    
    # Use edited resume_text if provided, otherwise use profile resume_text
    final_resume_text = resume_text or candidate.resume_text or ""
    
    candidate_data = {
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "email": user.email,
        "phone": candidate.phone or "",
        "location": candidate.location or "",
        "title": candidate.title or "",
        "linkedin": candidate.linkedin or "",
        "github": candidate.github or "",
        "profile_summary": final_resume_text,  # Use the edited resume text
        "skills": candidate.skills,
        "skills_list": json.loads(candidate.skills) if candidate.skills else [],
        "work_experience": candidate.work_experience,
        "work_experience_list": json.loads(candidate.work_experience) if candidate.work_experience else [],
        "education": candidate.education,
        "education_list": json.loads(candidate.education) if candidate.education else [],
        "certifications": json.loads(candidate.certifications) if candidate.certifications else [],
        "projects": json.loads(candidate.projects) if candidate.projects else [],
        "resume_text": final_resume_text,
    }

    company_name = getattr(job, 'company', getattr(job, 'recruiter', None))
    if company_name and not isinstance(company_name, str):
        company_name = "the company"

    # Generate combined PDF (cover letter + resume)
    pdf_path = pdf_service.generate_combined_pdf(
        candidate_data=candidate_data,
        cover_letter_text=cover_letter or "",
        job_title=job.title,
        company_name=company_name or "the company"
    )

    # Generate CV strength score
    cv_strength_score = min(100, int((match_score * 100) * 0.8 + 20))

    # Create application with PDF path
    application = Application(
        job_id=job_id,
        candidate_id=candidate.id,
        cover_letter=cover_letter or "",
        status=ApplicationStatus.PENDING,
        match_score=match_score,
        match_explanation=match_explanation
    )
    db.add(application)
    db.commit()
    db.refresh(application)

    # Store the PDF path on the candidate for retrieval
    candidate.resume_path = pdf_path
    db.commit()

    _ensure_notification(
        db, candidate.id, "application_follow_up",
        f"Follow up on {job.title}",
        "Your application has been submitted with a professionally formatted PDF.",
        job.id,
    )

    return {
        "application_id": application.id,
        "message": "Application submitted successfully with PDF documents",
        "match_score": round(match_score * 100, 1),
        "cv_strength_score": cv_strength_score,
        "pdf_path": pdf_path,
        "match_explanation": match_explanation
    }


@router.post("/me/optimize-resume")
async def optimize_resume(
        job_id: int = None,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    """
    Optimize the candidate's resume for a specific job and return an ATS-friendly PDF.
    """
    candidate = db.query(Candidate).filter(Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate profile not found")

    if not candidate.resume_text:
        raise HTTPException(status_code=400, detail="No resume found. Please upload a resume in your profile first.")

    job = None
    job_title = ""
    job_description = ""
    if job_id:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job_title = job.title
            job_description = job.description

    # Get Gemini to generate optimized resume content if available
    optimized_summary = candidate.profile_summary or ""
    optimized_skills = candidate.skills or "[]"
    optimized_experience = candidate.work_experience or "[]"

    try:
        from ..services.gemini_service import get_gemini_service
        gemini = get_gemini_service()

        if gemini and gemini.enabled:
            # Load resume training instructions
            resume_instructions = ""
            try:
                instructions_path = os.path.join(os.path.dirname(__file__), '..', '..', 'CV', 'resume.txt')
                with open(instructions_path, 'r', encoding='utf-8') as f:
                    resume_instructions = f.read()
            except Exception:
                pass

            skills_list = json.loads(candidate.skills) if candidate.skills else []
            
            # Extract keywords from job description
            job_keywords = []
            if job_description:
                # Simple keyword extraction - get important words from job description
                import re
                words = re.findall(r'\b[A-Za-z]{3,}\b', job_description.lower())
                # Filter out common words
                stop_words = {'the', 'and', 'for', 'with', 'that', 'this', 'will', 'you', 'your', 'are', 'have', 'has', 'been', 'from', 'not', 'but', 'what', 'all', 'were', 'when', 'there', 'their', 'would', 'could', 'should', 'does', 'did', 'can', 'may', 'our', 'out', 'who', 'get', 'has', 'had', 'his', 'her', 'its', 'into', 'more', 'most', 'other', 'some', 'such', 'than', 'then', 'them', 'these', 'they', 'this', 'those', 'through', 'under', 'very', 'well', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'within', 'without'}
                job_keywords = [w for w in words if w not in stop_words and len(w) > 3][:20]

            prompt = f"""You are an ATS resume optimization expert. Analyze the candidate's resume against the job requirements.

{resume_instructions if resume_instructions else ''}

CANDIDATE CURRENT INFO:
- Skills: {', '.join(skills_list[:15])}
- Experience: {candidate.experience_years or 0} years
- Current title: {candidate.title or 'Professional'}
- Summary: {candidate.profile_summary or ''}

{f'JOB TARGET: {job_title}' if job_title else ''}
{f'JOB DESCRIPTION: {job_description[:400]}' if job_description else ''}
{f'KEY JOB KEYWORDS TO MATCH: {", ".join(job_keywords)}' if job_keywords else ''}

TASK:
1. If the candidate's resume already contains most of the key job keywords, respond with: "Your resume looks good! It already includes key keywords like: [list 3-4 matching keywords]."
2. If optimization is needed, provide specific suggestions on what keywords or skills to add from the job description.
3. Keep the response concise (2-3 sentences max).

Do NOT generate a full resume. Just provide brief optimization feedback."""

            response = gemini.client.generate_content(prompt)
            optimized_text = response.text.strip()

            # Get user info
            user = db.query(User).filter(User.id == current_user.id).first()
            candidate_data = {
                "first_name": user.first_name or "",
                "last_name": user.last_name or "",
                "email": user.email,
                "phone": candidate.phone or "",
                "location": candidate.location or "",
                "title": candidate.title or "",
                "linkedin": candidate.linkedin or "",
                "github": candidate.github or "",
                "profile_summary": candidate.profile_summary or optimized_text,
                "skills": candidate.skills,
                "skills_list": skills_list,
                "work_experience": candidate.work_experience,
                "work_experience_list": json.loads(candidate.work_experience) if candidate.work_experience else [],
                "education": candidate.education,
                "education_list": json.loads(candidate.education) if candidate.education else [],
            }

            # Generate the PDF
            pdf_path = pdf_service.generate_resume_pdf(candidate_data)

            return {
                "optimized_text": optimized_text,
                "pdf_path": pdf_path,
                "message": "Resume analysis completed"
            }

    except Exception as e:
        logger.error(f"Error optimizing resume: {e}")

    # Fallback: Generate PDF with existing data
    user = db.query(User).filter(User.id == current_user.id).first()
    skills_list = json.loads(candidate.skills) if candidate.skills else []
    candidate_data = {
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "email": user.email,
        "phone": candidate.phone or "",
        "location": candidate.location or "",
        "title": candidate.title or "",
        "profile_summary": candidate.profile_summary or "",
        "skills": candidate.skills,
        "skills_list": skills_list,
        "work_experience": candidate.work_experience,
        "education": candidate.education,
    }
    pdf_path = pdf_service.generate_resume_pdf(candidate_data)

    return {
        "optimized_text": "Your resume has been processed. Upload a more detailed resume for better optimization suggestions.",
        "pdf_path": pdf_path,
        "message": "Resume PDF generated from existing profile data"
    }
