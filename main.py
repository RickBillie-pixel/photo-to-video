from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import uuid
from pathlib import Path
import tempfile
import shutil
import logging
from datetime import datetime
from PIL import Image
import json
from enum import Enum
from typing import Optional
import httpx  # ← NIEUWE IMPORT VOOR URL DOWNLOADS
from urllib.parse import urlparse

# Logging configuratie
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Foto Verlenger API - Async 4K")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path(tempfile.gettempdir()) / "uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "outputs"
JOBS_DIR = Path(tempfile.gettempdir()) / "jobs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

logger.info("=" * 80)
logger.info("🚀 FOTO VERLENGER API - ASYNC 4K - GESTART")
logger.info("=" * 80)
logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
logger.info(f"📁 Output directory: {OUTPUT_DIR}")
logger.info(f"📁 Jobs directory: {JOBS_DIR}")
logger.info(f"💰 Render Plan: Pro ($25/maand)")
logger.info("=" * 80)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def get_job_info(job_id: str) -> Optional[dict]:
    """Haal job informatie op"""
    job_file = JOBS_DIR / f"{job_id}.json"
    if not job_file.exists():
        return None
    
    try:
        with open(job_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Kan job info niet lezen: {e}")
        return None


def update_job_status(job_id: str, status: JobStatus, **kwargs):
    """Update job status"""
    job_file = JOBS_DIR / f"{job_id}.json"
    
    if job_file.exists():
        with open(job_file, 'r') as f:
            job_info = json.load(f)
    else:
        job_info = {"job_id": job_id, "created_at": datetime.now().isoformat()}
    
    job_info["status"] = status
    job_info["updated_at"] = datetime.now().isoformat()
    job_info.update(kwargs)
    
    with open(job_file, 'w') as f:
        json.dump(job_info, f, indent=2)
    
    logger.info(f"📝 Job {job_id[:8]} status: {status}")


def get_image_dimensions(image_path: str):
    """Haal de originele afmetingen van de afbeelding op"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            logger.info(f"📐 Originele afbeelding: {width}x{height} pixels")
            return width, height
    except Exception as e:
        logger.error(f"❌ Kan afbeelding niet lezen: {str(e)}")
        return None, None


async def download_image_from_url(url: str, save_path: str) -> tuple[bool, str, float]:
    """
    Download afbeelding van URL
    Returns: (success, error_message, file_size_mb)
    """
    try:
        logger.info(f"🌐 Downloading image from URL: {url[:60]}...")
        
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            return False, "Invalid URL scheme (must be http or https)", 0
        
        # Download with httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, follow_redirects=True)
            
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}: {response.reason_phrase}", 0
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False, f"URL does not point to an image (Content-Type: {content_type})", 0
            
            # Save file
            file_content = response.content
            file_size_mb = len(file_content) / 1024 / 1024
            
            with open(save_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"✅ Downloaded {file_size_mb:.2f} MB from URL")
            return True, "", file_size_mb
            
    except httpx.TimeoutException:
        return False, "Download timeout (30s exceeded)", 0
    except httpx.RequestError as e:
        return False, f"Download failed: {str(e)}", 0
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", 0


def create_video_background(job_id: str, input_path: str, output_path: str, duration: int, fps: int, keep_original_resolution: bool):
    """
    Background task voor video creatie
    """
    try:
        logger.info("=" * 80)
        logger.info(f"🎬 BACKGROUND JOB GESTART [ID: {job_id[:8]}]")
        logger.info("=" * 80)
        
        update_job_status(job_id, JobStatus.PROCESSING, progress=0)
        
        # Detecteer resolutie
        orig_width, orig_height = get_image_dimensions(input_path)
        
        if orig_width is None or orig_height is None:
            update_job_status(job_id, JobStatus.FAILED, error="Kan afbeelding dimensies niet bepalen")
            return
        
        if keep_original_resolution:
            width, height = orig_width, orig_height
            logger.info(f"✅ Gebruik ORIGINELE resolutie: {width}x{height}")
        else:
            width, height = 1920, 1080
            logger.info(f"⚠️  Gebruik standaard resolutie: {width}x{height}")
        
        update_job_status(job_id, JobStatus.PROCESSING, progress=5, width=width, height=height)
        
        estimated_frames = duration * fps
        estimated_size_mb = (width * height * 3 * estimated_frames) / (1024 * 1024 * 20)
        
        logger.info(f"📊 PARAMETERS:")
        logger.info(f"   ⏱️  Duur: {duration}s ({duration/60:.2f} min)")
        logger.info(f"   🎞️  FPS: {fps}")
        logger.info(f"   📐 Resolutie: {width}x{height}")
        logger.info(f"   🖼️  Frames: {estimated_frames:,}")
        logger.info(f"   📦 Geschat: ~{estimated_size_mb:.1f} MB")
        logger.info("-" * 80)
        
        start_time = datetime.now()
        
        # FFmpeg command
        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-t', str(duration),
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            '-vf', f'scale={width}:{height}',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        logger.info(f"🔧 FFmpeg: {' '.join(cmd[:8])}...")
        
        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor progress
        last_update = datetime.now()
        frame_count = 0
        
        for line in process.stderr:
            if 'frame=' in line:
                try:
                    frame_str = line.split('frame=')[1].split()[0]
                    frame_count = int(frame_str)
                    
                    # Update status elke 5 seconden
                    if (datetime.now() - last_update).total_seconds() >= 5:
                        progress = min(5 + (frame_count / estimated_frames) * 90, 95)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        fps_actual = frame_count / elapsed if elapsed > 0 else 0
                        
                        update_job_status(
                            job_id, 
                            JobStatus.PROCESSING, 
                            progress=int(progress),
                            frames_processed=frame_count,
                            total_frames=estimated_frames,
                            fps_actual=round(fps_actual, 1)
                        )
                        
                        logger.info(f"📊 Progress: {progress:.1f}% | Frame: {frame_count:,}/{estimated_frames:,} | FPS: {fps_actual:.1f}")
                        last_update = datetime.now()
                except:
                    pass
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"❌ FFmpeg gefaald voor job {job_id[:8]}")
            update_job_status(job_id, JobStatus.FAILED, error="FFmpeg processing failed", progress=0)
            return
        
        # Check output
        if not os.path.exists(output_path):
            logger.error(f"❌ Output niet aangemaakt voor job {job_id[:8]}")
            update_job_status(job_id, JobStatus.FAILED, error="Output file not created", progress=0)
            return
        
        # Success!
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        output_size = os.path.getsize(output_path)
        
        logger.info("=" * 80)
        logger.info(f"✅ JOB VOLTOOID [ID: {job_id[:8]}]")
        logger.info("=" * 80)
        logger.info(f"   ⏱️  Tijd: {processing_time:.2f}s")
        logger.info(f"   📦 Grootte: {output_size / 1024 / 1024:.2f} MB")
        logger.info(f"   ⚡ Snelheid: {duration / processing_time:.2f}x realtime")
        logger.info("=" * 80)
        
        update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress=100,
            output_file=str(output_path),
            processing_time=round(processing_time, 2),
            file_size_mb=round(output_size / 1024 / 1024, 2),
            frames_processed=frame_count,
            download_url=f"/download/{job_id}"
        )
        
        # Cleanup input
        if os.path.exists(input_path):
            os.remove(input_path)
            logger.info(f"🧹 Input cleaned up")
        
    except Exception as e:
        logger.error(f"❌ Error in background job {job_id[:8]}: {str(e)}")
        update_job_status(job_id, JobStatus.FAILED, error=str(e), progress=0)


@app.get("/")
@app.head("/")
async def root():
    logger.info("📍 Root endpoint")
    return {
        "message": "🎬 Foto Verlenger API - Async 4K",
        "status": "🟢 Online",
        "version": "4.1 - Async + URL Support",
        "workflow": {
            "step_1": "POST /extend → Krijg job_id direct terug (upload OF url)",
            "step_2": "GET /job/{job_id} → Check status (polling)",
            "step_3": "GET /download/{job_id} → Download video wanneer klaar"
        },
        "endpoints": {
            "/extend": "POST - Start video job (file upload OF image_url)",
            "/job/{job_id}": "GET - Check job status + progress",
            "/download/{job_id}": "GET - Download completed video",
            "/health": "GET - Health check",
            "/stats": "GET - Server stats"
        },
        "input_methods": {
            "file_upload": "Stuur 'file' parameter met image bestand",
            "url": "Stuur 'image_url' parameter met image URL"
        },
        "features": [
            "✅ Async processing - Geen timeouts in n8n!",
            "✅ 4K support (originele resolutie)",
            "✅ Real-time progress updates",
            "✅ Video's tot 30 minuten",
            "✅ CRF 18 quality",
            "✅ URL support - Download from any image URL"
        ]
    }


@app.post("/extend")
async def extend_photo_async(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    duration: int = Form(default=5),
    fps: int = Form(default=30),
    keep_original_resolution: bool = Form(default=True)
):
    """
    Start een async video job - returnt DIRECT met job_id
    
    Ondersteunt 2 input methodes:
    1. FILE UPLOAD: Stuur 'file' parameter met afbeelding
    2. IMAGE URL: Stuur 'image_url' parameter met URL naar afbeelding
    
    Perfect voor n8n - geen timeouts!
    """
    job_id = str(uuid.uuid4())
    
    logger.info("=" * 80)
    logger.info(f"📥 NIEUWE ASYNC REQUEST [Job ID: {job_id[:8]}]")
    logger.info("=" * 80)
    
    # Validatie: moet file OF url hebben
    if not file and not image_url:
        raise HTTPException(
            status_code=400, 
            detail="Geef een 'file' (upload) OF 'image_url' (URL) parameter"
        )
    
    if file and image_url:
        raise HTTPException(
            status_code=400,
            detail="Gebruik alleen 'file' OF 'image_url', niet beide"
        )
    
    # Validatie parameters
    if duration < 1 or duration > 1800:
        raise HTTPException(status_code=400, detail="Duur moet tussen 1-1800 seconden zijn")
    
    if fps < 1 or fps > 60:
        raise HTTPException(status_code=400, detail="FPS moet tussen 1-60 zijn")
    
    # Bestandsnamen
    file_id = str(uuid.uuid4())
    
    if file:
        input_filename = f"{file_id}_{file.filename}"
        source_type = "upload"
        source_name = file.filename
    else:
        # Haal extensie van URL of gebruik .jpg als fallback
        parsed_url = urlparse(image_url)
        url_path = parsed_url.path
        ext = os.path.splitext(url_path)[1] or '.jpg'
        input_filename = f"{file_id}_from_url{ext}"
        source_type = "url"
        source_name = image_url[:60] + "..." if len(image_url) > 60 else image_url
    
    output_filename = f"{file_id}_video_{duration}s.mp4"
    
    input_path = str(UPLOAD_DIR / input_filename)
    output_path = str(OUTPUT_DIR / output_filename)
    
    logger.info(f"📋 Source type: {source_type}")
    logger.info(f"📎 Source: {source_name}")
    logger.info(f"⏱️  Duur: {duration}s")
    logger.info(f"🎞️  FPS: {fps}")
    logger.info(f"📐 Keep original: {keep_original_resolution}")
    
    try:
        file_size_mb = 0
        
        # OPTIE 1: File upload
        if file:
            # Valideer content type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Alleen afbeeldingen toegestaan")
            
            logger.info("💾 Bestand uploaden...")
            file_content = await file.read()
            file_size_mb = len(file_content) / 1024 / 1024
            logger.info(f"📦 Grootte: {file_size_mb:.2f} MB")
            
            with open(input_path, "wb") as buffer:
                buffer.write(file_content)
        
        # OPTIE 2: Download van URL
        else:
            logger.info("🌐 Afbeelding downloaden van URL...")
            success, error_msg, file_size_mb = await download_image_from_url(image_url, input_path)
            
            if not success:
                raise HTTPException(status_code=400, detail=f"Download failed: {error_msg}")
            
            logger.info(f"✅ Downloaded: {file_size_mb:.2f} MB")
        
        # Create job
        update_job_status(
            job_id,
            JobStatus.PENDING,
            source_type=source_type,
            source=source_name,
            duration=duration,
            fps=fps,
            keep_original_resolution=keep_original_resolution,
            input_size_mb=round(file_size_mb, 2),
            progress=0
        )
        
        # Start background task
        background_tasks.add_task(
            create_video_background,
            job_id,
            input_path,
            output_path,
            duration,
            fps,
            keep_original_resolution
        )
        
        logger.info(f"✅ Job gestart in background")
        logger.info(f"🔗 Status URL: /job/{job_id}")
        logger.info(f"🔗 Download URL: /download/{job_id}")
        logger.info("=" * 80)
        
        # Return immediate response
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": "Video wordt gemaakt in de achtergrond",
                "job_id": job_id,
                "source_type": source_type,
                "status_url": f"/job/{job_id}",
                "download_url": f"/download/{job_id}",
                "estimated_time_seconds": max(duration / 60, 5),
                "poll_interval_seconds": 3
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        if os.path.exists(input_path):
            os.remove(input_path)
        raise
    except Exception as e:
        logger.error(f"❌ Error bij job start: {str(e)}")
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Check job status - gebruik dit voor polling in n8n
    """
    logger.info(f"🔍 Status check voor job {job_id[:8]}")
    
    job_info = get_job_info(job_id)
    
    if not job_info:
        logger.warning(f"❌ Job {job_id[:8]} niet gevonden")
        raise HTTPException(status_code=404, detail="Job niet gevonden")
    
    logger.info(f"📊 Job {job_id[:8]} status: {job_info.get('status')} - {job_info.get('progress', 0)}%")
    
    return job_info


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """
    Download de completed video
    """
    logger.info(f"⬇️  Download request voor job {job_id[:8]}")
    
    job_info = get_job_info(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job niet gevonden")
    
    if job_info["status"] != JobStatus.COMPLETED:
        status = job_info["status"]
        progress = job_info.get("progress", 0)
        
        if status == JobStatus.FAILED:
            error = job_info.get("error", "Unknown error")
            raise HTTPException(status_code=500, detail=f"Job failed: {error}")
        else:
            raise HTTPException(
                status_code=425,
                detail=f"Video nog niet klaar. Status: {status}, Progress: {progress}%"
            )
    
    output_file = job_info.get("output_file")
    
    if not output_file or not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="Video bestand niet gevonden")
    
    logger.info(f"✅ Downloading job {job_id[:8]}")
    
    return FileResponse(
        output_file,
        media_type="video/mp4",
        filename=f"video_{job_id[:8]}.mp4",
        headers={
            "X-Job-ID": job_id,
            "X-Processing-Time": str(job_info.get("processing_time", 0))
        }
    )


@app.get("/health")
@app.head("/health")
async def health_check():
    """Health check"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, text=True, timeout=5)
        ffmpeg_version = result.stdout.split('\n')[0]
        ffmpeg_status = "✅ Available"
    except Exception as e:
        ffmpeg_status = "❌ Not available"
        ffmpeg_version = str(e)
    
    # Count jobs
    total_jobs = len(list(JOBS_DIR.glob("*.json")))
    pending = 0
    processing = 0
    completed = 0
    failed = 0
    
    for job_file in JOBS_DIR.glob("*.json"):
        try:
            with open(job_file) as f:
                job = json.load(f)
                status = job.get("status")
                if status == "pending":
                    pending += 1
                elif status == "processing":
                    processing += 1
                elif status == "completed":
                    completed += 1
                elif status == "failed":
                    failed += 1
        except:
            pass
    
    return {
        "status": "🟢 Healthy",
        "version": "4.1 - URL Support",
        "ffmpeg": {
            "status": ffmpeg_status,
            "version": ffmpeg_version
        },
        "jobs": {
            "total": total_jobs,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed
        }
    }


@app.get("/stats")
async def stats():
    """Server statistieken"""
    upload_files = list(UPLOAD_DIR.glob("*"))
    output_files = list(OUTPUT_DIR.glob("*"))
    job_files = list(JOBS_DIR.glob("*.json"))
    
    return {
        "upload_dir": {
            "files": len(upload_files),
            "total_size_mb": round(sum(f.stat().st_size for f in upload_files) / 1024 / 1024, 2)
        },
        "output_dir": {
            "files": len(output_files),
            "total_size_mb": round(sum(f.stat().st_size for f in output_files) / 1024 / 1024, 2)
        },
        "jobs": {
            "total": len(job_files)
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Async Foto Verlenger API with URL support...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
