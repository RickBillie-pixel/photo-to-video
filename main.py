from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import subprocess
import os
import uuid
from pathlib import Path
import tempfile
import shutil
import logging
from datetime import datetime, timedelta
from PIL import Image
import json
from enum import Enum
from typing import Optional
import httpx
from urllib.parse import urlparse
import threading
import time

# Logging configuratie
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Foto Verlenger API - Video URL Output")

# CORS middleware - BELANGRIJK voor fal.ai en cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Directories - Gebruik /data voor persistent storage op Render
if os.path.exists('/data'):
    # Render persistent disk
    UPLOAD_DIR = Path('/data') / "uploads"
    OUTPUT_DIR = Path('/data') / "outputs"
    JOBS_DIR = Path('/data') / "jobs"
    logger.info("🗄️  Using Render persistent disk: /data")
else:
    # Fallback naar temp (local development)
    UPLOAD_DIR = Path(tempfile.gettempdir()) / "uploads"
    OUTPUT_DIR = Path(tempfile.gettempdir()) / "outputs"
    JOBS_DIR = Path(tempfile.gettempdir()) / "jobs"
    logger.info("⚠️  Using temporary storage (ephemeral)")

UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
JOBS_DIR.mkdir(exist_ok=True, parents=True)

# Mount static files voor directe video toegang
app.mount("/videos", StaticFiles(directory=str(OUTPUT_DIR)), name="videos")

logger.info("=" * 80)
logger.info("🚀 FOTO VERLENGER API - VIDEO URL OUTPUT - GESTART")
logger.info("=" * 80)
logger.info(f"📁 Upload directory: {UPLOAD_DIR}")
logger.info(f"📁 Output directory: {OUTPUT_DIR}")
logger.info(f"📁 Jobs directory: {JOBS_DIR}")
logger.info(f"🌐 Video serving: /videos (static files)")
logger.info("=" * 80)


# Cleanup thread voor oude bestanden (6 uur)
def cleanup_old_files():
    """Verwijdert video's ouder dan 6 uur"""
    while True:
        try:
            time.sleep(300)  # Check elke 5 minuten
            now = datetime.now()
            cutoff = now - timedelta(hours=6)
            
            deleted_count = 0
            for video_file in OUTPUT_DIR.glob("*.mp4"):
                file_time = datetime.fromtimestamp(video_file.stat().st_mtime)
                if file_time < cutoff:
                    video_file.unlink()
                    deleted_count += 1
                    
                    # Verwijder ook job info
                    job_id = video_file.stem.split('_')[0]
                    job_file = JOBS_DIR / f"{job_id}.json"
                    if job_file.exists():
                        job_file.unlink()
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} videos older than 6 hours")
                
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()


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
        
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            return False, "Invalid URL scheme (must be http or https)", 0
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, follow_redirects=True)
            
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}: {response.reason_phrase}", 0
            
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False, f"URL does not point to an image (Content-Type: {content_type})", 0
            
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


def create_video_background(job_id: str, input_path: str, output_filename: str, duration: int, fps: int, keep_original_resolution: bool, base_url: str):
    """
    Background task voor video creatie
    """
    output_path = str(OUTPUT_DIR / output_filename)
    
    try:
        logger.info("=" * 80)
        logger.info(f"🎬 BACKGROUND JOB GESTART [ID: {job_id[:8]}]")
        logger.info("=" * 80)
        
        update_job_status(job_id, JobStatus.PROCESSING, progress=0)
        
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
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        last_update = datetime.now()
        frame_count = 0
        
        for line in process.stderr:
            if 'frame=' in line:
                try:
                    frame_str = line.split('frame=')[1].split()[0]
                    frame_count = int(frame_str)
                    
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
        
        if not os.path.exists(output_path):
            logger.error(f"❌ Output niet aangemaakt voor job {job_id[:8]}")
            update_job_status(job_id, JobStatus.FAILED, error="Output file not created", progress=0)
            return
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        output_size = os.path.getsize(output_path)
        expires_at = end_time + timedelta(hours=16)
        
        # BELANGRIJKSTE: Video URL genereren
        video_url = f"{base_url}/videos/{output_filename}"
        
        logger.info("=" * 80)
        logger.info(f"✅ JOB VOLTOOID [ID: {job_id[:8]}]")
        logger.info("=" * 80)
        logger.info(f"   ⏱️  Tijd: {processing_time:.2f}s")
        logger.info(f"   📦 Grootte: {output_size / 1024 / 1024:.2f} MB")
        logger.info(f"   ⚡ Snelheid: {duration / processing_time:.2f}x realtime")
        logger.info(f"   🔗 Video URL: {video_url}")
        logger.info(f"   ⏰ Expires: {expires_at.isoformat()}")
        logger.info("=" * 80)
        
        update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress=100,
            output_file=str(output_path),
            output_filename=output_filename,
            processing_time=round(processing_time, 2),
            file_size_mb=round(output_size / 1024 / 1024, 2),
            frames_processed=frame_count,
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            video_url=video_url,
            expires_at=expires_at.isoformat(),
            expires_in_hours=16
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
        "message": "🎬 Foto Verlenger API - Direct Video URL",
        "status": "🟢 Online",
        "version": "5.0 - Direct Video URL for fal.ai",
        "workflow": {
            "step_1": "POST /extend → Krijg job_id (upload OF url)",
            "step_2": "GET /job/{job_id} → Check status + krijg video_url",
            "step_3": "Gebruik video_url direct in fal.ai!"
        },
        "endpoints": {
            "/extend": "POST - Start video job",
            "/job/{job_id}": "GET - Status + video_url",
            "/videos/{filename}": "GET - Direct video URL (voor fal.ai)",
            "/health": "GET - Health check"
        },
        "features": [
            "✅ Direct video URL - Perfect voor fal.ai!",
            "✅ Geen downloads nodig in n8n",
            "✅ 16 uur video beschikbaarheid",
            "✅ Persistent storage (Render Disk)",
            "✅ CORS enabled",
            "✅ 4K support",
            "✅ Tot 60 minuten video"
        ]
    }


@app.post("/extend")
async def extend_photo_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    duration: int = Form(default=5),
    fps: int = Form(default=30),
    keep_original_resolution: bool = Form(default=True)
):
    """
    Start async video job
    Returnt job_id → Poll /job/{job_id} → Krijg video_url voor fal.ai
    """
    job_id = str(uuid.uuid4())
    
    logger.info("=" * 80)
    logger.info(f"📥 NIEUWE ASYNC REQUEST [Job ID: {job_id[:8]}]")
    logger.info("=" * 80)
    
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
    
    if duration < 1 or duration > 3600:
        raise HTTPException(status_code=400, detail="Duur moet tussen 1-3600 seconden (60 minuten) zijn")
    
    if fps < 1 or fps > 60:
        raise HTTPException(status_code=400, detail="FPS moet tussen 1-60 zijn")
    
    # Get base URL from request
    base_url = str(request.base_url).rstrip('/')
    
    # Bestandsnamen
    file_id = str(uuid.uuid4())
    
    if file:
        input_filename = f"{file_id}_{file.filename}"
        source_type = "upload"
        source_name = file.filename
    else:
        parsed_url = urlparse(image_url)
        url_path = parsed_url.path
        ext = os.path.splitext(url_path)[1] or '.jpg'
        input_filename = f"{file_id}_from_url{ext}"
        source_type = "url"
        source_name = image_url[:60] + "..." if len(image_url) > 60 else image_url
    
    output_filename = f"{job_id}.mp4"  # Gebruik job_id als filename
    
    input_path = str(UPLOAD_DIR / input_filename)
    
    logger.info(f"📋 Source: {source_type} - {source_name}")
    logger.info(f"⏱️  Duur: {duration}s | FPS: {fps}")
    
    try:
        file_size_mb = 0
        
        if file:
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Alleen afbeeldingen toegestaan")
            
            logger.info("💾 Uploading...")
            file_content = await file.read()
            file_size_mb = len(file_content) / 1024 / 1024
            
            with open(input_path, "wb") as buffer:
                buffer.write(file_content)
        else:
            logger.info("🌐 Downloading...")
            success, error_msg, file_size_mb = await download_image_from_url(image_url, input_path)
            
            if not success:
                raise HTTPException(status_code=400, detail=f"Download failed: {error_msg}")
        
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
            output_filename,
            duration,
            fps,
            keep_original_resolution,
            base_url
        )
        
        logger.info(f"✅ Job started")
        logger.info(f"🔗 Status: /job/{job_id}")
        logger.info(f"🔗 Video URL: {base_url}/videos/{output_filename} (after completion)")
        logger.info("=" * 80)
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": "Video wordt gemaakt - Poll /job/{job_id} voor video_url",
                "job_id": job_id,
                "status_url": f"/job/{job_id}",
                "estimated_time_seconds": max(duration / 60, 5),
                "poll_interval_seconds": 5
            }
        )
        
    except HTTPException:
        if os.path.exists(input_path):
            os.remove(input_path)
        raise
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Check job status
    Wanneer completed: krijg video_url voor gebruik in fal.ai!
    """
    logger.info(f"🔍 Status check: {job_id[:8]}")
    
    job_info = get_job_info(job_id)
    
    if not job_info:
        raise HTTPException(status_code=404, detail="Job niet gevonden")
    
    status = job_info.get('status')
    progress = job_info.get('progress', 0)
    
    logger.info(f"📊 Job {job_id[:8]}: {status} - {progress}%")
    
    # Return job info (inclusief video_url als completed)
    return job_info


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
    
    total_jobs = len(list(JOBS_DIR.glob("*.json")))
    total_videos = len(list(OUTPUT_DIR.glob("*.mp4")))
    
    return {
        "status": "🟢 Healthy",
        "version": "5.0 - Direct Video URL",
        "ffmpeg": {
            "status": ffmpeg_status,
            "version": ffmpeg_version
        },
        "storage": {
            "total_jobs": total_jobs,
            "total_videos": total_videos,
            "cleanup_interval": "16 hours",
            "storage_type": "persistent" if os.path.exists('/data') else "ephemeral"
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting API with direct video URL support...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
