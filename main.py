from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import uuid
from pathlib import Path
import tempfile
import shutil

app = FastAPI(title="Foto Verlenger API")

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
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def create_video_from_image(input_path: str, output_path: str, duration: int = 5, fps: int = 30, width: int = 1920, height: int = 1080):
    """
    Creëert een video van een statische foto voor een bepaalde duur
    """
    cmd = [
        'ffmpeg',
        '-loop', '1',  # Loop de input afbeelding
        '-i', input_path,  # Input afbeelding
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',  # Schaal en centreer
        '-c:v', 'libx264',  # Video codec
        '-t', str(duration),  # Duur in seconden
        '-pix_fmt', 'yuv420p',  # Pixel format voor compatibiliteit
        '-r', str(fps),  # Frame rate
        '-y',  # Overschrijf output bestand
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Video conversie mislukt: {e.stderr}")


@app.get("/")
async def root():
    return {
        "message": "Foto Verlenger API - Maak een video van je foto",
        "endpoints": {
            "/extend": "POST - Upload foto en kies duur voor video"
        },
        "parameters": {
            "file": "Afbeelding bestand (JPG, PNG, etc.)",
            "duration": "Duur in seconden (standaard: 5, max: 1800 = 30 minuten)",
            "fps": "Frames per seconde (standaard: 30)",
            "width": "Video breedte in pixels (standaard: 1920)",
            "height": "Video hoogte in pixels (standaard: 1080)"
        }
    }


@app.post("/extend")
async def extend_photo(
    file: UploadFile = File(...),
    duration: int = Form(default=5),
    fps: int = Form(default=30),
    width: int = Form(default=1920),
    height: int = Form(default=1080)
):
    """
    Upload een foto en krijg een video terug van de gekozen duur
    
    Parameters:
    - file: De foto die je wilt verlengen
    - duration: Duur van de video in seconden (standaard: 5, max: 1800 = 30 minuten)
    - fps: Frames per seconde (standaard: 30)
    - width: Video breedte (standaard: 1920)
    - height: Video hoogte (standaard: 1080)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Alleen afbeeldingen zijn toegestaan")
    
    # Validatie
    if duration < 1 or duration > 1800:
        raise HTTPException(status_code=400, detail="Duur moet tussen 1 en 1800 seconden (30 minuten) zijn")
    
    if fps < 1 or fps > 60:
        raise HTTPException(status_code=400, detail="FPS moet tussen 1 en 60 zijn")
    
    # Unieke bestandsnamen genereren
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}_{file.filename}"
    output_filename = f"{file_id}_video_{duration}s.mp4"
    
    input_path = UPLOAD_DIR / input_filename
    output_path = OUTPUT_DIR / output_filename
    
    try:
        # Bestand opslaan
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Video creëren
        create_video_from_image(str(input_path), str(output_path), duration, fps, width, height)
        
        # Input bestand verwijderen
        input_path.unlink()
        
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=output_filename,
            background=None
        )
    
    except Exception as e:
        # Cleanup bij fout
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint voor Render"""
    # Check of FFmpeg beschikbaar is
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        ffmpeg_status = "available"
    except:
        ffmpeg_status = "not available"
    
    return {
        "status": "healthy",
        "ffmpeg": ffmpeg_status
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
