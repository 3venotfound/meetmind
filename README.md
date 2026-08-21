# MeetMind

MeetMind records one combined meeting video, extracts spoken meeting memory with
Gemini, finds whiteboard and slide evidence with OpenCV/EasyOCR, and stores the
validated result in a local SQLite database. The browser UI talks only to the
FastAPI backend; API keys and physical file paths never reach the browser.

## Architecture

- `frontend/`: static HTML/CSS/JavaScript client using `MediaRecorder`.
- `backend/`: FastAPI API, SQLite repositories, upload storage, orchestration,
  permanent evidence storage, and isolated subprocess adapters.
- `ai/`: Gemini recording extraction, change explanations, and evidence search.
- `cv/`: sampled-frame detection, OCR, and visual evidence generation.

AI and CV run in child Python processes. Their optional executable settings let
each worker use a separate environment while FastAPI stays lightweight and does
not import Gemini, OpenCV, Torch, or EasyOCR at startup.

## Requirements

Use Python **3.10.10**. On Windows, use three virtual environments so the large
CV stack cannot conflict with the API or Gemini SDK:

```powershell
py -3.10 -m venv backend\venv
py -3.10 -m venv ai\venv
py -3.10 -m venv cv\venv
backend\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
ai\venv\Scripts\python.exe -m pip install -r ai\requirements.txt
cv\venv\Scripts\python.exe -m pip install -r cv\requirements.txt
```

The CV installation is large because EasyOCR uses Torch and torchvision. Do not
commit any virtual environment or downloaded OCR model weights.

## Safe configuration

Copy `backend/.env.example` to `backend/.env`. Put `GEMINI_API_KEY` only in the
untracked `.env` file. Never put a real key in source, command arguments, or the
browser. For separate worker environments set:

```dotenv
AI_PYTHON_EXECUTABLE=C:\absolute\path\to\meetmind\ai\venv\Scripts\python.exe
CV_PYTHON_EXECUTABLE=C:\absolute\path\to\meetmind\cv\venv\Scripts\python.exe
```

Blank values use the Python executable running FastAPI. Database and storage
paths are resolved relative to `backend/`, independent of the launch directory.

## Run locally

Start the API:

```powershell
Set-Location backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger is at <http://127.0.0.1:8000/docs>. In a second terminal serve the UI
over HTTP (opening `index.html` through `file://` is not supported):

```powershell
Set-Location frontend
..\backend\venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1
```

Open <http://127.0.0.1:5500>. The configured CORS origins cover both
`127.0.0.1:5500` and `localhost:5500`. The tracked frontend is configured to
send API requests to the deployed Render API at
<https://meetmind-ux7u.onrender.com>; use local Swagger when testing the local
backend directly.

## Render and Netlify

Render must use `backend` as its root directory so the existing module layout
and requirements file are found. The compatible start command is:

```text
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Set this Render environment variable without a trailing slash on any origin:

```dotenv
CORS_ORIGINS=https://meetmiind.netlify.app,http://127.0.0.1:5500,http://localhost:5500
```

Netlify can publish the repository root (`index.html`) or the `frontend`
directory. Both tracked entrypoints use the same deployed API base URL. The
deployed frontend is <https://meetmiind.netlify.app/>.

## Tests

Automated tests use fake subprocess runners and never call Gemini, OpenCV,
EasyOCR, model downloads, or external networks:

```powershell
Set-Location backend
.\venv\Scripts\python.exe -m unittest discover -s tests -v
Set-Location ..
.\backend\venv\Scripts\python.exe -m compileall -q backend\app backend\tests ai cv
```

## EasyOCR weights and codec notes

The first real `easyocr.Reader(["en"])` initialization downloads English model
weights to EasyOCR's user model directory when internet access is available. To
prepare them before a demo, activate the CV environment once on a trusted
network and run:

```powershell
cv\venv\Scripts\python.exe -c "import easyocr; easyocr.Reader(['en'], gpu=False)"
```

This download is not part of automated tests and the weights must not be added
to Git. OpenCV container support depends on the codecs shipped with its FFmpeg
build. MP4/H.264 and WebM/VP8 are usually the safest inputs; some H.265 MP4 and
VP9 profiles may fail to decode. Browser MediaRecorder prefers WebM/VP9 or VP8.

## Short real smoke test

1. Configure the untracked `backend/.env`, including the Gemini key and worker
   executable paths.
2. Start FastAPI and the frontend server as above.
3. Open the frontend, add a participant, record 10–20 seconds while showing a
   high-contrast board or slide, then stop.
4. Wait for the synchronous `/process` request. Confirm Summary, Memory, and
   evidence display real stored results.
5. Check `GET /api/meetings/{meeting_id}` in Swagger and confirm the meeting is
   `processed`. Ask a question containing terms present in the transcript or OCR.

For a CV-only diagnostic after weights are prepared:

```powershell
Set-Location cv
.\venv\Scripts\python.exe -c "from pipeline import process_meeting_video; print(process_meeting_video(r'C:\path\short.webm','smoke'))"
```

Run this from a disposable directory if you do not want local frame/evidence
output beside the command.

## Known MVP limitations

- Processing is synchronous and may exceed browser/proxy timeouts for long video.
- Gemini supplies estimated fact timestamps; transcript lines have NULL timestamps
  because accurate segment timing is unavailable.
- Visual extraction uses one structured Gemini pass over OCR text and associates a
  fact with the best matching evidence image.
- Failed meetings are not retried in place; record a new meeting for the MVP.
- Codec availability and OCR accuracy vary by operating system, lighting, camera
  motion, board size, and installed OpenCV/Torch builds.
