import os
import io
import json
import logging
import hashlib
import csv
import urllib.parse
import time
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, url_for, redirect
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from gtts import gTTS
import base64
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import httpx
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import jwt
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from gradcam import generate_gradcam_overlay, find_target_layer

# Select CUDA when available; otherwise use CPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Image Preprocessing Utilities
# ---------------------------------------------------------------------------
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_DIMENSION_FOR_PROCESSING = 2048  # Resize larger images to this
SUPPORTED_FORMATS = {'JPEG', 'PNG', 'WEBP'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}

def preprocess_uploaded_image(image_bytes: bytes) -> Image.Image:
    """
    Preprocess an uploaded image for model inference.

    Handles:
    - EXIF orientation correction
    - RGBA/P/RGB mode conversion
    - Format validation (JPEG, PNG, WebP)
    - Dimension limits (resizes if too large)
    - Safe RGB conversion

    Returns:
        PIL Image in RGB mode, ready for model preprocessing

    Raises:
        ValueError: If image is invalid or unsupported format
    """
    try:
        # Open image without loading full data first
        image = Image.open(io.BytesIO(image_bytes))

        # Validate format
        if image.format not in SUPPORTED_FORMATS:
            raise ValueError(f'Unsupported image format: {image.format}. Supported: JPEG, PNG, WebP')

        # Load the image data
        image.load()

        # Check dimensions after loading - resize if too large
        width, height = image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            # Resize to max dimension while maintaining aspect ratio
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)

        # Apply EXIF orientation correction
        image = ImageOps.exif_transpose(image)

        # Resize if too large for processing (maintain aspect ratio)
        width, height = image.size
        if width > MAX_IMAGE_DIMENSION_FOR_PROCESSING or height > MAX_IMAGE_DIMENSION_FOR_PROCESSING:
            image.thumbnail((MAX_IMAGE_DIMENSION_FOR_PROCESSING, MAX_IMAGE_DIMENSION_FOR_PROCESSING), Image.LANCZOS)

        # Convert to RGB (handles RGBA, P, L, etc.)
        if image.mode != 'RGB':
            # For RGBA, composite on white background
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode == 'P':
                # Palette mode - convert via RGBA to handle transparency
                image = image.convert('RGBA')
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                # Other modes (L, CMYK, etc.)
                image = image.convert('RGB')

        return image

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f'Failed to process image: {str(e)}')


def validate_image_file(file_storage) -> tuple[bool, str | None]:
    """
    Validate uploaded file before processing.

    Returns:
        (is_valid, error_message)
    """
    if not file_storage or not file_storage.filename:
        return False, 'No file provided'

    filename = file_storage.filename.lower()
    if not ('.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS):
        return False, 'Invalid file type. Allowed: PNG, JPG, JPEG, WebP'

    # Check MIME type
    if file_storage.mimetype not in ALLOWED_MIME_TYPES:
        return False, f'Invalid MIME type: {file_storage.mimetype}. Allowed: image/jpeg, image/png, image/webp'

    # Check file size (Flask's MAX_CONTENT_LENGTH handles this but we check early)
    file_storage.seek(0, io.SEEK_END)
    size = file_storage.tell()
    file_storage.seek(0)
    if size > MAX_IMAGE_SIZE:
        return False, f'File too large: {size / (1024*1024):.1f}MB. Maximum: {MAX_IMAGE_SIZE / (1024*1024)}MB'
    if size == 0:
        return False, 'Empty file'

    return True, None


def validate_image_content(file_stream) -> tuple[bool, str | None]:
    """Verify file is actually an image by checking magic bytes."""
    file_stream.seek(0)
    header = file_stream.read(12)
    file_stream.seek(0)

    image_signatures = {
        b'\x89PNG\r\n\x1a\n': 'png',
        b'\xff\xd8\xff': 'jpg',
        b'RIFF': 'webp',  # WebP starts with RIFF
    }

    for sig, fmt in image_signatures.items():
        if header.startswith(sig):
            # Additional check for WebP - needs 'WEBP' at offset 8
            if fmt == 'webp':
                file_stream.seek(8)
                webp_header = file_stream.read(4)
                file_stream.seek(0)
                if webp_header != b'WEBP':
                    continue
            return True, fmt

    return False, None


class RetryableHTTPError(Exception):
    def __init__(self, status_code, message=''):
        self.status_code = status_code
        super().__init__(f'HTTP {status_code}: {message}')


class FatalHTTPError(Exception):
    def __init__(self, status_code, message=''):
        self.status_code = status_code
        super().__init__(f'HTTP {status_code}: {message}')


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
FATAL_STATUS_CODES = {401, 403, 404}


def _should_retry(exc):
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, RetryableHTTPError):
        return True
    return False


def _classify_http_error(status_code, base_message):
    if status_code in RETRYABLE_STATUS_CODES:
        raise RetryableHTTPError(status_code, base_message)
    elif status_code in FATAL_STATUS_CODES:
        raise FatalHTTPError(status_code, base_message)
    else:
        raise RetryableHTTPError(status_code, base_message)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openai/gpt-4.1-mini')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
NVIDIA_MODEL = os.getenv('NVIDIA_MODEL', 'meta/llama-3.3-70b-instruct')
# Total wall-clock budget for the whole advisory fallback chain of one request.
# Without it a request waits for every configured provider in turn (up to ~45s
# when all of them are unhealthy) even though the diagnosis itself is already
# computed locally in well under a second.
ADVISORY_LLM_BUDGET_S = float(os.getenv('ADVISORY_LLM_BUDGET_S', '8'))
PORT = int(os.getenv('PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
ALLOWED_ORIGINS = os.getenv(
    'ALLOWED_ORIGINS',
    'http://localhost:5000,https://ch-ai.netlify.app'
).split(',')

# Optional API keys: allow the app to start and return graceful errors when
# these services are not configured.
missing = []
if not OPENROUTER_API_KEY:
    missing.append('OPENROUTER_API_KEY')
if not OPENWEATHER_API_KEY:
    missing.append('OPENWEATHER_API_KEY')
if not GEMINI_API_KEY:
    missing.append('GEMINI_API_KEY')
if not NVIDIA_API_KEY:
    missing.append('NVIDIA_API_KEY')
if missing:
    logger.warning('Missing optional environment variables: %s', ', '.join(missing))

# ---------------------------------------------------------------------------
# Unified AI Service
# ---------------------------------------------------------------------------
class AIService:
    def __init__(self):
        self.nvidia_client = None
        self.openrouter_client = None
        self.gemini_model = None
        self._init_clients()

    def _init_clients(self):
        if NVIDIA_API_KEY:
            try:
                self.nvidia_client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=NVIDIA_API_KEY,
                    timeout=5.0,
                    max_retries=0,
                )
                logger.info('NVIDIA client initialized')
            except Exception as e:
                logger.error('Failed to initialize NVIDIA client: %s', e, exc_info=True)

        if OPENROUTER_API_KEY:
            try:
                self.openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY,
                    timeout=10.0,
                    max_retries=0,
                )
                logger.info('OpenRouter client initialized')
            except Exception as e:
                logger.error('Failed to initialize OpenRouter client: %s', e, exc_info=True)

        if GEMINI_API_KEY:
            try:
                self.gemini_model = genai.Client(
                    api_key=GEMINI_API_KEY,
                    http_options=types.HttpOptions(timeout=10000, retry_options=types.HttpRetryOptions(attempts=1)),
                )
                logger.info('Gemini client initialized')
            except Exception as e:
                logger.error('Failed to initialize Gemini client: %s', e, exc_info=True)

    def _mask_key(self, key):
        if not key:
            return 'None'
        return key[:8] + '...' if len(key) > 8 else key

    def _log_provider_attempt(self, provider, model, status=None, duration=None, error=None):
        parts = [f'provider={provider}', f'model={model}']
        if status is not None:
            parts.append(f'status={status}')
        if duration is not None:
            parts.append(f'duration_ms={duration}')
        if error:
            parts.append(f'error={error}')
        logger.info('AI request: %s', ', '.join(parts))

    @retry(
        stop=stop_after_attempt(1),
        wait=wait_exponential(multiplier=1, min=1, max=2),
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError, RetryableHTTPError)),
        reraise=True
    )
    def _call_nvidia(self, messages, max_tokens, temperature, timeout=None):
        if self.nvidia_client is None:
            raise ValueError('NVIDIA client not configured')
        start = time.time()
        extra = {} if timeout is None else {'timeout': timeout}
        try:
            response = self.nvidia_client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={'reasoning_effort': 'low'},
                **extra,
            )
            duration = int((time.time() - start) * 1000)
            content = response.choices[0].message.content
            if content is None:
                raise ValueError('NVIDIA returned empty response content')
            self._log_provider_attempt('NVIDIA', NVIDIA_MODEL, status=200, duration=duration)
            return content, None
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            body = None
            if hasattr(e, 'response') and e.response is not None:
                try:
                    body = e.response.text
                except Exception:
                    body = '<unreadable>'
            error_msg = f'NVIDIA failed (status={status}): {e}'
            if body:
                error_msg += f'. Response body: {body}'
            self._log_provider_attempt('NVIDIA', NVIDIA_MODEL, status=status, duration=duration, error=error_msg)
            if status and status not in RETRYABLE_STATUS_CODES and status not in FATAL_STATUS_CODES:
                raise
            _classify_http_error(status or 500, str(e))

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError, RetryableHTTPError)),
        reraise=True
    )
    def _call_openrouter(self, messages, max_tokens, temperature, timeout=None):
        if self.openrouter_client is None:
            raise ValueError('OpenRouter client not configured')
        start = time.time()
        extra = {} if timeout is None else {'timeout': timeout}
        try:
            response = self.openrouter_client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
            duration = int((time.time() - start) * 1000)
            content = response.choices[0].message.content
            if content is None:
                raise ValueError('OpenRouter returned empty response content')
            self._log_provider_attempt('OpenRouter', OPENROUTER_MODEL, status=200, duration=duration)
            return content, None
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            body = None
            if hasattr(e, 'response') and e.response is not None:
                try:
                    body = e.response.text
                except Exception:
                    body = '<unreadable>'
            error_msg = f'OpenRouter failed (status={status}): {e}'
            if body:
                error_msg += f'. Response body: {body}'
            self._log_provider_attempt('OpenRouter', OPENROUTER_MODEL, status=status, duration=duration, error=error_msg)
            if status and status not in RETRYABLE_STATUS_CODES and status not in FATAL_STATUS_CODES:
                raise
            _classify_http_error(status or 500, str(e))

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        retry=retry_if_exception_type((
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            RetryableHTTPError,
            httpx.TimeoutException,
            httpx.ConnectError,
        )),
        reraise=True
    )
    def _call_gemini(self, messages, max_tokens, temperature, timeout=None):
        if self.gemini_model is None:
            raise ValueError('Gemini client not configured')

        start = time.time()
        model_name = 'gemini-3.8-flash'

        try:
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"[System instruction: {content}]")
                else:
                    prompt_parts.append(content)

            full_prompt = '\n\n'.join(prompt_parts)
            timeout_seconds = (
                min(10.0, max(0.1, float(timeout)))
                if timeout is not None else 10.0
            )

            interaction = self.gemini_model.interactions.create(
                model=model_name,
                input=full_prompt,
                generation_config={
                    'max_output_tokens': max_tokens,
                    'temperature': temperature,
                    'thinking_level': 'low',
                },
                timeout=int(timeout_seconds * 1000),
            )

            duration = int((time.time() - start) * 1000)
            response_text = getattr(interaction, 'output_text', None)
            if not response_text or not response_text.strip():
                status_value = getattr(interaction, 'status', 'unknown')
                raise ValueError(
                    f'Gemini returned no final text (status={status_value})'
                )

            self._log_provider_attempt(
                'Gemini', model_name, status=200, duration=duration
            )
            return response_text.strip(), None

        except Exception as e:
            duration = int((time.time() - start) * 1000)
            status = getattr(e, 'code', None)
            if status is None:
                status = getattr(e, 'status_code', None)
            if status is None:
                response = getattr(e, 'response', None)
                status = getattr(response, 'status_code', None)

            error_msg = f'Gemini failed (status={status}): {e}'
            self._log_provider_attempt(
                'Gemini', model_name, status=status,
                duration=duration, error=error_msg
            )

            if status and status not in RETRYABLE_STATUS_CODES and status not in FATAL_STATUS_CODES:
                raise

            if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
                raise RetryableHTTPError(status or 500, str(e)) from e

            _classify_http_error(status or 500, str(e))


    def _local_fallback(self, query):
        query_lower = query.lower()
        for disease, remedy in REMEDY_MAP.items():
            disease_lower = disease.lower()
            if any(part in query_lower for part in disease_lower.split('___')):
                return f"Cloud AI is currently unavailable. Using offline agricultural knowledge.\n\n{disease}: {remedy}"
        return "Cloud AI is currently unavailable. Using offline agricultural knowledge. Please upload a leaf photo for disease-specific advice, or try again later."

    def generate_ai_response(self, messages, max_tokens=1024, temperature=0.7):
        providers = []
        if self.nvidia_client:
            providers.append(('NVIDIA', self._call_nvidia))
        if self.openrouter_client:
            providers.append(('OpenRouter', self._call_openrouter))
        if self.gemini_model:
            providers.append(('Gemini', self._call_gemini))

        deadline = time.time() + ADVISORY_LLM_BUDGET_S
        last_error = None
        for provider_name, provider_func in providers:
            remaining = deadline - time.time()
            if remaining <= 0:
                logger.warning(
                    'Advisory LLM budget of %ss exhausted before %s; '
                    'serving the local fallback instead',
                    ADVISORY_LLM_BUDGET_S, provider_name)
                last_error = f'budget of {ADVISORY_LLM_BUDGET_S}s exhausted'
                break
            try:
                logger.info('Trying provider: %s (budget_remaining=%.1fs)',
                            provider_name, remaining)
                content, error = provider_func(messages, max_tokens, temperature,
                                               timeout=remaining)
                if error:
                    last_error = error
                    logger.warning('Provider %s failed: %s', provider_name, error)
                    continue
                logger.info('Provider %s succeeded', provider_name)
                return content, None
            except Exception as e:
                last_error = str(e)
                logger.warning('Provider %s failed with exception: %s', provider_name, e, exc_info=True)
                continue

        logger.error('All providers failed or ran out of budget. Last error: %s', last_error)
        fallback = self._local_fallback(messages[-1].get('content', '') if messages else '')
        return None, fallback


ai_service = AIService()


def call_openrouter(messages, model=None, max_tokens=1024, temperature=0.7):
    return ai_service.generate_ai_response(messages, max_tokens, temperature)


def _call_gemini(messages, max_tokens=1024, temperature=0.7):
    return ai_service.generate_ai_response(messages, max_tokens, temperature)

# ---------------------------------------------------------------------------
# Optional database + auth scaffolding
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data.db')
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)


class PredictionRecord(Base):
    __tablename__ = 'prediction_records'
    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    remedy = Column(Text, nullable=True)
    lang = Column(String, nullable=False, default='en')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

Base.metadata.create_all(bind=engine)

JWT_SECRET = os.getenv('JWT_SECRET', 'krishivaani-local-test-secret-change-before-production-2026')
JWT_ALGO = 'HS256'


def create_token(user_id):
    exp = datetime.utcnow() + timedelta(days=7)
    return jwt.encode({"user_id": user_id, "exp": exp}, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Model Loading (robust path resolution)
# ---------------------------------------------------------------------------
# Resolve base dir relative to this file so the server works regardless of cwd
BASE_DIR = Path(__file__).parent.resolve()

# Candidate locations for the model and index file. This covers common layouts
# (repo root, model_files folder, or the current working directory).
CANDIDATE_MODEL_PATHS = [
    BASE_DIR / 'mobilenetv2_leaf_disease_final.pth',
    BASE_DIR / 'model_files' / 'mobilenetv2_leaf_disease_final.pth',
    Path.cwd() / 'mobilenetv2_leaf_disease_final.pth',
    Path.cwd() / 'model_files' / 'mobilenetv2_leaf_disease_final.pth',
]

CANDIDATE_INDEX_PATHS = [
    BASE_DIR / 'index_to_name.json',
    BASE_DIR / 'model_files' / 'index_to_name.json',
    Path.cwd() / 'index_to_name.json',
    Path.cwd() / 'model_files' / 'index_to_name.json',
]

MODEL_PATH = next((p for p in CANDIDATE_MODEL_PATHS if p.exists()), None)
INDEX_TO_NAME_PATH = next((p for p in CANDIDATE_INDEX_PATHS if p.exists()), None)

if MODEL_PATH is None:
    logger.error('Model file not found. Tried: %s', ', '.join(str(p) for p in CANDIDATE_MODEL_PATHS))
    raise RuntimeError('Model file not found. Please place mobilenetv2_leaf_disease_final.pth in the repo root or model_files/')

if INDEX_TO_NAME_PATH is None:
    logger.error('index_to_name.json not found. Tried: %s', ', '.join(str(p) for p in CANDIDATE_INDEX_PATHS))
    raise RuntimeError('index_to_name.json not found. Please ensure it exists in the repo root or model_files/')

logger.info(f'Using model file at: {MODEL_PATH}')
logger.info(f'Using index map at: {INDEX_TO_NAME_PATH}')

# The repo holds a second file with the same name under model_files/. It is NOT
# the audited production checkpoint: measured on the frozen 60-image external set
# it scores 2/60 (3.33%, 4.37% mean confidence) against the root checkpoint's
# 8/60 (13.33%). MODEL_PATH prefers the repo root, so this only matters if a
# deployment ships model_files/ without the root copy. Warn loudly rather than
# serving a different model quietly.
AUDITED_MODEL_SHA256 = os.getenv(
    'AUDITED_MODEL_SHA256',
    '9A6B3EB1F3D48FB34831E13C840A6D09E481A6D7588807390661DF74BC5FB850',
).strip().upper()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()


MODEL_SHA256 = _sha256(MODEL_PATH)

EXPECTED_PRODUCTION_MODEL_SHA256 = (
    '9A6B3EB1F3D48FB34831E13C840A6D09E481A6D7588807390661DF74BC5FB850'
)

# Ensure the configured hash cannot override the approved production hash.
if AUDITED_MODEL_SHA256 != EXPECTED_PRODUCTION_MODEL_SHA256:
    raise RuntimeError(
        'AUDITED_MODEL_SHA256 does not match the approved production hash. '
        'Refusing to start.'
    )

# Refuse to start if the wrong model checkpoint is selected.
if MODEL_SHA256 != EXPECTED_PRODUCTION_MODEL_SHA256:
    raise RuntimeError(
        f'Model checkpoint SHA256 mismatch. '
        f'Expected: {EXPECTED_PRODUCTION_MODEL_SHA256}; '
        f'Actual: {MODEL_SHA256}; '
        f'Loaded from: {MODEL_PATH}. '
        f'Refusing to start.'
    )

logger.info(
    'Model checkpoint matches the audited production SHA256'
)

# Load the class index mapping exactly once.
with open(INDEX_TO_NAME_PATH, 'r') as f:
    class_to_name = json.load(f)
    idx_to_class = {int(k): v for k, v in class_to_name.items()}
# ---------------------------------------------------------------------------
# Training Status and Background Training
# ---------------------------------------------------------------------------
TRAINING_STATUS_PATH = BASE_DIR / 'training_status.json'
training_lock = threading.Lock()
training_thread = None


def read_training_status():
    """Read the latest training status from the JSON file."""
    if not TRAINING_STATUS_PATH.exists():
        return {
            'status': 'not_started',
            'epoch': 0,
            'total_epochs': 0,
            'metrics': {},
            'updated_at': None,
        }
    try:
        with open(TRAINING_STATUS_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {
            'status': 'not_started',
            'epoch': 0,
            'total_epochs': 0,
            'metrics': {},
            'updated_at': None,
        }


def run_training_in_background():
    """Run the training script in a separate thread so the server stays responsive."""
    try:
        import train as train_module
        train_module.main()
    except Exception as e:
        logger.error('Training failed: %s', e, exc_info=True)
        # Write failure status
        failure_status = {
            'status': 'failed',
            'epoch': 0,
            'total_epochs': 0,
            'metrics': {},
            'error': str(e),
            'updated_at': int(time.time() * 1000),
        }
        with open(TRAINING_STATUS_PATH, 'w') as f:
            json.dump(failure_status, f)


def load_model(model_path):
    logger.info(f'Loading model from {model_path}')
    try:
        model = models.mobilenet_v2(weights=None, num_classes=len(idx_to_class))
    except TypeError:
        model = models.mobilenet_v2(num_classes=len(idx_to_class))

    raw = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(raw, dict):
        if 'state_dict' in raw:
            state_dict = raw['state_dict']
        elif 'model_state_dict' in raw:
            state_dict = raw['model_state_dict']
        else:
            state_dict = raw
    else:
        state_dict = raw

    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if isinstance(k, str) else k
        new_state[new_key] = v

    model_state = model.state_dict()
    filtered_state = {}
    for k, v in new_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            logger.debug(f'Skipping mismatched key: {k} (checkpoint shape: {v.shape}, model shape: {model_state.get(k, "N/A")})')

    model.load_state_dict(filtered_state, strict=False)
    missing = [k for k in model_state.keys() if k not in filtered_state]
    if missing:
        logger.warning('Some layers were randomly initialized: %s', missing[:10])

    model = model.to(DEVICE)
    model.eval()
    logger.info('Model loaded successfully on device: %s', DEVICE)
    return model


model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# Leaf-suitability gate
# ---------------------------------------------------------------------------
# The model is a 38-class leaf-disease classifier, so it will happily name a disease
# for a photograph of a face, a laptop or a wall, and it does so *confidently*: the
# measured external accuracy of the baseline is 13.33% at 54.05% mean confidence.
# Confidence therefore cannot detect non-leaf input. This gate decides leaf
# suitability first and withholds the diagnosis when it is not satisfied.
#
# Ordering is deliberate: quality -> leaf suitability -> classification -> advisory.
# A refused image yields no class, no confidence, no top-3, no Grad-CAM and no remedy.
#
# LEAF_GATE_ENABLED=0 disables the gate and restores ungated behaviour, which exists
# only so that the pre-gate behaviour can still be measured. It is off by default.
try:
    import leaf_gate as leaf_gate_module
    LEAF_GATE = leaf_gate_module.LeafGate.load()
    LEAF_GATE_ENABLED = (os.getenv('LEAF_GATE_ENABLED', '1').strip().lower()
                         in ('1', 'true', 'yes', 'on')) and LEAF_GATE.enabled
    if LEAF_GATE_ENABLED:
        logger.info('Leaf gate active: %s (accept >= %.4f, reject < %.4f)',
                    LEAF_GATE.source, LEAF_GATE.accept_threshold,
                    LEAF_GATE.reject_threshold)
    elif LEAF_GATE.source:
        logger.warning('Leaf gate DISABLED by LEAF_GATE_ENABLED although %s exists',
                       LEAF_GATE.source)
    else:
        logger.warning('Leaf gate inactive: no prototypes or config at %s',
                       leaf_gate_module.DEFAULT_CONFIG_PATH)
except Exception as exc:
    LEAF_GATE = None
    LEAF_GATE_ENABLED = False
    logger.error('Leaf gate could not be initialised (%s); /api/analyze is ungated', exc)

# ---------------------------------------------------------------------------
# Dedicated leaf / non-leaf classifier
# ---------------------------------------------------------------------------
# The heuristic gate above scores leaf-ness from a vegetation statistic, a texture
# statistic, a kNN embedding distance and a disease-posterior statistic. Measured over
# 456 studio leaves, 153 real-world leaves and 27 non-leaf photographs, that mixture
# separated real-world leaves from non-leaf images barely at all: the two distributions
# of kNN distance overlap, because the prototype bank is built only from
# PlantVillage studio imagery and a real field photograph sits as far from it as a
# photograph of a car does. The cost was severe - only 46% of genuine real-world leaf
# photographs could be diagnosed at a threshold that admitted no non-leaf image.
#
# This classifier is trained for exactly the leaf / non-leaf question on a dataset that
# includes real-world leaves, and it is the primary leaf-ness authority when present.
# The heuristic gate is retained as the fallback, because it is safe if blunt, and the
# image-quality rules in it are kept regardless: focus, exposure, resolution and
# blankness are different questions from leaf-ness and are unaffected by this change.
#
# Failure behaviour is deliberate and is the one place this differs from the code above.
# If neither the classifier nor the heuristic gate is available, the upload is REFUSED.
# The previous behaviour was to log the failure and serve ungated diagnoses, which means
# the one configuration where the gate is missing is the one configuration that hands
# out disease names for photographs of walls. A safety gate that fails open is worse
# than no gate, because it looks like protection.
LEAF_CLASSIFIER = None
LEAF_CLASSIFIER_ENABLED = (
    os.getenv('LEAF_CLASSIFIER_ENABLED', '1').strip().lower()
    in ('1', 'true', 'yes', 'on'))
if LEAF_CLASSIFIER_ENABLED:
    try:
        import leaf_classifier as leaf_classifier_module
        LEAF_CLASSIFIER = leaf_classifier_module.load()
        if LEAF_CLASSIFIER is None:
            logger.warning('leaf classifier unavailable; falling back to the heuristic '
                           'leaf gate, which is safe but has low real-world recall')
        else:
            logger.info('Leaf classifier active: %s', json.dumps(
                LEAF_CLASSIFIER.describe()))
    except Exception as exc:
        LEAF_CLASSIFIER = None
        logger.error('leaf classifier could not be initialised (%s); falling back to '
                     'the heuristic leaf gate', exc)
else:
    logger.info('leaf classifier DISABLED by LEAF_CLASSIFIER_ENABLED')

# Test-only bypass. The original gate had LEAF_GATE_ENABLED=0 for exactly this purpose:
# letting the ungated path be measured and letting plumbing tests run a synthetic image
# through the endpoint. That escape hatch is kept, because removing it would mean
# rewriting those tests to no longer test what they test, but it is now a single
# explicit flag rather than something reachable by disabling either validator, and it
# defaults to off. It must never be set in a deployed environment: with it on, a
# photograph of a wall receives a disease diagnosis.
LEAF_VALIDATION_BYPASS = (
    os.getenv('LEAF_VALIDATION_BYPASS', '0').strip().lower()
    in ('1', 'true', 'yes', 'on'))


def leaf_validation_authority():
    """Which validator decides leaf-ness for this request.

    Evaluated per request rather than once at import, so that toggling
    LEAF_CLASSIFIER_ENABLED or LEAF_GATE_ENABLED at runtime behaves the way the rest of
    this module's configuration does, and so the test harness can exercise each
    validator in isolation.
    """
    if LEAF_VALIDATION_BYPASS:
        return 'bypass'
    if LEAF_CLASSIFIER is not None and LEAF_CLASSIFIER_ENABLED:
        return 'classifier'
    if LEAF_GATE is not None and LEAF_GATE_ENABLED:
        return 'heuristic_gate'
    return 'none'


LEAF_VALIDATION_AUTHORITY = leaf_validation_authority()
logger.info('leaf validation authority: %s', LEAF_VALIDATION_AUTHORITY)

# Crop name per class index, used by the gate's class-coherence term.
LEAF_CROPS = [idx_to_class[i].split('___')[0] for i in range(len(idx_to_class))]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------------------------
# Remedy Map
# ---------------------------------------------------------------------------
REMEDY_MAP = {
    'Apple___Apple_scab': 'Apply fungicides and practice good sanitation by removing infected leaves.',
    'Apple___Black_rot': 'Prune infected branches and apply appropriate fungicides.',
    'Apple___Cedar_apple_rust': 'Remove nearby cedar trees if possible and apply fungicides.',
    'Apple___healthy': 'Your plant looks healthy! Maintain good growing conditions.',
    'Blueberry___healthy': 'Your blueberry plant looks healthy! Maintain good growing conditions.',
    'Cherry___Powdery_mildew': 'Apply sulfur or potassium bicarbonate-based fungicides.',
    'Cherry___healthy': 'Your cherry plant looks healthy! Maintain good growing conditions.',
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot': 'Use resistant varieties and apply appropriate fungicides.',
    'Corn___Common_rust': 'Plant resistant varieties and apply fungicides if necessary.',
    'Corn___Northern_Leaf_Blight': 'Rotate crops and use resistant varieties.',
    'Corn___healthy': 'Your corn plant looks healthy! Maintain good growing conditions.',
    'Grape___Black_rot': 'Apply fungicides and remove infected plant material.',
    'Grape___Esca__Black_Measles_': 'Prune infected vines and apply appropriate treatments.',
    'Grape___Leaf_blight__Isariopsis_Leaf_Spot_': 'Apply fungicides and improve air circulation.',
    'Grape___healthy': 'Your grape plant looks healthy! Maintain good growing conditions.',
    'Orange___Haunglongbing__Citrus_greening_': 'Remove infected trees and control psyllid vectors with insecticides.',
    'Peach___Bacterial_spot': 'Apply copper-based sprays and avoid overhead irrigation.',
    'Peach___healthy': 'Your peach plant looks healthy! Maintain good growing conditions.',
    'Pepper__bell___Bacterial_spot': 'Apply copper-based fungicides and avoid overhead irrigation.',
    'Pepper__bell___healthy': 'Your pepper plant looks healthy! Maintain good growing conditions.',
    'Potato___Early_blight': 'Rotate crops and apply fungicides preventatively.',
    'Potato___Late_blight': 'Remove and destroy infected plants, apply fungicides.',
    'Potato___healthy': 'Your potato plant looks healthy! Maintain good growing conditions.',
    'Raspberry___healthy': 'Your raspberry plant looks healthy! Maintain good growing conditions.',
    'Soybean___healthy': 'Your soybean plant looks healthy! Maintain good growing conditions.',
    'Squash___Powdery_mildew': 'Apply fungicides and ensure good air circulation.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves and apply fungicides if needed.',
    'Strawberry___healthy': 'Your strawberry plant looks healthy! Maintain good growing conditions.',
    'Tomato___Bacterial_spot': 'Apply copper-based fungicides, avoid overhead irrigation, and remove infected leaves.',
    'Tomato___Early_blight': 'Rotate crops, apply fungicides preventatively, and remove infected leaves.',
    'Tomato___Late_blight': 'Remove and destroy infected plants immediately, apply fungicides, and avoid wetting foliage.',
    'Tomato___Leaf_Mold': 'Improve air circulation, reduce humidity, and apply appropriate fungicides.',
    'Tomato___Septoria_leaf_spot': 'Remove infected leaves, avoid overhead watering, and apply fungicides.',
    'Tomato___Spider_mites_Two-spotted_spider_mite': 'Use insecticidal soap or miticides, and increase humidity around plants.',
    'Tomato___Target_Spot': 'Apply fungicides, practice crop rotation, and remove infected plant debris.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whitefly vectors with insecticides or reflective mulches, and remove infected plants.',
    'Tomato___Tomato_mosaic_virus': 'Remove and destroy infected plants, disinfect tools, and control aphid vectors.',
    'Tomato___healthy': 'Your tomato plant looks healthy! Continue good cultural practices and regular monitoring.',
}

import concurrent.futures

# Valid image extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_IMAGE_SIZE = 16 * 1024 * 1024  # 16MB
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))

# Request timeouts
ANALYZE_TIMEOUT_SECONDS = float(os.getenv('ANALYZE_TIMEOUT_SECONDS', '30'))
GRADCAM_TIMEOUT_SECONDS = float(os.getenv('GRADCAM_TIMEOUT_SECONDS', '10'))

# ---------------------------------------------------------------------------
# Optional confidence calibration (temperature scaling)
# ---------------------------------------------------------------------------
# Disabled unless CALIBRATION_FILE points at a temperature_*.json produced and
# validated by experiments/calibration/calibrate.py. Temperature scaling is a
# post-hoc transform of the logits: it never touches the checkpoint weights and
# it cannot change the argmax, so the predicted class is identical either way.
# The raw softmax confidence is ALWAYS still returned as `confidence`; the
# calibrated value is reported separately as `confidence_calibrated` so the two
# can be compared in production logs and in the frontend.
CALIBRATION_FILE = os.getenv('CALIBRATION_FILE')
CALIBRATION_TEMPERATURE = None
if CALIBRATION_FILE:
    try:
        with open(CALIBRATION_FILE, 'r') as fh:
            CALIBRATION_TEMPERATURE = float(json.load(fh)['temperature'])
        logger.info('Loaded calibration temperature T=%s from %s',
                    CALIBRATION_TEMPERATURE, CALIBRATION_FILE)
    except Exception as exc:
        logger.error('Could not load CALIBRATION_FILE %s (%s); '
                     'calibration stays disabled', CALIBRATION_FILE, exc)
        CALIBRATION_TEMPERATURE = None
else:
    logger.info('Confidence calibration disabled (set CALIBRATION_FILE to enable)')

# ---------------------------------------------------------------------------
# Optional: keep /api/analyze fully local
# ---------------------------------------------------------------------------
# The diagnosis endpoint enriches its answer with an LLM-written advisory. That
# makes a core prediction endpoint depend on third-party availability: the
# primary provider currently answers HTTP 410 (its model is end-of-life). The
# whole fallback chain is therefore bounded by ADVISORY_LLM_BUDGET_S (see the
# config block above), so a dead provider delays the response by at most that
# budget instead of being waited out provider by provider.
#
# Setting DISABLE_ADVISORY_LLM=1 keeps the endpoint entirely local. Static
# remedies from REMEDY_MAP are still used; only the network fallback and
# non-English translation are skipped, and the response is clearly marked so a
# caller can tell the difference. Default is off, so behaviour is unchanged
# unless explicitly enabled.
DISABLE_ADVISORY_LLM = os.getenv('DISABLE_ADVISORY_LLM', '').strip().lower() in (
    '1', 'true', 'yes', 'on')
if DISABLE_ADVISORY_LLM:
    logger.info('Advisory LLM disabled (DISABLE_ADVISORY_LLM): /api/analyze stays local')

# In-memory performance metrics (reset on server restart)
performance_metrics = {
    'total_predictions': 0,
    'total_inference_ms': 0.0,
    'total_preprocessing_ms': 0.0,
    'total_gradcam_ms': 0.0,
    'predictions_below_threshold': 0,
    'images_refused_by_gate': 0,
}


def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def predict_image(image_bytes):
    """Classify one image. Grad-CAM is not produced here.

    It is attached separately by `attach_gradcam`, after the leaf gate has accepted the
    image, so an attention overlay is never computed for a refused upload.
    """
    timings = {}

    # Preprocessing
    start = time.perf_counter()
    image = preprocess_uploaded_image(image_bytes)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    timings['preprocessing_ms'] = round((time.perf_counter() - start) * 1000, 2)

    # Inference. The backbone and the classifier are called separately so that the
    # pooled penultimate features can be handed to the leaf gate from the same single
    # forward pass. In eval mode this is exactly what model(batch) does internally:
    # MobileNetV2.forward is features -> adaptive_avg_pool2d -> flatten -> classifier,
    # and the dropout in front of the classifier is an identity in eval mode.
    start = time.perf_counter()
    with torch.no_grad():
        features = model.features(input_batch)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        output = model.classifier(pooled)
    timings['inference_ms'] = round((time.perf_counter() - start) * 1000, 2)

    _, predicted_idx = torch.max(output, 1)
    predicted_class = idx_to_class[predicted_idx.item()]
    probabilities = F.softmax(output, dim=1)
    confidence = float(probabilities.max().item())

    # Compute top-3 predictions for transparency
    top_k = 3
    top_probs, top_indices = torch.topk(probabilities, min(top_k, output.shape[1]))
    top_k_classes = []
    for i in range(len(top_probs[0])):
        idx = top_indices[0][i].item()
        top_k_classes.append({
            'class': idx_to_class[idx],
            'idx': idx,
            'confidence': round(float(top_probs[0][i].item()), 4)
        })

    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'predicted_idx': predicted_idx.item(),
        'top_k_classes': top_k_classes,
        'timings': timings,
        # Consumed by the leaf gate; stripped before the response leaves the server.
        '_embedding': pooled[0].detach().cpu().numpy().tolist(),
        '_probabilities': probabilities[0].detach().cpu().tolist(),
    }

    if CALIBRATION_TEMPERATURE:
        calibrated_probs = F.softmax(output / CALIBRATION_TEMPERATURE, dim=1)
        result['confidence_calibrated'] = float(calibrated_probs.max().item())

    timings['total_ms'] = round(sum(timings.values()), 2)
    logger.info(f"Timings: preprocessing={timings.get('preprocessing_ms', 0)}ms, "
                f"inference={timings.get('inference_ms', 0)}ms, "
                f"total={timings['total_ms']}ms")

    return result


def attach_gradcam(result, image_bytes):
    """Add the Grad-CAM overlay to a result that has already passed the leaf gate.

    Deliberately a separate step: an attention visualisation explains *why* the model
    named a disease, so producing one for an image that was refused would be work whose
    output must then be discarded, and the second forward pass is not free.
    """
    start = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(generate_gradcam_overlay, model, image_bytes)
            overlay_b64, heatmap_b64, _, gradcam_error = future.result(timeout=GRADCAM_TIMEOUT_SECONDS)
        result['gradcam_overlay'] = overlay_b64
        result['gradcam_heatmap'] = heatmap_b64
        if gradcam_error:
            result['gradcam_error'] = gradcam_error
    except concurrent.futures.TimeoutError:
        logger.error(f'Grad-CAM generation timed out after {GRADCAM_TIMEOUT_SECONDS}s')
        result['gradcam_error'] = 'Attention visualization timed out'
    except Exception as e:
        logger.error(f'Grad-CAM generation failed: {e}')
        result['gradcam_error'] = 'Failed to generate attention visualization'
    result['timings']['gradcam_ms'] = round((time.perf_counter() - start) * 1000, 2)
    return result


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE
testing = os.getenv('FLASK_TESTING', '0').strip().lower() in ('1', 'true', 'yes')

# Rate Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=['1000 per day', '200 per hour'],
    storage_uri='memory://'
)

# Restrict CORS to known origins
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# Ensure static directories exist
os.makedirs('static/images', exist_ok=True)


def translate_text(text, target_lang):
    if target_lang == 'en' or not text:
        return text
    prompt = (
        f"Translate the following agricultural advice into {target_lang}. "
        "Keep scientific names or pesticide ingredient names untranslated.\n\n"
        f"{text}"
    )
    content, error = call_openrouter(
        messages=[
            {"role": "system", "content": "You are a professional translator for agricultural content."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    if error or not content:
        return text
    return content.strip()


def generate_remedy_openrouter(disease_name, lang='en'):
    prompt = f"""
You are an expert plant pathologist. In one or two sentences, provide a practical field-level remedy for the following crop disease so that it is understandable by Indian farmers.
Return plain text with no bullet points or asterisks.
Language: {lang}
Disease: {disease_name}
"""
    content, error = call_openrouter(
        messages=[
            {"role": "system", "content": "You are an expert plant pathologist."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
        temperature=0.6,
    )
    if error or not content:
        return "Consult a local agricultural expert for specific treatment options."
    return content.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
@app.route('/index.html')
def serve_index():
    return send_file('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files with path traversal protection."""
    try:
        decoded_filename = urllib.parse.unquote(filename)
        static_dir = os.path.abspath(os.path.join(app.root_path, 'static'))
        file_path = os.path.abspath(os.path.join(static_dir, decoded_filename))

        if not file_path.startswith(static_dir + os.sep):
            logger.warning(f'Path traversal attempt blocked: {filename}')
            return jsonify({'error': 'Forbidden'}), 403

        if not os.path.isfile(file_path):
            return jsonify({'error': 'File not found'}), 404

        response = send_from_directory('static', decoded_filename)
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response
    except Exception as e:
        logger.error(f'Error serving static file {filename}: {e}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/images/<path:filename>')
def serve_legacy_images(filename):
    try:
        safe_name = os.path.basename(filename)
        return redirect(f'/static/images/{safe_name}')
    except Exception as e:
        logger.error(f'Error in legacy image handler: {e}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/main.js')
def serve_js():
    return send_file('main.js')


@app.route('/sw.js')
def serve_sw():
    return send_file('sw.js', mimetype='application/javascript')


@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='application/manifest+json')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/api/calibration', methods=['GET'])
@limiter.exempt
def calibration_status():
    """Report whether temperature-scaling calibration is active.

    `confidence` in /api/analyze is always the raw softmax value. When this
    endpoint reports `enabled: true`, responses additionally carry
    `confidence_calibrated` and `is_uncertain_calibrated`.
    """
    return jsonify({
        'enabled': CALIBRATION_TEMPERATURE is not None,
        'temperature': CALIBRATION_TEMPERATURE,
        'source_file': CALIBRATION_FILE if CALIBRATION_TEMPERATURE else None,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'uncertainty_rule': 'is_uncertain = confidence < confidence_threshold',
        'note': 'Temperature scaling is applied as softmax(logits / T). '
                'The raw confidence field is always present for comparison.',
    })


@app.route('/api/performance', methods=['GET'])
@limiter.exempt
def performance():
    """Return aggregated performance metrics for prediction inference."""
    total = performance_metrics['total_predictions']
    metrics = dict(performance_metrics)
    if total > 0:
        metrics['avg_inference_ms'] = round(performance_metrics['total_inference_ms'] / total, 2)
        metrics['avg_preprocessing_ms'] = round(performance_metrics['total_preprocessing_ms'] / total, 2)
        metrics['avg_gradcam_ms'] = round(performance_metrics['total_gradcam_ms'] / total, 2) if performance_metrics['total_gradcam_ms'] > 0 else 0.0
        metrics['uncertain_rate'] = round(performance_metrics['predictions_below_threshold'] / total * 100, 2)
    else:
        metrics['avg_inference_ms'] = 0.0
        metrics['avg_preprocessing_ms'] = 0.0
        metrics['avg_gradcam_ms'] = 0.0
        metrics['uncertain_rate'] = 0.0
    return jsonify(metrics)


@app.route('/api/training/status', methods=['GET'])
@limiter.exempt
def training_status():
    """Return the current training status and latest metrics."""
    return jsonify(read_training_status())


@app.route('/api/training/start', methods=['POST'])
@limiter.exempt
def start_training():
    """Start model training in a background thread.

    Returns immediately with a 202 Accepted response. Use `/api/training/status`
    to poll for progress updates.
    """
    global training_thread
    with training_lock:
        if training_thread is not None and training_thread.is_alive():
            return jsonify({'error': 'Training is already in progress', 'status': 'training'}), 409

        # Reset status file
        initial_status = {
            'status': 'training',
            'epoch': 0,
            'total_epochs': 15,
            'metrics': {},
            'updated_at': int(time.time() * 1000),
        }
        with open(TRAINING_STATUS_PATH, 'w') as f:
            json.dump(initial_status, f)

        training_thread = threading.Thread(target=run_training_in_background, daemon=True)
        training_thread.start()
        return jsonify({'message': 'Training started', 'status': 'training'}), 202


@app.route('/api/auth/register', methods=['POST'])
def register_user():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({'error': 'username and password are required'}), 400

    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            return jsonify({'error': 'User already exists'}), 400

        user = User(username=username, password_hash=generate_password_hash(password))
        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_token(user.id)
        return jsonify({
            'token': token,
            'user': {'id': user.id, 'username': user.username}
        })
    finally:
        db.close()


@app.route('/api/auth/login', methods=['POST'])
def login_user():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({'error': 'username and password are required'}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid credentials'}), 401

        token = create_token(user.id)
        return jsonify({
            'token': token,
            'user': {'id': user.id, 'username': user.username}
        })
    finally:
        db.close()


@app.route('/<path:path>')
def serve_file(path):
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    if os.path.exists(path):
        return send_file(path)
    return send_file('index.html')


def _classifier_verdict_to_decision(verdict):
    """Map a leaf-classifier verdict onto the leaf_gate decision shape.

    Both leaf validators return the same structure so that _gate_rejection_response, the
    frontend and the tests all have one contract to deal with. The classifier's own
    score is carried through as leaf_confidence, which keeps the response fields
    meaningful instead of inventing a parallel schema.
    """
    state = verdict.get('state')
    if state == 'NON_LEAF':
        state_name = 'NON_LEAF'
        error_type = leaf_gate_module.ERROR_NON_LEAF
    else:
        # Both LEAF_UNCERTAIN and an inference error land here. An image the classifier
        # could not evaluate must never reach the disease model, and "we could not
        # identify a leaf" is the honest description of that.
        state_name = 'UNCERTAIN'
        error_type = leaf_gate_module.ERROR_LEAF_UNCERTAIN
    return {
        'state': state_name,
        'is_leaf': verdict.get('is_leaf'),
        'leaf_confidence': verdict.get('leaf_probability'),
        'quality_ok': True,
        'error_type': error_type,
        'reason': verdict.get('reason'),
        'score_basis': 'leaf_classifier',
        'signals': {
            'validator': 'leaf_classifier',
            'leaf_probability': verdict.get('leaf_probability'),
            'accept_threshold': verdict.get('accept_threshold'),
            'reject_threshold': verdict.get('reject_threshold'),
        },
    }


def _gate_rejection_response(decision, lang, source):
    """Structured refusal carrying no diagnosis whatsoever.

    No class, no confidence, no top-3, no Grad-CAM and no remedy. The frontend needs a
    machine-readable reason so it can show the right guidance, and the reason code is
    part of the API contract, so it is returned alongside the human message.
    """
    error_type = decision.get('error_type')
    if error_type is None:
        error_type = (leaf_gate_module.ERROR_LEAF_UNCERTAIN
                      if decision.get('state') == 'UNCERTAIN'
                      else leaf_gate_module.ERROR_NON_LEAF)
    messages = leaf_gate_module.CAMERA_MESSAGES if source == 'camera' \
        else leaf_gate_module.MESSAGES
    reason = messages.get(error_type) or decision.get('reason') or \
        messages[leaf_gate_module.ERROR_NON_LEAF]
    response = {
        'error': reason,
        'error_type': error_type,
        'leaf_gate': {
            'state': decision.get('state'),
            'is_leaf': decision.get('is_leaf'),
            'leaf_confidence': decision.get('leaf_confidence'),
            'accept_threshold': (LEAF_GATE.accept_threshold if LEAF_GATE else None),
            'reject_threshold': (LEAF_GATE.reject_threshold if LEAF_GATE else None),
            'reason': decision.get('reason'),
            'score_basis': decision.get('score_basis'),
            'signals': decision.get('signals'),
        },
        'lang': lang,
    }
    if error_type == leaf_gate_module.ERROR_LEAF_UNCERTAIN:
        response['is_uncertain'] = True
    return response


def _analyze_single_image(image_file, lang='en', include_gradcam=False, source='upload'):
    # Validate file before reading
    is_valid, error_msg = validate_image_file(image_file)
    if not is_valid:
        raise ValueError(error_msg)

    # Check magic bytes
    is_valid, fmt = validate_image_content(image_file)
    if not is_valid:
        raise ValueError('File does not appear to be a valid image')

    image_bytes = image_file.read()

    # Decode once with improved preprocessing (EXIF, RGBA, etc.)
    try:
        image = preprocess_uploaded_image(image_bytes)
    except ValueError as e:
        unreadable = {'state': 'REJECTED', 'is_leaf': False, 'leaf_confidence': 0.0,
                      'quality_ok': False,
                      'error_type': leaf_gate_module.ERROR_UNREADABLE,
                      'reason': str(e), 'score_basis': 'quality', 'signals': None}
        return _gate_rejection_response(unreadable, lang, source)
    except Exception:
        unreadable = {'state': 'REJECTED', 'is_leaf': False, 'leaf_confidence': 0.0,
                      'quality_ok': False,
                      'error_type': leaf_gate_module.ERROR_UNREADABLE,
                      'reason': 'Failed to process image', 'score_basis': 'quality', 'signals': None}
        return _gate_rejection_response(unreadable, lang, source)

    prediction = None
    authority = leaf_validation_authority()

    # Stage 1: image quality. These rules are about focus, exposure, resolution and
    # blankness, not about leaf-ness, so they are kept regardless of which leaf
    # validator is in charge. They are also model-free, so a photograph of a wall is
    # refused before any neural network runs. Skipped only under the test bypass,
    # which is unguarded in every respect.
    if LEAF_GATE is not None and authority != 'bypass':
        prefiltered = LEAF_GATE.prefilter(image)
        if prefiltered is not None:
            performance_metrics['images_refused_by_gate'] += 1
            return _gate_rejection_response(prefiltered, lang, source)

    # Stage 2: leaf suitability, before the disease model is spent on the image.
    #
    # The dedicated classifier runs here, ahead of MobileNetV2. That ordering is the
    # point: a non-leaf image is refused for a fraction of the cost of a full
    # 38-class forward pass, and the disease model never runs on it at all, so there is
    # no prediction to discard.
    if authority == 'bypass':
        # Unguated, for plumbing tests and for measuring the pre-gate behaviour. Not a
        # deployment configuration.
        gate_summary = None
    elif authority == 'classifier':
        verdict = LEAF_CLASSIFIER.classify(image)
        logger.info('[LEAF VALIDATION] authority=%s classifier_loaded=True score=%s '
                    'accept_threshold=%s reject_threshold=%s verdict=%s',
                    authority,
                    round(verdict.get('leaf_probability', 0.0), 6),
                    LEAF_CLASSIFIER.accept_threshold,
                    LEAF_CLASSIFIER.reject_threshold,
                    verdict['state'])
        if verdict['state'] != 'LEAF':
            logger.info('[LEAF VALIDATION] %s -> BLOCKING DISEASE INFERENCE',
                        verdict['state'])
            performance_metrics['images_refused_by_gate'] += 1
            decision = _classifier_verdict_to_decision(verdict)
            return _gate_rejection_response(decision, lang, source)
        logger.info('[LEAF VALIDATION] LEAF -> continuing to disease classifier')
        gate_summary = {
            'state': 'VALID_LEAF',
            'is_leaf': True,
            'leaf_confidence': verdict['leaf_probability'],
            'accept_threshold': verdict.get('accept_threshold'),
            'reason': verdict.get('reason'),
            'validator': 'leaf_classifier',
            'validator_model': LEAF_CLASSIFIER.architecture,
        }
    elif authority == 'heuristic_gate':
        # Fallback path. Only taken when the classifier is unavailable; the heuristic
        # gate is safe but has poor real-world recall, which is why it is not primary.
        prediction = predict_image(image_bytes)
        decision = LEAF_GATE.assess(
            image,
            embedding=np.asarray(prediction['_embedding'], dtype=np.float32),
            probabilities=prediction['_probabilities'],
            crops=LEAF_CROPS,
        )
        logger.info('[LEAF VALIDATION] authority=%s score=%s accept_threshold=%s reject_threshold=%s verdict=%s',
                    authority,
                    decision.get('leaf_confidence'),
                    LEAF_GATE.accept_threshold,
                    LEAF_GATE.reject_threshold,
                    decision['state'])
        if decision['state'] != 'VALID_LEAF':
            logger.info('[LEAF VALIDATION] %s -> BLOCKING DISEASE INFERENCE', decision['state'])
            performance_metrics['images_refused_by_gate'] += 1
            return _gate_rejection_response(decision, lang, source)
        logger.info('[LEAF VALIDATION] LEAF -> continuing to disease classifier')
        gate_summary = {
            'state': decision['state'],
            'is_leaf': decision['is_leaf'],
            'leaf_confidence': decision['leaf_confidence'],
            'accept_threshold': LEAF_GATE.accept_threshold,
            'reason': decision['reason'],
            'validator': 'heuristic_gate',
            'signals': decision.get('signals'),
        }
    else:
        # No validator at all. Refuse rather than diagnose: this is the case the old
        # code handled by serving ungated answers.
        performance_metrics['images_refused_by_gate'] += 1
        unavailable = {
            'state': 'REJECTED', 'is_leaf': False, 'leaf_confidence': 0.0,
            'quality_ok': True,
            'error_type': leaf_gate_module.ERROR_LEAF_UNCERTAIN,
            'reason': 'No leaf validator is available, so this image cannot be safely '
                      'diagnosed.',
            'score_basis': 'validator_unavailable', 'signals': None,
        }
        return _gate_rejection_response(unavailable, lang, source)

    if prediction is None:
        prediction = predict_image(image_bytes)

    predicted_class = prediction['predicted_class']
    confidence = prediction['confidence']
    is_uncertain = confidence < CONFIDENCE_THRESHOLD

    remedy = REMEDY_MAP.get(predicted_class)
    advisory_source = 'static'
    if remedy is None:
        if DISABLE_ADVISORY_LLM:
            remedy = (f'No built-in remedy text for {predicted_class}. '
                      f'Consult a local agricultural expert for treatment advice.')
            advisory_source = 'unavailable'
        else:
            remedy = generate_remedy_openrouter(predicted_class, lang)
            advisory_source = 'llm'
    elif not DISABLE_ADVISORY_LLM:
        remedy = translate_text(remedy, lang)

    if is_uncertain:
        uncertain_remedy = (
            f"The AI model is not confident about this prediction "
            f"(confidence: {confidence*100:.1f}%).\n\n"
            f"Suggested condition: {predicted_class}.\n\n"
            f"For a more accurate diagnosis, try:\n"
            f"- Capturing a clearer image with even lighting\n"
            f"- Including both the leaf and any visible lesions in frame\n"
            f"- Consulting a local agricultural expert\n\n"
            f"Remedy (if applicable): {remedy}"
        )
    else:
        uncertain_remedy = None

    # Update performance metrics
    performance_metrics['total_predictions'] += 1
    timings = prediction.get('timings', {})
    performance_metrics['total_inference_ms'] += timings.get('inference_ms', 0)
    performance_metrics['total_preprocessing_ms'] += timings.get('preprocessing_ms', 0)
    performance_metrics['total_gradcam_ms'] += timings.get('gradcam_ms', 0)
    if is_uncertain:
        performance_metrics['predictions_below_threshold'] += 1

    result = {
        'class': predicted_class,
        'confidence': confidence,
        'is_uncertain': is_uncertain,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'remedy': uncertain_remedy if is_uncertain else remedy,
        'remedy_source': advisory_source,
        'lang': lang,
        'top_k_classes': prediction.get('top_k_classes', []),
        'timings': timings
    }
    if gate_summary is not None:
        result['leaf_gate'] = gate_summary
    if CALIBRATION_TEMPERATURE:
        calibrated_confidence = prediction.get('confidence_calibrated')
        result['confidence_calibrated'] = calibrated_confidence
        result['is_uncertain_calibrated'] = (
            calibrated_confidence < CONFIDENCE_THRESHOLD
            if calibrated_confidence is not None else None
        )
        result['calibration_temperature'] = CALIBRATION_TEMPERATURE
    if include_gradcam:
        attach_gradcam(prediction, image_bytes)
        result['gradcam_overlay'] = prediction.get('gradcam_overlay')
        result['gradcam_heatmap'] = prediction.get('gradcam_heatmap')
        if 'gradcam_error' in prediction:
            result['gradcam_error'] = prediction['gradcam_error']
        timings['gradcam_ms'] = prediction['timings'].get('gradcam_ms', 0)

    # The embedding and posterior are consumed by the gate and are far too large to
    # send to a browser; they must not survive into the response.
    prediction.pop('_embedding', None)
    prediction.pop('_probabilities', None)

    return result


def save_prediction_result(result):
    db = SessionLocal()
    try:
        record = PredictionRecord(
            class_name=result.get('class', 'unknown'),
            confidence=float(result.get('confidence', 0.0)),
            remedy=result.get('remedy'),
            lang=result.get('lang', 'en'),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except Exception as exc:
        logger.error('Failed to save prediction history: %s', exc, exc_info=True)
        return None
    finally:
        db.close()


@app.route('/api/analyze', methods=['POST'])
@limiter.limit('10 per minute', exempt_when=lambda: testing)
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    lang = request.form.get('lang', 'en')
    include_gradcam = request.form.get('include_gradcam', 'false').lower() == 'true'
    image_file = request.files['image']
    source = request.form.get('source', 'upload')

    # Run analysis with timeout to prevent hanging requests
    def run_analysis():
        return _analyze_single_image(image_file, lang, include_gradcam=include_gradcam, source=source)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_analysis)
            result = future.result(timeout=ANALYZE_TIMEOUT_SECONDS)

        # A refusal carries no diagnosis, so it must not be recorded as a prediction.
        # Every gate refusal uses 422 Unprocessable Entity: the upload was a valid
        # image, it just cannot be diagnosed. A single terminal code keeps the client
        # contract simple, and 422 is already treated as terminal by the offline retry
        # queue in main.js. 415 would have been wrong here: the media type is fine.
        if 'error_type' in result:
            return jsonify(result), 422
        save_prediction_result(result)
        return jsonify(result), 200
    except concurrent.futures.TimeoutError:
        logger.error(f'Analysis timed out after {ANALYZE_TIMEOUT_SECONDS}s')
        return jsonify({'error': 'Analysis timed out. Please try with a smaller image.'}), 504
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f'Analysis error: {e}', exc_info=True)
        return jsonify({'error': 'Failed to analyze image'}), 500


@app.route('/api/leaf-gate/status', methods=['GET'])
@limiter.exempt
def leaf_gate_status():
    """Report the gate configuration, so a client or operator can see why it decided
    what it decided without having to reproduce the decision."""
    if not LEAF_GATE or not LEAF_GATE.enabled:
        return jsonify({'enabled': False, 'reason': 'no prototypes or config found'}), 200
    return jsonify({
        'enabled': LEAF_GATE_ENABLED,
        'configured': True,
        'config_source': LEAF_GATE.source,
        'accept_threshold': LEAF_GATE.accept_threshold,
        'reject_threshold': LEAF_GATE.reject_threshold,
        'weights': LEAF_GATE.weights,
        'prototypes': int(LEAF_GATE.prototypes.shape[0]),
        'neighbours': LEAF_GATE.neighbours,
        'quality_rules': LEAF_GATE.quality_rules,
        'model_sha256': MODEL_SHA256,
        'audited_model_sha256': AUDITED_MODEL_SHA256,
        'model_matches_audited': MODEL_SHA256 == AUDITED_MODEL_SHA256,
        'refused_by_gate': performance_metrics['images_refused_by_gate'],
    }), 200


@app.route('/api/predictions', methods=['GET'])
def list_predictions():
    limit = request.args.get('limit', default=10, type=int)
    if limit < 1:
        limit = 10
    if limit > 50:
        limit = 50

    db = SessionLocal()
    try:
        records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit).all()
        payload = [
            {
                'id': record.id,
                'class_name': record.class_name,
                'confidence': round(record.confidence, 4),
                'remedy': record.remedy,
                'lang': record.lang,
                'created_at': record.created_at.isoformat() if record.created_at else None,
            }
            for record in records
        ]
        return jsonify({'predictions': payload})
    finally:
        db.close()


@app.route('/api/predictions/export', methods=['GET'])
def export_predictions():
    db = SessionLocal()
    try:
        records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).all()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['id', 'class_name', 'confidence', 'remedy', 'lang', 'created_at'])
        writer.writeheader()
        for record in records:
            writer.writerow({
                'id': record.id,
                'class_name': record.class_name,
                'confidence': round(record.confidence, 4),
                'remedy': record.remedy or '',
                'lang': record.lang,
                'created_at': record.created_at.isoformat() if record.created_at else '',
            })
        response = jsonify({
            'download_url': '/api/predictions/export.csv',
            'columns': ['id', 'class_name', 'confidence', 'remedy', 'lang', 'created_at'],
            'preview': [
                {
                    'id': record.id,
                    'class_name': record.class_name,
                    'confidence': round(record.confidence, 4),
                    'remedy': record.remedy or '',
                    'lang': record.lang,
                    'created_at': record.created_at.isoformat() if record.created_at else '',
                }
                for record in records[:3]
            ],
        })
        response.headers['X-Export-Ready'] = 'true'
        return response
    finally:
        db.close()


@app.route('/api/predictions/export.csv', methods=['GET'])
def export_predictions_csv():
    db = SessionLocal()
    try:
        records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc()).all()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['id', 'class_name', 'confidence', 'remedy', 'lang', 'created_at'])
        writer.writeheader()
        for record in records:
            writer.writerow({
                'id': record.id,
                'class_name': record.class_name,
                'confidence': round(record.confidence, 4),
                'remedy': record.remedy or '',
                'lang': record.lang,
                'created_at': record.created_at.isoformat() if record.created_at else '',
            })
        response = app.response_class(output.getvalue(), mimetype='text/csv')
        response.headers['Content-Disposition'] = 'attachment; filename=prediction_history.csv'
        return response
    finally:
        db.close()


@app.route('/api/predictions/analytics', methods=['GET'])
def prediction_analytics():
    db = SessionLocal()
    try:
        records = db.query(PredictionRecord).order_by(PredictionRecord.created_at.asc()).all()
        labels = []
        counts = []
        by_day = {}
        for record in records:
            day = record.created_at.strftime('%Y-%m-%d') if record.created_at else 'unknown'
            by_day[day] = by_day.get(day, 0) + 1
        for day in sorted(by_day):
            labels.append(day)
            counts.append(by_day[day])
        return jsonify({'series': [{'label': label, 'count': count} for label, count in zip(labels, counts)]})
    finally:
        db.close()


@app.route('/api/analyze-batch', methods=['POST'])
@limiter.limit('5 per minute')
def analyze_batch():
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400

    lang = request.form.get('lang', 'en')
    uploaded_files = request.files.getlist('images')
    if not uploaded_files:
        return jsonify({'error': 'No images uploaded'}), 400

    results = []
    for image_file in uploaded_files:
        try:
            result = _analyze_single_image(image_file, lang)
            # Refused images carry no diagnosis and must not enter the history.
            if 'error_type' not in result:
                save_prediction_result(result)
            result['filename'] = image_file.filename
            results.append(result)
        except ValueError as e:
            results.append({'error': str(e), 'filename': image_file.filename})
        except Exception as e:
            logger.error(f'Batch analysis error: {e}', exc_info=True)
            results.append({'error': 'Failed to analyze image', 'filename': image_file.filename})

    return jsonify({'results': results}), 200


@app.route('/api/weather', methods=['GET'])
def weather():
    city = request.args.get('city')
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    try:
        if lat and lon:
            try:
                float(lat)
                float(lon)
            except ValueError:
                return jsonify({'weather': 'Invalid coordinates provided.'}), 400
            params = {'lat': lat, 'lon': lon, 'appid': OPENWEATHER_API_KEY, 'units': 'metric'}
        elif city:
            params = {'q': city, 'appid': OPENWEATHER_API_KEY, 'units': 'metric'}
        else:
            params = {'q': 'Delhi', 'appid': OPENWEATHER_API_KEY, 'units': 'metric'}

        resp = requests.get('https://api.openweathermap.org/data/2.5/weather', params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        city_name = data.get('name', city or 'your area')
        weather_desc = data['weather'][0]['description']
        temp_c = data['main']['temp']
        weather_str = f"{weather_desc.capitalize()}, {temp_c}°C in {city_name}."

        return jsonify({'weather': weather_str})
    except requests.exceptions.Timeout:
        logger.error('OpenWeather API timeout')
        return jsonify({'weather': 'Weather service timed out. Please try again.'}), 504
    except requests.exceptions.HTTPError as e:
        logger.error(f'OpenWeather API error: {e.response.status_code}')
        return jsonify({'weather': 'Weather service temporarily unavailable.'}), 502
    except Exception as e:
        logger.error(f'Weather error: {e}')
        return jsonify({'weather': 'Error fetching weather data.'}), 500


@app.route('/api/tts', methods=['POST'])
@limiter.limit('20 per minute')
def tts():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400

    text = data.get('text', '')
    lang = data.get('lang', 'en')

    if not text or not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    if len(text) > 500:
        return jsonify({'error': 'Text too long. Maximum 500 characters.'}), 400

    try:
        tts_obj = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts_obj.write_to_fp(fp)
        audio_b64 = base64.b64encode(fp.getvalue()).decode('utf-8')
        return jsonify({'audio': audio_b64})
    except ValueError:
        return jsonify({'error': 'unsupported_lang'}), 400
    except Exception as e:
        logger.error(f'TTS error: {e}')
        return jsonify({'error': 'Text-to-speech failed'}), 500


@app.route('/api/chat', methods=['POST'])
@limiter.limit('10 per minute')
def chat():
    data = request.get_json()
    if not data:
        return jsonify({'reply': 'Invalid request format.'}), 400

    query = data.get('query', '').strip()
    lang = data.get('lang', 'en')
    lat = data.get('lat')
    lon = data.get('lon')

    if not query:
        return jsonify({'reply': 'Please enter a question.'}), 400

    if len(query) > 2000:
        return jsonify({'reply': 'Question too long. Please keep it under 2000 characters.'}), 400

    weather_context = ''
    try:
        if lat and lon:
            try:
                float(lat)
                float(lon)
            except ValueError:
                return jsonify({'reply': 'Invalid coordinates provided.'}), 400
            weather_params = {'lat': lat, 'lon': lon, 'appid': OPENWEATHER_API_KEY, 'units': 'metric'}
        else:
            weather_params = {'q': 'Delhi', 'appid': OPENWEATHER_API_KEY, 'units': 'metric'}

        weather_resp = requests.get('https://api.openweathermap.org/data/2.5/weather', params=weather_params, timeout=10)
        weather_resp.raise_for_status()
        weather_json = weather_resp.json()
        city = weather_json.get('name', 'your area')
        weather_desc = weather_json['weather'][0]['description']
        weather_temp = weather_json['main']['temp']
        weather_context = f"Current weather in {city}: {weather_desc}, {weather_temp}°C."
    except requests.exceptions.Timeout:
        weather_context = "Weather data unavailable due to timeout."
    except requests.exceptions.HTTPError:
        weather_context = "Weather data unavailable."
    except Exception:
        weather_context = "Weather data unavailable."

    prompt = f"""
You are CH AI, an Indian agriculture assistant.
- ONLY introduce yourself when the farmer greets you (e.g., "hi", "hello") or explicitly asks who you are.
- For all other queries, answer directly without any self-introduction.
- You may answer questions about crops, crop diseases, pests, weather conditions, storage or general agriculture.
Weather: {weather_context}
PreferredLanguage: {lang}

---
### FarmerQuestion
{query}

### Guidelines for YourResponse
- Reply in plain text with no Markdown or bullet asterisks.
- Keep it concise and actionable.

### YourResponse
"""

    content, error = call_openrouter(
        messages=[
            {"role": "system", "content": "You are CH AI, an Indian agriculture assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
    )

    if error:
        logger.warning('Chat AI failed: %s', error)
        return jsonify({
            "reply": error,
            "provider": "offline-fallback"
        }), 200

    return jsonify({"reply": content, "provider": "cloud"})


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    logger.info(f'Starting CH AI on port {PORT}, debug={FLASK_DEBUG}')
    app.run(host='0.0.0.0', port=PORT, debug=FLASK_DEBUG)
