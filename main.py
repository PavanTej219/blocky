"""
MediExtract FastAPI Backend with Google Cloud Vision API and OpenRouter Embeddings
Medical Report Processing, RAG System, and Doctor Consultation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import os
import json
import tempfile
import shutil
from datetime import datetime
import logging
from dotenv import load_dotenv
import base64
import numpy as np
import matplotlib.pyplot as plt
import io
import PyPDF2
import fitz

load_dotenv()

# Google Cloud Vision - Using REST API
import requests as http_requests

# Core imports
import qdrant_client
import google.generativeai as genai
from qdrant_client.models import Distance, VectorParams, PointStruct

# LlamaIndex imports
from llama_index.core.schema import Document
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.llms.gemini import Gemini as GeminiLLM
from llama_index.core.embeddings import BaseEmbedding

# Web scraping
import requests
from bs4 import BeautifulSoup
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# CONFIGURATION
# ================================

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # ADD THIS
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    COLLECTION_NAME = "medical_reports_db"
    UPLOAD_DIR = "temp_uploads"
    EMBEDDING_MODEL = "baai/bge-large-en-v1.5"
    EMBEDDING_DIMENSION = 1024  # BGE-large dimension
    
    NORMAL_RANGES = {
        'hemoglobin': {'specialty': 'Hematologist'},
        'glucose': {'specialty': 'Endocrinologist'},
        'cholesterol': {'specialty': 'Cardiologist'},
        'tsh': {'specialty': 'Endocrinologist'},
        'creatinine': {'specialty': 'Nephrologist'},
        'wbc': {'specialty': 'Hematologist'},
        'platelet': {'specialty': 'Hematologist'},
        'alt': {'specialty': 'Hepatologist'},
        'ast': {'specialty': 'Hepatologist'},
    }
    
    @classmethod
    def validate(cls):
        missing = []
        if not cls.GEMINI_API_KEY:  # CHANGE THIS
            missing.append("GEMINI_API_KEY")
        if not cls.QDRANT_URL:
            missing.append("QDRANT_URL")
        if not cls.QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")
        if not cls.GOOGLE_VISION_API_KEY:
            missing.append("GOOGLE_VISION_API_KEY")
        if not cls.OPENROUTER_API_KEY:
            missing.append("OPENROUTER_API_KEY")
        if not cls.GOOGLE_MAPS_API_KEY:
            missing.append("GOOGLE_MAPS_API_KEY")
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

os.makedirs(Config.UPLOAD_DIR, exist_ok=True)

# ================================
# OPENROUTER EMBEDDING CLASS
# ================================

class OpenRouterEmbedding(BaseEmbedding):
    """Custom embedding class using OpenRouter API"""
    
    # Configure Pydantic to allow extra fields
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "baai/bge-large-en-v1.5",
        **kwargs
    ):
        # Call parent init first
        super().__init__(**kwargs)
        
        # Now we can safely set our custom attributes
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "input": text
            }
            
            response = http_requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                return embedding
            else:
                raise Exception("No embedding returned from API")
                
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text"""
        return self._get_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for document text"""
        return self._get_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
            time.sleep(0.1)  # Rate limiting
        return embeddings
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of get_query_embedding"""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of get_text_embedding"""
        return self._get_text_embedding(text)

# ================================
# FASTAPI APP
# ================================

app = FastAPI(
    title="MediExtract API with Google Vision & OpenRouter",
    description="Medical Report Processing with Google Cloud Vision API and OpenRouter Embeddings",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# PYDANTIC MODELS
# ================================

class QueryRequest(BaseModel):
    query: str
    patient_name: Optional[str] = None

class DoctorSearchRequest(BaseModel):
    city: str
    state: str
    specialty: str

class DoctorInfo(BaseModel):
    name: str
    specialty: str
    hospital: Optional[str] = None
    experience: Optional[str] = None
    rating: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    profile_url: Optional[str] = None
    maps_url: Optional[str] = None  # Add this field

class DoctorSearchResponse(BaseModel):
    success: bool
    doctors: List[DoctorInfo]
    message: Optional[str] = None

class AbnormalTest(BaseModel):
    testName: str
    value: str
    normalRange: str
    specialty: str

class QueryResponse(BaseModel):
    response: str = ""
    success: bool = True
    is_comparison: bool = False
    table_data: Optional[Dict[str, Any]] = None
    abnormal_tests: Optional[List[AbnormalTest]] = None
    patient_name: Optional[str] = None

class MedicineInfo(BaseModel):
    name: str
    dosage: Optional[str] = "Not specified"  # Add default value
    timing: Optional[str] = "Not specified"   # Add default value
    duration: Optional[str] = "Not specified"  # Add default value
    instructions: Optional[str] = None
    buy_links: List[str] = []

class PrescriptionResult(BaseModel):
    success: bool
    doctor_name: Optional[str] = None
    patient_name: Optional[str] = None
    date: Optional[str] = None
    medicines: List[MedicineInfo] = []
    error: Optional[str] = None

class ProcessingResult(BaseModel):
    success: bool
    image_filename: str
    extracted_text: Optional[str] = None
    structured_json: Optional[dict] = None
    error: Optional[str] = None

class DatabaseStatus(BaseModel):
    exists: bool
    count: Optional[int] = None

class CompareReportsRequest(BaseModel):
    report1_id: Optional[str] = None  # If already in DB
    report2_id: Optional[str] = None  # If already in DB

class ComparisonData(BaseModel):
    patient_name: str
    report_date: str
    hospital_name: str
    test_results: List[Dict[str, Any]]

class ComparisonResponse(BaseModel):
    success: bool
    report1: Optional[ComparisonData] = None
    report2: Optional[ComparisonData] = None
    comparison_table: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ================================
# DOCTOR FINDER
# ================================

"""
Fixed DoctorFinder with robust Practo profile URL extraction
"""

class DoctorFinder:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def search_doctors(self, city: str, state: str, specialty: str) -> List[Dict[str, Any]]:
        """Main search - Practo first, then Google Maps"""
        doctors = []
        
        # Try Practo first
        try:
            doctors = self._search_practo(city, state, specialty)
            if doctors:
                logger.info(f"Found {len(doctors)} doctors via Practo")
                return doctors[:5]
        except Exception as e:
            logger.error(f"Practo search failed: {e}")
        
        # Fallback to Google Maps
        if not doctors:
            try:
                doctors = self._search_google_maps(city, state, specialty)
                if doctors:
                    logger.info(f"Found {len(doctors)} doctors via Google Maps")
            except Exception as e:
                logger.error(f"Google Maps search failed: {e}")
        
        # Last resort: Generate profiles
        if not doctors:
            doctors = self._generate_doctor_profiles(city, state, specialty)
        
        return doctors[:5]
    
    def _search_practo(self, city: str, state: str, specialty: str) -> List[Dict]:
        """Enhanced Practo scraping with multiple approaches"""
        doctors = []
        
        try:
            city_slug = city.lower().replace(' ', '-')
            specialty_slug = specialty.lower().replace(' ', '-')
            search_url = f"https://www.practo.com/{city_slug}/{specialty_slug}"
            
            logger.info(f"Searching Practo: {search_url}")
            
            # Add delay to avoid rate limiting
            time.sleep(random.uniform(2, 3))
            
            session = requests.Session()
            response = session.get(search_url, headers=self.headers, timeout=20)
            
            logger.info(f"Practo response status: {response.status_code}")
            
            if response.status_code != 200:
                logger.warning(f"Practo returned status {response.status_code}")
                return doctors
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Debug: Save HTML to file for inspection
            try:
                with open('/tmp/practo_debug.html', 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(str(soup.prettify()))
                logger.info("Saved HTML to /tmp/practo_debug.html for debugging")
            except:
                pass
            
            # Multiple strategies to find doctor profiles
            doctor_profile_links = []
            
            # Strategy 1: Look for links with /doctor/ in href
            all_links = soup.find_all('a', href=True)
            logger.info(f"Total links found on page: {len(all_links)}")
            
            for link in all_links:
                href = link.get('href', '')
                # Look for doctor profile patterns
                if '/doctor/' in href:
                    parts = href.split('/')
                    # Valid pattern: /city/doctor/name-specialty
                    if len(parts) >= 4:
                        if parts[-2] == 'doctor' or 'doctor' in href:
                            doctor_slug = parts[-1].split('?')[0]
                            # Avoid listing pages
                            if doctor_slug and doctor_slug not in [specialty_slug, city_slug, 'doctor']:
                                doctor_profile_links.append(link)
            
            logger.info(f"Strategy 1: Found {len(doctor_profile_links)} doctor links with /doctor/ pattern")
            
            # Strategy 2: Look for common Practo doctor card containers
            if not doctor_profile_links:
                # Common Practo class patterns for doctor cards
                card_patterns = [
                    {'class_': lambda x: x and 'doctor' in str(x).lower() and 'card' in str(x).lower()},
                    {'class_': lambda x: x and 'listing' in str(x).lower()},
                    {'class_': lambda x: x and 'profile' in str(x).lower() and 'card' in str(x).lower()},
                    {'attrs': {'data-qa-id': lambda x: x and 'doctor' in str(x).lower()}},
                ]
                
                for pattern in card_patterns:
                    cards = soup.find_all(['div', 'article', 'section'], **pattern)
                    logger.info(f"Found {len(cards)} cards with pattern: {pattern}")
                    
                    for card in cards:
                        links = card.find_all('a', href=True)
                        for link in links:
                            href = link.get('href', '')
                            if '/doctor/' in href or 'profile' in href.lower():
                                doctor_profile_links.append(link)
                    
                    if doctor_profile_links:
                        break
            
            logger.info(f"Strategy 2: Total doctor profile links: {len(doctor_profile_links)}")
            
            # Strategy 3: Look for any links with doctor names (contains text and href)
            if not doctor_profile_links:
                potential_links = []
                for link in all_links:
                    text = link.get_text(strip=True)
                    href = link.get('href', '')
                    # If link has substantial text (likely a name) and a valid href
                    if text and len(text) > 5 and len(text) < 100 and href and href.startswith('/'):
                        # Check if text looks like a name (contains spaces, title case)
                        if ' ' in text and any(word[0].isupper() for word in text.split() if word):
                            potential_links.append(link)
                
                logger.info(f"Strategy 3: Found {len(potential_links)} potential name links")
                
                # Filter potential links for those likely to be doctors
                for link in potential_links[:20]:
                    href = link.get('href', '')
                    if city_slug in href or specialty_slug in href:
                        doctor_profile_links.append(link)
            
            logger.info(f"Final: {len(doctor_profile_links)} doctor profile links to process")
            
            # Process each doctor profile link
            seen_urls = set()
            
            for link in doctor_profile_links[:15]:  # Process up to 15 links
                try:
                    href = link.get('href', '')
                    
                    # Build full URL
                    if href.startswith('http'):
                        profile_url = href
                    elif href.startswith('/'):
                        profile_url = f"https://www.practo.com{href}"
                    else:
                        continue
                    
                    # Extract base URL for deduplication
                    base_url = profile_url.split('?')[0]
                    
                    if base_url in seen_urls:
                        continue
                    seen_urls.add(base_url)
                    
                    # Skip invalid patterns
                    if any(x in profile_url for x in ['results_type', 'q=', '/search', '/consult']):
                        continue
                    
                    # Find parent container
                    parent = link.find_parent(['div', 'article', 'section'])
                    
                    # Extract doctor name
                    doctor_name = link.get_text(strip=True)
                    
                    # Try multiple methods to get name
                    if not doctor_name or len(doctor_name) < 3:
                        name_elem = link.find(['h2', 'h3', 'h4', 'span', 'div'])
                        doctor_name = name_elem.get_text(strip=True) if name_elem else None
                    
                    if not doctor_name and parent:
                        name_elems = parent.find_all(['h1', 'h2', 'h3', 'h4'])
                        for elem in name_elems:
                            text = elem.get_text(strip=True)
                            if text and 3 < len(text) < 100:
                                doctor_name = text
                                break
                    
                    # Extract from URL as last resort
                    if not doctor_name:
                        url_parts = profile_url.split('/')
                        if len(url_parts) >= 4:
                            doctor_slug = url_parts[-1].split('?')[0]
                            name_parts = doctor_slug.replace('-' + specialty_slug, '').split('-')
                            if len(name_parts) >= 2:
                                doctor_name = 'Dr. ' + ' '.join(word.capitalize() for word in name_parts)
                    
                    # Clean up name
                    if doctor_name:
                        doctor_name = doctor_name.replace('Book Appointment', '').replace('Consult Online', '')
                        doctor_name = doctor_name.replace('View Profile', '').strip()
                    
                    # Validate name
                    if not doctor_name or len(doctor_name) < 3:
                        continue
                    
                    invalid_names = ['view', 'profile', 'book', 'more', 'consult', 'appointment', 'call', 'clinic']
                    if doctor_name.lower() in invalid_names:
                        continue
                    
                    # Add "Dr." prefix if missing
                    if not doctor_name.lower().startswith('dr'):
                        doctor_name = f"Dr. {doctor_name}"
                    
                    # Extract other details
                    hospital = f'{city}, {state}'
                    rating = '4.5/5'
                    experience = '10+ years'
                    
                    if parent:
                        # Look for location/hospital
                        loc_keywords = ['clinic', 'hospital', 'address', 'location', 'area']
                        for elem in parent.find_all(['span', 'div', 'p']):
                            text = elem.get_text(strip=True)
                            elem_class = ' '.join(elem.get('class', [])).lower()
                            if any(kw in elem_class for kw in loc_keywords) and text and len(text) > 5:
                                hospital = text
                                break
                        
                        # Look for rating
                        for elem in parent.find_all(['span', 'div']):
                            text = elem.get_text(strip=True)
                            if text and any(c.isdigit() for c in text):
                                # Check if it looks like a rating (e.g., "4.5", "4.5/5", "95%")
                                if '.' in text or '/5' in text or '%' in text:
                                    rating = text if '/5' in text else f'{text}/5' if '.' in text else rating
                                    break
                        
                        # Look for experience
                        for elem in parent.find_all(['span', 'div', 'p']):
                            text = elem.get_text(strip=True).lower()
                            if 'year' in text and any(c.isdigit() for c in text):
                                experience = elem.get_text(strip=True)
                                break
                    
                    hospital_maps_url = f"https://www.google.com/maps/search/?api=1&query={hospital.replace(' ', '+')}"
                    
                    doctor_data = {
                        'name': doctor_name,
                        'specialty': specialty,
                        'hospital': hospital,
                        'rating': rating,
                        'experience': experience,
                        'profile_url': profile_url,
                        'maps_url': hospital_maps_url,
                        'phone': None,
                        'email': None
                    }
                    
                    doctors.append(doctor_data)
                    logger.info(f"Extracted: {doctor_name} - {profile_url}")
                    
                    if len(doctors) >= 5:
                        break
                        
                except Exception as e:
                    logger.error(f"Error parsing doctor link: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(doctors)} doctor profiles from Practo")
            
        except Exception as e:
            logger.error(f"Practo scraping error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return doctors
    
    def _search_google_maps(self, city: str, state: str, specialty: str) -> List[Dict]:
        """Search for doctors using Google Maps Places API"""
        doctors = []
        try:
            geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
            geocode_params = {
                'address': f"{city}, {state}, India",
                'key': Config.GOOGLE_MAPS_API_KEY
            }
            
            geocode_response = http_requests.get(geocode_url, params=geocode_params, timeout=10)
            if geocode_response.status_code != 200:
                return doctors
            
            geocode_data = geocode_response.json()
            if geocode_data['status'] != 'OK' or not geocode_data.get('results'):
                return doctors
            
            location = geocode_data['results'][0]['geometry']['location']
            lat, lng = location['lat'], location['lng']
            
            places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            
            specialty_keywords = {
                'cardiologist': 'cardiologist doctor',
                'endocrinologist': 'endocrinologist doctor',
                'hematologist': 'hematologist doctor blood specialist',
                'nephrologist': 'nephrologist kidney doctor',
                'hepatologist': 'hepatologist liver doctor',
            }
            
            search_keyword = specialty_keywords.get(specialty.lower(), f"{specialty} doctor")
            
            places_params = {
                'location': f"{lat},{lng}",
                'radius': 5000,
                'keyword': search_keyword,
                'type': 'doctor',
                'key': Config.GOOGLE_MAPS_API_KEY
            }
            
            places_response = http_requests.get(places_url, params=places_params, timeout=10)
            if places_response.status_code != 200:
                return doctors
            
            places_data = places_response.json()
            
            if places_data['status'] == 'OK':
                results = places_data.get('results', [])[:5]
                
                city_slug = city.lower().replace(' ', '-')
                specialty_slug = specialty.lower().replace(' ', '-')
                
                for place in results:
                    place_id = place.get('place_id')
                    
                    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                    details_params = {
                        'place_id': place_id,
                        'fields': 'name,formatted_address,formatted_phone_number,rating,url',
                        'key': Config.GOOGLE_MAPS_API_KEY
                    }
                    
                    details_response = http_requests.get(details_url, params=details_params, timeout=10)
                    
                    if details_response.status_code == 200:
                        details_data = details_response.json()
                        if details_data['status'] == 'OK':
                            result = details_data['result']
                            doctor_name = result.get('name', 'Dr. Unknown')
                            
                            # Try to search for this specific doctor on Practo
                            doctor_slug = doctor_name.lower().replace('dr.', '').replace('dr', '').strip()
                            doctor_slug = doctor_slug.replace(' ', '-')
                            doctor_slug = ''.join(c for c in doctor_slug if c.isalnum() or c == '-')
                            
                            # Try to construct potential Practo URL
                            potential_url = f"https://www.practo.com/{city_slug}/doctor/{doctor_slug}-{specialty_slug}"
                            
                            doctors.append({
                                'name': doctor_name,
                                'specialty': specialty,
                                'hospital': result.get('formatted_address', f'{city}, {state}'),
                                'rating': f"{result.get('rating', 4.5)}/5" if result.get('rating') else '4.5/5',
                                'experience': '10+ years',
                                'phone': result.get('formatted_phone_number'),
                                'email': None,
                                'profile_url': potential_url,
                                'maps_url': result.get('url', '#')
                            })
                            time.sleep(0.3)
            
        except Exception as e:
            logger.error(f"Google Maps search error: {e}")
        
        return doctors
    
    def _generate_doctor_profiles(self, city: str, state: str, specialty: str) -> List[Dict]:
        """Generate realistic doctor profiles using AI"""
        doctors = []
        try:
            gemini_api_key = Config.GEMINI_API_KEY
            gemini_api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
            
            prompt = f"""You generate realistic Indian doctor profiles in JSON format.

Generate 5 realistic doctor names for {specialty} specialists in {city}, {state}, India.
Return ONLY valid JSON array with this exact format:
[
  {{
    "name": "Dr. [First Last]",
    "hospital": "[Hospital Name], {city}",
    "experience": "5 - 10 years",
    "rating": "4.5/5"
  }}
]

Use common Indian doctor names. Keep hospital names realistic. Return only JSON, no markdown formatting."""

            # Use REST API
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1024,
                }
            }
            
            response = http_requests.post(
                gemini_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}")
                return doctors
            
            result = response.json()
            
            # Extract text
            result_text = ""
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    result_text = candidate['content']['parts'][0].get('text', '').strip()
            
            # Clean JSON
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]
            
            generated = json.loads(result_text.strip())
            
            city_slug = city.lower().replace(' ', '-')
            specialty_slug = specialty.lower().replace(' ', '-')
            
            for doc in generated:
                doctor_name = doc.get('name', 'Dr. Unknown')
                hospital_name = doc.get('hospital', f'{city}, {state}')
                
                # Create a realistic-looking Practo profile URL
                name_slug = doctor_name.lower().replace('dr.', '').replace('dr', '').strip()
                name_slug = name_slug.replace(' ', '-')
                name_slug = ''.join(c for c in name_slug if c.isalnum() or c == '-')
                
                profile_url = f"https://www.practo.com/{city_slug}/doctor/{name_slug}-{specialty_slug}"
                
                doctors.append({
                    'name': doctor_name,
                    'specialty': specialty,
                    'hospital': hospital_name,
                    'experience': doc.get('experience', '10+ years'),
                    'rating': doc.get('rating', '4.5/5'),
                    'phone': None,
                    'email': None,
                    'profile_url': profile_url,
                    'maps_url': f"https://www.google.com/maps/search/?api=1&query={hospital_name.replace(' ', '+')}"
                })
                
        except Exception as e:
            logger.error(f"Profile generation error: {e}")
        
        return doctors
# ================================
# MEDICAL OCR WITH GOOGLE VISION (REST API)
# ================================

# ================================
# MEDICAL OCR WITH GOOGLE VISION (REST API)
# ================================

class MedicalReportOCR:
    def __init__(self):
        self.api_key = Config.GOOGLE_VISION_API_KEY
        self.vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
        self.gemini_api_key = Config.GEMINI_API_KEY
        self.gemini_api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={self.gemini_api_key}"
        logger.info("MedicalReportOCR initialized with Google Vision and Gemini REST API")
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF to images using PyMuPDF (no system dependencies)"""
        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            temp_image_paths = []
            
            # Convert each page to image
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Render page to image (higher resolution for better OCR)
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Save as JPEG
                temp_image_path = os.path.join(
                    Config.UPLOAD_DIR, 
                    f"pdf_page_{page_num}_{datetime.now().timestamp()}.jpg"
                )
                pix.save(temp_image_path)
                temp_image_paths.append(temp_image_path)
            
            pdf_document.close()
            logger.info(f"Converted PDF to {len(temp_image_paths)} images using PyMuPDF")
            return temp_image_paths
            
        except Exception as e:
            logger.error(f"PDF conversion error: {e}")
            raise
    
    def extract_text(self, image_path: str, use_document_detection: bool = False) -> str:
        """
        Extract text using Google Vision REST API with API key
        
        Args:
            image_path: Path to the image file
            use_document_detection: If True, uses DOCUMENT_TEXT_DETECTION for handwritten text
                                   If False, uses TEXT_DETECTION for regular reports
        """
        try:
            # Read and encode image to base64
            with open(image_path, 'rb') as image_file:
                image_content = image_file.read()
            
            encoded_image = base64.b64encode(image_content).decode('utf-8')
            
            # Choose detection type based on parameter
            detection_type = "DOCUMENT_TEXT_DETECTION" if use_document_detection else "TEXT_DETECTION"
            
            # Prepare the request payload
            payload = {
                "requests": [
                    {
                        "image": {
                            "content": encoded_image
                        },
                        "features": [
                            {
                                "type": detection_type
                            }
                        ]
                    }
                ]
            }
            
            # Make the API request
            response = http_requests.post(
                self.vision_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Vision API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Check for errors in response
            if 'responses' in result and len(result['responses']) > 0:
                response_data = result['responses'][0]
                
                if 'error' in response_data:
                    raise Exception(f"Vision API error: {response_data['error']}")
                
                # For DOCUMENT_TEXT_DETECTION, use fullTextAnnotation
                if use_document_detection and 'fullTextAnnotation' in response_data:
                    full_text = response_data['fullTextAnnotation']['text']
                    logger.info(f"Extracted {len(full_text)} characters using DOCUMENT_TEXT_DETECTION")
                    return full_text
                # For TEXT_DETECTION, use textAnnotations
                elif 'textAnnotations' in response_data and len(response_data['textAnnotations']) > 0:
                    full_text = response_data['textAnnotations'][0]['description']
                    logger.info(f"Extracted {len(full_text)} characters using TEXT_DETECTION")
                    return full_text
                else:
                    logger.warning("No text found in image")
                    return ""
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    def _fallback_test_extraction(self, text: str) -> List[Dict]:
        """Fallback method to extract tests using pattern matching"""
        test_results = []
        
        # Common patterns for test results
        import re
        
        # Pattern 1: "Test Name : Value Unit Range"
        pattern1 = r'([A-Za-z\s\(\)]+)\s*[:]\s*([0-9\.]+)\s*([a-zA-Z/%]+)?\s*(?:Ref|Reference)?[:\s]*([0-9\.\-\s]+)?'
        
        # Pattern 2: "Test Name  Value  Range"
        pattern2 = r'([A-Za-z\s\(\)]{3,30})\s+([0-9\.]+)\s+([a-zA-Z/%]+)?\s+([0-9\.\-]+\s*-\s*[0-9\.]+)?'
        
        lines = text.split('\n')
        
        for line in lines:
            # Try pattern 1
            matches = re.finditer(pattern1, line)
            for match in matches:
                test_name = match.group(1).strip()
                value = match.group(2).strip()
                unit = match.group(3).strip() if match.group(3) else None
                ref_range = match.group(4).strip() if match.group(4) else None
                
                # Filter out non-test lines
                if len(test_name) > 2 and not any(x in test_name.lower() for x in ['hospital', 'patient', 'date', 'page', 'report']):
                    test_results.append({
                        'test_name': test_name,
                        'result_value': value,
                        'unit': unit,
                        'reference_range': ref_range
                    })
        
        logger.info(f"Fallback extraction found {len(test_results)} tests")
        return test_results
    
    def generate_json_with_gemini(self, extracted_text: str, image_filename: str):
        """Enhanced JSON generation with better prompting for Gemini"""
        if not extracted_text or len(extracted_text.strip()) < 10:
            return {'success': False, 'error': 'Insufficient text extracted'}
        
        max_length = 8000
        if len(extracted_text) > max_length:
            extracted_text = extracted_text[:max_length]
        
        prompt = f"""You are a medical data extraction expert. Extract ALL test results from this medical report.

EXTRACTED TEXT:
{extracted_text}

INSTRUCTIONS:
1. Extract EVERY test parameter you can find (e.g., Hemoglobin, Glucose, Cholesterol, WBC, RBC, Platelet Count, etc.)
2. Extract the result values with their units
3. Extract reference ranges if available
4. If a field is not found, use null
5. Return ONLY valid JSON, no markdown formatting, no explanation

OUTPUT FORMAT:
{{
  "hospital_info": {{
    "hospital_name": "string or null",
    "address": "string or null"
  }},
  "patient_info": {{
    "name": "string or null",
    "age": "string or null",
    "gender": "string or null"
  }},
  "doctor_info": {{
    "referring_doctor": "string or null"
  }},
  "report_info": {{
    "report_type": "string or null",
    "report_date": "string or null"
  }},
  "test_results": [
    {{
      "test_name": "Hemoglobin",
      "result_value": "14.5",
      "reference_range": "12-16",
      "unit": "g/dL"
    }},
    {{
      "test_name": "Total WBC Count",
      "result_value": "8500",
      "reference_range": "4000-11000",
      "unit": "cells/cumm"
    }}
  ]
}}

CRITICAL: Extract ALL tests found in the report. The test_results array should contain every single test parameter mentioned. Return ONLY the JSON object, nothing else."""

        try:
            # Use REST API
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 4096,  # Increased for more tests
                }
            }
            
            response = http_requests.post(
                self.gemini_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Extract text from response
            json_text = ""
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    json_text = candidate['content']['parts'][0].get('text', '').strip()
            
            if not json_text:
                raise Exception("Empty response from Gemini")
            
            # Clean JSON formatting
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0]
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0]
            
            json_text = json_text.strip()
            
            # Log for debugging
            logger.info(f"Gemini returned JSON length: {len(json_text)}")
            
            try:
                parsed_json = json.loads(json_text)
                
                # Validate test_results
                test_count = len(parsed_json.get('test_results', []))
                logger.info(f"Parsed {test_count} test results from report")
                
                if test_count == 0:
                    logger.warning("No test results found in Gemini response - trying fallback extraction")
                    # Fallback: try to extract tests directly from text
                    fallback_tests = self._fallback_test_extraction(extracted_text)
                    if fallback_tests:
                        parsed_json['test_results'] = fallback_tests
                        logger.info(f"Fallback extraction added {len(fallback_tests)} tests")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.error(f"Problematic JSON: {json_text[:500]}")
                # Create minimal structure with fallback extraction
                parsed_json = {
                    "hospital_info": {"hospital_name": None, "address": None},
                    "patient_info": {"name": None, "age": None, "gender": None},
                    "doctor_info": {"referring_doctor": None},
                    "report_info": {"report_type": "Medical Report", "report_date": None},
                    "test_results": self._fallback_test_extraction(extracted_text)
                }
            
            parsed_json['_metadata'] = {
                'source_image': image_filename,
                'extraction_method': 'google_vision_rest_api',
                'processing_timestamp': datetime.now().isoformat(),
                'model_used': 'gemini-2.0-flash-exp'
            }
            
            final_test_count = len(parsed_json.get('test_results', []))
            logger.info(f"Final JSON contains {final_test_count} test results")
            
            return {'success': True, 'json_data': parsed_json}
            
        except Exception as e:
            logger.error(f"Gemini processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def process_prescription(self, file_path: str):
        """Process handwritten prescription using DOCUMENT_TEXT_DETECTION"""
        try:
            # Extract text using DOCUMENT_TEXT_DETECTION for better handwriting recognition
            extracted_text = self.extract_text(file_path, use_document_detection=True)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                return {
                    'success': False,
                    'error': 'No text found in prescription'
                }
            
            # Use Gemini to structure the prescription data
            prompt = f"""You are a prescription extraction assistant. Extract prescription data. Never return null values. Use 'As directed' or 'Not specified' for missing fields.

Extract prescription information from this handwritten text:

TEXT: {extracted_text}

Return JSON with this format. IMPORTANT: Never use null values, always provide defaults:
{{
  "doctor_name": "string or 'Not specified'",
  "patient_name": "string or 'Not specified'",
  "date": "string or 'Not specified'",
  "medicines": [
    {{
      "name": "medicine name",
      "dosage": "dosage amount (e.g., 500mg, 10ml) or 'As directed' if not found",
      "timing": "when to take (e.g., Morning-Afternoon-Night, After meals) or 'As directed' if not found",
      "duration": "how long (e.g., 5 days, 2 weeks) or 'As directed' if not found",
      "instructions": "special instructions or 'None'"
    }}
  ]
}}

CRITICAL: If a field is unclear, use 'As directed' or 'Not specified' instead of null.
Return only valid JSON, no markdown formatting."""

            # Use REST API
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = http_requests.post(
                self.gemini_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Gemini API error: {response.status_code}")
            
            result = response.json()
            
            # Extract text
            json_text = ""
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    json_text = candidate['content']['parts'][0].get('text', '').strip()
            
            # Clean JSON
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0]
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0]
            
            parsed_data = json.loads(json_text.strip())
            
            # Ensure all medicine fields have default values
            for medicine in parsed_data.get('medicines', []):
                # Set defaults for None values
                if not medicine.get('dosage'):
                    medicine['dosage'] = 'As directed'
                if not medicine.get('timing'):
                    medicine['timing'] = 'As directed'
                if not medicine.get('duration'):
                    medicine['duration'] = 'As directed'
                if not medicine.get('name'):
                    medicine['name'] = 'Medicine name unclear'
                
                # Generate buy links
                medicine_name = medicine.get('name', '')
                if medicine_name and medicine_name != 'Medicine name unclear':
                    medicine['buy_links'] = [
                        f"https://www.1mg.com/search/all?name={medicine_name.replace(' ', '%20')}",
                        f"https://www.netmeds.com/catalogsearch/result/{medicine_name.replace(' ', '%20')}/all",
                        f"https://pharmeasy.in/search/all?name={medicine_name.replace(' ', '%20')}"
                    ]
                else:
                    medicine['buy_links'] = []
            
            return {
                'success': True,
                'doctor_name': parsed_data.get('doctor_name') or 'Not specified',
                'patient_name': parsed_data.get('patient_name') or 'Not specified',
                'date': parsed_data.get('date') or 'Not specified',
                'medicines': parsed_data.get('medicines', []),
                'extracted_text': extracted_text
            }
            
        except Exception as e:
            logger.error(f"Prescription processing error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_image(self, file_path: str):
        """Process regular medical report using TEXT_DETECTION with enhanced debugging"""
        image_filename = os.path.basename(file_path)
        
        try:
            # Check if file is PDF
            if file_path.lower().endswith('.pdf'):
                logger.info(f"Processing PDF: {image_filename}")
                # Convert PDF to images
                image_paths = self.convert_pdf_to_images(file_path)
                
                # Process all pages
                all_extracted_text = []
                all_json_data = []
                
                for img_path in image_paths:
                    try:
                        # Use TEXT_DETECTION for regular reports
                        extracted_text = self.extract_text(img_path, use_document_detection=False)
                        logger.info(f"Page extracted text length: {len(extracted_text)}")
                        
                        if extracted_text.strip():
                            all_extracted_text.append(extracted_text)
                            
                            gemini_result = self.generate_json_with_gemini(
                                extracted_text, 
                                f"{image_filename}_page_{len(all_json_data)+1}"
                            )
                            
                            if gemini_result['success']:
                                json_data = gemini_result['json_data']
                                test_count = len(json_data.get('test_results', []))
                                logger.info(f"Page {len(all_json_data)+1}: Extracted {test_count} tests")
                                all_json_data.append(json_data)
                    finally:
                        # Clean up temporary image
                        if os.path.exists(img_path):
                            os.unlink(img_path)
                
                # Combine results from all pages
                combined_text = "\n\n--- PAGE BREAK ---\n\n".join(all_extracted_text)
                
                # Use the first page's structured data or merge if needed
                primary_json = all_json_data[0] if all_json_data else {
                    "hospital_info": {"hospital_name": None, "address": None},
                    "patient_info": {"name": None, "age": None, "gender": None},
                    "doctor_info": {"referring_doctor": None},
                    "report_info": {"report_type": "Medical Report", "report_date": None},
                    "test_results": []
                }
                
                # Merge test results from all pages
                if len(all_json_data) > 1:
                    for json_data in all_json_data[1:]:
                        primary_json['test_results'].extend(
                            json_data.get('test_results', [])
                        )
                
                total_tests = len(primary_json.get('test_results', []))
                logger.info(f"PDF processing complete: {total_tests} total tests extracted")
                
                return {
                    'success': True,
                    'image_filename': image_filename,
                    'extracted_text': combined_text,
                    'structured_json': primary_json
                }
            
            else:
                logger.info(f"Processing image: {image_filename}")
                # Original image processing logic - use TEXT_DETECTION for regular reports
                extracted_text = self.extract_text(file_path, use_document_detection=False)
                logger.info(f"Extracted text length: {len(extracted_text)}")
                
                if not extracted_text.strip():
                    return {
                        'success': False,
                        'error': 'No text found in image',
                        'image_filename': image_filename
                    }
                
                # Log sample of extracted text
                logger.info(f"Text sample: {extracted_text[:200]}")
                
                gemini_result = self.generate_json_with_gemini(extracted_text, image_filename)
                
                if gemini_result['success']:
                    test_count = len(gemini_result['json_data'].get('test_results', []))
                    logger.info(f"Successfully extracted {test_count} tests from image")
                    
                    return {
                        'success': True,
                        'image_filename': image_filename,
                        'extracted_text': extracted_text,
                        'structured_json': gemini_result['json_data']
                    }
                else:
                    logger.error(f"JSON generation failed: {gemini_result.get('error')}")
                    return {
                        'success': False,
                        'error': gemini_result['error'],
                        'image_filename': image_filename,
                        'extracted_text': extracted_text[:500]
                    }
                    
        except Exception as e:
            logger.error(f"Processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'image_filename': image_filename
            }
# ================================
# RAG SYSTEM WITH OPENROUTER EMBEDDINGS
# ================================

class RAGSystem:
    def __init__(self):
        self.client = None
        self.vector_store = None
        self.embed_model = None
        self.gemini_api_key = None
        self.gemini_api_url = None
        self._init_components()
    
    def _init_components(self):
        try:
            self.client = qdrant_client.QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY
            )
            
            # Initialize OpenRouter embeddings
            self.embed_model = OpenRouterEmbedding(
                api_key=Config.OPENROUTER_API_KEY,
                model_name=Config.EMBEDDING_MODEL
            )
            
            # Store API key for REST API calls
            self.gemini_api_key = Config.GEMINI_API_KEY
            self.gemini_api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={self.gemini_api_key}"
            
            # Set embedding model for LlamaIndex
            Settings.embed_model = self.embed_model
            
            logger.info("RAG system initialized with OpenRouter embeddings and Gemini REST API")
            
        except Exception as e:
            logger.error(f"RAG initialization error: {e}")
            raise
    
    def _retrieve_context(self, query: str, top_k: int = 10) -> str:
        """Manually retrieve relevant context from Qdrant"""
        try:
            # Get query embedding
            query_embedding = self.embed_model._get_query_embedding(query)
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=Config.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Combine contexts
            contexts = []
            for result in search_results:
                if hasattr(result, 'payload'):
                    # Get document text - check different possible keys
                    text_content = (
                        result.payload.get('_node_content') or 
                        result.payload.get('text') or 
                        str(result.payload)
                    )
                    contexts.append(text_content)
            
            return "\n\n".join(contexts)
            
        except Exception as e:
            logger.error(f"Context retrieval error: {e}")
            return ""
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using Gemini REST API with retrieved context"""
        try:
            prompt = f"""Context from medical reports:
---------------------
{context}
---------------------

Answer questions about the medical reports based on the context above.

Instructions:
1. For test results: Include test name, value, unit, reference range
2. For abnormal values: Explain what it means and suggestions to improve
3. For dietary questions: Provide specific foods to eat and avoid
4. For lifestyle: Give practical recommendations (exercise, sleep, stress management)
5. For report comments: Cite the exact comments mentioned in the report
6. Be specific, practical, and evidence-based
7. If information is unavailable, state clearly
8. Use bullet points for clarity

Question: {query}

Answer:"""

            # Use REST API instead of SDK
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = http_requests.post(
                self.gemini_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Extract text from response
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    text = candidate['content']['parts'][0].get('text', '')
                    return text.strip()
            
            raise Exception("No response generated from Gemini")
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            raise
    
    def detect_abnormal_values(self, context: str) -> List[Dict]:
        abnormal_tests = []
        
        try:
            prompt = f"""You are a medical analyst. Identify abnormal results.

Analyze this medical data and identify abnormal test results:

{context}

Return as JSON array:
[
  {{
    "test_name": "Test Name",
    "value": "patient value",
    "normal_range": "normal range",
    "specialty": "recommended specialist"
  }}
]

Only abnormal values. If none, return []. Return only valid JSON, no markdown formatting."""

            # Use REST API
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1024,
                }
            }
            
            response = http_requests.post(
                self.gemini_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}")
                return abnormal_tests
            
            result = response.json()
            
            # Extract text
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    result_text = candidate['content']['parts'][0].get('text', '').strip()
                else:
                    return abnormal_tests
            else:
                return abnormal_tests
            
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]
            
            try:
                abnormal_data = json.loads(result_text.strip())
                
                for item in abnormal_data:
                    test_name_lower = item.get('test_name', '').lower()
                    specialty = item.get('specialty', 'General Physician')
                    
                    for known_test, info in Config.NORMAL_RANGES.items():
                        if known_test in test_name_lower:
                            specialty = info['specialty']
                            break
                    
                    abnormal_tests.append({
                        'testName': item.get('test_name', ''),
                        'value': item.get('value', ''),
                        'normalRange': item.get('normal_range', ''),
                        'specialty': specialty
                    })
            
            except json.JSONDecodeError:
                pass
        
        except Exception as e:
            logger.error(f"Abnormal detection error: {e}")
        
        return abnormal_tests
    
    def create_documents_from_reports(self, processed_reports: List[dict]):
        documents = []
        
        for report in processed_reports:
            if not report.get('success'):
                continue
            
            try:
                json_data = report['structured_json']
                text_parts = []
                
                hospital_info = json_data.get('hospital_info', {})
                if hospital_info.get('hospital_name'):
                    text_parts.append(f"Hospital: {hospital_info['hospital_name']}")
                
                patient_info = json_data.get('patient_info', {})
                if patient_info.get('name'):
                    text_parts.append(f"Patient: {patient_info['name']}")
                if patient_info.get('age'):
                    text_parts.append(f"Age: {patient_info['age']}")
                if patient_info.get('gender'):
                    text_parts.append(f"Gender: {patient_info['gender']}")
                
                report_info = json_data.get('report_info', {})
                if report_info.get('report_type'):
                    text_parts.append(f"Report Type: {report_info['report_type']}")
                if report_info.get('report_date'):
                    text_parts.append(f"Report Date: {report_info['report_date']}")
                
                test_results = json_data.get('test_results', [])
                for test in test_results:
                    if isinstance(test, dict) and test.get('test_name'):
                        test_text = f"Test: {test['test_name']}"
                        if test.get('result_value'):
                            test_text += f" Result: {test['result_value']}"
                        if test.get('reference_range'):
                            test_text += f" Reference: {test['reference_range']}"
                        text_parts.append(test_text)
                
                if 'extracted_text' in report:
                    text_parts.append(f"Original Text: {report['extracted_text']}")
                
                text_content = "\n".join(text_parts)
                
                document = Document(
                    text=text_content,
                    metadata={
                        'source_image': report['image_filename'],
                        'patient_name': patient_info.get('name', 'Unknown'),
                        'hospital_name': hospital_info.get('hospital_name', 'Unknown'),
                        'report_type': report_info.get('report_type', 'Medical Report'),
                        'report_date': report_info.get('report_date', 'Unknown')
                    }
                )
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Document creation error: {e}")
                continue
        
        return documents
    
    def setup_database(self, processed_reports: List[dict]):
        try:
            documents = self.create_documents_from_reports(processed_reports)
            
            if not documents:
                return False, "No valid documents"
            
            try:
                self.client.delete_collection(Config.COLLECTION_NAME)
            except:
                pass
            
            # Create collection with proper vector configuration
            self.client.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=Config.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            
            # Create vector store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=Config.COLLECTION_NAME
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Index documents with OpenRouter embeddings
            logger.info(f"Indexing {len(documents)} documents with OpenRouter embeddings...")
            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=False
            )
            
            logger.info(f"Successfully indexed {len(documents)} documents")
            
            return True, f"Successfully indexed {len(documents)} reports"
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")
            return False, str(e)
    
    def query(self, query_text: str, patient_name: Optional[str] = None):
        """Query using manual retrieval + Gemini"""
        try:
            # Enhance query with patient name if provided
            enhanced_query = f"For patient {patient_name}: {query_text}" if patient_name else query_text
            
            # Retrieve context from Qdrant
            context = self._retrieve_context(enhanced_query, top_k=10)
            
            if not context:
                return "No relevant information found in the database.", patient_name
            
            # Generate response using Gemini
            response = self._generate_response(enhanced_query, context)
            
            return response, patient_name
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise
    
    def generate_comparison_table(self, query_text: str, patient_name: Optional[str] = None):
        try:
            context, detected_patient = self.query(query_text, patient_name)
            
            prompt = f"""You are a medical report comparison assistant. Create clean comparison tables.

Create a comparison table in markdown format:

Medical Data:
{context}

Query: {query_text}

Format:
| Test Parameter | Report 1 (Date) | Report 2 (Date) |
| --- | --- | --- |
| Test Name | Value1 | Value2 |

Return only the markdown table, no additional text."""

            # Use REST API
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = http_requests.post(
                self.gemini_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Gemini API error: {response.status_code}")
            
            result = response.json()
            
            # Extract text
            table_text = ""
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    table_text = candidate['content']['parts'][0].get('text', '').strip()

            table_data = self._parse_table(table_text)
            abnormal_tests = self.detect_abnormal_values(context)
            
            return {
                'success': True,
                'response': '',
                'table_data': table_data,
                'is_comparison': True,
                'abnormal_tests': abnormal_tests,
                'patient_name': detected_patient
            }
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            return {
                'success': False,
                'response': str(e),
                'table_data': None,
                'is_comparison': False
            }
    
    def _parse_table(self, text: str):
        try:
            lines = [line.strip() for line in text.split('\n') if '|' in line]
            
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                rows = []
                
                for line in lines[1:]:
                    if all(c in '-|: ' for c in line):
                        continue
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if len(cells) == len(headers):
                        rows.append(cells)
                
                if rows:
                    return {'headers': headers, 'rows': rows}
        except Exception as e:
            logger.error(f"Table parsing error: {e}")
        
        return None
    
    def get_database_status(self):
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if Config.COLLECTION_NAME in collection_names:
                collection_info = self.client.get_collection(Config.COLLECTION_NAME)
                return {'exists': True, 'count': collection_info.points_count}
            else:
                return {'exists': False, 'count': 0}
                
        except Exception as e:
            logger.error(f"Database status error: {e}")
            return {'exists': False, 'count': 0}
    
    def generate_visualizations(self, processed_reports: List[dict]) -> Dict[str, Any]:
        """Generate visualization data from processed reports"""
        visualizations = []
        
        for report in processed_reports:
            if not report.get('success'):
                continue
                
            try:
                json_data = report['structured_json']
                test_results = json_data.get('test_results', [])
                patient_name = json_data.get('patient_info', {}).get('name', 'Unknown')
                
                if not test_results:
                    continue
                
                # Prepare data for visualization
                test_names = []
                test_values = []
                normal_ranges = []
                
                for test in test_results:
                    if isinstance(test, dict) and test.get('test_name') and test.get('result_value'):
                        # Extract numeric value
                        try:
                            value_str = str(test['result_value']).strip()
                            # Remove units and extract number
                            numeric_value = float(''.join(filter(lambda x: x.isdigit() or x == '.', value_str.split()[0])))
                            
                            test_names.append(test['test_name'])
                            test_values.append(numeric_value)
                            
                            # Try to extract normal range midpoint
                            ref_range = test.get('reference_range', '')
                            if ref_range and '-' in ref_range:
                                range_parts = ref_range.split('-')
                                if len(range_parts) == 2:
                                    try:
                                        low = float(''.join(filter(lambda x: x.isdigit() or x == '.', range_parts[0])))
                                        high = float(''.join(filter(lambda x: x.isdigit() or x == '.', range_parts[1])))
                                        normal_ranges.append((low + high) / 2)
                                    except:
                                        normal_ranges.append(None)
                                else:
                                    normal_ranges.append(None)
                            else:
                                normal_ranges.append(None)
                        except:
                            continue
                
                if test_names and test_values:
                    visualizations.append({
                        'patient_name': patient_name,
                        'report_filename': report['image_filename'],
                        'test_names': test_names,
                        'test_values': test_values,
                        'normal_ranges': normal_ranges,
                        'report_date': json_data.get('report_info', {}).get('report_date', 'Unknown')
                    })
                    
            except Exception as e:
                logger.error(f"Visualization generation error: {e}")
                continue
        
        return {'visualizations': visualizations}
    
    def get_all_reports(self) -> List[Dict[str, Any]]:
        """Get list of all reports in database"""
        try:
            db_status = self.get_database_status()
            if not db_status['exists']:
                return []
            
            # Scroll through all points in collection
            scroll_result = self.client.scroll(
                collection_name=Config.COLLECTION_NAME,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            reports = []
            seen_files = set()
            
            for point in scroll_result[0]:
                payload = point.payload
                source_image = payload.get('source_image', 'Unknown')
                
                # Avoid duplicates
                if source_image in seen_files:
                    continue
                seen_files.add(source_image)
                
                reports.append({
                    'id': str(point.id),
                    'patient_name': payload.get('patient_name', 'Unknown'),
                    'hospital_name': payload.get('hospital_name', 'Unknown'),
                    'report_type': payload.get('report_type', 'Medical Report'),
                    'report_date': payload.get('report_date', 'Unknown'),
                    'source_image': source_image
                })
            
            return reports
            
        except Exception as e:
            logger.error(f"Error fetching reports: {e}")
            return []
    
    def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get specific report data by ID with full test results"""
        try:
            point = self.client.retrieve(
                collection_name=Config.COLLECTION_NAME,
                ids=[report_id]
            )
            
            if not point:
                return None
            
            payload = point[0].payload
            
            # Parse the text content to extract test results
            text_content = payload.get('_node_content') or payload.get('text') or str(payload)
            
            # Extract test results from the stored text
            test_results = []
            lines = text_content.split('\n')
            
            for line in lines:
                if line.startswith('Test:'):
                    # Parse test line: "Test: TestName Result: Value Reference: Range"
                    parts = line.replace('Test:', '').strip()
                    
                    test_dict = {}
                    
                    # Extract test name
                    if 'Result:' in parts:
                        test_name = parts.split('Result:')[0].strip()
                        test_dict['test_name'] = test_name
                        
                        # Extract result value
                        remaining = parts.split('Result:')[1]
                        if 'Reference:' in remaining:
                            result_value = remaining.split('Reference:')[0].strip()
                            test_dict['result_value'] = result_value
                            
                            # Extract reference range
                            reference_range = remaining.split('Reference:')[1].strip()
                            test_dict['reference_range'] = reference_range
                        else:
                            test_dict['result_value'] = remaining.strip()
                            test_dict['reference_range'] = None
                    else:
                        test_dict['test_name'] = parts
                        test_dict['result_value'] = None
                        test_dict['reference_range'] = None
                    
                    test_dict['unit'] = None
                    test_results.append(test_dict)
            
            return {
                'patient_name': payload.get('patient_name', 'Unknown'),
                'hospital_name': payload.get('hospital_name', 'Unknown'),
                'report_type': payload.get('report_type', 'Medical Report'),
                'report_date': payload.get('report_date', 'Unknown'),
                'source_image': payload.get('source_image', 'Unknown'),
                'test_results': test_results  # Include parsed test results
            }
            
        except Exception as e:
            logger.error(f"Error fetching report: {e}")
            return None
    
    def compare_two_reports(self, report1_data: Dict, report2_data: Dict) -> Dict[str, Any]:
        """Compare two reports and generate comparison table"""
        try:
            # Extract test results from both reports
            tests1 = report1_data.get('test_results', [])
            tests2 = report2_data.get('test_results', [])
            
            logger.info(f"Report 1 has {len(tests1)} tests")
            logger.info(f"Report 2 has {len(tests2)} tests")
            
            if not tests1 or not tests2:
                return {
                    'success': False,
                    'error': f'Insufficient test data. Report 1: {len(tests1)} tests, Report 2: {len(tests2)} tests'
                }
            
            # Create mapping of test names to results (case-insensitive, normalized)
            def normalize_test_name(name):
                """Normalize test names for better matching"""
                if not name:
                    return ""
                # Convert to lowercase and remove extra spaces
                normalized = name.lower().strip()
                # Remove common variations
                normalized = normalized.replace('serum', '').replace('blood', '').strip()
                return normalized
            
            tests1_map = {}
            for test in tests1:
                if isinstance(test, dict) and test.get('test_name'):
                    key = normalize_test_name(test['test_name'])
                    tests1_map[key] = test
            
            tests2_map = {}
            for test in tests2:
                if isinstance(test, dict) and test.get('test_name'):
                    key = normalize_test_name(test['test_name'])
                    tests2_map[key] = test
            
            logger.info(f"Normalized test names - Report 1: {list(tests1_map.keys())}")
            logger.info(f"Normalized test names - Report 2: {list(tests2_map.keys())}")
            
            # Find common tests
            common_tests = set(tests1_map.keys()) & set(tests2_map.keys())
            
            logger.info(f"Common tests found: {len(common_tests)} - {list(common_tests)}")
            
            if not common_tests:
                # If no exact matches, try fuzzy matching
                common_tests = set()
                for key1 in tests1_map.keys():
                    for key2 in tests2_map.keys():
                        # Check if one is substring of another or vice versa
                        if (key1 in key2 or key2 in key1) and len(key1) > 3:
                            # Use the longer key as canonical
                            canonical_key = key1 if len(key1) >= len(key2) else key2
                            common_tests.add(canonical_key)
                            # Map both to the same canonical key
                            if canonical_key not in tests1_map:
                                tests1_map[canonical_key] = tests1_map[key1]
                            if canonical_key not in tests2_map:
                                tests2_map[canonical_key] = tests2_map[key2]
                
                logger.info(f"After fuzzy matching: {len(common_tests)} common tests")
            
            if not common_tests:
                return {
                    'success': False,
                    'error': f'No common tests found. Report 1 tests: {", ".join(list(tests1_map.keys())[:5])}. Report 2 tests: {", ".join(list(tests2_map.keys())[:5])}'
                }
            
            # Build comparison table
            headers = [
                'Test Parameter',
                f"Report 1 ({report1_data.get('report_date', 'N/A')})",
                f"Report 2 ({report2_data.get('report_date', 'N/A')})",
                'Reference Range',
                'Change'
            ]
            
            rows = []
            
            for test_name_key in sorted(common_tests):
                test1 = tests1_map[test_name_key]
                test2 = tests2_map[test_name_key]
                
                # Use the original test name (not normalized) for display
                display_name = test1.get('test_name', test_name_key)
                
                # Calculate change if both values are numeric
                change = 'N/A'
                try:
                    val1_str = str(test1.get('result_value', '')).strip()
                    val2_str = str(test2.get('result_value', '')).strip()
                    
                    # Extract first number from the string
                    val1_num = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', val1_str.split()[0]))
                    val2_num = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == '-', val2_str.split()[0]))
                    
                    if val1_num and val2_num:
                        val1 = float(val1_num)
                        val2 = float(val2_num)
                        
                        diff = val2 - val1
                        percent = (diff / val1 * 100) if val1 != 0 else 0
                        
                        if diff > 0:
                            change = f"↑ {abs(diff):.2f} (+{percent:.1f}%)"
                        elif diff < 0:
                            change = f"↓ {abs(diff):.2f} ({percent:.1f}%)"
                        else:
                            change = "No change"
                except Exception as e:
                    logger.warning(f"Could not calculate change for {display_name}: {e}")
                    pass
                
                rows.append([
                    display_name,
                    f"{test1.get('result_value', 'N/A')} {test1.get('unit', '') or ''}".strip(),
                    f"{test2.get('result_value', 'N/A')} {test2.get('unit', '') or ''}".strip(),
                    test1.get('reference_range') or test2.get('reference_range', 'N/A'),
                    change
                ])
            
            return {
                'success': True,
                'report1': {
                    'patient_name': report1_data.get('patient_name', 'Unknown'),
                    'report_date': report1_data.get('report_date', 'N/A'),
                    'hospital_name': report1_data.get('hospital_name', 'Unknown'),
                    'test_results': tests1
                },
                'report2': {
                    'patient_name': report2_data.get('patient_name', 'Unknown'),
                    'report_date': report2_data.get('report_date', 'N/A'),
                    'hospital_name': report2_data.get('hospital_name', 'Unknown'),
                    'test_results': tests2
                },
                'comparison_table': {
                    'headers': headers,
                    'rows': rows
                }
            }
            
        except Exception as e:
            logger.error(f"Comparison error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
# ================================
# GLOBAL INSTANCES
# ================================

ocr_processor = None
rag_system = None
doctor_finder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_processor, rag_system, doctor_finder
    
    try:
        Config.validate()
        ocr_processor = MedicalReportOCR()
        rag_system = RAGSystem()
        doctor_finder = DoctorFinder()
        logger.info("All components initialized with OpenRouter embeddings")
        yield
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

app.router.lifespan_context = lifespan

# ================================
# API ENDPOINTS
# ================================

@app.get("/")
async def root():
    return {
        "message": "MediExtract API with Google Vision & OpenRouter",
        "version": "4.0.0",
        "status": "running",
        "embedding_provider": "OpenRouter (baai/bge-large-en-v1.5)"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ocr_ready": ocr_processor is not None,
        "rag_ready": rag_system is not None,
        "embedding_model": Config.EMBEDDING_MODEL
    }

@app.get("/api/database/status", response_model=DatabaseStatus)
async def get_database_status():
    try:
        return rag_system.get_database_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-reports")
async def process_reports(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_reports = []
    
    for file in files:
        temp_path = None
        try:
            file_suffix = '.pdf' if file.content_type == 'application/pdf' else '.jpg'
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, dir=Config.UPLOAD_DIR) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
            
            result = ocr_processor.process_image(temp_path)
            result['original_filename'] = file.filename
            processed_reports.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            processed_reports.append({
                'success': False,
                'error': str(e),
                'image_filename': file.filename
            })
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    successful_reports = [r for r in processed_reports if r.get('success')]
    
    if successful_reports:
        rag_system.setup_database(processed_reports)
    
    viz_data = rag_system.generate_visualizations(processed_reports)
    
    return {
        "success": len(successful_reports) > 0,
        "total_count": len(processed_reports),
        "successful_count": len(successful_reports),
        "failed_count": len(processed_reports) - len(successful_reports),
        "results": processed_reports,
        "visualizations": viz_data.get('visualizations', [])
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_reports(request: QueryRequest):
    try:
        db_status = rag_system.get_database_status()
        if not db_status['exists']:
            raise HTTPException(status_code=400, detail="No data available. Upload reports first.")
        
        comparison_keywords = ['compare', 'comparison', 'tabular', 'table', 'versus', 'vs']
        is_comparison = any(kw in request.query.lower() for kw in comparison_keywords)
        
        if is_comparison:
            result = rag_system.generate_comparison_table(request.query, request.patient_name)
            return QueryResponse(**result)
        else:
            response, patient = rag_system.query(request.query, request.patient_name)
            abnormal_tests = rag_system.detect_abnormal_values(response)
            
            return QueryResponse(
                response=response,
                success=True,
                abnormal_tests=abnormal_tests if abnormal_tests else None,
                patient_name=patient
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/find-doctors", response_model=DoctorSearchResponse)
async def find_doctors(request: DoctorSearchRequest):
    try:
        if not all([request.city, request.state, request.specialty]):
            raise HTTPException(status_code=400, detail="City, state, and specialty required")
        
        doctors = doctor_finder.search_doctors(request.city, request.state, request.specialty)
        
        if not doctors:
            return DoctorSearchResponse(
                success=False,
                doctors=[],
                message=f"No doctors found for {request.specialty} in {request.city}"
            )
        
        doctor_list = [DoctorInfo(**doc) for doc in doctors]
        
        return DoctorSearchResponse(
            success=True,
            doctors=doctor_list,
            message=f"Found {len(doctor_list)} specialists"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config/maps-key")
async def get_maps_api_key():
    """Provide Google Maps API key to frontend"""
    return {
        "maps_api_key": Config.GOOGLE_MAPS_API_KEY
    }

@app.post("/api/process-prescription", response_model=PrescriptionResult)
async def process_prescription(file: UploadFile = File(...)):
    """Process handwritten prescription"""
    temp_path = None
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Only image files are supported")
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=Config.UPLOAD_DIR) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
        
        # Process prescription
        result = ocr_processor.process_prescription(temp_path)
        
        if result['success']:
            return PrescriptionResult(
                success=True,
                doctor_name=result.get('doctor_name'),
                patient_name=result.get('patient_name'),
                date=result.get('date'),
                medicines=[MedicineInfo(**med) for med in result.get('medicines', [])]
            )
        else:
            return PrescriptionResult(
                success=False,
                error=result.get('error', 'Unknown error')
            )
            
    except Exception as e:
        logger.error(f"Prescription endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/api/query-prescription")
async def query_prescription(request: QueryRequest):
    """Query about processed prescriptions using chat"""
    try:
        # You can extend this to store prescription data in Qdrant
        # For now, return a simple response
        return {
            "response": "Prescription chat feature - you can ask questions about medicines, dosages, and timings.",
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/api/reports/list")
async def list_all_reports():
    """Get list of all reports in database"""
    try:
        reports = rag_system.get_all_reports()
        return {
            "success": True,
            "reports": reports,
            "count": len(reports)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reports/compare")
async def compare_reports(
    report1_file: Optional[UploadFile] = File(None),
    report2_file: Optional[UploadFile] = File(None),
    report1_id: Optional[str] = None,
    report2_id: Optional[str] = None
):
    """Compare two reports - can use uploaded files or existing report IDs"""
    try:
        report1_data = None
        report2_data = None
        temp_paths = []
        
        # Process Report 1
        if report1_id:
            # Get from database
            logger.info(f"Fetching Report 1 from database with ID: {report1_id}")
            report1_data = rag_system.get_report_by_id(report1_id)
            if not report1_data:
                raise HTTPException(status_code=404, detail="Report 1 not found in database")
            logger.info(f"Report 1 loaded: {report1_data.get('patient_name')}, {len(report1_data.get('test_results', []))} tests")
        elif report1_file:
            # Process uploaded file
            logger.info(f"Processing uploaded Report 1: {report1_file.filename}")
            temp_path = None
            try:
                file_suffix = '.pdf' if report1_file.content_type == 'application/pdf' else '.jpg'
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, dir=Config.UPLOAD_DIR) as tmp:
                    shutil.copyfileobj(report1_file.file, tmp)
                    temp_path = tmp.name
                temp_paths.append(temp_path)
                
                result = ocr_processor.process_image(temp_path)
                if result['success']:
                    json_data = result['structured_json']
                    report1_data = {
                        'patient_name': json_data.get('patient_info', {}).get('name', 'Unknown'),
                        'report_date': json_data.get('report_info', {}).get('report_date', 'N/A'),
                        'hospital_name': json_data.get('hospital_info', {}).get('hospital_name', 'Unknown'),
                        'test_results': json_data.get('test_results', [])
                    }
                    logger.info(f"Report 1 processed: {report1_data.get('patient_name')}, {len(report1_data.get('test_results', []))} tests")
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to process Report 1: {result.get('error')}")
            except Exception as e:
                logger.error(f"Error processing Report 1: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing Report 1: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Report 1 file or ID required")
        
        # Process Report 2
        if report2_id:
            # Get from database
            logger.info(f"Fetching Report 2 from database with ID: {report2_id}")
            report2_data = rag_system.get_report_by_id(report2_id)
            if not report2_data:
                raise HTTPException(status_code=404, detail="Report 2 not found in database")
            logger.info(f"Report 2 loaded: {report2_data.get('patient_name')}, {len(report2_data.get('test_results', []))} tests")
        elif report2_file:
            # Process uploaded file
            logger.info(f"Processing uploaded Report 2: {report2_file.filename}")
            temp_path = None
            try:
                file_suffix = '.pdf' if report2_file.content_type == 'application/pdf' else '.jpg'
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, dir=Config.UPLOAD_DIR) as tmp:
                    shutil.copyfileobj(report2_file.file, tmp)
                    temp_path = tmp.name
                temp_paths.append(temp_path)
                
                result = ocr_processor.process_image(temp_path)
                if result['success']:
                    json_data = result['structured_json']
                    report2_data = {
                        'patient_name': json_data.get('patient_info', {}).get('name', 'Unknown'),
                        'report_date': json_data.get('report_info', {}).get('report_date', 'N/A'),
                        'hospital_name': json_data.get('hospital_info', {}).get('hospital_name', 'Unknown'),
                        'test_results': json_data.get('test_results', [])
                    }
                    logger.info(f"Report 2 processed: {report2_data.get('patient_name')}, {len(report2_data.get('test_results', []))} tests")
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to process Report 2: {result.get('error')}")
            except Exception as e:
                logger.error(f"Error processing Report 2: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing Report 2: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Report 2 file or ID required")
        
        # Compare reports
        logger.info("Comparing reports...")
        comparison_result = rag_system.compare_two_reports(report1_data, report2_data)
        
        # Cleanup temporary files
        for temp_path in temp_paths:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        
        logger.info(f"Comparison result: {comparison_result.get('success')}")
        return comparison_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare reports error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
