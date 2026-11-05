"""
F1 Performance Analysis & Telemetry System — Veri Yükleme ve Doğrulama Modülü (f1_data_loader.py)
==================================================================================================
Bu modül FastF1 önbellek yönetimini, sezon takvimi sorgulamalarını, seans verisi indirme işlemlerini,
detaylı hata yakalama/sınıflandırma mekanizmasını ve ham veri doğrulama (veri kalitesi, pist durumu
SC/VSC/Kırmızı Bayrak tespiti) süreçlerini analiz katmanından bağımsız olarak yürütür.

Temel İlkeler:
- FastF1 önbellek (cache) sistemi korunur.
- Veri indirme ve analiz süreçleri birbirinden ayrılmıştır.
- Ham veri (session.laps) asla yerinde (in-place) değiştirilmez; analiz katmanına her zaman kopya verilir.
- Eksik veriler sessizce sıfırla doldurulmaz ve mevcut olmayan kanallar uydurulmaz.
- FastF1 Session.load() içindeki @soft_exceptions tarafından yutulan gerçek istisnalar (ConnectionError,
  F1AuthError, SessionNotAvailableError vb.) özel log yakalayıcı ile tespit edilerek raporlanır.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import pickle
import re
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import fastf1
import fastf1._api as _ff1_api
import numpy as np
import pandas as pd
import requests

OFFICIAL_LIVETIMING_BASE_URL = "https://livetiming.formula1.com"
OFFICIAL_LIVETIMING_HOSTNAME = "livetiming.formula1.com"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "scratch", "fastf1_cache")

# Pist durumu (TrackStatus) standart FIA / FastF1 kod tanımları
TRACK_STATUS_LABELS: Dict[str, str] = {
    "1": "Pist Açık",
    "2": "Sarı Bayrak",
    "4": "Güvenlik Aracı (SC)",
    "5": "Kırmızı Bayrak",
    "6": "Sanal Güvenlik Aracı (VSC)",
    "7": "Sanal Güvenlik Aracı Bitişi (VSC)",
}

# Seans tipi etiketleri
SESSION_TYPE_OPTIONS: Dict[str, str] = {
    "R": "Ana Yarış",
    "Q": "Sıralama Turları",
    "S": "Sprint Yarışı",
    "SQ": "Sprint Sıralama",
    "SS": "Sprint Sıralama (Shootout)",
    "FP1": "1. Antrenman (FP1)",
    "FP2": "2. Antrenman (FP2)",
    "FP3": "3. Antrenman (FP3)",
}

# Seans adı ile resmi seans kodları arasındaki iki yönlü eşleştirmeler
SESSION_NAME_TO_CODE: Dict[str, str] = {
    "Race": "R",
    "Qualifying": "Q",
    "Sprint": "S",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SS",
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
}

CODE_TO_SESSION_NAME: Dict[str, str] = {
    "R": "Race",
    "Q": "Qualifying",
    "S": "Sprint",
    "SQ": "Sprint Qualifying",
    "SS": "Sprint Shootout",
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
}

# Türkçe / yaygın yarış adı eşleştirmeleri (Meksika GP ve ABD GP dahil)
EVENT_NAME_ALIASES: Dict[str, str] = {
    "meksika": "Mexico City Grand Prix",
    "meksika gp": "Mexico City Grand Prix",
    "mexico": "Mexico City Grand Prix",
    "mexico gp": "Mexico City Grand Prix",
    "mexican grand prix": "Mexico City Grand Prix",
    "mexico city grand prix": "Mexico City Grand Prix",
    "cota": "United States Grand Prix",
    "austin": "United States Grand Prix",
    "usa": "United States Grand Prix",
    "abd gp": "United States Grand Prix",
    "united states": "United States Grand Prix",
    "bahreyn": "Bahrain Grand Prix",
    "bahrain": "Bahrain Grand Prix",
    "avustralya": "Australian Grand Prix",
    "australia": "Australian Grand Prix",
    "monako": "Monaco Grand Prix",
    "italya": "Italian Grand Prix",
    "britanya": "British Grand Prix",
    "ispanya": "Spanish Grand Prix",
    "belçika": "Belgian Grand Prix",
    "hollanda": "Dutch Grand Prix",
    "azerbaycan": "Azerbaijan Grand Prix",
    "singapur": "Singapore Grand Prix",
    "japonya": "Japanese Grand Prix",
    "brezilya": "São Paulo Grand Prix",
    "abu dabi": "Abu Dhabi Grand Prix",
    "avusturya": "Austrian Grand Prix",
    "macaristan": "Hungarian Grand Prix",
    "kanada": "Canadian Grand Prix",
    "suudi arabistan": "Saudi Arabian Grand Prix",
}

# Çevrimdışı / yedek standart F1 takvim şablonu (API erişimi olmadığında veya gelecek yarışlar için)
FALLBACK_CALENDAR_EVENTS: List[Dict[str, Any]] = [
    {"RoundNumber": 1, "Country": "Australia", "Location": "Melbourne", "EventName": "Australian Grand Prix", "EventFormat": "conventional", "EventDate": "2026-03-08"},
    {"RoundNumber": 2, "Country": "China", "Location": "Shanghai", "EventName": "Chinese Grand Prix", "EventFormat": "sprint_qualifying", "EventDate": "2026-03-15"},
    {"RoundNumber": 3, "Country": "Japan", "Location": "Suzuka", "EventName": "Japanese Grand Prix", "EventFormat": "conventional", "EventDate": "2026-03-29"},
    {"RoundNumber": 4, "Country": "Bahrain", "Location": "Sakhir", "EventName": "Bahrain Grand Prix", "EventFormat": "conventional", "EventDate": "2026-04-12"},
    {"RoundNumber": 5, "Country": "Saudi Arabia", "Location": "Jeddah", "EventName": "Saudi Arabian Grand Prix", "EventFormat": "conventional", "EventDate": "2026-04-19"},
    {"RoundNumber": 6, "Country": "United States", "Location": "Miami", "EventName": "Miami Grand Prix", "EventFormat": "sprint_qualifying", "EventDate": "2026-05-03"},
    {"RoundNumber": 7, "Country": "Canada", "Location": "Montreal", "EventName": "Canadian Grand Prix", "EventFormat": "sprint_qualifying", "EventDate": "2026-05-24"},
    {"RoundNumber": 8, "Country": "Monaco", "Location": "Monaco", "EventName": "Monaco Grand Prix", "EventFormat": "conventional", "EventDate": "2026-06-07"},
    {"RoundNumber": 9, "Country": "Spain", "Location": "Barcelona", "EventName": "Spanish Grand Prix", "EventFormat": "conventional", "EventDate": "2026-06-14"},
    {"RoundNumber": 10, "Country": "Austria", "Location": "Spielberg", "EventName": "Austrian Grand Prix", "EventFormat": "conventional", "EventDate": "2026-06-28"},
    {"RoundNumber": 11, "Country": "United Kingdom", "Location": "Silverstone", "EventName": "British Grand Prix", "EventFormat": "sprint_qualifying", "EventDate": "2026-07-05"},
    {"RoundNumber": 12, "Country": "Belgium", "Location": "Spa-Francorchamps", "EventName": "Belgian Grand Prix", "EventFormat": "conventional", "EventDate": "2026-07-19"},
    {"RoundNumber": 13, "Country": "Hungary", "Location": "Budapest", "EventName": "Hungarian Grand Prix", "EventFormat": "conventional", "EventDate": "2026-07-26"},
    {"RoundNumber": 14, "Country": "Netherlands", "Location": "Zandvoort", "EventName": "Dutch Grand Prix", "EventFormat": "sprint_qualifying", "EventDate": "2026-08-23"},
    {"RoundNumber": 15, "Country": "Italy", "Location": "Monza", "EventName": "Italian Grand Prix", "EventFormat": "conventional", "EventDate": "2026-09-06"},
    {"RoundNumber": 16, "Country": "Spain", "Location": "Madrid", "EventName": "Madrid Grand Prix", "EventFormat": "conventional", "EventDate": "2026-09-13"},
    {"RoundNumber": 17, "Country": "Azerbaijan", "Location": "Baku", "EventName": "Azerbaijan Grand Prix", "EventFormat": "conventional", "EventDate": "2026-09-26"},
    {"RoundNumber": 18, "Country": "Singapore", "Location": "Marina Bay", "EventName": "Singapore Grand Prix", "EventFormat": "sprint_qualifying", "EventDate": "2026-10-11"},
    {"RoundNumber": 19, "Country": "United States", "Location": "Austin", "EventName": "United States Grand Prix", "EventFormat": "conventional", "EventDate": "2026-10-25"},
    {"RoundNumber": 20, "Country": "Mexico", "Location": "Mexico City", "EventName": "Mexico City Grand Prix", "EventFormat": "conventional", "EventDate": "2026-11-01"},
    {"RoundNumber": 21, "Country": "Brazil", "Location": "São Paulo", "EventName": "São Paulo Grand Prix", "EventFormat": "conventional", "EventDate": "2026-11-08"},
    {"RoundNumber": 22, "Country": "United States", "Location": "Las Vegas", "EventName": "Las Vegas Grand Prix", "EventFormat": "conventional", "EventDate": "2026-11-21"},
    {"RoundNumber": 23, "Country": "Qatar", "Location": "Lusail", "EventName": "Qatar Grand Prix", "EventFormat": "conventional", "EventDate": "2026-11-29"},
    {"RoundNumber": 24, "Country": "United Arab Emirates", "Location": "Yas Marina", "EventName": "Abu Dhabi Grand Prix", "EventFormat": "conventional", "EventDate": "2026-12-06"},
]

# Son seans yükleme denemelerinin tanısal (diagnostic) kayıtları
LAST_SESSION_LOAD_DIAGNOSTICS: Dict[Tuple[int, str, str], Dict[str, Any]] = {}


def _sanitize_sensitive_text(text: Any) -> str:
    """
    Günlük (log) veya hata mesajlarında bulunabilecek kimlik bilgilerini, e-posta adreslerini,
    parolaları ve Bearer / JWT erişim anahtarlarını (token) maskeler.
    """
    if text is None:
        return ""
    s = str(text)
    # Bearer token ve JWT dizgilerini maskele
    s = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", r"\1[REDACTED_TOKEN]", s)
    s = re.sub(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}", "[REDACTED_JWT]", s)
    # Ortam değişkeni değerleri tanımlıysa sızıntıyı önle
    for env_key in ("F1_PASSWORD", "F1_USERNAME"):
        env_val = os.environ.get(env_key)
        if env_val and len(env_val) >= 3:
            s = s.replace(env_val, f"[REDACTED_{env_key}]")
    return s


def get_fastf1_connection_and_auth_diagnostics(check_dns: bool = True) -> Dict[str, Any]:
    """
    FastF1 sürümünü, resmi zamanlama sunucusu adresini (fastf1._api.base_url),
    DNS çözümleme durumunu ve F1TV kimlik doğrulama durumunu (hiçbir kimlik bilgisi
    veya token ifşa etmeden) döndürür.
    """
    base_url = str(getattr(_ff1_api, "base_url", OFFICIAL_LIVETIMING_BASE_URL))
    hostname_matches_official = base_url.rstrip("/") == OFFICIAL_LIVETIMING_BASE_URL

    dns_resolved = False
    dns_error: Optional[str] = None
    if check_dns:
        try:
            socket.getaddrinfo(OFFICIAL_LIVETIMING_HOSTNAME, 443, proto=socket.IPPROTO_TCP)
            dns_resolved = True
        except Exception as dns_exc:
            dns_resolved = False
            dns_error = f"{type(dns_exc).__name__}: {dns_exc}"

    auth_method = "password"
    auth_status = "Not authenticated"
    subscription_status: Optional[str] = None
    try:
        from fastf1.internals.f1account import get_auth_method, token_manager

        auth_method = str(get_auth_method())
        stored_token = token_manager.get_stored()
        if stored_token is None:
            auth_status = "Not authenticated"
        elif stored_token.is_valid():
            auth_status = "Authenticated (Valid Token)"
            subscription_status = str(stored_token.claims.get("SubscriptionStatus") or "Active")
        else:
            auth_status = "Token Expired"
    except Exception:
        pass

    creds_configured = bool(os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD"))

    return {
        "fastf1_version": str(fastf1.__version__),
        "livetiming_base_url": base_url,
        "official_hostname": OFFICIAL_LIVETIMING_HOSTNAME,
        "hostname_matches_official": hostname_matches_official,
        "dns_resolved": dns_resolved,
        "dns_error": dns_error,
        "auth_method": auth_method,
        "auth_status": auth_status,
        "subscription_status": subscription_status,
        "credentials_env_configured": creds_configured,
    }


class _FastF1LogInterceptor(logging.Handler):
    """
    FastF1'in Session.load() sırasında @soft_exceptions dekoratörü ile yakalayıp
    yuttuğu gerçek istisnaları (exc_info) ve uyarı mesajlarını kaydeden özel Handler.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.exceptions: List[BaseException] = []
        self.warnings: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = _sanitize_sensitive_text(record.getMessage())
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            # _add_first_lap_time_from_ergast içindeki ikincil AttributeError ('_laps') gürültüsünü filtrele
            if not (isinstance(exc, AttributeError) and "_laps" in str(exc)):
                self.exceptions.append(exc)
        if record.levelno >= logging.WARNING:
            if "Failed to add first lap time from Ergast" not in msg:
                self.warnings.append(f"[{record.levelname}] {record.name}: {msg}")


def get_last_load_diagnostics(year: int, event_name: str, session_type: str = "R") -> Dict[str, Any]:
    """Belirtilen seans için son yükleme denemesinin detaylı hata/tanı bilgilerini döndürür."""
    resolved = resolve_event_name(event_name)
    norm = normalize_session_code(session_type, year=year)
    return (
        LAST_SESSION_LOAD_DIAGNOSTICS.get((int(year), resolved, str(norm)))
        or LAST_SESSION_LOAD_DIAGNOSTICS.get((int(year), resolved, str(session_type)))
        or {}
    )


def setup_fastf1_cache(cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """
    FastF1 önbellek dizinini güvenli bir şekilde oluşturur ve etkinleştirir.
    Resmi zamanlama sunucusu adresini (https://livetiming.formula1.com) asla değiştirmez.
    Arka plan / web sunucusu (Streamlit / HTTP Server) ortamında F1_USERNAME tanımlı değilse
    interaktif terminal input() çağrısının sunucuyu kilitlemesini önler.
    """
    os.makedirs(cache_dir, exist_ok=True)
    if hasattr(fastf1.Cache, "configure"):
        fastf1.Cache.configure(cache_dir=cache_dir)
    else:
        fastf1.Cache.enable_cache(cache_dir)

    # Arka plan süreçlerinde stdin.isatty() True olsa bile F1_USERNAME yoksa
    # input("F1 account email: ") ile web sunucu iş parçacığının bloklanmasını önle
    try:
        from fastf1.exceptions import F1AuthError
        from fastf1.internals.f1account import _AUTHENTICATORS, get_auth_method, token_manager
        from fastf1.internals.f1account.login import default_login

        def _non_blocking_auth():
            method = get_auth_method()
            if method == "browser":
                return _AUTHENTICATORS["browser"]()
            if not (os.environ.get("F1_USERNAME") and os.environ.get("F1_PASSWORD")):
                raise F1AuthError(
                    "No F1 account credentials available (F1_USERNAME / F1_PASSWORD are not configured). "
                    "Authenticate via 'python -m fastf1 auth f1tv --authenticate' or set environment variables."
                )
            return default_login()

        token_manager._default_auth_func = _non_blocking_auth
    except Exception:
        pass

    return cache_dir


def resolve_event_name(event_query: str) -> str:
    """
    Kullanıcının girdiği yarış adını (örn. 'Meksika GP', 'COTA', 'Mexico') resmi FastF1
    etkinlik adına dönüştürür.
    """
    if not event_query:
        return ""
    cleaned = str(event_query).strip()
    key = cleaned.lower()
    if key in EVENT_NAME_ALIASES:
        return EVENT_NAME_ALIASES[key]
    return cleaned


# Tarihsel Sprint Hafta Sonu Format Haritası (Yıllara göre format tipleri)
# 2021-2022: 'sprint' (FP1, Q, FP2, S, R)
# 2023: 'sprint_shootout' (FP1, Q, SS, S, R)
# 2024+: 'sprint_qualifying' (FP1, SQ, S, Q, R)
HISTORICAL_SPRINT_WEEKENDS: Dict[int, Dict[str, str]] = {
    2021: {
        "British Grand Prix": "sprint",
        "Italian Grand Prix": "sprint",
        "São Paulo Grand Prix": "sprint",
    },
    2022: {
        "Emilia Romagna Grand Prix": "sprint",
        "Austrian Grand Prix": "sprint",
        "São Paulo Grand Prix": "sprint",
    },
    2023: {
        "Azerbaijan Grand Prix": "sprint_shootout",
        "Austrian Grand Prix": "sprint_shootout",
        "Belgian Grand Prix": "sprint_shootout",
        "Qatar Grand Prix": "sprint_shootout",
        "United States Grand Prix": "sprint_shootout",
        "São Paulo Grand Prix": "sprint_shootout",
    },
    2024: {
        "Chinese Grand Prix": "sprint_qualifying",
        "Miami Grand Prix": "sprint_qualifying",
        "Austrian Grand Prix": "sprint_qualifying",
        "United States Grand Prix": "sprint_qualifying",
        "São Paulo Grand Prix": "sprint_qualifying",
        "Qatar Grand Prix": "sprint_qualifying",
    },
    2025: {
        "Chinese Grand Prix": "sprint_qualifying",
        "Miami Grand Prix": "sprint_qualifying",
        "Belgian Grand Prix": "sprint_qualifying",
        "United States Grand Prix": "sprint_qualifying",
        "São Paulo Grand Prix": "sprint_qualifying",
        "Qatar Grand Prix": "sprint_qualifying",
    },
    2026: {
        "Chinese Grand Prix": "sprint_qualifying",
        "Miami Grand Prix": "sprint_qualifying",
        "Canadian Grand Prix": "sprint_qualifying",
        "British Grand Prix": "sprint_qualifying",
        "Dutch Grand Prix": "sprint_qualifying",
        "Singapore Grand Prix": "sprint_qualifying",
    },
}


def normalize_session_code(identifier: str, year: Optional[int] = None) -> str:
    """
    Kullanıcı veya arayüzden gelen seans kodunu veya adını (örn. 'Sprint Shootout', 'SQ',
    'sprint', 'Ana Yarış', 'Practice 1') resmi seans koduna ('FP1', 'FP2', 'FP3', 'Q',
    'SQ', 'SS', 'S', 'R') dönüştürür.
    """
    if not identifier:
        return "R"
    s = str(identifier).strip()
    s_up = s.upper()
    if s_up in SESSION_TYPE_OPTIONS:
        if year in (2021, 2022) and s_up == "SQ":
            return "S"
        if year == 2023 and s_up == "SQ":
            return "SS"
        return s_up

    # İsim eşleme
    for name, code in SESSION_NAME_TO_CODE.items():
        if s.lower() == name.lower():
            if year in (2021, 2022) and code == "SQ":
                return "S"
            if year == 2023 and code == "SQ":
                return "SS"
            return code

    # Türkçe etiket eşleme
    for code, label in SESSION_TYPE_OPTIONS.items():
        if s.lower() == label.lower():
            if year in (2021, 2022) and code == "SQ":
                return "S"
            if year == 2023 and code == "SQ":
                return "SS"
            return code

    s_low = s.lower()
    if "shootout" in s_low:
        return "SS"
    if "sprint" in s_low and ("qualifying" in s_low or "sıralama" in s_low):
        return "SS" if year == 2023 else "SQ"
    if "sprint" in s_low:
        return "S"
    if "qualifying" in s_low or "sıralama" in s_low:
        return "Q"
    if "race" in s_low or "yarış" in s_low:
        return "R"
    if "practice 1" in s_low or "fp1" in s_low or "1. antrenman" in s_low:
        return "FP1"
    if "practice 2" in s_low or "fp2" in s_low or "2. antrenman" in s_low:
        return "FP2"
    if "practice 3" in s_low or "fp3" in s_low or "3. antrenman" in s_low:
        return "FP3"

    return s_up


def get_event_available_sessions(
    year: int,
    event_name: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[str, Dict[str, Any]]:
    """
    Seçilen yarış hafta sonu için yalnızca o yarışta mevcut olan gerçek seansları
    (Session1 ... Session5) döndürür.
    
    FastF1'in resmî takviminden (veya çevrimdışı fallback tablosundan) hafta sonu
    formatını (conventional, sprint, sprint_shootout, sprint_qualifying) tespit eder:
    - 2021-2022 Sprint formatı: FP1, Q, FP2, S, R (FP3 bulunmaz)
    - 2023 Sprint Shootout formatı: FP1, Q, SS, S, R (FP2 ve FP3 bulunmaz)
    - 2024+ Sprint Qualifying formatı: FP1, SQ, S, Q, R (FP2 ve FP3 bulunmaz)
    - Klasik (conventional) format: FP1, FP2, FP3, Q, R
    
    Dönüş formatı:
    {
        "FP1": {"code": "FP1", "name": "Practice 1", "label": "1. Antrenman (FP1)", "session_number": 1, ...},
        ...
    }
    """
    resolved = resolve_event_name(event_name)
    sched = get_season_schedule(year, include_testing=False, cache_dir=cache_dir)

    matched_row = None
    if sched is not None and not sched.empty and "EventName" in sched.columns:
        for _, r in sched.iterrows():
            ev_n = str(r.get("EventName") or "")
            if ev_n.lower() == resolved.lower() or resolved.lower() in ev_n.lower() or ev_n.lower() in resolved.lower():
                matched_row = r
                break

    available: Dict[str, Dict[str, Any]] = {}

    if matched_row is not None:
        has_session_cols = any(f"Session{idx}" in matched_row for idx in range(1, 6))
        if has_session_cols:
            for idx in range(1, 6):
                s_name = matched_row.get(f"Session{idx}")
                if pd.notna(s_name) and str(s_name).strip():
                    s_name_str = str(s_name).strip()
                    code = SESSION_NAME_TO_CODE.get(s_name_str)
                    if not code:
                        code = normalize_session_code(s_name_str, year=year)

                    label = SESSION_TYPE_OPTIONS.get(code, s_name_str)
                    available[code] = {
                        "code": code,
                        "name": s_name_str,
                        "label": label,
                        "full_label": f"{code} — {label}",
                        "session_number": idx,
                        "date": matched_row.get(f"Session{idx}Date"),
                        "date_utc": matched_row.get(f"Session{idx}DateUtc"),
                    }

    if available:
        return available

    # Takvimde seans sütunları yoksa veya bulunamadıysa: Format tespitine göre inşa et
    ev_format = "conventional"
    year_sprints = HISTORICAL_SPRINT_WEEKENDS.get(int(year), {})
    for sp_name, sp_fmt in year_sprints.items():
        if sp_name.lower() in resolved.lower() or resolved.lower() in sp_name.lower():
            ev_format = sp_fmt
            break
    else:
        for fb in FALLBACK_CALENDAR_EVENTS:
            if fb["EventName"].lower() == resolved.lower():
                ev_format = fb.get("EventFormat", "conventional")
                break

    if ev_format == "sprint_qualifying" or (year >= 2024 and "sprint" in resolved.lower()):
        ordered = [
            ("FP1", "Practice 1"),
            ("SQ", "Sprint Qualifying"),
            ("S", "Sprint"),
            ("Q", "Qualifying"),
            ("R", "Race"),
        ]
    elif ev_format == "sprint_shootout" or (year == 2023 and "sprint" in resolved.lower()):
        ordered = [
            ("FP1", "Practice 1"),
            ("Q", "Qualifying"),
            ("SS", "Sprint Shootout"),
            ("S", "Sprint"),
            ("R", "Race"),
        ]
    elif ev_format == "sprint" or (year in (2021, 2022) and "sprint" in resolved.lower()):
        ordered = [
            ("FP1", "Practice 1"),
            ("Q", "Qualifying"),
            ("FP2", "Practice 2"),
            ("S", "Sprint"),
            ("R", "Race"),
        ]
    else:
        ordered = [
            ("FP1", "Practice 1"),
            ("FP2", "Practice 2"),
            ("FP3", "Practice 3"),
            ("Q", "Qualifying"),
            ("R", "Race"),
        ]

    for idx, (code, name) in enumerate(ordered, start=1):
        label = SESSION_TYPE_OPTIONS.get(code, name)
        available[code] = {
            "code": code,
            "name": name,
            "label": label,
            "full_label": f"{code} — {label}",
            "session_number": idx,
            "date": None,
            "date_utc": None,
        }

    return available


def list_cached_sessions(cache_dir: str = DEFAULT_CACHE_DIR) -> List[Dict[str, Any]]:
    """
    Yerel FastF1 önbellek klasöründe (scratch/fastf1_cache) halihazırda indirilmiş
    olan yarış ve seansları listeler. Sadece gerçek tur verisi (_extended_timing_data.ff1pkl)
    içeren seansları geçerli kabul eder.
    """
    cached_items: List[Dict[str, Any]] = []
    if not os.path.isdir(cache_dir):
        return cached_items

    for year_entry in sorted(os.listdir(cache_dir), reverse=True):
        year_path = os.path.join(cache_dir, year_entry)
        if not os.path.isdir(year_path) or not year_entry.isdigit():
            continue
        year_int = int(year_entry)
        for event_dir in sorted(os.listdir(year_path)):
            event_path = os.path.join(year_path, event_dir)
            if not os.path.isdir(event_path):
                continue
            parts = event_dir.split("_", 1)
            event_date = parts[0] if len(parts) > 1 else ""
            event_name = parts[1].replace("_", " ") if len(parts) > 1 else event_dir.replace("_", " ")
            for session_dir in sorted(os.listdir(event_path)):
                session_path = os.path.join(event_path, session_dir)
                if not os.path.isdir(session_path):
                    continue
                s_parts = session_dir.split("_", 1)
                session_name = s_parts[1].replace("_", " ") if len(s_parts) > 1 else session_dir
                pkl_files = [f for f in os.listdir(session_path) if f.endswith(".ff1pkl")]
                if "_extended_timing_data.ff1pkl" in pkl_files:
                    cached_items.append(
                        {
                            "year": year_int,
                            "event_name": event_name,
                            "event_date": event_date,
                            "session_name": session_name,
                            "path": session_path,
                            "file_count": len(pkl_files),
                        }
                    )
    return cached_items


def get_season_schedule(
    year: int,
    include_testing: bool = False,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """
    Belirtilen sezonun (2018-2026+) yarış takvimini döndürür.
    Çevrimiçi/önbellek sorgusu başarısız olursa yerel önbellekteki yarışları ve
    standart F1 takvimini (Meksika GP ve ABD GP dahil) yedek olarak sunar.
    """
    setup_fastf1_cache(cache_dir)
    try:
        schedule = fastf1.get_event_schedule(year)
        if schedule is not None and not schedule.empty:
            df = schedule.copy()
            if not include_testing and "EventFormat" in df.columns:
                df = df[df["EventFormat"] != "testing"].copy()
            return df.reset_index(drop=True)
    except Exception:
        pass

    fallback_rows = [dict(item) for item in FALLBACK_CALENDAR_EVENTS]
    cached = [c for c in list_cached_sessions(cache_dir) if c["year"] == year]
    existing_names = {r["EventName"] for r in fallback_rows}
    for c in cached:
        if c["event_name"] not in existing_names:
            fallback_rows.insert(
                0,
                {
                    "RoundNumber": 0,
                    "Country": "",
                    "Location": "",
                    "EventName": c["event_name"],
                    "EventFormat": "conventional",
                    "EventDate": c.get("event_date", f"{year}-01-01"),
                },
            )
            existing_names.add(c["event_name"])

    return pd.DataFrame(fallback_rows)


def _build_offline_session_object(
    year: int,
    resolved_event: str,
    session_type: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Any:
    """
    Çevrimdışı (offline) durumda fastf1.get_event_schedule() ağ hatası verdiğinde,
    yerel önbellekteki klasör yapısından veya yedek takvimden geçerli bir
    fastf1.core.Session nesnesi inşa eder.
    """
    from fastf1.core import Session
    from fastf1.events import Event

    norm_code = normalize_session_code(session_type, year=year)
    available = get_event_available_sessions(year, resolved_event, cache_dir=cache_dir)

    if norm_code not in available:
        raise ValueError(
            f"Session type '{session_type}' does not exist for this event"
        )

    target_session_info = available[norm_code]
    target_session_name = target_session_info["name"]

    # Önce yerel önbellekte eşleşen yarış tarihini ara
    event_date_str = f"{year}-03-08"
    round_num = 1
    location_str = ""
    country_str = ""
    for fb in FALLBACK_CALENDAR_EVENTS:
        if fb["EventName"].lower() == resolved_event.lower():
            event_date_str = str(fb.get("EventDate", f"{year}-03-08")).replace("2026", str(year))
            round_num = int(fb.get("RoundNumber", 1))
            location_str = str(fb.get("Location", ""))
            country_str = str(fb.get("Country", ""))
            break

    for c in list_cached_sessions(cache_dir):
        if c["year"] == year and c["event_name"].lower() == resolved_event.lower():
            if c.get("event_date"):
                event_date_str = c["event_date"]
            break

    ev_dt = pd.to_datetime(event_date_str)

    # Format tipini belirle
    ev_format = "conventional"
    year_sprints = HISTORICAL_SPRINT_WEEKENDS.get(int(year), {})
    for sp_name, sp_fmt in year_sprints.items():
        if sp_name.lower() in resolved_event.lower() or resolved_event.lower() in sp_name.lower():
            ev_format = sp_fmt
            break
    else:
        for fb in FALLBACK_CALENDAR_EVENTS:
            if fb["EventName"].lower() == resolved_event.lower():
                ev_format = fb.get("EventFormat", "conventional")
                break

    event_dict = {
        "RoundNumber": round_num,
        "Country": country_str,
        "Location": location_str,
        "OfficialEventName": f"FORMULA 1 {resolved_event.upper()} {year}",
        "EventDate": ev_dt,
        "EventName": resolved_event,
        "EventFormat": ev_format,
        "F1ApiSupport": True,
    }

    # Session1 ... Session5 alanlarını mevcut seanslara göre yerleştir
    for idx in range(1, 6):
        s_match = next((meta for meta in available.values() if meta.get("session_number") == idx), None)
        s_name = s_match["name"] if s_match else ""
        days_offset = 3 - idx if idx <= 2 else (1 if idx <= 4 else 0)
        s_date = ev_dt - pd.Timedelta(days=days_offset)
        event_dict[f"Session{idx}"] = s_name
        event_dict[f"Session{idx}Date"] = s_date
        event_dict[f"Session{idx}DateUtc"] = s_date

    ev_series = Event(event_dict, year=year)
    return Session(event=ev_series, session_name=target_session_name, f1_api_support=True)


def _fetch_openf1_json_list(
    url: str,
    timeout: int = 20,
    max_retries: int = 4,
    allow_empty_on_rate_limit: bool = False,
) -> List[Dict[str, Any]]:
    """
    OpenF1 API uç noktasından JSON listesi çeker.
    HTTP 429 (Too Many Requests — 3 istek/sn sınırı) veya geçici 5xx hatalarında
    otomatik bekleme (backoff) ile yeniden dener ve dönen verinin sözlük listesi
    olduğunu doğrular.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries - 1:
                    time.sleep(0.55 * (attempt + 1))
                    continue
                if allow_empty_on_rate_limit:
                    return []
                resp.raise_for_status()
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            if isinstance(payload, dict) and "detail" in payload:
                if attempt < max_retries - 1:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                if allow_empty_on_rate_limit:
                    return []
                raise requests.HTTPError(f"OpenF1 API mesajı: {payload.get('detail')}")
            return []
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
    if allow_empty_on_rate_limit:
        return []
    if last_exc is not None:
        raise last_exc
    return []


def _hydrate_session_cache_from_openf1(
    session: Any,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[bool, Optional[str]]:
    """
    FastF1 3.9.0'da livetiming.formula1.com F1TV aboneliği gerektirdiğinde (HTTP 401)
    ve livetiming-mirror.fastf1.dev HTTP 404 döndürdüğünde, resmi F1 zamanlama akışını
    açık erişimli OpenF1 API (https://api.openf1.org/v1) üzerinden indirip FastF1 Stage-2
    (.ff1pkl) önbellek dosyalarını oluşturur.
    """
    year = int(session.event.year)
    event_name = str(session.event["EventName"])
    location = str(session.event.get("Location") or "")
    country = str(session.event.get("Country") or "")
    event_date = pd.to_datetime(session.event.get("EventDate"), errors="coerce")
    session_name = str(session.name)

    all_sessions = _fetch_openf1_json_list(f"https://api.openf1.org/v1/sessions?year={year}", timeout=15)
    if not all_sessions:
        return False, f"OpenF1 üzerinde {year} sezonuna ait seans kaydı bulunamadı."

    # Seans tipi eşleştirme (Sprint Yarışı, Sprint Sıralama/Shootout, Sıralama Turları, Ana Yarış)
    sname_norm = session_name.lower().strip()
    candidates = []
    for s in all_sessions:
        s_open_name = str(s.get("session_name") or "").lower().strip()
        if sname_norm == "sprint":
            # Sprint yarışı kesinlikle Sprint Qualifying veya Shootout ile karışmamalıdır
            if s_open_name == "sprint":
                candidates.append(s)
        elif sname_norm in ("sprint qualifying", "sprint shootout"):
            # Sprint Sıralama / Shootout seansı sprint yarışıyla karışmamalıdır
            if s_open_name in ("sprint qualifying", "sprint shootout") or (
                "sprint" in s_open_name and ("shootout" in s_open_name or "qualifying" in s_open_name)
            ):
                candidates.append(s)
        elif sname_norm == "qualifying":
            # Ana sıralama turları, sprint sıralama ile karışmamalıdır
            if s_open_name == "qualifying" or (
                "qualifying" in s_open_name and "sprint" not in s_open_name and "shootout" not in s_open_name
            ):
                candidates.append(s)
        elif sname_norm == "race":
            # Ana yarış, sprint yarışı ile karışmamalıdır
            if s_open_name == "race" or ("race" in s_open_name and "sprint" not in s_open_name):
                candidates.append(s)
        elif s_open_name == sname_norm:
            candidates.append(s)

    if not candidates:
        return False, f"OpenF1 üzerinde '{session_name}' tipinde seans bulunamadı."

    # Etkinlik tarihi ve lokasyon yakınlığına göre en uygun seansı seç
    matched_session = None
    best_score = -999999.0
    for s in candidates:
        score = 0.0
        s_loc = str(s.get("location") or "").lower()
        s_circuit = str(s.get("circuit_short_name") or "").lower()
        s_country = str(s.get("country_name") or "").lower()
        if location and (location.lower() in s_loc or s_loc in location.lower() or location.lower() in s_circuit):
            score += 50.0
        if country and (country.lower() in s_country or s_country in country.lower()):
            score += 20.0
        if pd.notna(event_date) and s.get("date_start"):
            s_dt = pd.to_datetime(s["date_start"], utc=True, errors="coerce")
            if pd.notna(s_dt):
                days_diff = abs((s_dt.tz_localize(None) - event_date.tz_localize(None)).total_seconds()) / 86400.0
                score -= days_diff * 10.0
                if days_diff <= 4.0:
                    score += 100.0
        if score > best_score:
            best_score = score
            matched_session = s

    if matched_session is None or best_score < 20.0:
        return False, f"{year} {event_name} ({session_name}) seansı çevrimiçi veri sağlayıcısında bulunamadı."

    sk = int(matched_session["session_key"])
    time.sleep(0.25)
    laps = _fetch_openf1_json_list(f"https://api.openf1.org/v1/laps?session_key={sk}", timeout=20)
    if not laps:
        return (
            False,
            f"{year} {event_name} ({session_name}) seansı için henüz yayınlanmış tur verisi bulunmuyor (seans henüz gerçekleşmemiş olabilir).",
        )

    time.sleep(0.25)
    drivers = _fetch_openf1_json_list(f"https://api.openf1.org/v1/drivers?session_key={sk}", timeout=15)
    time.sleep(0.25)
    stints = _fetch_openf1_json_list(
        f"https://api.openf1.org/v1/stints?session_key={sk}", timeout=15, allow_empty_on_rate_limit=True
    )
    time.sleep(0.25)
    pits = _fetch_openf1_json_list(
        f"https://api.openf1.org/v1/pit?session_key={sk}", timeout=15, allow_empty_on_rate_limit=True
    )
    time.sleep(0.25)
    rcm = _fetch_openf1_json_list(
        f"https://api.openf1.org/v1/race_control?session_key={sk}", timeout=15, allow_empty_on_rate_limit=True
    )

    cache_subdir = os.path.join(cache_dir, session.api_path[8:])
    os.makedirs(cache_subdir, exist_ok=True)

    df_laps_raw = pd.DataFrame(laps)
    session_start_utc = pd.to_datetime(matched_session["date_start"], utc=True).tz_localize(None)
    # İlk turun gerçek başlangıç zamanını (lights out) bul
    lap1_dates = pd.to_datetime(
        df_laps_raw.loc[df_laps_raw["lap_number"] == 1, "date_start"], utc=True, errors="coerce"
    ).dropna()
    if not lap1_dates.empty:
        lights_out_utc = lap1_dates.min().tz_localize(None)
    else:
        lights_out_utc = session_start_utc

    session_end_utc = pd.to_datetime(
        matched_session.get("date_end") or matched_session["date_start"], utc=True
    ).tz_localize(None)
    t0_utc = lights_out_utc - pd.to_timedelta(1, unit="h")
    session_start_td = lights_out_utc - t0_utc

    def _write_pkl(name: str, data: Any) -> None:
        p = os.path.join(cache_subdir, f"{name}.ff1pkl")
        with open(p, "wb") as f:
            pickle.dump({"version": 15, "data": data}, f)

    # 1. session_info.ff1pkl
    session_info_dict = {
        "Meeting": {
            "Key": int(matched_session.get("meeting_key") or 0),
            "Name": event_name,
            "OfficialName": str(session.event.get("OfficialEventName") or event_name),
            "Location": str(matched_session.get("location") or location),
            "Number": int(session.event.get("RoundNumber") or 1),
            "Country": {
                "Key": int(matched_session.get("country_key") or 0),
                "Code": str(matched_session.get("country_code") or ""),
                "Name": str(matched_session.get("country_name") or country),
            },
            "Circuit": {
                "Key": int(matched_session.get("circuit_key") or 0),
                "ShortName": str(matched_session.get("circuit_short_name") or location),
            },
        },
        "SessionStatus": "Inactive",
        "ArchiveStatus": {"Status": "Complete"},
        "Key": sk,
        "Type": str(matched_session.get("session_type") or "Race"),
        "Name": str(matched_session.get("session_name") or session_name),
        "StartDate": lights_out_utc.to_pydatetime(),
        "EndDate": session_end_utc.to_pydatetime(),
        "GmtOffset": pd.to_timedelta(matched_session.get("gmt_offset") or "00:00:00").to_pytimedelta(),
        "Path": session.api_path[8:],
    }
    _write_pkl("session_info", session_info_dict)

    # 2. driver_info.ff1pkl
    driver_info_dict: Dict[str, Any] = {}
    for idx, d in enumerate(drivers):
        drv_str = str(d["driver_number"])
        driver_info_dict[drv_str] = {
            "RacingNumber": drv_str,
            "BroadcastName": str(d.get("broadcast_name") or d.get("name_acronym") or drv_str),
            "FullName": str(d.get("full_name") or d.get("name_acronym") or drv_str),
            "Tla": str(d.get("name_acronym") or drv_str),
            "Line": idx + 1,
            "TeamName": str(d.get("team_name") or ""),
            "TeamColour": str(d.get("team_colour") or "888888"),
            "FirstName": str(d.get("first_name") or d.get("name_acronym") or drv_str),
            "LastName": str(d.get("last_name") or ""),
            "Reference": str(d.get("name_acronym") or drv_str),
            "HeadshotUrl": str(d.get("headshot_url") or ""),
            "CountryCode": str(d.get("country_code") or ""),
        }
    _write_pkl("driver_info", driver_info_dict)

    # 3. track_status_data.ff1pkl & race_control_messages.ff1pkl
    ts_times = [datetime.timedelta(seconds=0)]
    ts_status = ["1"]
    ts_msg = ["AllClear"]

    rcm_data: Dict[str, List[Any]] = {
        "Time": [],
        "Category": [],
        "Message": [],
        "Status": [],
        "Flag": [],
        "Scope": [],
        "Sector": [],
        "RacingNumber": [],
        "Lap": [],
    }
    for row in sorted(rcm, key=lambda x: str(x.get("date") or "")):
        dt_utc = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
        if pd.isna(dt_utc):
            continue
        dt_naive = dt_utc.tz_localize(None)
        td_val = (dt_naive - t0_utc).to_pytimedelta()
        msg_str = str(row.get("message") or "")
        cat_str = str(row.get("category") or "Other")
        flag_str = str(row.get("flag") or "")

        rcm_data["Time"].append(dt_naive.to_pydatetime())
        rcm_data["Category"].append(cat_str)
        rcm_data["Message"].append(msg_str)
        rcm_data["Status"].append(None)
        rcm_data["Flag"].append(flag_str if flag_str else None)
        rcm_data["Scope"].append(str(row.get("scope")) if row.get("scope") else None)
        rcm_data["Sector"].append(int(row["sector"]) if row.get("sector") is not None else None)
        rcm_data["RacingNumber"].append(str(row["driver_number"]) if row.get("driver_number") is not None else None)
        rcm_data["Lap"].append(int(row["lap_number"]) if row.get("lap_number") is not None else None)

        msg_up = msg_str.upper()
        new_code, new_label = None, None
        if "VSC DEPLOYED" in msg_up or "VIRTUAL SAFETY CAR DEPLOYED" in msg_up:
            new_code, new_label = "6", "VSCDeployed"
        elif "VSC ENDING" in msg_up or "VIRTUAL SAFETY CAR ENDING" in msg_up:
            new_code, new_label = "7", "VSCEnding"
        elif "SAFETY CAR DEPLOYED" in msg_up:
            new_code, new_label = "4", "SCDeployed"
        elif "SAFETY CAR IN THIS LAP" in msg_up:
            new_code, new_label = "1", "AllClear"
        elif flag_str == "RED":
            new_code, new_label = "5", "Red"
        elif flag_str in ("YELLOW", "DOUBLE YELLOW"):
            new_code, new_label = "2", "Yellow"
        elif flag_str in ("CLEAR", "GREEN"):
            new_code, new_label = "1", "AllClear"

        if new_code and new_code != ts_status[-1]:
            ts_times.append(td_val)
            ts_status.append(new_code)
            ts_msg.append(new_label)

    _write_pkl("track_status_data", {"Time": ts_times, "Status": ts_status, "Message": ts_msg})
    _write_pkl("race_control_messages", rcm_data)

    # 4. _extended_timing_data.ff1pkl & timing_app_data.ff1pkl
    pit_in_lookup = set()
    for p in pits:
        if p.get("driver_number") is not None and p.get("lap_number") is not None:
            pit_in_lookup.add((int(p["driver_number"]), int(p["lap_number"])))

    stints_by_drv: Dict[int, List[Dict[str, Any]]] = {}
    for st in sorted(stints, key=lambda x: (int(x.get("driver_number") or 0), int(x.get("stint_number") or 0))):
        d_num = int(st.get("driver_number") or 0)
        stints_by_drv.setdefault(d_num, []).append(st)
    for d_num, st_list in stints_by_drv.items():
        for i in range(len(st_list) - 1):
            if st_list[i].get("lap_end") is not None:
                pit_in_lookup.add((d_num, int(st_list[i]["lap_end"])))

    laps_rows: List[Dict[str, Any]] = []
    app_rows: List[Dict[str, Any]] = []
    lap_utc_windows: Dict[str, Tuple[str, str]] = {}

    for d_num, drv_df in df_laps_raw.groupby("driver_number"):
        drv_str = str(int(d_num))
        drv_df = drv_df.sort_values("lap_number").reset_index(drop=True)
        drv_stints = stints_by_drv.get(int(d_num), [])

        running_td = session_start_td
        best_lap_sec = float("inf")
        pit_count = 0
        lap_end_tds: Dict[int, Tuple[pd.Timedelta, pd.Timedelta]] = {}

        for _, row in drv_df.iterrows():
            lap_no = int(row["lap_number"])
            s1_sec = float(row["duration_sector_1"]) if pd.notna(row.get("duration_sector_1")) else np.nan
            s2_sec = float(row["duration_sector_2"]) if pd.notna(row.get("duration_sector_2")) else np.nan
            s3_sec = float(row["duration_sector_3"]) if pd.notna(row.get("duration_sector_3")) else np.nan
            lap_sec = float(row["lap_duration"]) if pd.notna(row.get("lap_duration")) else np.nan

            if np.isnan(lap_sec) and not (np.isnan(s1_sec) or np.isnan(s2_sec) or np.isnan(s3_sec)):
                lap_sec = s1_sec + s2_sec + s3_sec
            elif not np.isnan(lap_sec) and not (np.isnan(s1_sec) or np.isnan(s2_sec) or np.isnan(s3_sec)):
                if abs((s1_sec + s2_sec + s3_sec) - lap_sec) < 0.05:
                    s3_sec = max(0.001, lap_sec - s1_sec - s2_sec)

            dt_start = pd.to_datetime(row.get("date_start"), utc=True, errors="coerce")
            start_td = session_start_td if lap_no == 1 else running_td

            if not np.isnan(lap_sec) and lap_sec > 0:
                lap_td = pd.to_timedelta(round(lap_sec, 6), unit="s")
                end_td = start_td + lap_td
            else:
                lap_td = pd.NaT
                end_td = start_td + pd.to_timedelta(100, unit="s")

            running_td = end_td
            lap_end_tds[lap_no] = (start_td, end_td)

            utc_s = dt_start.tz_localize(None) if pd.notna(dt_start) else (t0_utc + start_td)
            utc_e = utc_s + (lap_td if pd.notna(lap_td) else pd.to_timedelta(100, unit="s"))
            lap_utc_windows[f"{drv_str}:{lap_no}"] = (
                utc_s.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                utc_e.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            )

            s1_td = pd.to_timedelta(round(s1_sec, 6), unit="s") if not np.isnan(s1_sec) else pd.NaT
            s2_td = pd.to_timedelta(round(s2_sec, 6), unit="s") if not np.isnan(s2_sec) else pd.NaT
            s3_td = pd.to_timedelta(round(s3_sec, 6), unit="s") if not np.isnan(s3_sec) else pd.NaT

            is_pit_in = (int(d_num), lap_no) in pit_in_lookup
            is_race_or_sprint = sname_norm in ("race", "sprint")
            is_pit_out = bool(row.get("is_pit_out_lap")) or (lap_no > 1 and (int(d_num), lap_no - 1) in pit_in_lookup)
            if is_pit_out:
                pit_count += 1

            # Yarış/Sprint'te 1. tur grid kalkışıdır; Antrenman/Sıralama'da 1. tur pit çıkışıdır
            if is_pit_out and (lap_no > 1 or not is_race_or_sprint):
                pit_out_td = start_td + pd.to_timedelta(12, unit="s")
            else:
                pit_out_td = pd.NaT
            pit_in_td = (end_td - pd.to_timedelta(5, unit="s")) if is_pit_in else pd.NaT

            is_pb = False
            if not np.isnan(lap_sec) and not is_pit_in and not is_pit_out and lap_sec <= best_lap_sec:
                best_lap_sec = lap_sec
                is_pb = True

            s1_sess = (start_td + s1_td) if pd.notna(s1_td) else pd.NaT
            s2_sess = (s1_sess + s2_td) if (pd.notna(s1_sess) and pd.notna(s2_td)) else pd.NaT
            s3_sess = end_td if pd.notna(lap_td) else pd.NaT

            laps_rows.append(
                {
                    "Time": end_td,
                    "Driver": drv_str,
                    "LapTime": lap_td,
                    "NumberOfLaps": int(lap_no),
                    "NumberOfPitStops": int(pit_count),
                    "PitOutTime": pit_out_td,
                    "PitInTime": pit_in_td,
                    "Sector1Time": s1_td,
                    "Sector2Time": s2_td,
                    "Sector3Time": s3_td,
                    "Sector1SessionTime": s1_sess,
                    "Sector2SessionTime": s2_sess,
                    "Sector3SessionTime": s3_sess,
                    "SpeedI1": float(row["i1_speed"]) if pd.notna(row.get("i1_speed")) else np.nan,
                    "SpeedI2": float(row["i2_speed"]) if pd.notna(row.get("i2_speed")) else np.nan,
                    "SpeedFL": np.nan,
                    "SpeedST": float(row["st_speed"]) if pd.notna(row.get("st_speed")) else np.nan,
                    "IsPersonalBest": bool(is_pb),
                }
            )

        seen_stints = set()
        if not drv_stints:
            drv_stints = [{"stint_number": 1, "lap_start": 1, "tyre_age_at_start": 0, "compound": "UNKNOWN"}]
        for st_idx, st in enumerate(drv_stints):
            st_no = int(st.get("stint_number") or (st_idx + 1)) - 1
            if st_no in seen_stints:
                continue
            seen_stints.add(st_no)
            l_start = int(st.get("lap_start") or 1)
            if l_start in lap_end_tds:
                st_time = lap_end_tds[l_start][1] - pd.to_timedelta(1, unit="ms")
            elif st_no == 0:
                st_time = pd.to_timedelta(0, unit="s")
            else:
                st_time = session_start_td
            start_age = float(st.get("tyre_age_at_start") or 0.0)
            app_rows.append(
                {
                    "LapNumber": float(l_start),
                    "Driver": drv_str,
                    "LapTime": pd.NaT,
                    "Stint": int(st_no),
                    "TotalLaps": start_age,
                    "Compound": str(st.get("compound") or "UNKNOWN").upper(),
                    "New": bool(start_age == 0),
                    "TyresNotChanged": False,
                    "Time": st_time,
                    "LapFlags": 0.0,
                    "LapCountTime": None,
                    "StartLaps": start_age,
                    "Outlap": None,
                }
            )

    laps_data_df = pd.DataFrame(laps_rows).sort_values(["Time", "Driver"]).reset_index(drop=True)
    stream_data_df = pd.DataFrame(
        {
            "Time": pd.Series(dtype="timedelta64[ns]"),
            "Driver": pd.Series(dtype="object"),
            "Position": pd.Series(dtype="int64"),
            "GapToLeader": pd.Series(dtype="object"),
            "IntervalToPositionAhead": pd.Series(dtype="object"),
        }
    )
    split_times = [datetime.timedelta(0), datetime.timedelta(days=1), datetime.timedelta(days=1)]
    _write_pkl("_extended_timing_data", (laps_data_df, stream_data_df, split_times))

    app_data_df = pd.DataFrame(app_rows).sort_values(["Time", "Driver"]).reset_index(drop=True)
    _write_pkl("timing_app_data", app_data_df)

    # 5. session_status_data.ff1pkl & lap_count.ff1pkl
    max_end_td = laps_data_df["Time"].max() if not laps_data_df.empty else (session_end_utc - t0_utc)
    session_status_dict = {
        "Time": [
            datetime.timedelta(seconds=10),
            session_start_td.to_pytimedelta(),
            (max_end_td + pd.to_timedelta(5, unit="s")).to_pytimedelta(),
        ],
        "Status": ["Inactive", "Started", "Finished"],
    }
    _write_pkl("session_status_data", session_status_dict)

    max_lap = int(df_laps_raw["lap_number"].max()) if not df_laps_raw.empty else 0
    _write_pkl(
        "lap_count",
        {
            "Time": [session_start_td.to_pytimedelta()],
            "TotalLaps": [max_lap],
            "CurrentLap": [1],
        },
    )

    with open(os.path.join(cache_subdir, "openf1_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_key": sk,
                "t0_utc": t0_utc.isoformat(),
                "lap_utc_windows": lap_utc_windows,
            },
            f,
        )

    return True, None


def _classify_session_load_failure(
    year: int,
    resolved_event: str,
    session_type: str,
    interceptor: _FastF1LogInterceptor,
    fallback_err: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Yakalanan istisnalardan:
    1. DNS çözümleme hatalarını (DNS_FAILURE — NameResolutionError / gaierror),
    2. F1TV kimlik doğrulama / yetkilendirme hatalarını (AUTH_FAILURE — HTTP 401/403 / F1AuthError),
    3. Eksik veya henüz yayınlanmamış seans verisini (MISSING_SESSION_DATA — SessionNotAvailableError / 404),
    4. Diğer ağ ve zaman aşımı hatalarını (NETWORK_ERROR)
    birbirinden ayırır ve hiçbir kimlik bilgisi veya erişim anahtarı (token) ifşa etmeden raporlar.
    """
    conn_diag = get_fastf1_connection_and_auth_diagnostics(check_dns=True)
    all_excs = interceptor.exceptions
    primary_exc = all_excs[0] if all_excs else None
    exc_type_name = type(primary_exc).__name__ if primary_exc is not None else "SessionNotAvailableError"
    exc_msg_raw = _sanitize_sensitive_text(
        str(primary_exc) if primary_exc is not None else (fallback_err or "Seans verisi yüklenemedi.")
    )
    clean_fallback = _sanitize_sensitive_text(fallback_err) if fallback_err else None
    clean_logs = [_sanitize_sensitive_text(w) for w in interceptor.warnings[:10]]

    combined_text = " ".join([exc_msg_raw] + clean_logs + ([clean_fallback] if clean_fallback else []))

    if any(
        k in combined_text
        for k in (
            "NameResolutionError",
            "gaierror",
            "nodename nor servname",
            "Failed to resolve",
            "Temporary failure in name resolution",
        )
    ) or not conn_diag["dns_resolved"]:
        category = "DNS_FAILURE"
        category_label = "DNS Çözümleme Hatası (Sunucu Adresi Çözümlenemedi)"
        summary_msg = (
            f"[{exc_type_name}] Resmi F1 zamanlama sunucusu ({conn_diag['livetiming_base_url']}) için DNS "
            f"çözümlemesi başarısız oldu. Lütfen DNS/ağ bağlantınızı kontrol edin. "
            f"Teknik Detay: {conn_diag.get('dns_error') or exc_msg_raw}"
        )
    elif any(
        k in combined_text
        for k in (
            "Read timed out",
            "ConnectTimeoutError",
            "ConnectionResetError",
            "ConnectionRefusedError",
        )
    ):
        category = "NETWORK_ERROR"
        category_label = "Ağ Bağlantı / Zaman Aşımı Hatası"
        summary_msg = (
            f"[{exc_type_name}] {year} {resolved_event} ({session_type}) indirilirken ağ zaman aşımı veya "
            f"bağlantı kopması oluştu ({conn_diag['livetiming_base_url']}). Teknik Detay: {exc_msg_raw}"
        )
    elif (
        "Failed to authenticate with Formula 1 account" in combined_text
        or "F1AuthError" in combined_text
        or "401" in combined_text
        or "403" in combined_text
    ) and not (clean_fallback and "henüz yayınlanmış tur verisi bulunmuyor" in clean_fallback):
        category = "AUTH_FAILURE"
        category_label = "F1TV Kimlik Doğrulama / Yetkilendirme Hatası (HTTP 401/403)"
        summary_msg = (
            f"[{exc_type_name}] Resmi FastF1 sunucusu ({conn_diag['livetiming_base_url']}) bu seans için "
            f"F1TV hesap kimlik doğrulaması gerektirdi (Durum: {conn_diag['auth_status']}). "
            f"Terminalde 'python -m fastf1 auth f1tv --status' ile durumu kontrol edebilir veya "
            f"'python -m fastf1 auth f1tv --authenticate' ile oturum açabilirsiniz. "
            f"{('Yedek sağlayıcı detayı: ' + clean_fallback) if clean_fallback else exc_msg_raw}"
        )
    else:
        category = "MISSING_SESSION_DATA"
        category_label = "Seans Verisi Mevcut Değil / Henüz Yayınlanmadı"
        summary_msg = (
            f"[{exc_type_name}] {year} {resolved_event} ({session_type}) için zamanlama verisi çevrimiçi "
            f"sağlayıcıda mevcut değil (seans henüz gerçekleşmemiş olabilir). "
            f"{clean_fallback or exc_msg_raw}"
        )

    return {
        "status": "error",
        "source": "failed",
        "error_category": category,
        "error_category_label": category_label,
        "exception_type": exc_type_name,
        "exception_message": exc_msg_raw,
        "fallback_error": clean_fallback,
        "captured_logs": clean_logs,
        "formatted_error": summary_msg,
        "fastf1_version": conn_diag["fastf1_version"],
        "livetiming_base_url": conn_diag["livetiming_base_url"],
        "dns_resolved": conn_diag["dns_resolved"],
        "auth_status": conn_diag["auth_status"],
    }


def load_f1_session(
    year: int,
    event_name: str,
    session_type: str = "R",
    load_telemetry: bool = True,
    load_weather: bool = False,
    load_messages: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Belirtilen sezon, Grand Prix ve seans tipi için FastF1 Session nesnesini yükler.

    İşleyiş:
    1. Önce yerel önbellekte (_extended_timing_data.ff1pkl) seansın hazır olup olmadığını kontrol eder.
    2. FastF1 Session.load() çağrısını özel _FastF1LogInterceptor ile izleyerek @soft_exceptions
       tarafından gizlenen gerçek istisnaları (ConnectionError, F1AuthError, SessionNotAvailableError) yakalar.
    3. Eğer seans yerel önbellekte yoksa ve livetiming.formula1.com F1TV kimlik doğrulaması (HTTP 401)
       nedeniyle tur verisini döndürmezse, otomatik olarak açık erişimli OpenF1 API üzerinden seans
       verisini indirip yerel önbelleğe (.ff1pkl) yazar ve Session.load() işlemini tamamlar.
    """
    setup_fastf1_cache(cache_dir)
    resolved_event = resolve_event_name(event_name)
    normalized_session = normalize_session_code(session_type, year=year)
    diag_key = (int(year), resolved_event, str(normalized_session))

    # 1. Etkinlik (Event / Session) çözümleme
    try:
        session = fastf1.get_session(year, resolved_event, normalized_session)
    except Exception as lookup_exc:
        # Hafta sonu formatında bu seansın bulunup bulunmadığını kontrol et
        valid_sessions = get_event_available_sessions(year, resolved_event, cache_dir=cache_dir)
        if normalized_session not in valid_sessions or "does not exist for this event" in str(lookup_exc).lower():
            valid_str = ", ".join([f"{c} ({m['label']})" for c, m in valid_sessions.items()])
            err_msg = (
                f"[ValueError] '{session_type}' seansı {year} {resolved_event} programında bulunmamaktadır. "
                f"Bu yarış hafta sonu için geçerli seanslar: {valid_str}."
            )
            LAST_SESSION_LOAD_DIAGNOSTICS[diag_key] = {
                "status": "error",
                "source": "failed",
                "error_category": "INVALID_SESSION_FOR_EVENT",
                "error_category_label": "Geçersiz Seans Tipi (Bu Hafta Sonu Programında Bu Seans Yok)",
                "exception_type": "ValueError",
                "exception_message": str(lookup_exc),
                "captured_logs": [],
                "formatted_error": err_msg,
            }
            return None, err_msg

        # Çevrimdışı durumda yerel önbellek / yedek takvim üzerinden Session oluşturmayı dene
        try:
            session = _build_offline_session_object(year, resolved_event, normalized_session, cache_dir=cache_dir)
        except Exception as offline_exc:
            exc_type = type(lookup_exc).__name__
            err_msg = f"[{exc_type}] Etkinlik takvimde çözümlenemedi ({year} {resolved_event} - {session_type}): {lookup_exc}"
            LAST_SESSION_LOAD_DIAGNOSTICS[diag_key] = {
                "status": "error",
                "source": "failed",
                "error_category": "EVENT_LOOKUP_ERROR",
                "error_category_label": "Etkinlik / Takvim Çözümleme Hatası (Event Lookup Failure)",
                "exception_type": exc_type,
                "exception_message": str(lookup_exc),
                "captured_logs": [],
                "formatted_error": err_msg,
            }
            return None, err_msg

    cache_subdir = os.path.join(cache_dir, session.api_path[8:])
    has_local_timing_cache = os.path.isfile(os.path.join(cache_subdir, "_extended_timing_data.ff1pkl"))
    has_local_car_cache = os.path.isfile(os.path.join(cache_subdir, "car_data.ff1pkl"))

    # 2. FastF1 Session.load() çağrısını log yakalayıcı ile çalıştır
    interceptor = _FastF1LogInterceptor()
    ff1_logger = logging.getLogger("fastf1")
    prev_level = ff1_logger.level
    ff1_logger.addHandler(interceptor)
    ff1_logger.setLevel(logging.DEBUG)

    try:
        # Eğer yerel önbellekte timing var ama tam car_data.ff1pkl yoksa (OpenF1 hibrit önbelleği),
        # gereksiz 401/404 telemetri isteklerini önlemek için telemetry=has_local_car_cache kullan
        effective_load_tel = load_telemetry if (not has_local_timing_cache or has_local_car_cache) else False
        session.load(
            laps=True,
            telemetry=effective_load_tel,
            weather=load_weather,
            messages=load_messages,
        )
    except Exception as direct_exc:
        interceptor.exceptions.append(direct_exc)
    finally:
        ff1_logger.removeHandler(interceptor)
        ff1_logger.setLevel(prev_level)

    # 3. Tur verisinin (_laps) yüklenip yüklenmediğini kontrol et
    laps_loaded = hasattr(session, "_laps") and session._laps is not None and not session._laps.empty

    if not laps_loaded:
        # Eğer birincil FastF1 LiveTiming sunucusu 401/404 döndürdüyse, OpenF1 üzerinden indirip önbelleğe yaz
        fallback_err: Optional[str] = None
        try:
            hydrated_ok, hydrate_msg = _hydrate_session_cache_from_openf1(session, cache_dir=cache_dir)
            if hydrated_ok:
                session.load(
                    laps=True,
                    telemetry=False,
                    weather=False,
                    messages=True,
                )
                laps_loaded = hasattr(session, "_laps") and session._laps is not None and not session._laps.empty
            else:
                fallback_err = hydrate_msg
        except Exception as fb_exc:
            interceptor.exceptions.append(fb_exc)
            fallback_err = f"{type(fb_exc).__name__}: {fb_exc}"

        if not laps_loaded:
            diag = _classify_session_load_failure(year, resolved_event, session_type, interceptor, fallback_err)
            LAST_SESSION_LOAD_DIAGNOSTICS[diag_key] = diag
            return None, diag["formatted_error"]

        LAST_SESSION_LOAD_DIAGNOSTICS[diag_key] = {
            "status": "ok",
            "source": "openf1_fallback",
            "error_category": "NONE",
            "error_category_label": "Çevrimiçi İndirildi ve Önbelleğe Alındı (OpenF1 + FastF1 Cache)",
            "exception_type": None,
            "exception_message": None,
            "captured_logs": [],
            "formatted_error": None,
        }
        return session, None

    LAST_SESSION_LOAD_DIAGNOSTICS[diag_key] = {
        "status": "ok",
        "source": "local_cache" if has_local_timing_cache else "fastf1_upstream",
        "error_category": "NONE",
        "error_category_label": "Yerel Önbellekten Yüklendi" if has_local_timing_cache else "FastF1 Sunucusundan İndirildi",
        "exception_type": None,
        "exception_message": None,
        "captured_logs": [],
        "formatted_error": None,
    }
    return session, None


def extract_driver_lap_telemetry(
    session: Any,
    driver_laps: pd.DataFrame,
    lap_selection: str = "Fastest",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[pd.DataFrame, Optional[int], Optional[float], Optional[str]]:
    """
    Seçilen sürücünün belirtilen turu (veya 'Fastest' en hızlı turu) için telemetri
    DataFrame'ini (Distance, Speed, Throttle, Brake, nGear, RPM, DRS, Time) döndürür.

    Önce FastF1'in yerleşik `target_lap.get_telemetry().add_distance()` metodunu dener;
    eğer seans OpenF1 üzerinden önbelleğe alınmışsa ilgili turun 3.7 Hz telemetrisini
    OpenF1 `/v1/car_data` uç noktasından indirip yerel önbelleğe kaydederek döndürür.
    """
    if driver_laps is None or driver_laps.empty:
        return pd.DataFrame(), None, None, "Sürücüye ait tur verisi bulunamadı."

    lap_num_val: Optional[int] = None
    lap_sec_val: Optional[float] = None
    try:
        if lap_selection in ("Fastest", "En Hızlı Tur", "En Hızlı Tur (Fastest)", ""):
            target_lap = driver_laps.pick_fastest()
            if target_lap is None or (isinstance(target_lap, pd.Series) and target_lap.empty):
                target_lap = driver_laps.pick_fastest(only_by_time=True)
            if target_lap is None or (isinstance(target_lap, pd.Series) and target_lap.empty):
                return pd.DataFrame(), None, None, "En hızlı tur seçilemedi."
        else:
            lap_no = int(lap_selection)
            matched = driver_laps[driver_laps["LapNumber"] == lap_no]
            if matched.empty:
                return pd.DataFrame(), lap_no, None, f"Tur {lap_no} bulunamadı."
            target_lap = matched.iloc[0]

        lap_num_val = (
            int(target_lap["LapNumber"])
            if "LapNumber" in target_lap and pd.notna(target_lap["LapNumber"])
            else None
        )
        lap_sec_val = None
        if "LapTime" in target_lap and pd.notna(target_lap["LapTime"]):
            lt = target_lap["LapTime"]
            lap_sec_val = float(lt.total_seconds()) if hasattr(lt, "total_seconds") else float(lt)

        # 1. Yerleşik FastF1 telemetrisi mevcutsa doğrudan kullan
        try:
            tel = target_lap.get_telemetry().add_distance()
            if tel is not None and not tel.empty:
                return tel.copy(), lap_num_val, lap_sec_val, None
        except Exception:
            pass

        # 2. OpenF1 tur bazlı telemetri önbelleği / indirme yedeklemesi
        if session is None or lap_num_val is None:
            return pd.DataFrame(), lap_num_val, lap_sec_val, "Telemetri verisi yüklenemedi."

        drv_num_str = str(target_lap.get("DriverNumber") or "").strip()
        cache_subdir = os.path.join(cache_dir, session.api_path[8:])
        lap_tel_cache_file = os.path.join(cache_subdir, f"openf1_lap_tel_{drv_num_str}_{lap_num_val}.pkl")

        if os.path.isfile(lap_tel_cache_file):
            with open(lap_tel_cache_file, "rb") as f:
                cached_tel = pickle.load(f)
            if isinstance(cached_tel, pd.DataFrame) and not cached_tel.empty:
                return cached_tel.copy(), lap_num_val, lap_sec_val, None

        meta_path = os.path.join(cache_subdir, "openf1_meta.json")
        if not os.path.isfile(meta_path):
            return pd.DataFrame(), lap_num_val, lap_sec_val, "Telemetri önbellek üst verisi (openf1_meta.json) bulunamadı."

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        sk = meta.get("session_key")
        win = meta.get("lap_utc_windows", {}).get(f"{drv_num_str}:{lap_num_val}")
        if not sk or not win:
            return pd.DataFrame(), lap_num_val, lap_sec_val, f"Sürücü #{drv_num_str} Tur {lap_num_val} zaman penceresi bulunamadı."

        url = f"https://api.openf1.org/v1/car_data?session_key={sk}&driver_number={drv_num_str}&date>={win[0]}&date<={win[1]}"
        raw_rows = _fetch_openf1_json_list(url, timeout=20)
        if not raw_rows:
            return pd.DataFrame(), lap_num_val, lap_sec_val, f"Tur {lap_num_val} için telemetri örneği bulunamadı."

        raw_df = pd.DataFrame(raw_rows)
        raw_df["Date"] = pd.to_datetime(raw_df["date"], utc=True, errors="coerce").dt.tz_localize(None)
        raw_df = raw_df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        if raw_df.empty:
            return pd.DataFrame(), lap_num_val, lap_sec_val, "Telemetri zaman damgaları ayrıştırılamadı."

        t_zero = raw_df["Date"].iloc[0]
        time_td = raw_df["Date"] - t_zero
        dt_sec = time_td.dt.total_seconds().diff().fillna(0.0).clip(lower=0.0, upper=2.0)

        speed_kmh = pd.to_numeric(raw_df["speed"], errors="coerce").fillna(0.0).astype(float)
        throttle_pct = pd.to_numeric(raw_df["throttle"], errors="coerce").fillna(0.0).astype(float)
        brake_raw_num = pd.to_numeric(raw_df["brake"], errors="coerce").fillna(0.0)
        brake_bool = (brake_raw_num > 0).astype(bool)
        n_gear = pd.to_numeric(raw_df["n_gear"], errors="coerce").fillna(0).astype(int)
        rpm = pd.to_numeric(raw_df["rpm"], errors="coerce").fillna(0).astype(int)
        drs = pd.to_numeric(raw_df["drs"], errors="coerce").fillna(0).astype(int)

        # Mesafe (Distance) hesabı: d = cumsum(v_m_s * dt)
        dist_m = ((speed_kmh / 3.6) * dt_sec).cumsum()

        tel_df = pd.DataFrame(
            {
                "Date": raw_df["Date"],
                "Time": time_td,
                "SessionTime": time_td,
                "Speed": speed_kmh,
                "RPM": rpm,
                "nGear": n_gear,
                "Throttle": throttle_pct,
                "Brake": brake_bool,
                "DRS": drs,
                "Distance": dist_m,
            }
        )

        os.makedirs(cache_subdir, exist_ok=True)
        with open(lap_tel_cache_file, "wb") as f:
            pickle.dump(tel_df, f)

        return tel_df.copy(), lap_num_val, lap_sec_val, None
    except Exception as exc:
        return pd.DataFrame(), lap_num_val, lap_sec_val, f"{type(exc).__name__}: {exc}"


def get_session_drivers_info(session: Any) -> List[Dict[str, str]]:
    """
    Yüklenmiş bir seanstaki sürücülerin kısaltma, tam ad ve takım bilgilerini döndürür.
    """
    drivers_list: List[Dict[str, str]] = []
    if session is None:
        return drivers_list

    seen = set()
    try:
        driver_nums = list(session.drivers)
    except Exception:
        driver_nums = []

    for drv in driver_nums:
        try:
            info = session.get_driver(drv)
            abbr = str(info.get("Abbreviation", "")).strip()
            if not abbr or abbr in seen:
                continue
            seen.add(abbr)
            full_name = str(info.get("FullName", abbr)).strip()
            team_name = str(info.get("TeamName", "")).strip()
            team_color = str(info.get("TeamColor", "")).strip()
            drivers_list.append(
                {
                    "abbreviation": abbr,
                    "driver_number": str(drv),
                    "full_name": full_name,
                    "team_name": team_name,
                    "team_color": f"#{team_color}" if team_color and not team_color.startswith("#") else team_color,
                }
            )
        except Exception:
            continue

    if not drivers_list:
        try:
            laps = session.laps
            if "Driver" in laps.columns:
                for abbr in laps["Driver"].dropna().unique():
                    abbr_str = str(abbr).strip()
                    if abbr_str and abbr_str not in seen:
                        seen.add(abbr_str)
                        drivers_list.append(
                            {
                                "abbreviation": abbr_str,
                                "driver_number": "",
                                "full_name": abbr_str,
                                "team_name": "",
                                "team_color": "",
                            }
                        )
        except Exception:
            pass

    return drivers_list


def get_raw_driver_laps(session: Any, driver_code: str) -> pd.DataFrame:
    """
    Ham veriyi (session.laps) değiştirmeden, seçilen sürücünün turlarının
    bağımsız bir kopyasını (.copy()) döndürür.
    """
    if session is None or not driver_code:
        return pd.DataFrame()
    try:
        laps = session.laps
        if hasattr(laps, "pick_drivers"):
            drv_laps = laps.pick_drivers(driver_code)
        elif hasattr(laps, "pick_driver"):
            drv_laps = laps.pick_driver(driver_code)
        else:
            drv_laps = laps[laps["Driver"] == driver_code]
        return drv_laps.copy()
    except Exception:
        return pd.DataFrame()


def decode_track_status(status_val: Any) -> List[str]:
    """
    FastF1 TrackStatus alanında birden fazla kod birleşik bulunabilir (örn. '12', '24', '16').
    Her karakter kodunu okunabilir açıklamaya çevirir.
    """
    if status_val is None or pd.isna(status_val):
        return ["Bilinmiyor"]
    s = str(status_val).strip()
    if not s:
        return ["Bilinmiyor"]
    labels = []
    for ch in s:
        label = TRACK_STATUS_LABELS.get(ch)
        if label and label not in labels:
            labels.append(label)
    return labels if labels else [f"Kod: {s}"]


def has_neutralized_track_status(status_val: Any) -> bool:
    """
    Bir turun Güvenlik Aracı (4), Kırmızı Bayrak (5) veya Sanal Güvenlik Aracı (6, 7)
    altında geçip geçmediğini döndürür.
    """
    if status_val is None or pd.isna(status_val):
        return False
    s = str(status_val).strip()
    return any(code in s for code in ("4", "5", "6", "7"))


def summarize_session_conditions(session: Any) -> Dict[str, Any]:
    """
    Seans genelinde pist durumlarını (SC, VSC, Kırmızı Bayrak, Sarı Bayrak turları),
    mevcut veri sütunlarını ve hava durumu özetini çıkarır.
    """
    summary: Dict[str, Any] = {
        "total_session_laps": 0,
        "sc_laps": [],
        "vsc_laps": [],
        "red_flag_laps": [],
        "yellow_flag_laps": [],
        "available_lap_columns": [],
        "weather_available": False,
        "weather_summary": {},
    }
    if session is None:
        return summary

    try:
        laps = session.laps
        if laps is not None and not laps.empty:
            summary["available_lap_columns"] = list(laps.columns)
            if "LapNumber" in laps.columns:
                summary["total_session_laps"] = int(laps["LapNumber"].dropna().max() or 0)

            if "TrackStatus" in laps.columns and "LapNumber" in laps.columns:
                for lap_num, grp in laps.groupby("LapNumber"):
                    statuses = "".join(grp["TrackStatus"].dropna().astype(str).unique())
                    lap_int = int(lap_num)
                    if "4" in statuses:
                        summary["sc_laps"].append(lap_int)
                    if "6" in statuses or "7" in statuses:
                        summary["vsc_laps"].append(lap_int)
                    if "5" in statuses:
                        summary["red_flag_laps"].append(lap_int)
                    if "2" in statuses:
                        summary["yellow_flag_laps"].append(lap_int)
    except Exception:
        pass

    try:
        weather = getattr(session, "weather_data", None)
        if weather is not None and not weather.empty:
            summary["weather_available"] = True
            w_info: Dict[str, Any] = {}
            if "TrackTemp" in weather.columns:
                w_info["track_temp_mean_c"] = round(float(weather["TrackTemp"].dropna().mean()), 1)
                w_info["track_temp_min_c"] = round(float(weather["TrackTemp"].dropna().min()), 1)
                w_info["track_temp_max_c"] = round(float(weather["TrackTemp"].dropna().max()), 1)
            if "AirTemp" in weather.columns:
                w_info["air_temp_mean_c"] = round(float(weather["AirTemp"].dropna().mean()), 1)
            if "Rainfall" in weather.columns:
                w_info["rainfall_detected"] = bool(weather["Rainfall"].dropna().any())
            summary["weather_summary"] = w_info
    except Exception:
        pass

    return summary
