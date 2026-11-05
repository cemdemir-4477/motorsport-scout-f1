"""
F1 Performance Analysis & Telemetry Dashboard (app.py)
======================================================
FastF1 tabanlı kapsamlı Formula 1 sürücü performansı, stint gelişimi, lastik kullanımı,
sektör/teorik optimal tur, sürücü fark analizi ve ortak mesafe eksenli telemetri dashboard'u.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from f1_data_loader import (
    DEFAULT_CACHE_DIR,
    SESSION_TYPE_OPTIONS,
    extract_driver_lap_telemetry,
    get_event_available_sessions,
    get_last_load_diagnostics,
    get_raw_driver_laps,
    get_season_schedule,
    get_session_drivers_info,
    list_cached_sessions,
    load_f1_session,
    normalize_session_code,
    setup_fastf1_cache,
    summarize_session_conditions,
)
from f1_performance import (
    STINT_DEGRADATION_METHODOLOGY_WARNING,
    TTEST_METHODOLOGY_WARNING,
    analyze_driver_stints,
    build_sector_comparison_table,
    calculate_consistency_score,
    calculate_lap_by_lap_deltas,
    calculate_pace_score,
    calculate_sector_zscores,
    calculate_theoretical_optimal,
    compare_drivers_ttest,
    compare_matching_stints,
    filter_clean_laps,
    format_laptime_str,
    to_seconds_series,
)
from f1_telemetry import (
    BRAKE_ACTIVITY_DISCLAIMER,
    align_and_compare_telemetry,
)

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="F1 Performance Analysis & Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
)

setup_fastf1_cache(DEFAULT_CACHE_DIR)

# Lastik hamuru görsel renk ve sembol haritası
COMPOUND_STYLES: Dict[str, Dict[str, str]] = {
    "SOFT": {"color": "#ef4444", "symbol": "circle", "label": "Soft (Yumuşak)"},
    "MEDIUM": {"color": "#eab308", "symbol": "diamond", "label": "Medium (Orta)"},
    "HARD": {"color": "#94a3b8", "symbol": "square", "label": "Hard (Sert)"},
    "INTERMEDIATE": {"color": "#22c55e", "symbol": "triangle-up", "label": "Intermediate (Geçiş)"},
    "WET": {"color": "#3b82f6", "symbol": "cross", "label": "Wet (Yağmur)"},
    "UNKNOWN": {"color": "#a855f7", "symbol": "x", "label": "Bilinmiyor"},
}

DRIVER_PALETTE = ("#0284c7", "#ea580c", "#16a34a")


def header_with_info(title: str, info_text: str, level: str = "h3") -> None:
    """Başlığın yanına metodolojik açıklama içeren bir bilgi popover'ı ekler."""
    col1, col2 = st.columns([0.94, 0.06])
    with col1:
        if level == "h2":
            st.subheader(title)
        elif level == "h3":
            st.markdown(f"### {title}")
        else:
            st.markdown(f"**{title}**")
    with col2:
        with st.popover("ℹ️"):
            st.markdown(info_text)


@st.cache_data(show_spinner=False)
def cached_get_schedule(year: int) -> pd.DataFrame:
    return get_season_schedule(year, include_testing=False, cache_dir=DEFAULT_CACHE_DIR)


@st.cache_resource(show_spinner=False)
def cached_load_session(year: int, event_name: str, session_type: str) -> Tuple[Optional[Any], Optional[str]]:
    return load_f1_session(
        year=year,
        event_name=event_name,
        session_type=session_type,
        load_telemetry=True,
        load_weather=False,
        load_messages=False,
        cache_dir=DEFAULT_CACHE_DIR,
    )


def apply_user_lap_filters(
    laps_df: pd.DataFrame,
    selected_stint: str,
    selected_compound: str,
) -> pd.DataFrame:
    """Kullanıcının yan menüden seçtiği Stint ve Tyre Compound filtrelerini uygular."""
    if laps_df is None or laps_df.empty:
        return pd.DataFrame() if laps_df is None else laps_df.copy()

    out = laps_df.copy()
    if selected_stint != "Tümü" and "Stint" in out.columns:
        try:
            st_val = int(selected_stint)
            out = out[out["Stint"] == st_val].copy()
        except ValueError:
            pass

    if selected_compound != "Tümü" and "Compound" in out.columns:
        out = out[out["Compound"].astype(str).str.upper() == selected_compound.upper()].copy()

    return out


def extract_selected_lap_telemetry(
    session_obj: Any,
    driver_laps: pd.DataFrame,
    lap_selection: str,
) -> Tuple[pd.DataFrame, Optional[int], Optional[float], Optional[str]]:
    """
    Seçilen turun ('En Hızlı Tur (Fastest)' veya belirli tur numarası) telemetrisini,
    tur numarasını, tur süresini (sn) ve olası hata mesajını döndürür.
    Ham veriyi değiştirmez; FastF1 veya OpenF1 yedek önbelleğini kullanır.
    """
    return extract_driver_lap_telemetry(
        session=session_obj,
        driver_laps=driver_laps,
        lap_selection=lap_selection,
        cache_dir=DEFAULT_CACHE_DIR,
    )


# ==============================================================================
# ANA BAŞLIK VE YAN MENÜ (KULLANICI KONTROLLERİ — BÖLÜM 6)
# ==============================================================================

st.title("🏎️ F1 Performans ve Telemetri Analiz Sistemi")
st.markdown(
    "Bu sistem; **Tur Temposunu**, **Stint gelişimini ve Lastik Hamuru kullanımını**, "
    "**Sektör ve Teorik Optimal Tur performansını**, **sürücüler arası Tur Süresi/Stint farklarını** ve "
    "**ortak mesafe ekseninde hizalanmış telemetri (Fren Aktivitesi dahil) davranışlarını** "
    "metodolojik doğrulama ile analiz eder."
)

st.sidebar.header("🎛️ Kullanıcı Kontrolleri")

# Yerel önbellekte bulunan yarışları tespit et
cached_sessions_list = list_cached_sessions(DEFAULT_CACHE_DIR)
if cached_sessions_list:
    cached_labels = [f"{c['year']} {c['event_name']} ({c['session_name']})" for c in cached_sessions_list]
    with st.sidebar.expander("💾 Önbellekte Hazır Yarışlar", expanded=False):
        for lbl in cached_labels:
            st.caption(f"• {lbl}")

# 1. Sezon
years_list = list(range(2026, 2017, -1))
default_year_idx = years_list.index(2025) if 2025 in years_list else 0
selected_year = st.sidebar.selectbox("1. Sezon", years_list, index=default_year_idx)

# 2. Grand Prix (2026 ve Meksika GP dahil)
schedule_df = cached_get_schedule(selected_year)
events_list = schedule_df["EventName"].dropna().unique().tolist() if not schedule_df.empty and "EventName" in schedule_df.columns else []
if "Mexico City Grand Prix" not in events_list:
    events_list.append("Mexico City Grand Prix")

default_event_idx = 0
for idx_e, ev_name in enumerate(events_list):
    if "Australian" in str(ev_name):
        default_event_idx = idx_e
        break

selected_event = st.sidebar.selectbox(
    "2. Grand Prix (Yarış)",
    events_list,
    index=default_event_idx,
    help="2026 sezonu ve Mexico City Grand Prix (Meksika GP) dahil tüm yarışlar seçilebilir.",
)

# 3. Seans Tipi (Seçilen Grand Prix hafta sonuna göre dinamik)
available_sessions = get_event_available_sessions(selected_year, selected_event)
session_codes = list(available_sessions.keys())
default_sess_idx = session_codes.index("R") if "R" in session_codes else 0

selected_session_type = st.sidebar.selectbox(
    "3. Seans Tipi",
    session_codes,
    index=default_sess_idx,
    format_func=lambda code: available_sessions.get(code, {}).get("full_label") or f"{code} — {SESSION_TYPE_OPTIONS.get(code, code)}",
    help="Seçilen yarış hafta sonunun formatına (Klasik veya Sprint) uygun gerçek seanslar listelenir.",
)

# Seansı Yükle
with st.sidebar:
    with st.spinner("Seans verisi yükleniyor..."):
        session_obj, session_load_err = cached_load_session(
            selected_year, selected_event, selected_session_type
        )

drivers_info = get_session_drivers_info(session_obj) if session_obj is not None else []
driver_codes = [d["abbreviation"] for d in drivers_info]
driver_label_map = {
    d["abbreviation"]: f"{d['abbreviation']} ({d['full_name']})" if d.get("full_name") and d["full_name"] != d["abbreviation"] else d["abbreviation"]
    for d in drivers_info
}

if not driver_codes:
    st.sidebar.error("Seçilen seans için sürücü listesi alınamadı.")
    diag = get_last_load_diagnostics(selected_year, selected_event, selected_session_type)
    if session_load_err:
        st.error(f"**Seans Yükleme Hatası ({diag.get('error_category', 'UNKNOWN')} — {diag.get('error_category_label', '')}):** {session_load_err}")
        if diag.get("exception_type") or diag.get("exception_message"):
            with st.expander("🔎 Teknik Tanılama Detayları (İstisna & FastF1 Kayıtları)", expanded=True):
                st.write(f"**Hata Sınıfı:** `{diag.get('error_category')}` — {diag.get('error_category_label')}")
                st.write(f"**FastF1 Sürümü:** `{diag.get('fastf1_version', '3.9.0')}` | **Resmi Sunucu:** `{diag.get('livetiming_base_url', 'https://livetiming.formula1.com')}`")
                st.write(f"**DNS Çözümleme Durumu:** `{'Çözümlendi (OK)' if diag.get('dns_resolved') else 'Çözümlenemedi'}` | **F1TV Kimlik Doğrulama Durumu:** `{diag.get('auth_status', 'Not authenticated')}`")
                st.write(f"**Yakalanan İstisna Tipi:** `{diag.get('exception_type')}`")
                st.code(str(diag.get("exception_message", "")))
                if diag.get("captured_logs"):
                    st.code("\n".join(diag["captured_logs"]))
        if st.button("🔄 Önbelleği Temizle ve Seansı Çevrimiçi İndirmeyi Tekrar Dene"):
            cached_load_session.clear()
            st.rerun()
    st.stop()

# 4. 1. Sürücü & 5. 2. Sürücü & 9. Referans Sürücü
default_d1_idx = 0
default_d2_idx = 1 if len(driver_codes) > 1 else 0
for idx_d, code in enumerate(driver_codes):
    if code == "VER":
        default_d1_idx = idx_d
    elif code in ("NOR", "LEC") and idx_d != default_d1_idx:
        default_d2_idx = idx_d

driver1 = st.sidebar.selectbox(
    "4. 1. Sürücü",
    driver_codes,
    index=default_d1_idx,
    format_func=lambda c: driver_label_map.get(c, c),
)

driver2 = st.sidebar.selectbox(
    "5. 2. Sürücü",
    driver_codes,
    index=default_d2_idx if default_d2_idx != default_d1_idx else (1 if len(driver_codes) > 1 else 0),
    format_func=lambda c: driver_label_map.get(c, c),
)

ref_driver = st.sidebar.selectbox(
    "9. Referans Sürücü",
    driver_codes,
    index=driver_codes.index(driver1) if driver1 in driver_codes else 0,
    format_func=lambda c: driver_label_map.get(c, c),
    help="Tur Temposu, Stint Medyan farkları ve Sektör Z-Skorları için referans alınacak sürücü.",
)

# Ham sürücü turlarını çek (Ham veri korunur, kopyaları alınır)
raw_d1_all = get_raw_driver_laps(session_obj, driver1)
raw_d2_all = get_raw_driver_laps(session_obj, driver2)
raw_ref_all = get_raw_driver_laps(session_obj, ref_driver)

# Mevcut Stint, Compound ve LapNumber seçeneklerini oluştur
all_stints_union = sorted(
    set(
        (raw_d1_all["Stint"].dropna().astype(int).tolist() if "Stint" in raw_d1_all.columns else [])
        + (raw_d2_all["Stint"].dropna().astype(int).tolist() if "Stint" in raw_d2_all.columns else [])
    )
)
stint_options = ["Tümü"] + [str(s) for s in all_stints_union]

all_compounds_union = sorted(
    {
        str(c).upper()
        for c in (
            (raw_d1_all["Compound"].dropna().tolist() if "Compound" in raw_d1_all.columns else [])
            + (raw_d2_all["Compound"].dropna().tolist() if "Compound" in raw_d2_all.columns else [])
        )
        if str(c).upper() not in ("", "NAN", "NONE")
    }
)
compound_options = ["Tümü"] + all_compounds_union

all_laps_union = sorted(
    set(
        (raw_d1_all["LapNumber"].dropna().astype(int).tolist() if "LapNumber" in raw_d1_all.columns else [])
        + (raw_d2_all["LapNumber"].dropna().astype(int).tolist() if "LapNumber" in raw_d2_all.columns else [])
    )
)
lap_options = ["En Hızlı Tur"] + [str(lno) for lno in all_laps_union]

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtre ve Telemetri Parametreleri")

# 7. Stint
selected_stint = st.sidebar.selectbox(
    "7. Stint",
    stint_options,
    index=0,
    help="Belirli bir stinte odaklanmak için seçin veya 'Tümü' ile tüm yarışı inceleyin.",
)

# 8. Lastik Hamuru
selected_compound = st.sidebar.selectbox(
    "8. Lastik Hamuru",
    compound_options,
    index=0,
)

# 6. Telemetri Turu
selected_lap_mode = st.sidebar.selectbox(
    "6. Telemetri Turu",
    lap_options,
    index=0,
    help="Telemetri karşılaştırması için 'En Hızlı Tur' veya belirli bir tur numarası seçin.",
)

# 10. Fren Aktivitesi Penceresi
selected_brake_window = st.sidebar.selectbox(
    "10. Fren Aktivitesi Penceresi",
    [10, 20, 30],
    index=1,
    format_func=lambda w: f"{w} metre",
    help="Fren Aktivitesi (BrakeActivity ve SmoothedBrakeActivity) hesabı için mesafe bazlı pencere genişliği.",
)

# Çeyrekler Açıklığı (IQR) Filtreleme Modu
iqr_by_stint = st.sidebar.toggle(
    "Çeyrekler Açıklığı (IQR) Filtresini Stint Bazında Uygula",
    value=True,
    help="Açık olduğunda her stint kendi dağılımı içinde (Q1-1.5*IQR, Q3+1.5*IQR) filtrelenir; yakıt yükü ve hamur farklarının temiz turları yanlışlıkla elemesini önler.",
)

if driver1 == driver2:
    st.warning("⚠️ Karşılaştırma için lütfen birbirinden farklı iki sürücü (1. Sürücü ve 2. Sürücü) seçin.")

# ==============================================================================
# VERİ HAZIRLAMA VE HESAPLAMA KATMANI (SEÇİMLER DEĞİŞTİĞİNDE OTOMATİK ÇALIŞIR)
# ==============================================================================

# Kullanıcı filtrelerini (Stint / Compound) uygula
raw_d1 = apply_user_lap_filters(raw_d1_all, selected_stint, selected_compound)
raw_d2 = apply_user_lap_filters(raw_d2_all, selected_stint, selected_compound)
raw_ref = apply_user_lap_filters(raw_ref_all, selected_stint, selected_compound)

# Temiz turları ve filtreleme raporlarını hesapla
clean_d1, filter_rep_d1 = filter_clean_laps(raw_d1, by_stint=iqr_by_stint, return_report=True)
clean_d2, filter_rep_d2 = filter_clean_laps(raw_d2, by_stint=iqr_by_stint, return_report=True)
clean_ref, filter_rep_ref = filter_clean_laps(raw_ref, by_stint=iqr_by_stint, return_report=True)

# Tüm yarışın también stint analizleri için (tüm turlar üzerinden) temiz turlar
clean_d1_full, _ = filter_clean_laps(raw_d1_all, by_stint=iqr_by_stint, return_report=True)
clean_d2_full, _ = filter_clean_laps(raw_d2_all, by_stint=iqr_by_stint, return_report=True)
clean_ref_full, _ = filter_clean_laps(raw_ref_all, by_stint=iqr_by_stint, return_report=True)

# Seansın en hızlı tur süresi ve referans sürücü medyanı
all_session_lap_sec = to_seconds_series(session_obj.laps["LapTime"]).dropna()
fastest_lap_seconds = float(all_session_lap_sec.min()) if not all_session_lap_sec.empty else np.nan
ref_median_seconds = float(to_seconds_series(clean_ref["LapTime"]).dropna().median()) if not clean_ref.empty else np.nan

# 1. Tur Temposu Metrikleri
pace_det_d1 = calculate_pace_score(clean_d1, fastest_lap_seconds, ref_median_seconds=ref_median_seconds, return_details=True)
pace_det_d2 = calculate_pace_score(clean_d2, fastest_lap_seconds, ref_median_seconds=ref_median_seconds, return_details=True)

# 2. Tur Tutarlılığı Metrikleri
cons_det_d1 = calculate_consistency_score(clean_d1, by_stint=iqr_by_stint, return_details=True)
cons_det_d2 = calculate_consistency_score(clean_d2, by_stint=iqr_by_stint, return_details=True)

# 3. İstatistiksel Karşılaştırma (Welch t-Testi)
ttest_det = compare_drivers_ttest(clean_d1, clean_d2, return_details=True)

# 4. Stint Analizi (Seçili filtreye göre ve tüm yarışa göre)
stints_ref_df = analyze_driver_stints(raw_ref, clean_ref, driver_name=ref_driver)
stints_d1_df = analyze_driver_stints(raw_d1, clean_d1, driver_name=driver1, ref_stints_df=stints_ref_df)
stints_d2_df = analyze_driver_stints(raw_d2, clean_d2, driver_name=driver2, ref_stints_df=stints_ref_df)
matching_stints_df = compare_matching_stints(stints_d1_df, stints_d2_df, driver1, driver2)

# 5. Sürücü Fark Analizi (Tur Bazlı)
lap_deltas_df = calculate_lap_by_lap_deltas(raw_d1, raw_d2, clean_d1, clean_d2, driver1, driver2)

# 6. Sektör ve Teorik Optimal Tur Analizi
sector_comp_df = build_sector_comparison_table(clean_d1, clean_d2, clean_ref, driver1, driver2, ref_driver)
opt_det_d1 = calculate_theoretical_optimal(clean_d1, return_details=True)
opt_det_d2 = calculate_theoretical_optimal(clean_d2, return_details=True)

# 7. Seans Koşulları Özeti (SC, VSC, Kırmızı Bayrak, Hava Durumu)
session_cond = summarize_session_conditions(session_obj)

# ==============================================================================
# BÖLÜM A: GENEL PERFORMANS KARTLARI
# ==============================================================================

st.markdown("---")
header_with_info(
    f"📌 A. Genel Performans Kartları ({selected_year} {selected_event} — {SESSION_TYPE_OPTIONS.get(selected_session_type, selected_session_type)})",
    "Her iki sürücü için temiz tur filtrelemesi sonrası hesaplanan temel performans göstergeleri. "
    "Tüm kartlarda kullanılan geçerli örneklem sayısı (n) açıkça belirtilmiştir.",
    level="h2",
)

if selected_stint != "Tümü" or selected_compound != "Tümü":
    st.info(
        f"🔎 **Aktif Filtre:** Stint = `{selected_stint}` | Lastik Hamuru = `{selected_compound}` "
        f"(Tüm yarışı görmek için yan menüden 'Tümü' seçebilirsiniz)."
    )

col_card_d1, col_card_d2 = st.columns(2)

for col_ui, drv_code, p_det, c_det, f_rep, st_df, accent in (
    (col_card_d1, driver1, pace_det_d1, cons_det_d1, filter_rep_d1, stints_d1_df, DRIVER_PALETTE[0]),
    (col_card_d2, driver2, pace_det_d2, cons_det_d2, filter_rep_d2, stints_d2_df, DRIVER_PALETTE[1]),
):
    with col_ui:
        st.markdown(
            f"<div style='border-left: 5px solid {accent}; padding-left: 12px; margin-bottom: 8px;'>"
            f"<h3 style='margin:0;'>🏎️ {driver_label_map.get(drv_code, drv_code)}</h3>"
            f"<small>Geçerli Temiz Tur: <b>n = {f_rep['clean_laps_count']}</b> / Toplam {f_rep['total_raw_laps']} Tur "
            f"({f_rep['total_raw_laps'] - f_rep['clean_laps_count']} tur filtrelendi)</small></div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3 = st.columns(3)
        pace_f_val = p_det.get("pace_score_vs_fastest_pct", np.nan)
        pace_r_val = p_det.get("pace_score_vs_ref_pct", np.nan)
        m1.metric(
            "Tur Temposu (En Hızlı Tura Göre)",
            f"%{pace_f_val:.3f}" if pd.notna(pace_f_val) else "-",
            delta=f"{p_det.get('delta_vs_fastest_sec', 0.0):+.3f} sn" if pd.notna(p_det.get("delta_vs_fastest_sec")) else None,
            delta_color="inverse",
            help=f"Seansın En Hızlı Turuna ({format_laptime_str(fastest_lap_seconds)}) göre yüzde Medyan Tur Temposu farkı (n={p_det.get('sample_size', 0)}).",
        )
        m2.metric(
            "Medyan Tur Süresi",
            format_laptime_str(p_det.get("median_lap_sec")),
            delta=f"Referans Sürücü ({ref_driver}): {p_det.get('delta_vs_ref_sec', 0.0):+.3f} sn (%{pace_r_val:+.2f})"
            if pd.notna(p_det.get("delta_vs_ref_sec"))
            else None,
            delta_color="inverse",
            help=f"Ortalama Tur Süresi: {format_laptime_str(p_det.get('mean_lap_sec'))} | n={p_det.get('sample_size', 0)}",
        )
        m3.metric(
            "En Hızlı Tur",
            format_laptime_str(p_det.get("best_lap_sec")),
            help=f"Teorik Optimal Tur: {format_laptime_str(opt_det_d1['theoretical_optimal_sec'] if drv_code == driver1 else opt_det_d2['theoretical_optimal_sec'])}",
        )

        m4, m5, m6 = st.columns(3)
        cv_val = c_det.get("cv_pct", np.nan)
        std_val = c_det.get("std_sec", np.nan)
        stint_cv_val = c_det.get("stint_weighted_cv_pct", np.nan)
        m4.metric(
            "Tur Tutarlılığı — Değişim Katsayısı (CV)",
            f"%{cv_val:.3f}" if pd.notna(cv_val) else "-",
            delta=f"Stint İçi CV (%): %{stint_cv_val:.3f}" if pd.notna(stint_cv_val) else None,
            delta_color="off",
            help=f"Standart Sapma: {std_val:.3f} sn | Çeyrekler Açıklığı (IQR): {c_det.get('iqr_sec', np.nan):.3f} sn",
        )
        m5.metric(
            "Geçerli Tur Sayısı (n)",
            f"{f_rep['clean_laps_count']} / {f_rep['total_raw_laps']}",
            help="Analize dahil edilen temiz tur sayısı / toplam atılan tur sayısı.",
        )
        m6.metric(
            "Stint Sayısı",
            f"{len(st_df)}",
            help="Filtre kapsamındaki toplam Stint sayısı.",
        )

# Ayrıntılı Genel Özet Tablosu (CV, Standart Sapma, Medyan, Ortalama ve Referans Sürücü Tur Temposu ayrı ayrı)
with st.expander("📋 Detaylı Genel Performans Metrikleri Tablosu (Ortalama, Medyan, Standart Sapma, Değişim Katsayısı (CV), Tur Temposu)", expanded=False):
    summary_table_df = pd.DataFrame(
        [
            {
                "Sürücü": driver1,
                "Toplam Tur": filter_rep_d1["total_raw_laps"],
                "Geçerli Temiz Tur (n)": filter_rep_d1["clean_laps_count"],
                "Filtrelenen Tur": filter_rep_d1["total_raw_laps"] - filter_rep_d1["clean_laps_count"],
                "Stint Sayısı": len(stints_d1_df),
                "Ortalama Tur Süresi (sn)": pace_det_d1["mean_lap_sec"],
                "Medyan Tur Süresi (sn)": pace_det_d1["median_lap_sec"],
                "En Hızlı Tur (sn)": pace_det_d1["best_lap_sec"],
                "Standart Sapma (sn)": cons_det_d1["std_sec"],
                "Çeyrekler Açıklığı — IQR (sn)": cons_det_d1["iqr_sec"],
                "Tur Tutarlılığı — CV (%)": cons_det_d1["cv_pct"],
                "Stint Ağırlıklı CV (%)": cons_det_d1["stint_weighted_cv_pct"],
                "Tur Temposu (En Hızlı Tura Göre) (%)": pace_det_d1["pace_score_vs_fastest_pct"],
                f"Tur Temposu (Referans Sürücü: {ref_driver}) (%)": pace_det_d1["pace_score_vs_ref_pct"],
                f"Medyan Fark (Referans Sürücü: {ref_driver}) (sn)": pace_det_d1["delta_vs_ref_sec"],
            },
            {
                "Sürücü": driver2,
                "Toplam Tur": filter_rep_d2["total_raw_laps"],
                "Geçerli Temiz Tur (n)": filter_rep_d2["clean_laps_count"],
                "Filtrelenen Tur": filter_rep_d2["total_raw_laps"] - filter_rep_d2["clean_laps_count"],
                "Stint Sayısı": len(stints_d2_df),
                "Ortalama Tur Süresi (sn)": pace_det_d2["mean_lap_sec"],
                "Medyan Tur Süresi (sn)": pace_det_d2["median_lap_sec"],
                "En Hızlı Tur (sn)": pace_det_d2["best_lap_sec"],
                "Standart Sapma (sn)": cons_det_d2["std_sec"],
                "Çeyrekler Açıklığı — IQR (sn)": cons_det_d2["iqr_sec"],
                "Tur Tutarlılığı — CV (%)": cons_det_d2["cv_pct"],
                "Stint Ağırlıklı CV (%)": cons_det_d2["stint_weighted_cv_pct"],
                "Tur Temposu (En Hızlı Tura Göre) (%)": pace_det_d2["pace_score_vs_fastest_pct"],
                f"Tur Temposu (Referans Sürücü: {ref_driver}) (%)": pace_det_d2["pace_score_vs_ref_pct"],
                f"Medyan Fark (Referans Sürücü: {ref_driver}) (sn)": pace_det_d2["delta_vs_ref_sec"],
            },
        ]
    )
    st.dataframe(summary_table_df.style.format(precision=3), use_container_width=True)

# ==============================================================================
# ANA SEKMELER (B, C, D, E, F BÖLÜMLERİ)
# ==============================================================================

st.markdown("---")
tab_stint, tab_diff, tab_telemetry, tab_sector, tab_quality = st.tabs(
    [
        "🛞 B. Stint & Lastik Hamuru Analizi",
        "⚖️ C. Sürücü Fark Analizi",
        "📡 D. Telemetri Analizi",
        "⏱️ E. Sektör & Teorik Optimal Tur",
        "🛡️ F. Veri Doğrulama & Güvenilirlik",
    ]
)

# ------------------------------------------------------------------------------
# BÖLÜM B: STINT ANALİZİ
# ------------------------------------------------------------------------------
with tab_stint:
    header_with_info(
        "🛞 Stint Bazlı Performans ve Lastik Hamuru Kullanımı Analizi",
        "Tur Sürelerinin tur numarasına göre gelişimi, Stint ayrımları, Lastik Hamurları, "
        "pit-stop noktaları ve her Stint için Ortalama/Medyan çizgileri.",
        level="h2",
    )

    st.warning(f"📌 **{STINT_DEGRADATION_METHODOLOGY_WARNING}**")

    if session_cond["sc_laps"] or session_cond["vsc_laps"] or session_cond["red_flag_laps"]:
        cond_parts = []
        if session_cond["sc_laps"]:
            cond_parts.append(f"**Güvenlik Aracı (SC) Turları:** {session_cond['sc_laps']}")
        if session_cond["vsc_laps"]:
            cond_parts.append(f"**Sanal Güvenlik Aracı (VSC) Turları:** {session_cond['vsc_laps']}")
        if session_cond["red_flag_laps"]:
            cond_parts.append(f"**Kırmızı Bayrak Turları:** {session_cond['red_flag_laps']}")
        st.info("🚨 **Seans İçi Nötralizasyon Tespiti:** " + " | ".join(cond_parts))

    # Stint Tur Süresi / Tur Numarası Plotly Grafiği
    fig_stint = go.Figure()

    stint_dash_styles = ["solid", "dash", "dot", "dashdot", "longdash"]

    for drv_idx, (drv_code, raw_df_drv, clean_df_drv, st_df_drv, base_color) in enumerate(
        [
            (driver1, raw_d1, clean_d1, stints_d1_df, DRIVER_PALETTE[0]),
            (driver2, raw_d2, clean_d2, stints_d2_df, DRIVER_PALETTE[1]),
        ]
    ):
        if clean_df_drv.empty or "LapNumber" not in clean_df_drv.columns:
            continue

        c_plot = clean_df_drv.copy()
        c_plot["LapTime_Sec"] = to_seconds_series(c_plot["LapTime"])
        c_plot = c_plot.dropna(subset=["LapNumber", "LapTime_Sec"]).sort_values("LapNumber")

        if "Stint" not in c_plot.columns:
            c_plot["Stint"] = 1

        for st_no, grp in c_plot.groupby("Stint"):
            st_int = int(st_no)
            comp = (
                str(grp["Compound"].dropna().iloc[0]).upper()
                if "Compound" in grp.columns and grp["Compound"].notna().any()
                else "UNKNOWN"
            )
            comp_style = COMPOUND_STYLES.get(comp, COMPOUND_STYLES["UNKNOWN"])
            dash_style = stint_dash_styles[(st_int - 1) % len(stint_dash_styles)]

            tyre_life_vals = grp["TyreLife"].fillna(-1).astype(int) if "TyreLife" in grp.columns else pd.Series(-1, index=grp.index)

            fig_stint.add_trace(
                go.Scatter(
                    x=grp["LapNumber"],
                    y=grp["LapTime_Sec"],
                    mode="lines+markers",
                    name=f"{drv_code} — Stint {st_int} ({comp})",
                    legendgroup=f"{drv_code}_S{st_int}",
                    line=dict(color=base_color, width=2.2, dash=dash_style),
                    marker=dict(
                        color=comp_style["color"],
                        symbol=comp_style["symbol"],
                        size=9,
                        line=dict(color=base_color, width=1.8),
                    ),
                    customdata=np.stack([tyre_life_vals.values, np.full(len(grp), comp)], axis=-1),
                    hovertemplate=(
                        f"<b>{drv_code} (Stint {st_int})</b><br>"
                        "Tur: %{x}<br>"
                        "Tur Süresi: %{y:.3f} sn<br>"
                        "Lastik Hamuru: %{customdata[1]} (Lastik Yaşı: %{customdata[0]})<extra></extra>"
                    ),
                )
            )

            # Stint Ortalama ve Medyan Yatay Çizgileri
            if len(grp) >= 2:
                st_min_lap = float(grp["LapNumber"].min())
                st_max_lap = float(grp["LapNumber"].max())
                st_med_val = float(grp["LapTime_Sec"].median())
                st_mean_val = float(grp["LapTime_Sec"].mean())

                fig_stint.add_trace(
                    go.Scatter(
                        x=[st_min_lap, st_max_lap],
                        y=[st_med_val, st_med_val],
                        mode="lines",
                        name=f"{drv_code} S{st_int} Medyan ({st_med_val:.2f} sn)",
                        legendgroup=f"{drv_code}_S{st_int}",
                        showlegend=False,
                        line=dict(color=base_color, width=1.3, dash="dot"),
                        hovertemplate=f"{drv_code} Stint {st_int} Medyan: {st_med_val:.3f} sn<extra></extra>",
                    )
                )
                fig_stint.add_trace(
                    go.Scatter(
                        x=[st_min_lap, st_max_lap],
                        y=[st_mean_val, st_mean_val],
                        mode="lines",
                        name=f"{drv_code} S{st_int} Ortalama ({st_mean_val:.2f} sn)",
                        legendgroup=f"{drv_code}_S{st_int}",
                        showlegend=False,
                        line=dict(color=base_color, width=1.0, dash="dash"),
                        hovertemplate=f"{drv_code} Stint {st_int} Ortalama: {st_mean_val:.3f} sn<extra></extra>",
                    )
                )

        # Pit-stop noktalarını (PitInTime olan turları) işaretle
        if "PitInTime" in raw_df_drv.columns and "LapNumber" in raw_df_drv.columns:
            pit_in_laps = raw_df_drv[raw_df_drv["PitInTime"].notna()]["LapNumber"].dropna().astype(int).unique()
            for p_lap in pit_in_laps:
                fig_stint.add_vline(
                    x=float(p_lap),
                    line_width=1.4,
                    line_dash="dashdot",
                    line_color=base_color,
                    annotation_text=f"PIT ({drv_code} T{p_lap})",
                    annotation_position="top left" if drv_idx == 0 else "bottom right",
                )

    fig_stint.update_layout(
        title="Tur Süresi / Tur Numarası Gelişimi (Stint, Lastik Hamuru, Pit-Stop ve Medyan/Ortalama Çizgileri)",
        xaxis_title="Tur Numarası (LapNumber)",
        yaxis_title="Temiz Tur Süresi (Saniye)",
        hovermode="x unified",
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_stint, use_container_width=True)
    st.caption(
        "🎨 **Görsel Rehber:** Çizgi rengi sürücüyü (Mavi: 1. Sürücü, Turuncu: 2. Sürücü), nokta içi rengi ve şekli "
        "Lastik Hamurunu (🔴 Daire: Soft, 🟡 Elmas: Medium, ⚪ Kare: Hard, 🟢 Üçgen: Inter, 🔵 Artı: Wet), "
        "noktalı/kesikli yatay çizgiler her Stintin Medyan ve Ortalamasını, dikey çizgiler ise Pit-Stop turlarını gösterir."
    )

    # Stint Metrikleri Karşılaştırma Tablosu
    st.subheader(f"📊 Stint Metrikleri Tablosu (Referans Sürücü: {ref_driver})")
    combined_stints_df = pd.concat([stints_d1_df, stints_d2_df], ignore_index=True)
    if combined_stints_df.empty:
        st.warning("Seçilen filtreler için Stint verisi bulunamadı.")
    else:
        display_stint_df = combined_stints_df.rename(
            columns={
                "Driver": "Sürücü",
                "Stint": "Stint",
                "Compound": "Lastik Hamuru",
                "StartLap": "Başlangıç Turu",
                "EndLap": "Bitiş Turu",
                "ValidLapCount": "Geçerli Tur (n)",
                "TotalLaps": "Toplam Tur",
                "TyreLifeStart": "Başlangıç Lastik Yaşı",
                "TyreLifeEnd": "Bitiş Lastik Yaşı",
                "MeanLapTime_Sec": "Ortalama Tur Süresi (sn)",
                "MedianLapTime_Sec": "Medyan Tur Süresi (sn)",
                "BestLapTime_Sec": "En Hızlı Tur (sn)",
                "WorstLapTime_Sec": "En Yavaş Tur (sn)",
                "StdLapTime_Sec": "Standart Sapma (sn)",
                "CV_Pct": "Değişim Katsayısı — CV (%)",
                "First3Mean_Sec": "İlk 3 Tur Ortalaması (sn)",
                "Last3Mean_Sec": "Son 3 Tur Ortalaması (sn)",
                "Last3_Minus_First3_Sec": "Performans Kaybı: Son 3 - İlk 3 (sn)",
                "LinearTrend_SecPerLap": "Eğim (sn/tur)",
                "DiffToRefMedian_Sec": f"Referans Sürücü ({ref_driver}) Medyan Farkı (sn)",
                "SC_VSC_LapCount": "SC/VSC Turu",
                "SampleNote": "Örneklem & Durum Notu",
            }
        )
        st.dataframe(display_stint_df.style.format(precision=3), use_container_width=True)

        # Yetersiz örneklem uyarısı
        low_sample_stints = combined_stints_df[combined_stints_df["ValidLapCount"] < 3]
        if not low_sample_stints.empty:
            st.warning(
                "⚠️ **Stint Örneklem Uyarısı:** Bazı stintlerde geçerli temiz tur sayısı 3'ten azdır "
                f"({len(low_sample_stints)} stint). Bu stintlerde İlk 3 / Son 3 Tur Ortalaması hesaplanmamıştır."
            )

    # Aynı Lastik Hamuru ve Benzer Lastik Yaşındaki Stintlerin Karşılaştırması
    st.subheader("🔄 Aynı Lastik Hamuru ve Benzer Lastik Yaşındaki Stintlerin Karşılaştırması")
    if matching_stints_df.empty:
        st.info(
            "Bu filtre kapsamında iki sürücünün aynı Lastik Hamuru ve benzer başlangıç Lastik Yaşına (±6 tur) "
            "sahip eşleşen Stinti bulunamadı."
        )
    else:
        st.dataframe(matching_stints_df.style.format(precision=3), use_container_width=True)

    # Stint Bazlı Tur Tutarlılığı Karşılaştırması
    stint_cons_col_map = {
        "Compound": "Lastik Hamuru",
        "SampleSize_n": "Geçerli Tur (n)",
        "Mean_Sec": "Ortalama Tur Süresi (sn)",
        "Median_Sec": "Medyan Tur Süresi (sn)",
        "Std_Sec": "Standart Sapma (sn)",
        "IQR_Sec": "Çeyrekler Açıklığı — IQR (sn)",
        "CV_Pct": "Değişim Katsayısı — CV (%)",
    }
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        st.markdown(f"**{driver1} — Stint Bazında Tur Tutarlılığı (Değişim Katsayısı — CV (%), Standart Sapma, Medyan, Ortalama)**")
        if not cons_det_d1["stint_breakdown"].empty:
            st.dataframe(cons_det_d1["stint_breakdown"].rename(columns=stint_cons_col_map).style.format(precision=3), use_container_width=True)
    with col_sc2:
        st.markdown(f"**{driver2} — Stint Bazında Tur Tutarlılığı (Değişim Katsayısı — CV (%), Standart Sapma, Medyan, Ortalama)**")
        if not cons_det_d2["stint_breakdown"].empty:
            st.dataframe(cons_det_d2["stint_breakdown"].rename(columns=stint_cons_col_map).style.format(precision=3), use_container_width=True)


# ------------------------------------------------------------------------------
# BÖLÜM C: SÜRÜCÜ FARK ANALİZİ
# ------------------------------------------------------------------------------
with tab_diff:
    header_with_info(
        f"⚖️ Sürücü Fark Analizi ({driver1} vs {driver2})",
        f"Tur bazlı, Stint bazlı ve Sektör Zaman Farkları. Tüm grafiklerde Fark = {driver1} - {driver2} olarak tanımlanmıştır:\n"
        f"- **Negatif Fark (< 0):** {driver1} daha hızlıdır (daha kısa sürede tamamlamıştır).\n"
        f"- **Pozitif Fark (> 0):** {driver2} daha hızlıdır ({driver1} zaman kaybetmiştir).",
        level="h2",
    )

    if lap_deltas_df.empty:
        st.warning("İki sürücü arasında ortak tur numarası bulunamadı.")
    else:
        # 1. Tur Bazlı Zaman Farkı ve Kümülatif Fark Grafiği
        clean_only_toggle = st.checkbox(
            "Yalnızca her iki sürücünün de temiz olduğu turları göster",
            value=True,
        )
        plot_deltas = lap_deltas_df[lap_deltas_df["Both_Clean"]].copy() if clean_only_toggle else lap_deltas_df.copy()

        if plot_deltas.empty:
            st.warning("Her iki sürücünün aynı anda temiz tur attığı ortak tur bulunamadı. Kutucuğu kaldırarak tüm turları görebilirsiniz.")
        else:
            fig_delta = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.10,
                subplot_titles=(
                    f"Tur Bazlı Tur Süresi Farkı: Δt = {driver1} - {driver2} (sn)  [Negatif (<0): {driver1} Hızlı | Pozitif (>0): {driver2} Hızlı]",
                    f"Kümülatif Tur Süresi Farkı (sn)  [Aşağı Eğim: {driver1} Avantaj Kazanıyor | Yukarı Eğim: {driver2} Avantaj Kazanıyor]",
                ),
            )

            bar_colors = np.where(plot_deltas["Delta_Sec"] < 0, DRIVER_PALETTE[0], DRIVER_PALETTE[1])
            fig_delta.add_trace(
                go.Bar(
                    x=plot_deltas["LapNumber"],
                    y=plot_deltas["Delta_Sec"],
                    name=f"Tur Süresi Farkı ({driver1} - {driver2})",
                    marker_color=bar_colors,
                    customdata=plot_deltas["Faster_Driver"],
                    hovertemplate=(
                        "Tur: %{x}<br>"
                        f"Δt ({driver1} - {driver2}): %{{y:+.3f}} sn<br>"
                        "Daha Hızlı Sürücü: <b>%{customdata}</b><extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            fig_delta.add_hline(y=0.0, line_dash="solid", line_color="#475569", line_width=1.5, row=1, col=1)

            # Grafik üzerinde pozitif ve negatif fark açıklaması
            fig_delta.add_annotation(
                xref="paper",
                yref="y1",
                x=0.01,
                y=float(plot_deltas["Delta_Sec"].min()) * 0.85 if plot_deltas["Delta_Sec"].min() < 0 else -0.2,
                text=f"⬇️ Negatif (< 0 sn): {driver1} daha hızlı",
                showarrow=False,
                bgcolor="rgba(2, 132, 199, 0.15)",
                bordercolor=DRIVER_PALETTE[0],
                font=dict(size=11),
            )
            fig_delta.add_annotation(
                xref="paper",
                yref="y1",
                x=0.01,
                y=float(plot_deltas["Delta_Sec"].max()) * 0.85 if plot_deltas["Delta_Sec"].max() > 0 else 0.2,
                text=f"⬆️ Pozitif (> 0 sn): {driver2} daha hızlı",
                showarrow=False,
                bgcolor="rgba(234, 88, 12, 0.15)",
                bordercolor=DRIVER_PALETTE[1],
                font=dict(size=11),
            )

            cum_col = "Cumulative_Delta_Clean_Sec" if clean_only_toggle else "Cumulative_Delta_All_Sec"
            fig_delta.add_trace(
                go.Scatter(
                    x=plot_deltas["LapNumber"],
                    y=plot_deltas[cum_col],
                    mode="lines+markers",
                    name="Kümülatif Fark (sn)",
                    line=dict(color="#8b5cf6", width=2.5),
                    hovertemplate="Tur: %{x}<br>Kümülatif Fark: %{y:+.3f} sn<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig_delta.add_hline(y=0.0, line_dash="dash", line_color="#475569", row=2, col=1)

            fig_delta.update_yaxes(title_text="Δ Tur Süresi (sn)", row=1, col=1)
            fig_delta.update_yaxes(title_text="Kümülatif Δ (sn)", row=2, col=1)
            fig_delta.update_xaxes(title_text="Tur Numarası (LapNumber)", row=2, col=1)
            fig_delta.update_layout(height=620, hovermode="x unified", showlegend=False)

            st.plotly_chart(fig_delta, use_container_width=True)

        # 2. Stint Bazlı Ortalama ve Medyan Fark Grafiği
        col_diff_stint, col_diff_sec = st.columns(2)
        with col_diff_stint:
            st.subheader("📊 Stint Bazlı Ortalama & Medyan Fark")
            if not stints_d1_df.empty and not stints_d2_df.empty:
                st_merged = pd.merge(
                    stints_d1_df[["Stint", "Compound", "MeanLapTime_Sec", "MedianLapTime_Sec", "ValidLapCount"]],
                    stints_d2_df[["Stint", "Compound", "MeanLapTime_Sec", "MedianLapTime_Sec", "ValidLapCount"]],
                    on="Stint",
                    suffixes=(f"_{driver1}", f"_{driver2}"),
                )
                if not st_merged.empty:
                    st_merged["Mean_Diff_Sec"] = (
                        st_merged[f"MeanLapTime_Sec_{driver1}"] - st_merged[f"MeanLapTime_Sec_{driver2}"]
                    )
                    st_merged["Median_Diff_Sec"] = (
                        st_merged[f"MedianLapTime_Sec_{driver1}"] - st_merged[f"MedianLapTime_Sec_{driver2}"]
                    )
                    st_merged["StintLabel"] = st_merged.apply(
                        lambda r: f"Stint {int(r['Stint'])} ({r[f'Compound_{driver1}']}/{r[f'Compound_{driver2}']})",
                        axis=1,
                    )
                    fig_st_diff = go.Figure()
                    fig_st_diff.add_trace(
                        go.Bar(
                            x=st_merged["StintLabel"],
                            y=st_merged["Mean_Diff_Sec"],
                            name="Ortalama Fark (sn)",
                            marker_color="#0284c7",
                        )
                    )
                    fig_st_diff.add_trace(
                        go.Bar(
                            x=st_merged["StintLabel"],
                            y=st_merged["Median_Diff_Sec"],
                            name="Medyan Fark (sn)",
                            marker_color="#f59e0b",
                        )
                    )
                    fig_st_diff.add_hline(y=0.0, line_dash="solid", line_color="#475569")
                    fig_st_diff.update_layout(
                        title=f"Stint Farkı: {driver1} - {driver2} (<0: {driver1} Hızlı, >0: {driver2} Hızlı)",
                        yaxis_title="Fark (Saniye)",
                        barmode="group",
                        height=380,
                    )
                    st.plotly_chart(fig_st_diff, use_container_width=True)
                else:
                    st.info("Ortak Stint numarası bulunamadı.")

        with col_diff_sec:
            st.subheader("⏱️ Sektör Zaman Farkı (Saniye)")
            if not sector_comp_df.empty:
                mean_col = f"Ort. Fark ({driver1} - {driver2}) (sn)"
                med_col = f"Medyan Fark ({driver1} - {driver2}) (sn)"
                best_col = f"En İyi Fark ({driver1} - {driver2}) (sn)"
                fig_sec_diff = go.Figure()
                fig_sec_diff.add_trace(
                    go.Bar(x=sector_comp_df["Sektör"], y=sector_comp_df[mean_col], name="Ortalama Sektör Zaman Farkı (sn)")
                )
                fig_sec_diff.add_trace(
                    go.Bar(x=sector_comp_df["Sektör"], y=sector_comp_df[med_col], name="Medyan Sektör Zaman Farkı (sn)")
                )
                fig_sec_diff.add_trace(
                    go.Bar(x=sector_comp_df["Sektör"], y=sector_comp_df[best_col], name="En Hızlı Sektör Zaman Farkı (sn)")
                )
                fig_sec_diff.add_hline(y=0.0, line_dash="solid", line_color="#475569")
                fig_sec_diff.update_layout(
                    title=f"Sektör Zaman Farkları: {driver1} - {driver2} (<0: {driver1} Hızlı, >0: {driver2} Hızlı)",
                    yaxis_title="Sektör Zaman Farkı (Saniye)",
                    barmode="group",
                    height=380,
                )
                st.plotly_chart(fig_sec_diff, use_container_width=True)

        # 3. Welch t-Testi Yardımcı Analiz Kartı
        st.markdown("---")
        header_with_info(
            "🔬 Yardımcı İstatistiksel Analiz: Welch t-Testi",
            TTEST_METHODOLOGY_WARNING,
            level="h3",
        )
        p_val = ttest_det.get("p_value", np.nan)
        t_stat = ttest_det.get("t_stat", np.nan)
        if pd.isna(p_val):
            st.warning(f"Welch t-Testi hesaplanamadı: {ttest_det.get('sample_size_warning', 'Yetersiz veri.')}")
        else:
            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.metric("Welch t-Testi İstatistiği (t)", f"{t_stat:.4f}", help=f"n1={ttest_det['n_driver1']}, n2={ttest_det['n_driver2']}")
            col_t2.metric("p-Değeri", f"{p_val:.5f}", delta="p < 0.05 (İstatistiksel Fark)" if p_val < 0.05 else "p >= 0.05 (Anlamlı Fark Yok)")
            col_t3.metric(
                f"Ortalama Fark ({driver1} - {driver2})",
                f"{ttest_det['mean_diff_sec']:+.3f} sn",
                delta=f"Medyan Fark: {ttest_det['median_diff_sec']:+.3f} sn",
                delta_color="off",
            )
            if ttest_det.get("sample_size_warning"):
                st.warning(f"⚠️ {ttest_det['sample_size_warning']}")
            st.caption(f"ℹ️ {ttest_det['methodology_warning']}")


# ------------------------------------------------------------------------------
# BÖLÜM D: TELEMETRİ ANALİZİ
# ------------------------------------------------------------------------------
with tab_telemetry:
    header_with_info(
        f"📡 Ortak Mesafe Eksenli Telemetri ve Fren Aktivitesi Analizi ({selected_lap_mode})",
        "Tüm telemetri kanalları ortak bir Distance (metre) eksenine hizalanmıştır. "
        "Sürekli kanallar (Speed, Throttle, RPM, BrakeActivity, SmoothedBrakeActivity) doğrusal interpolasyonla; "
        "kesikli kanallar (BrakeRaw, nGear, DRS) ise fiziksel anlamı korumak amacıyla kesikli basamak "
        "yöntemiyle hizalanmıştır.",
        level="h2",
    )

    st.info(f"ℹ️ **{BRAKE_ACTIVITY_DISCLAIMER}** (Aktif Mesafe Penceresi: **{selected_brake_window} m**)")

    raw_tel_1, lap_no_1, lap_sec_1, err_tel_1 = extract_selected_lap_telemetry(session_obj, raw_d1_all, selected_lap_mode)
    raw_tel_2, lap_no_2, lap_sec_2, err_tel_2 = extract_selected_lap_telemetry(session_obj, raw_d2_all, selected_lap_mode)

    if err_tel_1 or err_tel_2 or raw_tel_1.empty or raw_tel_2.empty:
        st.warning(
            f"Telemetri verisi yüklenemedi. "
            f"{driver1}: {err_tel_1 or 'Tamam'} | {driver2}: {err_tel_2 or 'Tamam'}"
        )
    else:
        tel_comp = align_and_compare_telemetry(
            raw_tel_d1=raw_tel_1,
            raw_tel_d2=raw_tel_2,
            driver1_name=driver1,
            driver2_name=driver2,
            window_m=float(selected_brake_window),
            distance_step_m=2.0,
        )

        aligned_df = tel_comp["aligned_df"]
        t_sum = tel_comp["summary_metrics"]

        # Telemetri Üst Özet Kartları
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric(
            f"{driver1} Seçilen Tur",
            f"Tur {lap_no_1} ({format_laptime_str(lap_sec_1)})",
            help=f"Ham telemetri örnek sayısı: n = {t_sum.get('d1_raw_samples_n', 0)}",
        )
        tc2.metric(
            f"{driver2} Seçilen Tur",
            f"Tur {lap_no_2} ({format_laptime_str(lap_sec_2)})",
            help=f"Ham telemetri örnek sayısı: n = {t_sum.get('d2_raw_samples_n', 0)}",
        )
        tc3.metric(
            "Toplam Fren Süresi (BrakeDuration)",
            f"{driver1}: {t_sum.get('d1_brake_duration_sec', 0):.2f} sn | {driver2}: {t_sum.get('d2_brake_duration_sec', 0):.2f} sn",
            help="Tur boyunca frenin aktif (BrakeRaw=1) kaldığı toplam süre.",
        )
        tc4.metric(
            "Toplam Fren Mesafesi (BrakingDistance)",
            f"{driver1}: {t_sum.get('d1_braking_distance_m', 0):.0f} m | {driver2}: {t_sum.get('d2_braking_distance_m', 0):.0f} m",
            help=f"Fren olay sayısı: {driver1}={t_sum.get('d1_brake_events_count', 0)}, {driver2}={t_sum.get('d2_brake_events_count', 0)}",
        )

        # Kanal Açma / Kapama Seçici
        CHANNEL_METADATA: Dict[str, Dict[str, Any]] = {
            "Speed": {"title": "Hız — Speed (km/h)", "yaxis": "km/h", "shape": "linear", "range": None},
            "SpeedDelta": {"title": f"Hız Farkı — ΔSpeed ({driver1} - {driver2}) (km/h)", "yaxis": "Δ km/h", "shape": "linear", "range": None},
            "Throttle": {"title": "Gaz Pedalı — Throttle (%)", "yaxis": "Gaz Pedalı (%)", "shape": "linear", "range": [-5, 105]},
            "SmoothedBrakeActivity": {
                "title": f"Yumuşatılmış Fren Aktivitesi — SmoothedBrakeActivity (%) [Pencere: {selected_brake_window} m, Basınç Değildir]",
                "yaxis": "Fren Aktivitesi (%)",
                "shape": "linear",
                "range": [-5, 105],
            },
            "BrakeActivity": {
                "title": f"Fren Aktivitesi — BrakeActivity (%) [Pencere: {selected_brake_window} m, Basınç Değildir]",
                "yaxis": "Fren Aktivitesi (%)",
                "shape": "linear",
                "range": [-5, 105],
            },
            "BrakeRaw": {
                "title": "Ham Fren Durumu — Brake (0=Kapalı, 1=Açık) [Kesikli]",
                "yaxis": "Durum (0/1)",
                "shape": "hv",
                "range": [-0.15, 1.15],
            },
            "nGear": {"title": "Vites — nGear (1-8) [Kesikli]", "yaxis": "Vites", "shape": "hv", "range": [0.5, 8.5]},
            "RPM": {"title": "Motor Devri — RPM (devir/dk)", "yaxis": "RPM", "shape": "linear", "range": None},
            "DRS": {"title": "DRS Durumu [Kesikli]", "yaxis": "DRS Kodu", "shape": "hv", "range": None},
        }

        default_selected_channels = [
            "Speed",
            "SpeedDelta",
            "Throttle",
            "SmoothedBrakeActivity",
            "BrakeRaw",
            "nGear",
            "RPM",
            "DRS",
        ]
        active_channels = st.multiselect(
            "🎛️ Görüntülenecek Telemetri Kanallarını Seçin (Aç / Kapat):",
            options=list(CHANNEL_METADATA.keys()),
            default=default_selected_channels,
            format_func=lambda k: CHANNEL_METADATA[k]["title"],
        )

        if not active_channels:
            st.info("Grafiği görüntülemek için yukarıdan en az bir telemetri kanalı seçin.")
        else:
            n_rows = len(active_channels)
            fig_tel = make_subplots(
                rows=n_rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=max(0.02, 0.18 / n_rows),
                subplot_titles=[CHANNEL_METADATA[ch]["title"] for ch in active_channels],
            )

            for r_idx, ch_key in enumerate(active_channels, start=1):
                meta = CHANNEL_METADATA[ch_key]
                show_leg = r_idx == 1

                if ch_key == "SpeedDelta":
                    fig_tel.add_trace(
                        go.Scatter(
                            x=aligned_df["Distance"],
                            y=aligned_df["SpeedDelta_kmh"],
                            mode="lines",
                            name=f"ΔSpeed ({driver1} - {driver2})",
                            line=dict(color="#8b5cf6", width=1.8),
                            showlegend=False,
                            hovertemplate=f"Mesafe: %{{x:.0f}} m<br>ΔHız ({driver1}-{driver2}): %{{y:+.1f}} km/h<extra></extra>",
                        ),
                        row=r_idx,
                        col=1,
                    )
                    fig_tel.add_hline(y=0.0, line_dash="dash", line_color="#475569", row=r_idx, col=1)
                else:
                    col_d1 = f"{ch_key}_D1"
                    col_d2 = f"{ch_key}_D2"
                    if col_d1 in aligned_df.columns:
                        fig_tel.add_trace(
                            go.Scatter(
                                x=aligned_df["Distance"],
                                y=aligned_df[col_d1],
                                mode="lines",
                                name=f"{driver1} (Tur {lap_no_1})",
                                legendgroup=driver1,
                                showlegend=show_leg,
                                line=dict(color=DRIVER_PALETTE[0], width=2.0, shape=meta["shape"]),
                            ),
                            row=r_idx,
                            col=1,
                        )
                    if col_d2 in aligned_df.columns:
                        fig_tel.add_trace(
                            go.Scatter(
                                x=aligned_df["Distance"],
                                y=aligned_df[col_d2],
                                mode="lines",
                                name=f"{driver2} (Tur {lap_no_2})",
                                legendgroup=driver2,
                                showlegend=show_leg,
                                line=dict(color=DRIVER_PALETTE[1], width=2.0, shape=meta["shape"]),
                            ),
                            row=r_idx,
                            col=1,
                        )

                fig_tel.update_yaxes(title_text=meta["yaxis"], range=meta["range"], row=r_idx, col=1)

            fig_tel.update_xaxes(title_text="Pist Mesafesi — Distance (metre)", row=n_rows, col=1)
            fig_tel.update_layout(
                height=max(420, 215 * n_rows),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_tel, use_container_width=True)

        # Sürüş Bölgeleri (Gaz Pedalı / Fren / Süzülme) ve Fren Olayları Tablosu
        brake_col_map = {
            "EventNo": "Fren Olayı No",
            "StartDistance_m": "Başlangıç Mesafesi (m)",
            "EndDistance_m": "Bitiş Mesafesi (m)",
            "BrakingDistanceRange": "Frenleme Mesafe Aralığı",
            "BrakingDistance_m": "Frenleme Mesafesi (m)",
            "BrakeDuration_sec": "Fren Süresi (sn)",
            "EntrySpeed_kmh": "Giriş Hızı (km/h)",
            "MinSpeed_kmh": "Minimum Hız (km/h)",
            "SpeedDrop_kmh": "Hız Düşüşü (km/h)",
            "MinGear": "Minimum Vites",
            "SampleCount": "Örneklem (n)",
        }
        col_zone, col_events = st.columns([0.45, 0.55])
        with col_zone:
            st.subheader("🚦 Gaz Pedalı / Fren Kullanım Bölgeleri Karşılaştırması")
            if not tel_comp["zone_comparison_df"].empty:
                st.dataframe(tel_comp["zone_comparison_df"], use_container_width=True)
            if pd.notna(t_sum.get("max_abs_speed_diff_kmh")):
                st.info(
                    f"💡 **En Büyük Hız Farkı:** **{t_sum['max_abs_speed_diff_kmh']:.1f} km/h** "
                    f"(Mesafe: **{t_sum['max_abs_speed_diff_distance_m']:.0f} m**, "
                    f"Daha Hızlı Sürücü: **{t_sum['max_abs_speed_diff_faster_driver']}**)"
                )

        with col_events:
            st.subheader("🛑 Fren Olayları (BrakeEvents & BrakingDistance)")
            ev_tab1, ev_tab2 = st.tabs([f"{driver1} Fren Bölgeleri", f"{driver2} Fren Bölgeleri"])
            with ev_tab1:
                ev1 = tel_comp["driver1_brake"]["brake_events_df"]
                if not ev1.empty:
                    st.dataframe(ev1.rename(columns=brake_col_map), use_container_width=True)
                else:
                    st.info("Fren olayı tespit edilmedi.")
            with ev_tab2:
                ev2 = tel_comp["driver2_brake"]["brake_events_df"]
                if not ev2.empty:
                    st.dataframe(ev2.rename(columns=brake_col_map), use_container_width=True)
                else:
                    st.info("Fren olayı tespit edilmedi.")


# ------------------------------------------------------------------------------
# BÖLÜM E: SEKTÖR VE TEORİK OPTİMAL TUR ANALİZİ
# ------------------------------------------------------------------------------
with tab_sector:
    header_with_info(
        f"⏱️ Sektör Performansı ve Teorik Optimal Tur Analizi (Referans Sürücü: {ref_driver})",
        "S1, S2 ve S3 sektör süreleri, saniye bazında Sektör Zaman Farkları, Referans Sürücüye göre Z-Skorları "
        "ve en iyi sektörlerin hangi turlardan geldiğini gösteren Teorik Optimal Tur tablosu.",
        level="h2",
    )

    st.subheader(f"📊 S1 / S2 / S3 Sektör Süreleri, Sektör Zaman Farkları ve Z-Skorları (Referans Sürücü: {ref_driver})")
    if not sector_comp_df.empty:
        sector_col_map = {
            f"{driver1} Ort. (sn)": f"{driver1} Ortalama (sn)",
            f"{driver2} Ort. (sn)": f"{driver2} Ortalama (sn)",
            f"{ref_driver} (Ref) Ort. (sn)": f"Referans Sürücü ({ref_driver}) Ortalama (sn)",
            f"{driver1} En İyi (sn)": f"{driver1} En Hızlı (sn)",
            f"{driver2} En İyi (sn)": f"{driver2} En Hızlı (sn)",
            f"Ort. Fark ({driver1} - {driver2}) (sn)": f"Ortalama Sektör Zaman Farkı ({driver1} - {driver2}) (sn)",
            f"Medyan Fark ({driver1} - {driver2}) (sn)": f"Medyan Sektör Zaman Farkı ({driver1} - {driver2}) (sn)",
            f"En İyi Fark ({driver1} - {driver2}) (sn)": f"En Hızlı Sektör Zaman Farkı ({driver1} - {driver2}) (sn)",
            f"{driver1} Z-Skor (Ref: {ref_driver})": f"{driver1} Z-Skoru (Referans Sürücü: {ref_driver})",
            f"{driver2} Z-Skor (Ref: {ref_driver})": f"{driver2} Z-Skoru (Referans Sürücü: {ref_driver})",
        }
        st.dataframe(sector_comp_df.rename(columns=sector_col_map).style.format(precision=3), use_container_width=True)

        col_z_chart, col_s_chart = st.columns(2)
        with col_z_chart:
            z1_col = f"{driver1} Z-Skor (Ref: {ref_driver})"
            z2_col = f"{driver2} Z-Skor (Ref: {ref_driver})"
            fig_z = go.Figure()
            fig_z.add_trace(go.Bar(x=sector_comp_df["Sektör"], y=sector_comp_df[z1_col], name=driver1, marker_color=DRIVER_PALETTE[0]))
            fig_z.add_trace(go.Bar(x=sector_comp_df["Sektör"], y=sector_comp_df[z2_col], name=driver2, marker_color=DRIVER_PALETTE[1]))
            fig_z.add_hline(y=0.0, line_dash="dash", line_color="#475569")
            fig_z.update_layout(
                title=f"Sektör Z-Skorları (Referans Sürücü: {ref_driver}) — Negatif = Referanstan Daha Hızlı",
                yaxis_title="Z-Skoru (Standart Sapma)",
                barmode="group",
                height=360,
            )
            st.plotly_chart(fig_z, use_container_width=True)

        with col_s_chart:
            m1_col = f"{driver1} Ort. (sn)"
            m2_col = f"{driver2} Ort. (sn)"
            fig_sec_abs = go.Figure()
            fig_sec_abs.add_trace(go.Bar(x=sector_comp_df["Sektör"], y=sector_comp_df[m1_col], name=f"{driver1} Ortalama", marker_color=DRIVER_PALETTE[0]))
            fig_sec_abs.add_trace(go.Bar(x=sector_comp_df["Sektör"], y=sector_comp_df[m2_col], name=f"{driver2} Ortalama", marker_color=DRIVER_PALETTE[1]))
            fig_sec_abs.update_layout(
                title="Ortalama Sektör Süreleri Karşılaştırması (Saniye)",
                yaxis_title="Süre (sn)",
                barmode="group",
                height=360,
            )
            st.plotly_chart(fig_sec_abs, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Teorik Optimal Tur ve En İyi Sektörlerin Geldiği Kaynak Turlar")

    opt_rows: List[Dict[str, Any]] = []
    sector_origin_rows: List[Dict[str, Any]] = []

    for drv_code, o_det in ((driver1, opt_det_d1), (driver2, opt_det_d2)):
        secs = o_det.get("sectors", {})
        s1_info = secs.get("Sector1Time", {})
        s2_info = secs.get("Sector2Time", {})
        s3_info = secs.get("Sector3Time", {})

        opt_rows.append(
            {
                "Sürücü": drv_code,
                "Örneklem (n)": o_det.get("sample_size", 0),
                "En Hızlı Tur (sn)": o_det.get("actual_best_sec", np.nan),
                "En Hızlı Tur (Formatlı)": format_laptime_str(o_det.get("actual_best_sec")),
                "En Hızlı Tur No": f"Tur {o_det.get('actual_best_lap_number')} ({o_det.get('actual_best_compound') or '-'})"
                if o_det.get("actual_best_lap_number")
                else "-",
                "Teorik Optimal Tur (sn)": o_det.get("theoretical_optimal_sec", np.nan),
                "Teorik Optimal Tur (Formatlı)": format_laptime_str(o_det.get("theoretical_optimal_sec")),
                "Fark / Gelişim Potansiyeli (En Hızlı Tur - Teorik Optimal Tur) (sn)": o_det.get("development_potential_sec", np.nan),
                "En İyi S1 (Kaynak Tur)": f"{s1_info.get('best_sector_sec', np.nan):.3f} sn (Tur {s1_info.get('from_lap_number')}, S{s1_info.get('from_stint')}, {s1_info.get('from_compound')})"
                if pd.notna(s1_info.get("best_sector_sec"))
                else "-",
                "En İyi S2 (Kaynak Tur)": f"{s2_info.get('best_sector_sec', np.nan):.3f} sn (Tur {s2_info.get('from_lap_number')}, S{s1_info.get('from_stint')}, {s2_info.get('from_compound')})"
                if pd.notna(s2_info.get("best_sector_sec"))
                else "-",
                "En İyi S3 (Kaynak Tur)": f"{s3_info.get('best_sector_sec', np.nan):.3f} sn (Tur {s3_info.get('from_lap_number')}, S{s3_info.get('from_stint')}, {s3_info.get('from_compound')})"
                if pd.notna(s3_info.get("best_sector_sec"))
                else "-",
            }
        )

        for s_col, s_label in (("Sector1Time", "Sektör 1 (S1)"), ("Sector2Time", "Sektör 2 (S2)"), ("Sector3Time", "Sektör 3 (S3)")):
            s_d = secs.get(s_col, {})
            sector_origin_rows.append(
                {
                    "Sürücü": drv_code,
                    "Sektör": s_label,
                    "En Hızlı Sektör Süresi (sn)": s_d.get("best_sector_sec", np.nan),
                    "Geldiği Tur (LapNumber)": s_d.get("from_lap_number"),
                    "Geldiği Stint": s_d.get("from_stint"),
                    "Lastik Hamuru": s_d.get("from_compound") or "-",
                    "Lastik Yaşı": s_d.get("from_tyre_life"),
                    "En Hızlı Turdaki Süresi (sn)": s_d.get("actual_best_lap_sector_sec", np.nan),
                    "En Hızlı Turda Bu Sektördeki Kayıp (sn)": s_d.get("loss_in_actual_best_sec", np.nan),
                }
            )

    df_opt_summary = pd.DataFrame(opt_rows)
    df_sec_origins = pd.DataFrame(sector_origin_rows)

    st.dataframe(df_opt_summary.style.format(precision=3), use_container_width=True)
    st.markdown("**🔍 En İyi Sektörlerin Geldiği Turlar ve En Hızlı Turdaki Sektör Kayıpları**")
    st.dataframe(df_sec_origins.style.format(precision=3), use_container_width=True)


# ------------------------------------------------------------------------------
# BÖLÜM F: VERİ DOĞRULAMA VE GÜVENİLİRLİK RAPORU
# ------------------------------------------------------------------------------
with tab_quality:
    header_with_info(
        "🛡️ Veri Doğrulama, Filtreleme Nedenleri ve Metodolojik Güvenilirlik Raporu",
        "Analiz sırasında elenen turların sayısal dökümü, her turun elenme gerekçesi, "
        "örneklem büyüklüğü kontrolleri ve pist/hava durumu kayıtları.",
        level="h2",
    )

    filter_summary_df = pd.DataFrame(
        [
            {
                "Sürücü": driver1,
                "Toplam Ham Tur": filter_rep_d1["total_raw_laps"],
                "Geçerli Temiz Tur (n)": filter_rep_d1["clean_laps_count"],
                "Toplam Filtrelenen Tur": filter_rep_d1["total_raw_laps"] - filter_rep_d1["clean_laps_count"],
                "1. Eksik/Geçersiz Süre (NaN)": filter_rep_d1["missing_or_invalid_laptime"],
                "2. Pit Giriş/Çıkış (Pit-In/Out)": filter_rep_d1["pit_in_out_laps"],
                "3. Silinen Tur (Deleted)": filter_rep_d1["deleted_laps"],
                "4. SC / VSC / Kırmızı Bayrak": filter_rep_d1["sc_vsc_red_laps"],
                "5. Çeyrekler Açıklığı (IQR) Aykırı Değer": filter_rep_d1["iqr_outlier_laps"],
                "Çeyrekler Açıklığı (IQR) Modu": "Stint Bazlı" if filter_rep_d1["by_stint_iqr"] else "Seans Genel",
            },
            {
                "Sürücü": driver2,
                "Toplam Ham Tur": filter_rep_d2["total_raw_laps"],
                "Geçerli Temiz Tur (n)": filter_rep_d2["clean_laps_count"],
                "Toplam Filtrelenen Tur": filter_rep_d2["total_raw_laps"] - filter_rep_d2["clean_laps_count"],
                "1. Eksik/Geçersiz Süre (NaN)": filter_rep_d2["missing_or_invalid_laptime"],
                "2. Pit Giriş/Çıkış (Pit-In/Out)": filter_rep_d2["pit_in_out_laps"],
                "3. Silinen Tur (Deleted)": filter_rep_d2["deleted_laps"],
                "4. SC / VSC / Kırmızı Bayrak": filter_rep_d2["sc_vsc_red_laps"],
                "5. Çeyrekler Açıklığı (IQR) Aykırı Değer": filter_rep_d2["iqr_outlier_laps"],
                "Çeyrekler Açıklığı (IQR) Modu": "Stint Bazlı" if filter_rep_d2["by_stint_iqr"] else "Seans Genel",
            },
        ]
    )
    st.subheader("📋 Tur Filtreleme Dökümü")
    st.dataframe(filter_summary_df, use_container_width=True)

    reasons_col_map = {
        "LapNumber": "Tur No",
        "Stint": "Stint",
        "Compound": "Lastik Hamuru",
        "LapTime_Sec": "Tur Süresi (sn)",
        "Reason": "Filtreleme Nedeni",
    }
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        st.markdown(f"**{driver1} — Filtrelenen Turlar ve Nedenleri**")
        ex1_df = pd.DataFrame(filter_rep_d1["reasons_by_lap"])
        if not ex1_df.empty:
            st.dataframe(ex1_df.rename(columns=reasons_col_map), use_container_width=True)
        else:
            st.success("Filtrelenen tur bulunmuyor.")

    with col_ex2:
        st.markdown(f"**{driver2} — Filtrelenen Turlar ve Nedenleri**")
        ex2_df = pd.DataFrame(filter_rep_d2["reasons_by_lap"])
        if not ex2_df.empty:
            st.dataframe(ex2_df.rename(columns=reasons_col_map), use_container_width=True)
        else:
            st.success("Filtrelenen tur bulunmuyor.")

    st.markdown("---")
    st.subheader("🌡️ Pist Durumu ve Metodolojik İlkeler")
    if session_cond["weather_available"]:
        ws = session_cond["weather_summary"]
        st.markdown(
            f"- **Ortalama Pist Sıcaklığı:** `{ws.get('track_temp_mean_c', '-')} °C` "
            f"(Min: `{ws.get('track_temp_min_c', '-')} °C`, Maks: `{ws.get('track_temp_max_c', '-')} °C`)\n"
            f"- **Ortalama Hava Sıcaklığı:** `{ws.get('air_temp_mean_c', '-')} °C`\n"
            f"- **Yağış Tespiti:** `{'Evet' if ws.get('rainfall_detected') else 'Hayır'}`"
        )
    st.markdown(
        """
        **Uygulanan Veri Güvenilirliği Kuralları:**
        1. **Ham Veri Bütünlüğü:** `session.laps` ve ham telemetri tabloları hiçbir adımda yerinde değiştirilmez; tüm dönüşümler bağımsız kopyalar üzerinde yürütülür.
        2. **Sıfırla Doldurmama İlkesi:** Eksik Tur Süresi veya sektör süreleri (`NaN`) asla `0` ile doldurulmaz; her metrik yalnızca geçerli gözlemler üzerinden hesaplanır ve örneklem sayısı (`n`) raporlanır.
        3. **Kesikli ve Sürekli Telemetri Ayrımı:** `BrakeRaw` (0/1), `nGear` (1-8) ve `DRS` gibi kesikli kanallarda doğrusal interpolasyon uygulanmaz; ara vites veya yapay fren basıncı üretilmez.
        4. **Nedensellik Çıkarımı Kısıtı:** Stint içi Tur Süresi eğimleri (Performans Kaybı) ve Welch t-Testi sonuçları; yakıt yükü, trafik, ERS/Lift&Coast ve pist evrimi etkilerinden bağımsız nedensel kanıt olarak sunulmaz.
        """
    )
