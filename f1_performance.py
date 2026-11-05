"""
F1 Performance Analysis & Telemetry System — Performans, Sektör ve Stint Analiz Motoru (f1_performance.py)
===========================================================================================================
Bu modül FastF1 tur (laps) verileri üzerinde:
1. Çok aşamalı temiz tur filtreleme (Pit-in/out, Deleted, TrackStatus SC/VSC/Red Flag, Stint bazlı veya Genel IQR) ve filtreleme raporu,
2. Pace skoru (hem yarışın en hızlı turuna hem de referans sürücü medyanına göre),
3. Tutarlılık analizi (Genel ve Stint bazında CV %, standart sapma, medyan, ortalama, IQR),
4. İstatistiksel karşılaştırma (Welch's t-test + otokorelasyon/bağımsızlık metodolojik uyarısı),
5. Sektör analizi (Referans sürücüye göre Z-skorları + saniye bazında ortalama/medyan/en iyi sektör farkları),
6. Teorik optimal tur analizi (En iyi sektörlerin hangi turlardan, stintten ve lastik hamurundan geldiği),
7. Stint bazlı performans ve lastik kullanımı analizi (İlk 3 / Son 3 tur farkı, eşdeğer stint karşılaştırması),
8. Sürücü fark analizi (Tur bazlı, stint bazlı ve sektör bazlı zaman farkları)
işlemlerini gerçekleştirir.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import fastf1
import numpy as np
import pandas as pd
from scipy import stats

# Önbellek dizinini güvenli şekilde oluştur ve etkinleştir
_DEFAULT_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch", "fastf1_cache")
os.makedirs(_DEFAULT_CACHE_PATH, exist_ok=True)
if hasattr(fastf1.Cache, "configure"):
    fastf1.Cache.configure(cache_dir=_DEFAULT_CACHE_PATH)
else:
    fastf1.Cache.enable_cache(_DEFAULT_CACHE_PATH)

# Metodolojik uyarı sabitleri
TTEST_METHODOLOGY_WARNING = (
    "Metodolojik Not: Yarış içerisindeki ardışık tur süresi gözlemleri; yakıt yükü azalması, lastik performans kaybı "
    "ve pist yüzeyi gelişimi nedeniyle birbirinden bağımsız (i.i.d.) değildir ve zaman serisi otokorelasyonu içerir. "
    "Welch t-Testi (p-Değeri ve güven aralığı) sonucu kesin bir nedensel kanıt olarak değil, yardımcı/keşifsel bir istatistik olarak yorumlanmalıdır."
)

STINT_DEGRADATION_METHODOLOGY_WARNING = (
    "Metodolojik Not: Stint içindeki tur süresi değişimi ve gözlenen performans kaybı yalnızca lastik aşınmasından "
    "kaynaklanmaz. Azalan yakıt yükü (tur başına ~0.03-0.06 sn hızlanma etkisi), kirli hava ve trafik, "
    "Güvenlik Aracı (SC/VSC) sonrası lastik sıcaklığı değişimleri, enerji yönetimi ve "
    "pist yüzey gelişimi tur sürelerini doğrudan etkiler. Lastik yaşı ve performans kaybı üzerine nedensel çıkarım yapılırken bu etkenler birlikte değerlendirilmelidir."
)


# ==============================================================================
# YARDIMCI DÖNÜŞÜM VE GÜVENLİK FONKSİYONLARI
# ==============================================================================

def to_seconds_series(series: Optional[pd.Series]) -> pd.Series:
    """
    Bir Pandas serisini (Timedelta veya sayısal) güvenli bir şekilde saniye (float)
    serisine dönüştürür.<= 0 veya sonsuz değerleri NaN yapar.
    Ham seriyi değiştirmez.
    """
    if series is None or not isinstance(series, pd.Series):
        return pd.Series(dtype=float)
    if series.empty:
        return pd.Series(index=series.index, dtype=float)

    s = series.copy()
    if pd.api.types.is_timedelta64_dtype(s):
        sec = s.dt.total_seconds().astype(float)
    elif pd.api.types.is_numeric_dtype(s):
        sec = pd.to_numeric(s, errors="coerce").astype(float)
    else:
        # Objelerden timedelta veya float dönüşümü dene
        try:
            sec = pd.to_timedelta(s, errors="coerce").dt.total_seconds().astype(float)
            if sec.isna().all():
                sec = pd.to_numeric(s, errors="coerce").astype(float)
        except Exception:
            sec = pd.to_numeric(s, errors="coerce").astype(float)

    sec = sec.where((sec > 0) & np.isfinite(sec), np.nan)
    return sec


def format_laptime_str(seconds: Optional[float]) -> str:
    """Saniye cinsinden süreyi 'D:SS.mmm' (örn. 1:32.456) formatına çevirir."""
    if seconds is None or pd.isna(seconds) or not np.isfinite(seconds) or seconds <= 0:
        return "-"
    minutes = int(seconds // 60)
    rem_sec = seconds - (minutes * 60)
    if minutes > 0:
        return f"{minutes}:{rem_sec:06.3f}"
    return f"{rem_sec:.3f} sn"


# ==============================================================================
# 1. TEMİZ TUR FİLTRELEME VE FİLTRELEME RAPORU
# ==============================================================================

def filter_clean_laps(
    laps_data: pd.DataFrame,
    by_stint: bool = False,
    exclude_pit: bool = True,
    exclude_deleted: bool = True,
    exclude_sc_vsc_red: bool = True,
    exclude_yellow: bool = False,
    exclude_inaccurate: bool = False,
    iqr_multiplier: float = 1.5,
    min_laps_for_iqr: int = 4,
    return_report: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Tur zamanlarını çok aşamalı doğrulama (geçersiz/eksik süre, Pit-in, Pit-out,
    Deleted lap, SC/VSC/Kırmızı Bayrak TrackStatus) ve IQR (Çeyrekler Açıklığı)
    yöntemine göre filtreleyerek temiz turları bulur.

    Ham veriyi (laps_data) kesinlikle değiştirmez; yeni bir DataFrame kopyası döndürür.

    Args:
        laps_data (pd.DataFrame): Bir pilota ait tur zamanlarını içeren FastF1 verisi.
        by_stint (bool): True ise IQR filtresini her Stint içinde ayrı ayrı uygular.
        exclude_pit (bool): Pit-in ve Pit-out turlarını eler.
        exclude_deleted (bool): Pist sınırları vb. nedeniyle silinen (Deleted=True) turları eler.
        exclude_sc_vsc_red (bool): Güvenlik Aracı (4), Kırmızı Bayrak (5) ve VSC (6, 7) turlarını eler.
        exclude_yellow (bool): Sarı bayrak (2) içeren turları da eler (varsayılan False).
        exclude_inaccurate (bool): FastF1 IsAccurate=False olan turları eler.
        iqr_multiplier (float): IQR çarpanı (varsayılan 1.5).
        min_laps_for_iqr (int): IQR uygulanabilmesi için gereken minimum tur sayısı.
        return_report (bool): True ise (clean_laps_df, filter_report_dict) tuple döndürür.

    Returns:
        pd.DataFrame veya Tuple[pd.DataFrame, Dict[str, Any]]: Temiz turlar (ve opsiyonel filtreleme raporu).
    """
    empty_report: Dict[str, Any] = {
        "total_raw_laps": 0,
        "missing_or_invalid_laptime": 0,
        "pit_in_out_laps": 0,
        "deleted_laps": 0,
        "sc_vsc_red_laps": 0,
        "yellow_flag_laps": 0,
        "inaccurate_laps": 0,
        "iqr_outlier_laps": 0,
        "clean_laps_count": 0,
        "by_stint_iqr": bool(by_stint),
        "iqr_multiplier": float(iqr_multiplier),
        "reasons_by_lap": [],
    }

    if laps_data is None or not isinstance(laps_data, pd.DataFrame) or laps_data.empty:
        empty_df = laps_data.iloc[0:0].copy() if isinstance(laps_data, pd.DataFrame) else pd.DataFrame()
        return (empty_df, empty_report) if return_report else empty_df

    if "LapTime" not in laps_data.columns:
        empty_df = laps_data.iloc[0:0].copy()
        empty_report["total_raw_laps"] = len(laps_data)
        empty_report["missing_or_invalid_laptime"] = len(laps_data)
        return (empty_df, empty_report) if return_report else empty_df

    df = laps_data.copy()
    total_raw = len(df)
    lap_seconds = to_seconds_series(df["LapTime"])

    # 1. Geçerli LapTime kontrolü
    valid_time_mask = lap_seconds.notna()
    missing_time_count = int((~valid_time_mask).sum())

    # 2. Pit-in ve Pit-out kontrolü
    pit_mask = pd.Series(False, index=df.index)
    if exclude_pit:
        if "PitInTime" in df.columns:
            pit_mask = pit_mask | df["PitInTime"].notna()
        if "PitOutTime" in df.columns:
            pit_mask = pit_mask | df["PitOutTime"].notna()
    pit_excluded_mask = valid_time_mask & pit_mask
    pit_count = int(pit_excluded_mask.sum())

    # 3. Silinen tur (Deleted) kontrolü
    deleted_mask = pd.Series(False, index=df.index)
    if exclude_deleted and "Deleted" in df.columns:
        deleted_mask = df["Deleted"].eq(True)
    deleted_excluded_mask = valid_time_mask & (~pit_mask) & deleted_mask
    deleted_count = int(deleted_excluded_mask.sum())

    # 4. Pist durumu (TrackStatus: SC=4, Red=5, VSC=6,7) kontrolü
    sc_vsc_red_mask = pd.Series(False, index=df.index)
    yellow_mask = pd.Series(False, index=df.index)
    if "TrackStatus" in df.columns:
        ts_str = df["TrackStatus"].fillna("").astype(str)
        if exclude_sc_vsc_red:
            sc_vsc_red_mask = ts_str.apply(lambda s: any(c in s for c in ("4", "5", "6", "7")))
        if exclude_yellow:
            yellow_mask = ts_str.apply(lambda s: "2" in s)

    remaining_before_ts = valid_time_mask & (~pit_mask) & (~deleted_mask)
    sc_excluded_mask = remaining_before_ts & sc_vsc_red_mask
    sc_count = int(sc_excluded_mask.sum())

    remaining_before_yellow = remaining_before_ts & (~sc_vsc_red_mask)
    yellow_excluded_mask = remaining_before_yellow & yellow_mask
    yellow_count = int(yellow_excluded_mask.sum())

    # 5. IsAccurate kontrolü (opsiyonel)
    inacc_mask = pd.Series(False, index=df.index)
    if exclude_inaccurate and "IsAccurate" in df.columns:
        inacc_mask = ~df["IsAccurate"].eq(True)
    remaining_before_acc = remaining_before_yellow & (~yellow_mask)
    inacc_excluded_mask = remaining_before_acc & inacc_mask
    inacc_count = int(inacc_excluded_mask.sum())

    # Ön filtreyi geçen aday turlar
    candidate_mask = remaining_before_acc & (~inacc_mask)
    candidate_indices = df.index[candidate_mask]

    if len(candidate_indices) == 0:
        empty_df = df.iloc[0:0].copy()
        report = {
            "total_raw_laps": total_raw,
            "missing_or_invalid_laptime": missing_time_count,
            "pit_in_out_laps": pit_count,
            "deleted_laps": deleted_count,
            "sc_vsc_red_laps": sc_count,
            "yellow_flag_laps": yellow_count,
            "inaccurate_laps": inacc_count,
            "iqr_outlier_laps": 0,
            "clean_laps_count": 0,
            "by_stint_iqr": bool(by_stint),
            "iqr_multiplier": float(iqr_multiplier),
            "reasons_by_lap": _build_lap_exclusion_reasons(
                df,
                valid_time_mask,
                pit_excluded_mask,
                deleted_excluded_mask,
                sc_excluded_mask,
                yellow_excluded_mask,
                inacc_excluded_mask,
                pd.Series(False, index=df.index),
            ),
        }
        return (empty_df, report) if return_report else empty_df

    # 6. IQR Filtrelemesi (Genel veya Stint Bazlı)
    iqr_keep_mask = pd.Series(False, index=df.index)

    if by_stint and "Stint" in df.columns and df.loc[candidate_indices, "Stint"].notna().any():
        for stint_val, stint_grp in df.loc[candidate_indices].groupby("Stint", dropna=False):
            grp_idx = stint_grp.index
            grp_sec = lap_seconds.loc[grp_idx].dropna()
            if len(grp_sec) < min_laps_for_iqr:
                # Yeterli örneklem yoksa IQR aykırı değer silmesi yapmadan aday turları koru
                iqr_keep_mask.loc[grp_idx] = True
            else:
                q1 = float(np.percentile(grp_sec, 25))
                q3 = float(np.percentile(grp_sec, 75))
                iqr = q3 - q1
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr
                in_bounds = (lap_seconds.loc[grp_idx] >= lower_bound) & (lap_seconds.loc[grp_idx] <= upper_bound)
                iqr_keep_mask.loc[grp_idx] = in_bounds
    else:
        cand_sec = lap_seconds.loc[candidate_indices].dropna()
        if len(cand_sec) < min_laps_for_iqr:
            iqr_keep_mask.loc[candidate_indices] = True
        else:
            q1 = float(np.percentile(cand_sec, 25))
            q3 = float(np.percentile(cand_sec, 75))
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            in_bounds = (lap_seconds.loc[candidate_indices] >= lower_bound) & (
                lap_seconds.loc[candidate_indices] <= upper_bound
            )
            iqr_keep_mask.loc[candidate_indices] = in_bounds

    iqr_outlier_mask = candidate_mask & (~iqr_keep_mask)
    iqr_outlier_count = int(iqr_outlier_mask.sum())

    final_mask = candidate_mask & iqr_keep_mask
    clean_laps = df.loc[final_mask].copy()

    if return_report:
        report = {
            "total_raw_laps": total_raw,
            "missing_or_invalid_laptime": missing_time_count,
            "pit_in_out_laps": pit_count,
            "deleted_laps": deleted_count,
            "sc_vsc_red_laps": sc_count,
            "yellow_flag_laps": yellow_count,
            "inaccurate_laps": inacc_count,
            "iqr_outlier_laps": iqr_outlier_count,
            "clean_laps_count": int(len(clean_laps)),
            "by_stint_iqr": bool(by_stint),
            "iqr_multiplier": float(iqr_multiplier),
            "reasons_by_lap": _build_lap_exclusion_reasons(
                df,
                valid_time_mask,
                pit_excluded_mask,
                deleted_excluded_mask,
                sc_excluded_mask,
                yellow_excluded_mask,
                inacc_excluded_mask,
                iqr_outlier_mask,
            ),
        }
        return clean_laps, report

    return clean_laps


def _build_lap_exclusion_reasons(
    df: pd.DataFrame,
    valid_time_mask: pd.Series,
    pit_excluded_mask: pd.Series,
    deleted_excluded_mask: pd.Series,
    sc_excluded_mask: pd.Series,
    yellow_excluded_mask: pd.Series,
    inacc_excluded_mask: pd.Series,
    iqr_outlier_mask: pd.Series,
) -> List[Dict[str, Any]]:
    """Her elenen turun hangi nedenle elendiğini açıklayan kayıt listesi üretir."""
    records: List[Dict[str, Any]] = []
    lap_sec = to_seconds_series(df["LapTime"]) if "LapTime" in df.columns else pd.Series(np.nan, index=df.index)

    for idx in df.index:
        reason = None
        if not valid_time_mask.loc[idx]:
            reason = "Eksik veya Geçersiz Tur Süresi (NaN)"
        elif pit_excluded_mask.loc[idx]:
            reason = "Pit Giriş / Çıkış Turu (Pit-In / Pit-Out)"
        elif deleted_excluded_mask.loc[idx]:
            reason = "Silinen Tur (Deleted Lap)"
        elif sc_excluded_mask.loc[idx]:
            reason = "Güvenlik Aracı / VSC / Kırmızı Bayrak (TrackStatus)"
        elif yellow_excluded_mask.loc[idx]:
            reason = "Sarı Bayrak (TrackStatus)"
        elif inacc_excluded_mask.loc[idx]:
            reason = "Zamanlama Doğruluğu Düşük (IsAccurate=False)"
        elif iqr_outlier_mask.loc[idx]:
            reason = "Çeyrekler Açıklığı (IQR) Aykırı Değer (Trafik / Sürüş Hatası / Anomali)"

        if reason is not None:
            lap_no = df.loc[idx, "LapNumber"] if "LapNumber" in df.columns else np.nan
            stint_no = df.loc[idx, "Stint"] if "Stint" in df.columns else np.nan
            comp = df.loc[idx, "Compound"] if "Compound" in df.columns else "-"
            records.append(
                {
                    "LapNumber": int(lap_no) if pd.notna(lap_no) else None,
                    "Stint": int(stint_no) if pd.notna(stint_no) else None,
                    "Compound": str(comp) if pd.notna(comp) else "-",
                    "LapTime_Sec": round(float(lap_sec.loc[idx]), 3) if pd.notna(lap_sec.loc[idx]) else np.nan,
                    "Reason": reason,
                }
            )
    return records


# ==============================================================================
# 2. PACE SKORU HESAPLAMA (YARIŞ EN HIZLI TURU VE REFERANS SÜRÜCÜ MEDYANINA GÖRE)
# ==============================================================================

def calculate_pace_score(
    clean_laps: pd.DataFrame,
    fastest_lap_seconds: float,
    ref_median_seconds: Optional[float] = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    Temiz turların medyan süresinin, yarıştaki en hızlı tura (ve opsiyonel olarak
    seçilen referans sürücünün medyan temposuna) göre yüzde ve saniye farkını hesaplar.
    Daha düşük skor, daha iyi (hızlı) bir pace anlamına gelir.

    Args:
        clean_laps (pd.DataFrame): Pilotun temiz turlarını içeren veri.
        fastest_lap_seconds (float): Yarıştaki en hızlı turun saniye cinsinden değeri.
        ref_median_seconds (Optional[float]): Referans sürücünün medyan tur süresi (saniye).
        return_details (bool): True ise tüm pace metriklerini içeren sözlük döndürür.

    Returns:
        float veya Dict[str, Any]: Yüzde cinsinden pace skoru (veya detaylı metrik sözlüğü).
    """
    details: Dict[str, Any] = {
        "pace_score_vs_fastest_pct": np.nan,
        "delta_vs_fastest_sec": np.nan,
        "pace_score_vs_ref_pct": np.nan,
        "delta_vs_ref_sec": np.nan,
        "median_lap_sec": np.nan,
        "mean_lap_sec": np.nan,
        "best_lap_sec": np.nan,
        "fastest_lap_sec": float(fastest_lap_seconds) if fastest_lap_seconds is not None and pd.notna(fastest_lap_seconds) else np.nan,
        "ref_median_sec": float(ref_median_seconds) if ref_median_seconds is not None and pd.notna(ref_median_seconds) else np.nan,
        "sample_size": 0,
    }

    if clean_laps is None or not isinstance(clean_laps, pd.DataFrame) or clean_laps.empty or "LapTime" not in clean_laps.columns:
        return details if return_details else np.nan

    lap_times = to_seconds_series(clean_laps["LapTime"]).dropna()
    if lap_times.empty:
        return details if return_details else np.nan

    median_time = float(lap_times.median())
    mean_time = float(lap_times.mean())
    best_time = float(lap_times.min())

    details["median_lap_sec"] = median_time
    details["mean_lap_sec"] = mean_time
    details["best_lap_sec"] = best_time
    details["sample_size"] = int(len(lap_times))

    pace_score = np.nan
    if (
        fastest_lap_seconds is not None
        and pd.notna(fastest_lap_seconds)
        and np.isfinite(fastest_lap_seconds)
        and fastest_lap_seconds > 0
    ):
        pace_score = float(((median_time - fastest_lap_seconds) / fastest_lap_seconds) * 100.0)
        details["pace_score_vs_fastest_pct"] = pace_score
        details["delta_vs_fastest_sec"] = float(median_time - fastest_lap_seconds)

    if (
        ref_median_seconds is not None
        and pd.notna(ref_median_seconds)
        and np.isfinite(ref_median_seconds)
        and ref_median_seconds > 0
    ):
        details["pace_score_vs_ref_pct"] = float(((median_time - ref_median_seconds) / ref_median_seconds) * 100.0)
        details["delta_vs_ref_sec"] = float(median_time - ref_median_seconds)

    if return_details:
        return details
    return pace_score


# ==============================================================================
# 3. TUTARLILIK (CONSISTENCY) SKORU (GENEL VE STINT BAZLI)
# ==============================================================================

def calculate_consistency_score(
    clean_laps: pd.DataFrame,
    by_stint: bool = False,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    Temiz turların Varyasyon Katsayısını (CV = std/mean * 100), standart sapmasını,
    medyanını, ortalamasını ve stint bazlı tutarlılık metriklerini hesaplar.
    Düşük CV, daha istikrarlı bir sürüş anlamına gelir.

    Args:
        clean_laps (pd.DataFrame): Pilotun temiz turlarını içeren veri.
        by_stint (bool): True ise ve return_details=False ise stintlerin ağırlıklı ortalama CV'sini döndürür.
        return_details (bool): True ise CV, std, medyan, ortalama, IQR ve stint bazlı tabloyu içeren sözlük döndürür.

    Returns:
        float veya Dict[str, Any]: Yüzde cinsinden tutarlılık skoru (CV %) veya detaylı sözlük.
    """
    empty_details: Dict[str, Any] = {
        "cv_pct": np.nan,
        "std_sec": np.nan,
        "mean_sec": np.nan,
        "median_sec": np.nan,
        "iqr_sec": np.nan,
        "min_sec": np.nan,
        "max_sec": np.nan,
        "sample_size": 0,
        "stint_weighted_cv_pct": np.nan,
        "stint_weighted_std_sec": np.nan,
        "stint_breakdown": pd.DataFrame(),
    }

    if clean_laps is None or not isinstance(clean_laps, pd.DataFrame) or clean_laps.empty or "LapTime" not in clean_laps.columns:
        return empty_details if return_details else np.nan

    df = clean_laps.copy()
    df["_lap_sec"] = to_seconds_series(df["LapTime"])
    valid_df = df.dropna(subset=["_lap_sec"])

    if valid_df.empty:
        return empty_details if return_details else np.nan

    lap_times = valid_df["_lap_sec"]
    n_total = len(lap_times)
    mean_time = float(lap_times.mean())
    median_time = float(lap_times.median())
    min_time = float(lap_times.min())
    max_time = float(lap_times.max())
    q1 = float(np.percentile(lap_times, 25))
    q3 = float(np.percentile(lap_times, 75))
    iqr_sec = float(q3 - q1)

    if n_total >= 2 and mean_time > 0:
        std_time = float(lap_times.std(ddof=1))
        cv = float((std_time / mean_time) * 100.0)
    else:
        std_time = np.nan
        cv = np.nan

    # Stint bazlı tutarlılık hesaplama
    stint_rows: List[Dict[str, Any]] = []
    if "Stint" in valid_df.columns and valid_df["Stint"].notna().any():
        for stint_val, grp in valid_df.groupby("Stint"):
            s_times = grp["_lap_sec"].dropna()
            n_s = len(s_times)
            if n_s == 0:
                continue
            s_mean = float(s_times.mean())
            s_med = float(s_times.median())
            s_std = float(s_times.std(ddof=1)) if n_s >= 2 else np.nan
            s_cv = float((s_std / s_mean) * 100.0) if (n_s >= 2 and s_mean > 0 and pd.notna(s_std)) else np.nan
            s_q1 = float(np.percentile(s_times, 25))
            s_q3 = float(np.percentile(s_times, 75))
            comp_vals = grp["Compound"].dropna().unique() if "Compound" in grp.columns else []
            comp_str = str(comp_vals[0]) if len(comp_vals) > 0 else "UNKNOWN"
            stint_rows.append(
                {
                    "Stint": int(stint_val),
                    "Compound": comp_str,
                    "SampleSize": int(n_s),
                    "Mean_Sec": s_mean,
                    "Median_Sec": s_med,
                    "Std_Sec": s_std,
                    "CV_Pct": s_cv,
                    "IQR_Sec": float(s_q3 - s_q1),
                    "Min_Sec": float(s_times.min()),
                    "Max_Sec": float(s_times.max()),
                }
            )

    stint_df = pd.DataFrame(stint_rows)
    stint_weighted_cv = np.nan
    stint_weighted_std = np.nan
    if not stint_df.empty:
        valid_stints = stint_df.dropna(subset=["CV_Pct", "Std_Sec"])
        if not valid_stints.empty:
            weights = valid_stints["SampleSize"].values
            stint_weighted_cv = float(np.average(valid_stints["CV_Pct"].values, weights=weights))
            stint_weighted_std = float(np.average(valid_stints["Std_Sec"].values, weights=weights))

    details: Dict[str, Any] = {
        "cv_pct": cv,
        "std_sec": std_time,
        "mean_sec": mean_time,
        "median_sec": median_time,
        "iqr_sec": iqr_sec,
        "min_sec": min_time,
        "max_sec": max_time,
        "sample_size": int(n_total),
        "stint_weighted_cv_pct": stint_weighted_cv,
        "stint_weighted_std_sec": stint_weighted_std,
        "stint_breakdown": stint_df,
    }

    if return_details:
        return details

    if by_stint and pd.notna(stint_weighted_cv):
        return stint_weighted_cv
    return cv


# ==============================================================================
# 4. İSTATİSTİKSEL KARŞILAŞTIRMA (WELCH'S T-TEST VE METODOLOJİK UYARI)
# ==============================================================================

def compare_drivers_ttest(
    clean_laps_driver1: pd.DataFrame,
    clean_laps_driver2: pd.DataFrame,
    return_details: bool = False,
) -> Union[Tuple[float, str], Dict[str, Any]]:
    """
    İki pilotun temiz tur zamanı dağılımlarını Welch's bağımsız örneklem t-testiyle
    (equal_var=False) karşılaştırır. Ardışık turların bağımsız (i.i.d.) olmadığını
    ve örneklem büyüklüğü kısıtlarını açıkça raporlar.

    Args:
        clean_laps_driver1 (pd.DataFrame): 1. pilotun temiz turları.
        clean_laps_driver2 (pd.DataFrame): 2. pilotun temiz turları.
        return_details (bool): True ise detaylı istatistik sözlüğü döndürür.

    Returns:
        Tuple[float, str] veya Dict[str, Any]: (p_value, warning_msg) veya detaylı sözlük.
    """
    times_d1 = (
        to_seconds_series(clean_laps_driver1["LapTime"]).dropna().values
        if isinstance(clean_laps_driver1, pd.DataFrame) and "LapTime" in clean_laps_driver1.columns
        else np.array([], dtype=float)
    )
    times_d2 = (
        to_seconds_series(clean_laps_driver2["LapTime"]).dropna().values
        if isinstance(clean_laps_driver2, pd.DataFrame) and "LapTime" in clean_laps_driver2.columns
        else np.array([], dtype=float)
    )

    n1 = int(len(times_d1))
    n2 = int(len(times_d2))

    warnings_list: List[str] = []
    if n1 < 2 or n2 < 2:
        warnings_list.append(
            f"Uyarı: Welch t-Testi için her iki sürücüde de en az 2 geçerli tur gereklidir (Mevcut: n1={n1}, n2={n2})."
        )
        warnings_list.append(TTEST_METHODOLOGY_WARNING)
        combined_warning = " | ".join(warnings_list)
        details = {
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_driver1": n1,
            "n_driver2": n2,
            "mean_driver1_sec": float(times_d1.mean()) if n1 > 0 else np.nan,
            "mean_driver2_sec": float(times_d2.mean()) if n2 > 0 else np.nan,
            "median_driver1_sec": float(np.median(times_d1)) if n1 > 0 else np.nan,
            "median_driver2_sec": float(np.median(times_d2)) if n2 > 0 else np.nan,
            "mean_diff_sec": np.nan,
            "median_diff_sec": np.nan,
            "is_significant_005": False,
            "sample_size_warning": warnings_list[0],
            "methodology_warning": TTEST_METHODOLOGY_WARNING,
            "warning_msg": combined_warning,
        }
        return details if return_details else (np.nan, combined_warning)

    sample_warning = ""
    if n1 < 30 or n2 < 30:
        sample_warning = (
            f"Uyarı: Örneklem sayısı 30'dan küçük (n1={n1}, n2={n2}), Welch t-Testi sonuçları güvenilir olmayabilir."
        )
        warnings_list.append(sample_warning)

    warnings_list.append(TTEST_METHODOLOGY_WARNING)
    combined_warning = " | ".join(warnings_list)

    # Sıfır varyans (tüm turlar birebir aynı) kontrolü
    std1 = float(np.std(times_d1, ddof=1))
    std2 = float(np.std(times_d2, ddof=1))
    if std1 == 0.0 and std2 == 0.0:
        t_stat = 0.0 if times_d1[0] == times_d2[0] else np.inf
        p_value = 1.0 if times_d1[0] == times_d2[0] else 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t_res = stats.ttest_ind(times_d1, times_d2, equal_var=False)
            t_stat = float(t_res.statistic) if pd.notna(t_res.statistic) else np.nan
            p_value = float(t_res.pvalue) if pd.notna(t_res.pvalue) else np.nan

    mean1 = float(np.mean(times_d1))
    mean2 = float(np.mean(times_d2))
    med1 = float(np.median(times_d1))
    med2 = float(np.median(times_d2))

    details = {
        "t_stat": t_stat,
        "p_value": p_value,
        "n_driver1": n1,
        "n_driver2": n2,
        "mean_driver1_sec": mean1,
        "mean_driver2_sec": mean2,
        "median_driver1_sec": med1,
        "median_driver2_sec": med2,
        "mean_diff_sec": float(mean1 - mean2),
        "median_diff_sec": float(med1 - med2),
        "is_significant_005": bool(pd.notna(p_value) and p_value < 0.05),
        "sample_size_warning": sample_warning,
        "methodology_warning": TTEST_METHODOLOGY_WARNING,
        "warning_msg": combined_warning,
    }

    if return_details:
        return details
    return p_value, combined_warning


# ==============================================================================
# 5. SEKTÖR ANALİZİ (Z-SKORU VE SANİYE BAZINDA SEKTÖR FARKLARI)
# ==============================================================================

def calculate_sector_zscores(
    driver_laps: pd.DataFrame,
    ref_driver_laps: pd.DataFrame,
    return_details: bool = False,
) -> Dict[str, Any]:
    """
    Pilotun sektör sürelerini, referans pilotun sektör sürelerine göre Z-skoru
    ve saniye bazında farklar (Ortalama, Medyan, En İyi) olarak ayrıştırır.

    Geriye dönük uyumluluk için 'Sektör 1 Z-Skor', 'Sektör 2 Z-Skor', 'Sektör 3 Z-Skor'
    anahtarlarını her zaman korur ve ek olarak saniye farkı anahtarlarını içerir.

    Args:
        driver_laps (pd.DataFrame): İncelenen pilotun temiz turları.
        ref_driver_laps (pd.DataFrame): Referans pilotun temiz turları.
        return_details (bool): True ise sektör bazlı detaylı alt sözlükleri de ekler.

    Returns:
        Dict[str, Any]: Sektör Z-skorları ve saniye farklarını içeren sözlük.
    """
    result: Dict[str, Any] = {}
    sector_details: Dict[int, Dict[str, Any]] = {}

    drv_valid = isinstance(driver_laps, pd.DataFrame) and not driver_laps.empty
    ref_valid = isinstance(ref_driver_laps, pd.DataFrame) and not ref_driver_laps.empty

    for i in (1, 2, 3):
        sector_col = f"Sector{i}Time"
        z_key = f"Sektör {i} Z-Skor"
        diff_mean_key = f"Sektör {i} Fark (sn)"
        diff_med_key = f"Sektör {i} Medyan Fark (sn)"

        if not drv_valid or not ref_valid or sector_col not in driver_laps.columns or sector_col not in ref_driver_laps.columns:
            result[z_key] = np.nan
            result[diff_mean_key] = np.nan
            result[diff_med_key] = np.nan
            sector_details[i] = {
                "sector": i,
                "driver_mean_sec": np.nan,
                "driver_median_sec": np.nan,
                "driver_best_sec": np.nan,
                "driver_std_sec": np.nan,
                "driver_n": 0,
                "ref_mean_sec": np.nan,
                "ref_median_sec": np.nan,
                "ref_best_sec": np.nan,
                "ref_std_sec": np.nan,
                "ref_n": 0,
                "delta_mean_sec": np.nan,
                "delta_median_sec": np.nan,
                "delta_best_sec": np.nan,
                "z_score": np.nan,
            }
            continue

        ref_times = to_seconds_series(ref_driver_laps[sector_col]).dropna()
        drv_times = to_seconds_series(driver_laps[sector_col]).dropna()

        n_ref = int(len(ref_times))
        n_drv = int(len(drv_times))

        if n_ref == 0 or n_drv == 0:
            result[z_key] = np.nan
            result[diff_mean_key] = np.nan
            result[diff_med_key] = np.nan
            sector_details[i] = {
                "sector": i,
                "driver_mean_sec": float(drv_times.mean()) if n_drv > 0 else np.nan,
                "driver_median_sec": float(drv_times.median()) if n_drv > 0 else np.nan,
                "driver_best_sec": float(drv_times.min()) if n_drv > 0 else np.nan,
                "driver_std_sec": float(drv_times.std(ddof=1)) if n_drv >= 2 else np.nan,
                "driver_n": n_drv,
                "ref_mean_sec": float(ref_times.mean()) if n_ref > 0 else np.nan,
                "ref_median_sec": float(ref_times.median()) if n_ref > 0 else np.nan,
                "ref_best_sec": float(ref_times.min()) if n_ref > 0 else np.nan,
                "ref_std_sec": float(ref_times.std(ddof=1)) if n_ref >= 2 else np.nan,
                "ref_n": n_ref,
                "delta_mean_sec": np.nan,
                "delta_median_sec": np.nan,
                "delta_best_sec": np.nan,
                "z_score": np.nan,
            }
            continue

        ref_mean = float(ref_times.mean())
        ref_med = float(ref_times.median())
        ref_best = float(ref_times.min())
        ref_std = float(ref_times.std(ddof=1)) if n_ref >= 2 else np.nan

        drv_mean = float(drv_times.mean())
        drv_med = float(drv_times.median())
        drv_best = float(drv_times.min())
        drv_std = float(drv_times.std(ddof=1)) if n_drv >= 2 else np.nan

        delta_mean = float(drv_mean - ref_mean)
        delta_med = float(drv_med - ref_med)
        delta_best = float(drv_best - ref_best)

        result[diff_mean_key] = delta_mean
        result[diff_med_key] = delta_med

        if pd.isna(ref_std) or ref_std <= 0:
            z_val = np.nan
        else:
            z_score_series = (drv_times - ref_mean) / ref_std
            z_val = float(z_score_series.mean())

        result[z_key] = z_val

        sector_details[i] = {
            "sector": i,
            "driver_mean_sec": drv_mean,
            "driver_median_sec": drv_med,
            "driver_best_sec": drv_best,
            "driver_std_sec": drv_std,
            "driver_n": n_drv,
            "ref_mean_sec": ref_mean,
            "ref_median_sec": ref_med,
            "ref_best_sec": ref_best,
            "ref_std_sec": ref_std,
            "ref_n": n_ref,
            "delta_mean_sec": delta_mean,
            "delta_median_sec": delta_med,
            "delta_best_sec": delta_best,
            "z_score": z_val,
        }

    if return_details:
        result["sector_details"] = sector_details

    return result


def build_sector_comparison_table(
    laps_d1: pd.DataFrame,
    laps_d2: pd.DataFrame,
    ref_laps: Optional[pd.DataFrame] = None,
    driver1_name: str = "Driver 1",
    driver2_name: str = "Driver 2",
    ref_driver_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    İki sürücünün S1, S2 ve S3 sektör performanslarını (Ortalama, Medyan, En İyi,
    birbirlerine göre saniye farkı ve seçilen referans sürücüye göre Z-skorları)
    karşılaştırmalı bir DataFrame olarak üretir.

    Referans tur havuzu verilmediğinde (None veya boş), iki sürücünün temiz turlarının
    birleşik havuzunu referans alarak 0.0 Z-skor sorununu önler.
    """
    if ref_laps is None or not isinstance(ref_laps, pd.DataFrame) or ref_laps.empty:
        ref_laps = pd.concat([laps_d1, laps_d2], ignore_index=True) if (
            (laps_d1 is not None and not laps_d1.empty) or (laps_d2 is not None and not laps_d2.empty)
        ) else pd.DataFrame()
        if not ref_driver_name:
            ref_driver_name = f"{driver1_name}+{driver2_name} Havuzu"

    if not ref_driver_name:
        ref_driver_name = "Referans"

    z_d1 = calculate_sector_zscores(laps_d1, ref_laps, return_details=True)
    z_d2 = calculate_sector_zscores(laps_d2, ref_laps, return_details=True)

    det_d1 = z_d1.get("sector_details", {})
    det_d2 = z_d2.get("sector_details", {})

    rows: List[Dict[str, Any]] = []
    for i in (1, 2, 3):
        s1 = det_d1.get(i, {})
        s2 = det_d2.get(i, {})

        m1 = s1.get("driver_mean_sec", np.nan)
        m2 = s2.get("driver_mean_sec", np.nan)
        med1 = s1.get("driver_median_sec", np.nan)
        med2 = s2.get("driver_median_sec", np.nan)
        b1 = s1.get("driver_best_sec", np.nan)
        b2 = s2.get("driver_best_sec", np.nan)

        rows.append(
            {
                "Sektör": f"Sektör {i} (S{i})",
                f"{driver1_name} Ort. (sn)": m1,
                f"{driver2_name} Ort. (sn)": m2,
                f"Ort. Fark ({driver1_name} - {driver2_name}) (sn)": (m1 - m2) if (pd.notna(m1) and pd.notna(m2)) else np.nan,
                f"{driver1_name} Medyan (sn)": med1,
                f"{driver2_name} Medyan (sn)": med2,
                f"Medyan Fark ({driver1_name} - {driver2_name}) (sn)": (med1 - med2) if (pd.notna(med1) and pd.notna(med2)) else np.nan,
                f"{driver1_name} En İyi (sn)": b1,
                f"{driver2_name} En İyi (sn)": b2,
                f"En İyi Fark ({driver1_name} - {driver2_name}) (sn)": (b1 - b2) if (pd.notna(b1) and pd.notna(b2)) else np.nan,
                f"{driver1_name} Z-Skor (Ref: {ref_driver_name})": s1.get("z_score", np.nan),
                f"{driver2_name} Z-Skor (Ref: {ref_driver_name})": s2.get("z_score", np.nan),
                f"{driver1_name} Örneklem (n)": s1.get("driver_n", 0),
                f"{driver2_name} Örneklem (n)": s2.get("driver_n", 0),
            }
        )
    return pd.DataFrame(rows)


# ==============================================================================
# 6. TEORİK OPTİMAL TUR ANALİZİ (KAYNAK TUR VE HAMUR BİLGİSİYLE)
# ==============================================================================

def calculate_theoretical_optimal(
    clean_laps_df: pd.DataFrame,
    sector_columns: Optional[List[str]] = None,
    return_details: bool = False,
) -> Union[Tuple[float, float, float], Dict[str, Any]]:
    """
    Her sektörde pilotun attığı en iyi (minimum) zamanı bulup toplayarak teorik optimal turu hesaplar.
    Gerçek en iyi turuyla arasındaki farkı saniye cinsinden (gelişim potansiyeli) ve
    en iyi sektörlerin hangi turlardan (LapNumber, Stint, Compound) geldiğini döndürür.

    Args:
        clean_laps_df (pd.DataFrame): Pilotun temiz turları.
        sector_columns (Optional[List[str]]): Sektör sütun isimleri (varsayılan ['Sector1Time', 'Sector2Time', 'Sector3Time']).
        return_details (bool): True ise sektör kaynak turlarını da içeren detaylı sözlük döndürür.

    Returns:
        Tuple[float, float, float] veya Dict[str, Any]:
            (teorik_optimal_saniye, gercek_en_iyi_saniye, gelisim_potansiyeli_saniye) veya detaylı sözlük.
    """
    if sector_columns is None:
        sector_columns = ["Sector1Time", "Sector2Time", "Sector3Time"]

    empty_details: Dict[str, Any] = {
        "theoretical_optimal_sec": np.nan,
        "actual_best_sec": np.nan,
        "development_potential_sec": np.nan,
        "actual_best_lap_number": None,
        "actual_best_stint": None,
        "actual_best_compound": None,
        "sectors": {},
        "sample_size": 0,
    }

    if clean_laps_df is None or not isinstance(clean_laps_df, pd.DataFrame) or clean_laps_df.empty:
        return empty_details if return_details else (np.nan, np.nan, np.nan)

    df = clean_laps_df.copy()

    # Gerçek en iyi turu bul
    actual_best_time = np.nan
    actual_best_idx = None
    if "LapTime" in df.columns:
        lap_sec = to_seconds_series(df["LapTime"])
        valid_laps = lap_sec.dropna()
        empty_details["sample_size"] = int(len(valid_laps))
        if not valid_laps.empty:
            actual_best_idx = valid_laps.idxmin()
            actual_best_time = float(valid_laps.loc[actual_best_idx])

    actual_best_lap_no = (
        int(df.loc[actual_best_idx, "LapNumber"])
        if actual_best_idx is not None and "LapNumber" in df.columns and pd.notna(df.loc[actual_best_idx, "LapNumber"])
        else None
    )
    actual_best_stint = (
        int(df.loc[actual_best_idx, "Stint"])
        if actual_best_idx is not None and "Stint" in df.columns and pd.notna(df.loc[actual_best_idx, "Stint"])
        else None
    )
    actual_best_compound = (
        str(df.loc[actual_best_idx, "Compound"])
        if actual_best_idx is not None and "Compound" in df.columns and pd.notna(df.loc[actual_best_idx, "Compound"])
        else None
    )

    sector_info: Dict[str, Dict[str, Any]] = {}
    theoretical_optimal_time = 0.0
    all_sectors_valid = True

    for idx_s, col in enumerate(sector_columns, start=1):
        if col not in df.columns:
            all_sectors_valid = False
            sector_info[col] = {
                "sector_index": idx_s,
                "best_sector_sec": np.nan,
                "from_lap_number": None,
                "from_stint": None,
                "from_compound": None,
                "from_tyre_life": None,
                "actual_best_lap_sector_sec": np.nan,
                "loss_in_actual_best_sec": np.nan,
            }
            continue

        sec_series = to_seconds_series(df[col]).dropna()
        if sec_series.empty:
            all_sectors_valid = False
            sector_info[col] = {
                "sector_index": idx_s,
                "best_sector_sec": np.nan,
                "from_lap_number": None,
                "from_stint": None,
                "from_compound": None,
                "from_tyre_life": None,
                "actual_best_lap_sector_sec": np.nan,
                "loss_in_actual_best_sec": np.nan,
            }
            continue

        best_idx = sec_series.idxmin()
        best_sec = float(sec_series.loc[best_idx])
        theoretical_optimal_time += best_sec

        from_lap = (
            int(df.loc[best_idx, "LapNumber"])
            if "LapNumber" in df.columns and pd.notna(df.loc[best_idx, "LapNumber"])
            else None
        )
        from_stint = (
            int(df.loc[best_idx, "Stint"])
            if "Stint" in df.columns and pd.notna(df.loc[best_idx, "Stint"])
            else None
        )
        from_comp = (
            str(df.loc[best_idx, "Compound"])
            if "Compound" in df.columns and pd.notna(df.loc[best_idx, "Compound"])
            else None
        )
        from_life = (
            int(df.loc[best_idx, "TyreLife"])
            if "TyreLife" in df.columns and pd.notna(df.loc[best_idx, "TyreLife"])
            else None
        )

        act_lap_sec_val = np.nan
        if actual_best_idx is not None:
            full_sec_series = to_seconds_series(df[col])
            if actual_best_idx in full_sec_series.index and pd.notna(full_sec_series.loc[actual_best_idx]):
                act_lap_sec_val = float(full_sec_series.loc[actual_best_idx])

        sector_info[col] = {
            "sector_index": idx_s,
            "best_sector_sec": best_sec,
            "from_lap_number": from_lap,
            "from_stint": from_stint,
            "from_compound": from_comp,
            "from_tyre_life": from_life,
            "actual_best_lap_sector_sec": act_lap_sec_val,
            "loss_in_actual_best_sec": float(act_lap_sec_val - best_sec) if pd.notna(act_lap_sec_val) else np.nan,
        }

    if not all_sectors_valid or len(sector_columns) == 0:
        theoretical_optimal_time = np.nan

    development_potential = (
        float(actual_best_time - theoretical_optimal_time)
        if pd.notna(actual_best_time) and pd.notna(theoretical_optimal_time)
        else np.nan
    )

    details: Dict[str, Any] = {
        "theoretical_optimal_sec": theoretical_optimal_time,
        "actual_best_sec": actual_best_time,
        "development_potential_sec": development_potential,
        "actual_best_lap_number": actual_best_lap_no,
        "actual_best_stint": actual_best_stint,
        "actual_best_compound": actual_best_compound,
        "sectors": sector_info,
        "sample_size": empty_details["sample_size"],
    }

    if return_details:
        return details
    return theoretical_optimal_time, actual_best_time, development_potential


# ==============================================================================
# 7. STINT BAZLI ANALİZ MODÜLÜ (BÖLÜM 2)
# ==============================================================================

def analyze_driver_stints(
    raw_laps: pd.DataFrame,
    clean_laps: pd.DataFrame,
    driver_name: str = "",
    ref_stints_df: Optional[pd.DataFrame] = None,
    min_valid_laps_warning: int = 4,
    fuel_effect_sec_per_lap: float = 0.035,
) -> pd.DataFrame:
    """
    FastF1 laps DataFrame içerisindeki Stint, Compound, TyreLife, LapNumber ve LapTime
    alanlarını kullanarak her stint için kapsamlı performans ve lastik metriklerini hesaplar.

    Hesaplanan Metrikler:
    - Driver, Stint, Compound
    - StartLap, EndLap, TotalLaps, ValidLapCount (n)
    - TyreLifeStart, TyreLifeEnd
    - MeanLapTime_Sec, MedianLapTime_Sec, BestLapTime_Sec, WorstLapTime_Sec
    - StdLapTime_Sec, CV_Pct (%)
    - First3Mean_Sec (İlk 3 geçerli turun ortalaması)
    - Last3Mean_Sec (Son 3 geçerli turun ortalaması)
    - Last3_Minus_First3_Sec (Son 3 tur ort. - İlk 3 tur ort. saniye farkı)
    - LinearTrend_SecPerLap (Tur başına doğrusal değişim eğimi)
    - FuelCorrectedTrend_SecPerLap (Yakıt yükü azalması ~0.035 sn/tur etkisinden arındırılmış gerçek aşınma eğimi)
    - DiffToRefMedian_Sec (Referans sürücünün aynı stint medyanına göre farkı)
    - SC_VSC_LapCount (Stint içinde SC/VSC/Kırmızı bayrak görülen tur sayısı)
    - SampleStatus / Warning (Örneklem yeterliliği ve metodolojik notlar)
    """
    columns_order = [
        "Driver",
        "Stint",
        "Compound",
        "StartLap",
        "EndLap",
        "TotalLaps",
        "ValidLapCount",
        "TyreLifeStart",
        "TyreLifeEnd",
        "MeanLapTime_Sec",
        "MedianLapTime_Sec",
        "BestLapTime_Sec",
        "BestLapNumber",
        "WorstLapTime_Sec",
        "WorstLapNumber",
        "StdLapTime_Sec",
        "CV_Pct",
        "First3Mean_Sec",
        "Last3Mean_Sec",
        "Last3_Minus_First3_Sec",
        "LinearTrend_SecPerLap",
        "FuelCorrectedTrend_SecPerLap",
        "DiffToRefMedian_Sec",
        "SC_VSC_LapCount",
        "SampleNote",
    ]

    if raw_laps is None or not isinstance(raw_laps, pd.DataFrame) or raw_laps.empty:
        return pd.DataFrame(columns=columns_order)

    raw_df = raw_laps.copy()
    clean_df = clean_laps.copy() if isinstance(clean_laps, pd.DataFrame) else pd.DataFrame()

    if "Stint" not in raw_df.columns or raw_df["Stint"].dropna().empty:
        raw_df["Stint"] = 1
        if not clean_df.empty and "Stint" not in clean_df.columns:
            clean_df["Stint"] = 1

    raw_df["_lap_sec"] = to_seconds_series(raw_df["LapTime"]) if "LapTime" in raw_df.columns else np.nan
    if not clean_df.empty and "LapTime" in clean_df.columns:
        clean_df["_lap_sec"] = to_seconds_series(clean_df["LapTime"])
    else:
        clean_df["_lap_sec"] = pd.Series(dtype=float)

    stint_numbers = sorted(raw_df["Stint"].dropna().unique())
    rows: List[Dict[str, Any]] = []

    for stint_val in stint_numbers:
        stint_int = int(stint_val)
        r_grp = raw_df[raw_df["Stint"] == stint_val].copy()
        if "LapNumber" in r_grp.columns:
            r_grp = r_grp.sort_values("LapNumber")

        c_grp = (
            clean_df[clean_df["Stint"] == stint_val].copy()
            if not clean_df.empty and "Stint" in clean_df.columns
            else pd.DataFrame()
        )
        if not c_grp.empty and "LapNumber" in c_grp.columns:
            c_grp = c_grp.sort_values("LapNumber")

        # Lastik hamuru (Compound)
        compound_str = "UNKNOWN"
        if "Compound" in r_grp.columns:
            valid_comps = r_grp["Compound"].dropna().astype(str)
            valid_comps = valid_comps[~valid_comps.isin(["", "nan", "None", "UNKNOWN"])]
            if not valid_comps.empty:
                compound_str = str(valid_comps.mode().iloc[0])

        # Tur aralığı
        start_lap = int(r_grp["LapNumber"].dropna().min()) if "LapNumber" in r_grp.columns and r_grp["LapNumber"].notna().any() else None
        end_lap = int(r_grp["LapNumber"].dropna().max()) if "LapNumber" in r_grp.columns and r_grp["LapNumber"].notna().any() else None
        total_laps = int(len(r_grp))

        # Lastik yaşı başlangıç ve bitiş
        tyre_start = (
            int(r_grp["TyreLife"].dropna().min())
            if "TyreLife" in r_grp.columns and r_grp["TyreLife"].notna().any()
            else None
        )
        tyre_end = (
            int(r_grp["TyreLife"].dropna().max())
            if "TyreLife" in r_grp.columns and r_grp["TyreLife"].notna().any()
            else None
        )

        # SC / VSC / Red Flag tur sayısı
        sc_vsc_count = 0
        if "TrackStatus" in r_grp.columns:
            sc_vsc_count = int(
                r_grp["TrackStatus"]
                .fillna("")
                .astype(str)
                .apply(lambda s: any(code in s for code in ("4", "5", "6", "7")))
                .sum()
            )

        # Geçerli (temiz) turlar üzerinden istatistikler
        valid_c = c_grp.dropna(subset=["_lap_sec"]) if not c_grp.empty and "_lap_sec" in c_grp.columns else pd.DataFrame()
        n_valid = int(len(valid_c))

        if n_valid == 0:
            rows.append(
                {
                    "Driver": driver_name,
                    "Stint": stint_int,
                    "Compound": compound_str,
                    "StartLap": start_lap,
                    "EndLap": end_lap,
                    "TotalLaps": total_laps,
                    "ValidLapCount": 0,
                    "TyreLifeStart": tyre_start,
                    "TyreLifeEnd": tyre_end,
                    "MeanLapTime_Sec": np.nan,
                    "MedianLapTime_Sec": np.nan,
                    "BestLapTime_Sec": np.nan,
                    "BestLapNumber": None,
                    "WorstLapTime_Sec": np.nan,
                    "WorstLapNumber": None,
                    "StdLapTime_Sec": np.nan,
                    "CV_Pct": np.nan,
                    "First3Mean_Sec": np.nan,
                    "Last3Mean_Sec": np.nan,
                    "Last3_Minus_First3_Sec": np.nan,
                    "LinearTrend_SecPerLap": np.nan,
                    "FuelCorrectedTrend_SecPerLap": np.nan,
                    "DiffToRefMedian_Sec": np.nan,
                    "SC_VSC_LapCount": sc_vsc_count,
                    "SampleNote": "Yetersiz örneklem (n=0 geçerli temiz tur)",
                }
            )
            continue

        times = valid_c["_lap_sec"]
        mean_sec = float(times.mean())
        median_sec = float(times.median())

        best_idx = times.idxmin()
        worst_idx = times.idxmax()
        best_sec = float(times.loc[best_idx])
        worst_sec = float(times.loc[worst_idx])

        best_lap_no = (
            int(valid_c.loc[best_idx, "LapNumber"])
            if "LapNumber" in valid_c.columns and pd.notna(valid_c.loc[best_idx, "LapNumber"])
            else None
        )
        worst_lap_no = (
            int(valid_c.loc[worst_idx, "LapNumber"])
            if "LapNumber" in valid_c.columns and pd.notna(valid_c.loc[worst_idx, "LapNumber"])
            else None
        )

        std_sec = float(times.std(ddof=1)) if n_valid >= 2 else np.nan
        cv_pct = float((std_sec / mean_sec) * 100.0) if (n_valid >= 2 and mean_sec > 0 and pd.notna(std_sec)) else np.nan

        # İlk 3 ve Son 3 turun ortalaması ve farkı
        if n_valid >= 3:
            first3_mean = float(times.iloc[:3].mean())
            last3_mean = float(times.iloc[-3:].mean())
            last3_minus_first3 = float(last3_mean - first3_mean)
        else:
            first3_mean = np.nan
            last3_mean = np.nan
            last3_minus_first3 = np.nan

        # Doğrusal trend (saniye / tur) ve yakıt etkisinden arındırılmış aşınma eğimi
        linear_slope = np.nan
        fuel_corrected_slope = np.nan
        if n_valid >= 3:
            x_seq = np.arange(1, n_valid + 1, dtype=float)
            y_seq = times.values.astype(float)
            slope, _ = np.polyfit(x_seq, y_seq, 1)
            linear_slope = float(slope)
            if pd.notna(fuel_effect_sec_per_lap):
                fuel_corrected_slope = float(slope + fuel_effect_sec_per_lap)

        # Örneklem notu
        notes: List[str] = []
        if n_valid < 3:
            notes.append(f"Düşük örneklem (n={n_valid} < 3): İlk 3/Son 3 tur farkı hesaplanamadı.")
        elif n_valid < 6:
            notes.append(f"Kısa stint (n={n_valid}): İlk 3 ve son 3 tur pencereleri kısmen örtüşüyor.")
        elif n_valid < min_valid_laps_warning:
            notes.append(f"Sınırlı örneklem (n={n_valid}).")
        else:
            notes.append(f"Yeterli örneklem (n={n_valid}).")

        if sc_vsc_count > 0:
            notes.append(f"Bu stintte {sc_vsc_count} tur SC/VSC/Kırmızı Bayrak altında geçti.")

        rows.append(
            {
                "Driver": driver_name,
                "Stint": stint_int,
                "Compound": compound_str,
                "StartLap": start_lap,
                "EndLap": end_lap,
                "TotalLaps": total_laps,
                "ValidLapCount": n_valid,
                "TyreLifeStart": tyre_start,
                "TyreLifeEnd": tyre_end,
                "MeanLapTime_Sec": mean_sec,
                "MedianLapTime_Sec": median_sec,
                "BestLapTime_Sec": best_sec,
                "BestLapNumber": best_lap_no,
                "WorstLapTime_Sec": worst_sec,
                "WorstLapNumber": worst_lap_no,
                "StdLapTime_Sec": std_sec,
                "CV_Pct": cv_pct,
                "First3Mean_Sec": first3_mean,
                "Last3Mean_Sec": last3_mean,
                "Last3_Minus_First3_Sec": last3_minus_first3,
                "LinearTrend_SecPerLap": linear_slope,
                "FuelCorrectedTrend_SecPerLap": fuel_corrected_slope,
                "DiffToRefMedian_Sec": np.nan,
                "SC_VSC_LapCount": sc_vsc_count,
                "SampleNote": " ".join(notes),
            }
        )

    result_df = pd.DataFrame(rows, columns=columns_order)

    # Referans sürücü stintleriyle medyan farkı hesapla
    if ref_stints_df is not None and isinstance(ref_stints_df, pd.DataFrame) and not ref_stints_df.empty:
        ref_map = (
            ref_stints_df.dropna(subset=["Stint", "MedianLapTime_Sec"])
            .set_index("Stint")["MedianLapTime_Sec"]
            .to_dict()
        )
        for idx in result_df.index:
            st_no = result_df.loc[idx, "Stint"]
            med_val = result_df.loc[idx, "MedianLapTime_Sec"]
            if st_no in ref_map and pd.notna(med_val) and pd.notna(ref_map[st_no]):
                result_df.loc[idx, "DiffToRefMedian_Sec"] = float(med_val - ref_map[st_no])
    elif not result_df.empty:
        # Referans verilmediyse kendi 1. stintine göre değil, 0.0 (kendisi referans ise) bırakılabilir
        pass

    return result_df


def compare_matching_stints(
    stints_d1: pd.DataFrame,
    stints_d2: pd.DataFrame,
    driver1_name: str,
    driver2_name: str,
    max_start_age_diff: int = 6,
) -> pd.DataFrame:
    """
    İki sürücünün aynı lastik hamuru (Compound) ve benzer başlangıç lastik yaşına
    (TyreLifeStart) sahip stintlerini eşleştirerek karşılaştırır.
    """
    if (
        stints_d1 is None
        or stints_d2 is None
        or stints_d1.empty
        or stints_d2.empty
    ):
        return pd.DataFrame()

    matches: List[Dict[str, Any]] = []
    used_d2_stints = set()

    for _, row1 in stints_d1.iterrows():
        comp1 = str(row1.get("Compound", "UNKNOWN"))
        if comp1 in ("UNKNOWN", "-", "nan") or pd.isna(row1.get("MedianLapTime_Sec")):
            continue
        age1 = row1.get("TyreLifeStart")
        stint1 = int(row1["Stint"])

        # Aynı hamura sahip adayları bul
        candidates = stints_d2[
            (stints_d2["Compound"].astype(str) == comp1)
            & (stints_d2["MedianLapTime_Sec"].notna())
            & (~stints_d2["Stint"].isin(used_d2_stints))
        ].copy()

        if candidates.empty:
            continue

        # Önce aynı stint numarası ve benzer yaşa öncelik ver
        best_cand = None
        best_score = 999999.0
        for _, row2 in candidates.iterrows():
            age2 = row2.get("TyreLifeStart")
            age_diff = abs(int(age1) - int(age2)) if (pd.notna(age1) and pd.notna(age2)) else 0
            if age_diff > max_start_age_diff:
                continue
            stint_diff = abs(stint1 - int(row2["Stint"]))
            lap_overlap_dist = abs(
                (row1.get("StartLap") or 0) - (row2.get("StartLap") or 0)
            )
            score = stint_diff * 10 + age_diff * 2 + lap_overlap_dist * 0.5
            if score < best_score:
                best_score = score
                best_cand = row2

        if best_cand is not None:
            stint2 = int(best_cand["Stint"])
            used_d2_stints.add(stint2)
            med1 = float(row1["MedianLapTime_Sec"])
            med2 = float(best_cand["MedianLapTime_Sec"])
            mean1 = float(row1["MeanLapTime_Sec"])
            mean2 = float(best_cand["MeanLapTime_Sec"])
            deg1 = row1.get("Last3_Minus_First3_Sec", np.nan)
            deg2 = best_cand.get("Last3_Minus_First3_Sec", np.nan)
            fuel_trend1 = row1.get("FuelCorrectedTrend_SecPerLap", np.nan)
            fuel_trend2 = best_cand.get("FuelCorrectedTrend_SecPerLap", np.nan)

            matches.append(
                {
                    "Lastik Hamuru": comp1,
                    f"{driver1_name} Stint": f"Stint {stint1} (Tur {row1.get('StartLap')}-{row1.get('EndLap')}, n={row1.get('ValidLapCount')})",
                    f"{driver2_name} Stint": f"Stint {stint2} (Tur {best_cand.get('StartLap')}-{best_cand.get('EndLap')}, n={best_cand.get('ValidLapCount')})",
                    f"{driver1_name} Lastik Yaşı": f"{row1.get('TyreLifeStart')}-{row1.get('TyreLifeEnd')}",
                    f"{driver2_name} Lastik Yaşı": f"{best_cand.get('TyreLifeStart')}-{best_cand.get('TyreLifeEnd')}",
                    f"{driver1_name} Medyan Tur Süresi (sn)": med1,
                    f"{driver2_name} Medyan Tur Süresi (sn)": med2,
                    f"Medyan Tur Süresi Farkı ({driver1_name} - {driver2_name}) (sn)": med1 - med2,
                    f"{driver1_name} Ortalama Tur Süresi (sn)": mean1,
                    f"{driver2_name} Ortalama Tur Süresi (sn)": mean2,
                    f"Ortalama Tur Süresi Farkı ({driver1_name} - {driver2_name}) (sn)": mean1 - mean2,
                    f"{driver1_name} Performans Kaybı: Son 3 - İlk 3 (sn)": deg1,
                    f"{driver2_name} Performans Kaybı: Son 3 - İlk 3 (sn)": deg2,
                    f"{driver1_name} Yakıt Düzeltilmiş Aşınma Eğimi (sn/tur)": fuel_trend1,
                    f"{driver2_name} Yakıt Düzeltilmiş Aşınma Eğimi (sn/tur)": fuel_trend2,
                    f"Aşınma Eğimi Farkı ({driver1_name} - {driver2_name}) (sn/tur)": (fuel_trend1 - fuel_trend2) if (pd.notna(fuel_trend1) and pd.notna(fuel_trend2)) else np.nan,
                }
            )

    return pd.DataFrame(matches)


# ==============================================================================
# 8. SÜRÜCÜ FARK ANALİZİ (TUR BAZLI VE STINT BAZLI FARKLAR)
# ==============================================================================

def calculate_lap_by_lap_deltas(
    raw_laps_d1: pd.DataFrame,
    raw_laps_d2: pd.DataFrame,
    clean_laps_d1: Optional[pd.DataFrame] = None,
    clean_laps_d2: Optional[pd.DataFrame] = None,
    driver1_name: str = "Driver 1",
    driver2_name: str = "Driver 2",
) -> pd.DataFrame:
    """
    İki sürücünün ortak tur numaraları (LapNumber) üzerinden tur bazlı zaman farkını
    (Delta_Sec = Driver1_LapTime - Driver2_LapTime) ve kümülatif farkını hesaplar.

    Yorumlama kuralı:
    - Delta_Sec < 0 (Negatif): Driver 1 o turda daha hızlıdır (daha düşük tur süresi).
    - Delta_Sec > 0 (Pozitif): Driver 2 o turda daha hızlıdır (Driver 1 daha yavaş).
    """
    if (
        raw_laps_d1 is None
        or raw_laps_d2 is None
        or raw_laps_d1.empty
        or raw_laps_d2.empty
        or "LapNumber" not in raw_laps_d1.columns
        or "LapNumber" not in raw_laps_d2.columns
    ):
        return pd.DataFrame()

    d1 = raw_laps_d1.copy()
    d2 = raw_laps_d2.copy()

    d1["LapTime_Sec_D1"] = to_seconds_series(d1["LapTime"])
    d2["LapTime_Sec_D2"] = to_seconds_series(d2["LapTime"])

    clean_set_d1 = (
        set(clean_laps_d1["LapNumber"].dropna().astype(int).tolist())
        if isinstance(clean_laps_d1, pd.DataFrame) and not clean_laps_d1.empty and "LapNumber" in clean_laps_d1.columns
        else set()
    )
    clean_set_d2 = (
        set(clean_laps_d2["LapNumber"].dropna().astype(int).tolist())
        if isinstance(clean_laps_d2, pd.DataFrame) and not clean_laps_d2.empty and "LapNumber" in clean_laps_d2.columns
        else set()
    )

    cols_d1 = ["LapNumber", "LapTime_Sec_D1"]
    for extra in ("Stint", "Compound", "TyreLife", "TrackStatus"):
        if extra in d1.columns:
            cols_d1.append(extra)

    cols_d2 = ["LapNumber", "LapTime_Sec_D2"]
    for extra in ("Stint", "Compound", "TyreLife", "TrackStatus"):
        if extra in d2.columns:
            cols_d2.append(extra)

    merged = pd.merge(
        d1[cols_d1].drop_duplicates(subset=["LapNumber"]),
        d2[cols_d2].drop_duplicates(subset=["LapNumber"]),
        on="LapNumber",
        how="inner",
        suffixes=("_D1", "_D2"),
    ).sort_values("LapNumber").reset_index(drop=True)

    if merged.empty:
        return merged

    merged["Delta_Sec"] = merged["LapTime_Sec_D1"] - merged["LapTime_Sec_D2"]
    merged["Both_Clean"] = merged["LapNumber"].apply(
        lambda lno: (int(lno) in clean_set_d1) and (int(lno) in clean_set_d2) if pd.notna(lno) else False
    )
    merged["Clean_Delta_Sec"] = merged["Delta_Sec"].where(merged["Both_Clean"], np.nan)
    merged["Cumulative_Delta_All_Sec"] = merged["Delta_Sec"].fillna(0.0).cumsum()
    merged["Cumulative_Delta_Clean_Sec"] = merged["Clean_Delta_Sec"].fillna(0.0).cumsum()
    merged["Faster_Driver"] = np.where(
        merged["Delta_Sec"].isna(),
        "Veri Yok",
        np.where(
            merged["Delta_Sec"] < 0,
            driver1_name,
            np.where(merged["Delta_Sec"] > 0, driver2_name, "Eşit"),
        ),
    )

    return merged


# ==============================================================================
# 9. EN HIZLI TUR TELEMETRİ VE PİST DOMİNANSI (LAZY FASTEST LAP TELEMETRY)
# ==============================================================================

def get_fastest_lap_telemetry_comparison(
    session: Any,
    driver1: str,
    driver2: str,
    clean_laps_d1: Optional[pd.DataFrame] = None,
    clean_laps_d2: Optional[pd.DataFrame] = None,
    cache_dir: Optional[str] = None,
    brake_window_m: float = 20.0,
) -> Dict[str, Any]:
    """
    İki sürücünün en hızlı temiz turlarına (Fastest Lap) ait telemetri verilerini
    (Speed, Throttle, Brake, nGear, DRS, Distance, X, Y) tembel yükleme (lazy loading)
    ile çeker, ortak mesafe ekseninde hizalar ve mikro-sektör pist dominansı (track dominance)
    hesaplar.
    """
    from f1_data_loader import DEFAULT_CACHE_DIR, extract_driver_lap_telemetry
    from f1_telemetry import align_and_compare_telemetry

    c_dir = cache_dir or DEFAULT_CACHE_DIR

    laps_1 = clean_laps_d1 if (clean_laps_d1 is not None and not clean_laps_d1.empty) else (
        session.laps.pick_drivers(driver1).copy() if (session is not None and hasattr(session, "laps") and session.laps is not None) else pd.DataFrame()
    )
    laps_2 = clean_laps_d2 if (clean_laps_d2 is not None and not clean_laps_d2.empty) else (
        session.laps.pick_drivers(driver2).copy() if (session is not None and hasattr(session, "laps") and session.laps is not None) else pd.DataFrame()
    )

    raw_tel_1, lap_no_1, lap_sec_1, err1 = extract_driver_lap_telemetry(session, laps_1, "Fastest", cache_dir=c_dir)
    raw_tel_2, lap_no_2, lap_sec_2, err2 = extract_driver_lap_telemetry(session, laps_2, "Fastest", cache_dir=c_dir)

    tel_comp = align_and_compare_telemetry(
        raw_tel_1,
        raw_tel_2,
        driver1_name=driver1,
        driver2_name=driver2,
        window_m=brake_window_m,
        distance_step_m=2.0,
    )

    aligned = tel_comp.get("aligned_df", pd.DataFrame())
    has_spatial_coords = False
    if not aligned.empty and "X" in aligned.columns and "Y" in aligned.columns:
        if aligned["X"].notna().any() and aligned["Y"].notna().any():
            has_spatial_coords = True

    # Pist dominansı (Track Dominance) mikro-sektörlerini hesapla
    dominance_summary = {
        "driver1_faster_pct": 0.0,
        "driver2_faster_pct": 0.0,
        "equal_pct": 0.0,
        "driver1_faster_distance_m": 0.0,
        "driver2_faster_distance_m": 0.0,
        "total_distance_m": 0.0,
    }

    if not aligned.empty and "SpeedDelta_kmh" in aligned.columns:
        valid_spds = aligned.dropna(subset=["SpeedDelta_kmh"])
        n_pts = len(valid_spds)
        if n_pts > 0:
            d1_faster = int((valid_spds["SpeedDelta_kmh"] > 0).sum())
            d2_faster = int((valid_spds["SpeedDelta_kmh"] < 0).sum())
            eq = int((valid_spds["SpeedDelta_kmh"] == 0).sum())
            dominance_summary["driver1_faster_pct"] = float((d1_faster / n_pts) * 100.0)
            dominance_summary["driver2_faster_pct"] = float((d2_faster / n_pts) * 100.0)
            dominance_summary["equal_pct"] = float((eq / n_pts) * 100.0)
            tot_dist = float(valid_spds["Distance"].max() - valid_spds["Distance"].min()) if "Distance" in valid_spds.columns else 0.0
            dominance_summary["total_distance_m"] = tot_dist
            dominance_summary["driver1_faster_distance_m"] = float((d1_faster / n_pts) * tot_dist)
            dominance_summary["driver2_faster_distance_m"] = float((d2_faster / n_pts) * tot_dist)

    return {
        "driver1": driver1,
        "driver2": driver2,
        "lap_number_1": lap_no_1,
        "lap_number_2": lap_no_2,
        "lap_sec_1": lap_sec_1,
        "lap_sec_2": lap_sec_2,
        "raw_telemetry_1": raw_tel_1,
        "raw_telemetry_2": raw_tel_2,
        "error_1": err1,
        "error_2": err2,
        "tel_comp": tel_comp,
        "aligned_df": aligned,
        "has_spatial_coordinates": has_spatial_coords,
        "dominance_summary": dominance_summary,
        "summary_metrics": tel_comp.get("summary_metrics", {}),
    }


# ==============================================================================
# 10. MERKEZİ ANALİZ BORU HATTI (UNIFIED ANALYSIS PIPELINE)
# ==============================================================================

def run_full_race_analysis(
    session: Any,
    driver1: str = "NOR",
    driver2: str = "VER",
    ref_driver: Optional[str] = None,
    by_stint: bool = True,
    fuel_effect_sec_per_lap: float = 0.035,
    iqr_multiplier: float = 1.5,
    min_laps_for_iqr: int = 4,
    load_telemetry: bool = True,
    brake_window_m: float = 20.0,
    cache_dir: Optional[str] = None,
    raw_d1_override: Optional[pd.DataFrame] = None,
    raw_d2_override: Optional[pd.DataFrame] = None,
    raw_ref_override: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Çifte veri yüklemeyi ve tekrarlı döngüleri önlemek için tüm yarış analizlerini
    (Temiz tur filtreleme, Pace skoru, Tutarlılık, Welch t-Testi, Stint analizi ve
    yakıt düzeltilmiş aşınma eğimi, Eşdeğer stint karşılaştırması, Sektör analizi ve
    Z-skorları, Teorik optimal tur, Tur bazlı farklar ve En hızlı tur telemetri karşılaştırması)
    tek seferde çalıştıran merkezi boru hattı (unified pipeline).
    """
    from f1_data_loader import DEFAULT_CACHE_DIR
    c_dir = cache_dir or DEFAULT_CACHE_DIR

    # 1. Ham tur verilerini çıkar
    if session is not None and hasattr(session, "laps") and session.laps is not None and not session.laps.empty:
        all_laps = session.laps
    else:
        all_laps = pd.DataFrame()

    if raw_d1_override is not None and isinstance(raw_d1_override, pd.DataFrame):
        raw_d1 = raw_d1_override.copy()
    elif not all_laps.empty:
        raw_d1 = all_laps.pick_drivers(driver1).copy()
    else:
        raw_d1 = pd.DataFrame()

    if raw_d2_override is not None and isinstance(raw_d2_override, pd.DataFrame):
        raw_d2 = raw_d2_override.copy()
    elif not all_laps.empty:
        raw_d2 = all_laps.pick_drivers(driver2).copy()
    else:
        raw_d2 = pd.DataFrame()

    # 2. Temiz turları filtrele
    clean_d1, rep_d1 = filter_clean_laps(
        raw_d1,
        by_stint=by_stint,
        iqr_multiplier=iqr_multiplier,
        min_laps_for_iqr=min_laps_for_iqr,
        return_report=True,
    )
    clean_d2, rep_d2 = filter_clean_laps(
        raw_d2,
        by_stint=by_stint,
        iqr_multiplier=iqr_multiplier,
        min_laps_for_iqr=min_laps_for_iqr,
        return_report=True,
    )

    # 3. Referans tur havuzu (0.0 Z-score sorununu önlemek için birleşik havuz veya üçüncü sürücü)
    if raw_ref_override is not None and isinstance(raw_ref_override, pd.DataFrame) and not raw_ref_override.empty:
        raw_ref = raw_ref_override.copy()
        clean_ref, rep_ref = filter_clean_laps(
            raw_ref,
            by_stint=by_stint,
            iqr_multiplier=iqr_multiplier,
            min_laps_for_iqr=min_laps_for_iqr,
            return_report=True,
        )
        ref_name = ref_driver or "Referans"
    elif ref_driver and ref_driver not in (driver1, "COMBINED", "MEDIAN") and not all_laps.empty:
        raw_ref = all_laps.pick_drivers(ref_driver).copy()
        clean_ref, rep_ref = filter_clean_laps(
            raw_ref,
            by_stint=by_stint,
            iqr_multiplier=iqr_multiplier,
            min_laps_for_iqr=min_laps_for_iqr,
            return_report=True,
        )
        ref_name = ref_driver
    else:
        # İki sürücünün birleşik temiz tur havuzu
        clean_ref = pd.concat([clean_d1, clean_d2], ignore_index=True) if (
            (not clean_d1.empty) or (not clean_d2.empty)
        ) else pd.DataFrame()
        raw_ref = pd.concat([raw_d1, raw_d2], ignore_index=True) if (
            (not raw_d1.empty) or (not raw_d2.empty)
        ) else pd.DataFrame()
        rep_ref = None
        ref_name = ref_driver if (ref_driver and ref_driver not in ("COMBINED", "MEDIAN")) else f"{driver1}+{driver2} Havuzu"

    # 4. Yarışın en hızlı turu ve referans medyanı
    all_session_sec = to_seconds_series(all_laps["LapTime"]).dropna() if not all_laps.empty else pd.Series(dtype=float)
    fastest_lap_seconds = float(all_session_sec.min()) if not all_session_sec.empty else np.nan

    clean_ref_sec = to_seconds_series(clean_ref["LapTime"]).dropna() if not clean_ref.empty else pd.Series(dtype=float)
    ref_med_sec = float(clean_ref_sec.median()) if not clean_ref_sec.empty else np.nan

    # 5. Pace Skoru
    pace_d1 = calculate_pace_score(clean_d1, fastest_lap_seconds, ref_median_seconds=ref_med_sec, return_details=True)
    pace_d2 = calculate_pace_score(clean_d2, fastest_lap_seconds, ref_median_seconds=ref_med_sec, return_details=True)

    # 6. Tutarlılık Skoru
    cons_d1 = calculate_consistency_score(clean_d1, by_stint=by_stint, return_details=True)
    cons_d2 = calculate_consistency_score(clean_d2, by_stint=by_stint, return_details=True)

    # 7. Welch's t-Testi
    ttest_det = compare_drivers_ttest(clean_d1, clean_d2, return_details=True)

    # 8. Stint Analizi ve Yakıt Düzeltilmiş Aşınma Eğimi
    stints_ref = analyze_driver_stints(raw_ref, clean_ref, driver_name=ref_name, fuel_effect_sec_per_lap=fuel_effect_sec_per_lap)
    stints_d1 = analyze_driver_stints(raw_d1, clean_d1, driver_name=driver1, ref_stints_df=stints_ref, fuel_effect_sec_per_lap=fuel_effect_sec_per_lap)
    stints_d2 = analyze_driver_stints(raw_d2, clean_d2, driver_name=driver2, ref_stints_df=stints_ref, fuel_effect_sec_per_lap=fuel_effect_sec_per_lap)
    matching_stints = compare_matching_stints(stints_d1, stints_d2, driver1, driver2)

    # 9. Sektör Karşılaştırması ve Teorik Optimal Tur
    sector_comp = build_sector_comparison_table(clean_d1, clean_d2, clean_ref, driver1, driver2, ref_name)
    opt_d1 = calculate_theoretical_optimal(clean_d1, return_details=True)
    opt_d2 = calculate_theoretical_optimal(clean_d2, return_details=True)

    # 10. Tur Bazlı Zaman Farkları
    lap_deltas = calculate_lap_by_lap_deltas(raw_d1, raw_d2, clean_d1, clean_d2, driver1, driver2)

    # 11. Telemetri Karşılaştırması (Opsiyonel / Lazy)
    tel_comp_res = None
    if load_telemetry and session is not None:
        try:
            tel_comp_res = get_fastest_lap_telemetry_comparison(
                session=session,
                driver1=driver1,
                driver2=driver2,
                clean_laps_d1=clean_d1,
                clean_laps_d2=clean_d2,
                cache_dir=c_dir,
                brake_window_m=brake_window_m,
            )
        except Exception as exc:
            tel_comp_res = {
                "error": str(exc),
                "aligned_df": pd.DataFrame(),
                "has_spatial_coordinates": False,
                "dominance_summary": {},
            }

    return {
        "session": session,
        "driver1": driver1,
        "driver2": driver2,
        "ref_driver": ref_name,
        "raw_d1": raw_d1,
        "raw_d2": raw_d2,
        "clean_d1": clean_d1,
        "clean_d2": clean_d2,
        "clean_ref": clean_ref,
        "report_d1": rep_d1,
        "report_d2": rep_d2,
        "report_ref": rep_ref,
        "pace_d1": pace_d1,
        "pace_d2": pace_d2,
        "consistency_d1": cons_d1,
        "consistency_d2": cons_d2,
        "ttest": ttest_det,
        "stints_d1": stints_d1,
        "stints_d2": stints_d2,
        "stints_ref": stints_ref,
        "matching_stints": matching_stints,
        "sector_comparison": sector_comp,
        "optimal_d1": opt_d1,
        "optimal_d2": opt_d2,
        "lap_deltas": lap_deltas,
        "telemetry_comparison": tel_comp_res,
        "fastest_lap_seconds": fastest_lap_seconds,
        "ref_median_seconds": ref_med_sec,
    }


# ==============================================================================
# KOMUT SATIRI (CLI) TEST VE ÖRNEK ÇALIŞTIRMA
# ==============================================================================

def main() -> None:
    print("Veri indiriliyor / önbellekten okunuyor...")
    session = fastf1.get_session(2025, "Australian Grand Prix", "R")
    session.load(telemetry=False, weather=False, messages=False)

    driver1 = "NOR"
    driver2 = "VER"
    print(f"Tekil Veri Akışı (Unified Pipeline) ile analiz ediliyor ({driver1} vs {driver2})...")

    analysis = run_full_race_analysis(
        session=session,
        driver1=driver1,
        driver2=driver2,
        ref_driver=None,  # Otomatik birleşik havuz: 0.0 Z-skor sorununu önler
        by_stint=True,
        fuel_effect_sec_per_lap=0.035,
        load_telemetry=False,
    )

    pace_d1 = analysis["pace_d1"]
    pace_d2 = analysis["pace_d2"]
    cons_d1 = analysis["consistency_d1"]
    cons_d2 = analysis["consistency_d2"]
    rep_d1 = analysis["report_d1"]
    rep_d2 = analysis["report_d2"]
    ttest_det = analysis["ttest"]
    opt_d1 = analysis["optimal_d1"]
    opt_d2 = analysis["optimal_d2"]
    sec_comp = analysis["sector_comparison"]
    st_d1 = analysis["stints_d1"]
    st_d2 = analysis["stints_d2"]
    matching_st = analysis["matching_stints"]

    results = [
        {
            "Sürücü": driver1,
            "Toplam / Temiz Tur": f"{rep_d1['total_raw_laps']} / {rep_d1['clean_laps_count']}",
            "Medyan Tur Süresi (sn)": round(pace_d1["median_lap_sec"], 3),
            "Tur Temposu (%)": round(pace_d1["pace_score_vs_fastest_pct"], 3),
            "Tur Tutarlılığı — CV (%)": round(cons_d1["cv_pct"], 3),
            "Standart Sapma (sn)": round(cons_d1["std_sec"], 3),
            "En Hızlı Tur (sn)": round(opt_d1["actual_best_sec"], 3),
            "Teorik Optimal Tur (sn)": round(opt_d1["theoretical_optimal_sec"], 3),
        },
        {
            "Sürücü": driver2,
            "Toplam / Temiz Tur": f"{rep_d2['total_raw_laps']} / {rep_d2['clean_laps_count']}",
            "Medyan Tur Süresi (sn)": round(pace_d2["median_lap_sec"], 3),
            "Tur Temposu (%)": round(pace_d2["pace_score_vs_fastest_pct"], 3),
            "Tur Tutarlılığı — CV (%)": round(cons_d2["cv_pct"], 3),
            "Standart Sapma (sn)": round(cons_d2["std_sec"], 3),
            "En Hızlı Tur (sn)": round(opt_d2["actual_best_sec"], 3),
            "Teorik Optimal Tur (sn)": round(opt_d2["theoretical_optimal_sec"], 3),
        },
    ]

    print("\n--- 1. Sürücü Performans Metrikleri ---")
    print(pd.DataFrame(results).to_markdown(index=False))

    print("\n--- 2. Sektör Karşılaştırma Tablosu (Birleşik Havuz Z-Skorları Dahil) ---")
    print(sec_comp.to_markdown(index=False))

    print("\n--- 3. Teorik Optimal Tur ve Sektör Kaynakları ---")
    for d_code, o_det in [(driver1, opt_d1), (driver2, opt_d2)]:
        print(f"[{d_code}] En Hızlı Tur: {format_laptime_str(o_det['actual_best_sec'])} | Teorik Optimal: {format_laptime_str(o_det['theoretical_optimal_sec'])} | Potansiyel Kazanç: {o_det['development_potential_sec']:.3f} sn")
        for s_col, s_info in o_det.get("sectors", {}).items():
            print(f"  - {s_col}: {s_info.get('best_sector_sec', np.nan):.3f} sn (Tur {s_info.get('from_lap_number')}, Stint {s_info.get('from_stint')}, {s_info.get('from_compound')})")

    print("\n--- 4. Stint Analizi ve Yakıt Düzeltilmiş Aşınma Eğimi (Fuel-Corrected Stint Degradation) ---")
    stints_combined = pd.concat([st_d1, st_d2], ignore_index=True)
    stints_view_cols = [
        "Driver", "Stint", "Compound", "ValidLapCount", "MeanLapTime_Sec", "MedianLapTime_Sec",
        "LinearTrend_SecPerLap", "FuelCorrectedTrend_SecPerLap", "Last3_Minus_First3_Sec"
    ]
    avail_view_cols = [c for c in stints_view_cols if c in stints_combined.columns]
    print(stints_combined[avail_view_cols].to_markdown(index=False))

    if not matching_st.empty:
        print("\n--- 5. Eşdeğer Stint Karşılaştırması ---")
        print(matching_st.to_markdown(index=False))

    print("\n--- 6. İstatistiksel Karşılaştırma (Welch t-Testi) ---")
    print(f"Karşılaştırılan Sürücüler: {driver1} vs {driver2}")
    print(f"t-İstatistiği: {ttest_det.get('t_stat', np.nan):.4f} | p-Değeri: {ttest_det.get('p_value', np.nan):.5f}")
    if ttest_det.get("warning"):
        print(f"Uyarı: {ttest_det['warning']}")
    print(f"Metodolojik Not: {TTEST_METHODOLOGY_WARNING}")
    print(f"Stint Notu: {STINT_DEGRADATION_METHODOLOGY_WARNING}")

    # 7. Analizleri HTML olarak oluştur ve localhost üzerinde yayınla
    from f1_html_dashboard import serve_html_on_localhost
    serve_html_on_localhost(
        port=8050,
        open_browser=True,
        block=True,
        initial_year=2025,
        initial_event="Australian Grand Prix",
        initial_driver1=driver1,
        initial_driver2=driver2,
    )


if __name__ == "__main__":
    main()
