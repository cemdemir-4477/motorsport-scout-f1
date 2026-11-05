"""
F1 Performance Analysis & Telemetry System — Fren İşleme ve Telemetri Karşılaştırma Modülü (f1_telemetry.py)
=============================================================================================================
Bu modül FastF1 telemetri verileri üzerinde:
1. Boolean (0/1) yapıdaki ham Brake kanalını bozmadan koruma (BrakeRaw),
2. Yapay fren basıncı üretmeden mesafe penceresi (10m, 20m, 30m) bazlı:
   - BrakeActivity (%): Mesafe penceresindeki fren aktif örneklerinin oranı,
   - SmoothedBrakeActivity (%): Mesafe çekirdekli Gaussian yumuşatılmış fren aktivitesi,
   - BrakeDuration (sn): Frenin aktif kaldığı toplam süre,
   - BrakeEvents (DataFrame): Frenin devreye girdiği ve çıktığı bölgeler,
   - BrakingDistance (m): Her fren olayının başladığı-bittiği mesafe aralığı ve toplam fren mesafesi,
3. İki sürücünün telemetrisini ortak Distance (m) eksenine hizalama:
   - Sürekli (continuous) kanallar (Speed, Throttle, RPM, BrakeActivity, SmoothedBrakeActivity) için doğrusal interpolasyon,
   - Kesikli (discrete) kanallar (BrakeRaw, nGear, DRS) için basamaklı / en yakın komşu (step/discrete) hizalama,
4. Hız farkı (SpeedDelta) ve Gaz / Fren / Coasting sürüş bölgesi karşılaştırma
işlemlerini yürütür.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BRAKE_ACTIVITY_DISCLAIMER = (
    "Metodolojik Uyarı: FastF1 'Brake' kanalı ikili (0=Kapalı, 1=Açık) bir sensör sinyali sunar; "
    "hidrolik fren basıncı (bar) veya doğrudan yavaşlama ivmesi ölçümü içermez. Burada gösterilen Fren Aktivitesi "
    "('BrakeActivity (%)' ve 'SmoothedBrakeActivity (%)') değerleri fren basıncı DEĞİLDİR; seçilen mesafe penceresi "
    "(10m, 20m, 30m) içindeki frenin aktif olduğu örneklerin yerel oranını (%) ifade eder."
)

CONTINUOUS_TELEMETRY_CHANNELS = (
    "Speed",
    "Throttle",
    "RPM",
    "BrakeActivity",
    "SmoothedBrakeActivity",
    "X",
    "Y",
)

DISCRETE_TELEMETRY_CHANNELS = (
    "Brake",
    "BrakeRaw",
    "nGear",
    "DRS",
    "DRS_Active",
)


# ==============================================================================
# 1. YARDIMCI ZAMAN VE MESAFE ÇIKARIMI
# ==============================================================================

def _extract_time_seconds(df: pd.DataFrame) -> pd.Series:
    """Telemetri DataFrame'inden saniye cinsinden göreli zaman dizisini çıkarır."""
    for col in ("Time", "SessionTime"):
        if col in df.columns:
            s = df[col]
            if pd.api.types.is_timedelta64_dtype(s):
                sec = s.dt.total_seconds().astype(float)
                if sec.notna().any():
                    return sec - float(sec.dropna().iloc[0])
            elif pd.api.types.is_numeric_dtype(s):
                sec = pd.to_numeric(s, errors="coerce").astype(float)
                if sec.notna().any():
                    return sec - float(sec.dropna().iloc[0])
    # Eğer Time sütunu yoksa אך Distance ve Speed varsa dt = dDistance / (Speed / 3.6) ile tahmin et
    if "Distance" in df.columns and "Speed" in df.columns:
        dist = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
        spd_ms = pd.to_numeric(df["Speed"], errors="coerce").astype(float) / 3.6
        dd = dist.diff().fillna(0.0).clip(lower=0.0)
        safe_spd = spd_ms.where(spd_ms > 1.0, np.nan)
        dt = (dd / safe_spd).fillna(0.0)
        return dt.cumsum()

    return pd.Series(np.nan, index=df.index, dtype=float)


def _normalize_raw_brake_series(brake_series: pd.Series) -> pd.Series:
    """
    FastF1 Brake kanalını ham 0/1 (int) değerlerine dönüştürür.
    NaN değerleri 0 yapmadan önce geçerli indekslerde güvenli dönüşüm sağlar.
    """
    if brake_series is None or brake_series.empty:
        return pd.Series(dtype=float)

    s = brake_series.copy()
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    num = pd.to_numeric(s, errors="coerce")
    # FastF1'de Brake genellikle True/False veya 0/1 (bazen 0/100) olarak gelir
    # > 0 olan aktif fren örneklerini 1, <= 0 olanları 0 olarak kodla (NaN korunur)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = num.notna()
    out.loc[valid] = (num.loc[valid] > 0).astype(int)
    return out


# ==============================================================================
# 2. FREN TELEMETRİSİ İŞLEME MODÜLÜ (BÖLÜM 3)
# ==============================================================================

def process_brake_telemetry(
    telemetry_df: pd.DataFrame,
    window_m: float = 20.0,
) -> Dict[str, Any]:
    """
    FastF1 telemetri verisindeki Boolean Brake kanalını işleyerek türetilmiş
    fren metriklerini ve olay tablosunu üretir. Ham DataFrame'i kesinlikle değiştirmez.

    Üretilen Değişkenler (processed_df içinde):
    - BrakeRaw: Ham Boolean fren durumu (0 veya 1)
    - Brake: Orijinal ham fren sütunu (değiştirilmeden korunur)
    - BrakeActivity: Mesafe penceresi (window_m) içindeki fren aktif örneklerinin oranı (%0 - %100)
    - SmoothedBrakeActivity: Mesafe bazlı Gaussian çekirdekle yumuşatılmış fren aktivitesi (%0 - %100)

    Özet Metrikler:
    - brake_duration_sec: Frenin tur boyunca aktif kaldığı toplam süre (saniye)
    - total_braking_distance_m: Frenin tur boyunca aktif kaldığı toplam mesafe (metre)
    - brake_events_df: Her frenleme olayının başlangıç/bitiş mesafesi, mesafe aralığı (BrakingDistance),
      süresi (BrakeDuration) ve hız değişimi
    - brake_events_count: Toplam frenleme olayı sayısı
    - window_m: Kullanılan mesafe penceresi (m)
    - sample_size: Geçerli telemetri örneklem sayısı (n)
    - disclaimer: Metodolojik uyarı metni
    """
    if window_m <= 0:
        raise ValueError("window_m (fren yumuşatma mesafe penceresi) pozitif bir değer olmalıdır.")

    empty_result: Dict[str, Any] = {
        "processed_df": pd.DataFrame(),
        "brake_events_df": pd.DataFrame(),
        "brake_duration_sec": np.nan,
        "total_braking_distance_m": np.nan,
        "brake_sample_ratio_pct": np.nan,
        "brake_events_count": 0,
        "window_m": float(window_m),
        "sample_size": 0,
        "is_boolean_channel": True,
        "disclaimer": BRAKE_ACTIVITY_DISCLAIMER,
    }

    if telemetry_df is None or not isinstance(telemetry_df, pd.DataFrame) or telemetry_df.empty:
        return empty_result

    if "Distance" not in telemetry_df.columns or "Brake" not in telemetry_df.columns:
        return empty_result

    # Ham veriyi korumak için kopya oluştur
    df = telemetry_df.copy()
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    df = df.dropna(subset=["Distance"]).sort_values("Distance").reset_index(drop=True)
    if df.empty:
        return empty_result

    df["_Time_Sec"] = _extract_time_seconds(df)
    raw_brake = _normalize_raw_brake_series(df["Brake"])
    df["BrakeRaw"] = raw_brake

    valid_mask = df["BrakeRaw"].notna()
    n_samples = int(valid_mask.sum())
    if n_samples == 0:
        return empty_result

    distances = df["Distance"].values.astype(float)
    brake_vals = df["BrakeRaw"].fillna(0.0).values.astype(float)
    half_w = float(window_m) / 2.0

    # 1. BrakeActivity (%): [d_i - W/2, d_i + W/2] mesafe penceresindeki fren aktif örneklerinin oranı
    left_idx = np.searchsorted(distances, distances - half_w, side="left")
    right_idx = np.searchsorted(distances, distances + half_w, side="right")

    # Kümülatif toplam ile O(N) sürede pencere ortalaması hesapla
    cumsum_brake = np.concatenate(([0.0], np.cumsum(brake_vals)))
    counts = np.maximum(1, right_idx - left_idx)
    window_sums = cumsum_brake[right_idx] - cumsum_brake[left_idx]
    brake_activity_pct = (window_sums / counts) * 100.0

    # 2. SmoothedBrakeActivity (%): Mesafe bazlı Gaussian smoothing (sigma = window_m / 2.5)
    sigma_m = max(1.0, float(window_m) / 2.5)
    smoothed_activity_pct = np.zeros_like(brake_activity_pct, dtype=float)
    radius_m = 3.0 * sigma_m
    g_left = np.searchsorted(distances, distances - radius_m, side="left")
    g_right = np.searchsorted(distances, distances + radius_m, side="right")

    for i in range(len(distances)):
        s_i, e_i = g_left[i], g_right[i]
        if e_i <= s_i:
            smoothed_activity_pct[i] = brake_activity_pct[i]
        else:
            d_local = distances[s_i:e_i] - distances[i]
            weights = np.exp(-0.5 * (d_local / sigma_m) ** 2)
            w_sum = np.sum(weights)
            if w_sum > 0:
                smoothed_activity_pct[i] = float(np.sum(weights * brake_activity_pct[s_i:e_i]) / w_sum)
            else:
                smoothed_activity_pct[i] = brake_activity_pct[i]

    df["BrakeActivity"] = np.clip(brake_activity_pct, 0.0, 100.0)
    df["SmoothedBrakeActivity"] = np.clip(smoothed_activity_pct, 0.0, 100.0)

    # 3. BrakeEvents, BrakeDuration ve BrakingDistance çıkarımı
    events_df, total_duration_sec, total_distance_m = extract_brake_events(df)

    sample_ratio_pct = float(np.mean(brake_vals[valid_mask.values]) * 100.0)

    return {
        "processed_df": df,
        "brake_events_df": events_df,
        "brake_duration_sec": total_duration_sec,
        "total_braking_distance_m": total_distance_m,
        "brake_sample_ratio_pct": sample_ratio_pct,
        "brake_events_count": int(len(events_df)),
        "window_m": float(window_m),
        "sample_size": n_samples,
        "is_boolean_channel": True,
        "disclaimer": BRAKE_ACTIVITY_DISCLAIMER,
    }


def extract_brake_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
    """
    İşlenmiş telemetri DataFrame'inden frenin devreye girdiği (0 -> 1) ve
    devreden çıktığı (1 -> 0) her frenleme olayını (BrakeEvents), toplam frenleme
    süresini (BrakeDuration) ve toplam frenleme mesafesini (BrakingDistance) çıkarır.
    """
    event_cols = [
        "EventNo",
        "StartDistance_m",
        "EndDistance_m",
        "BrakingDistanceRange",
        "BrakingDistance_m",
        "BrakeDuration_sec",
        "EntrySpeed_kmh",
        "MinSpeed_kmh",
        "SpeedDrop_kmh",
        "MinGear",
        "SampleCount",
    ]

    if df is None or df.empty or "Distance" not in df.columns:
        return pd.DataFrame(columns=event_cols), 0.0, 0.0

    brake_col = "BrakeRaw" if "BrakeRaw" in df.columns else "Brake"
    if brake_col not in df.columns:
        return pd.DataFrame(columns=event_cols), 0.0, 0.0

    dist = df["Distance"].values.astype(float)
    time_sec = (
        df["_Time_Sec"].values.astype(float)
        if "_Time_Sec" in df.columns
        else _extract_time_seconds(df).values.astype(float)
    )
    brake_active = (_normalize_raw_brake_series(df[brake_col]).fillna(0.0).values > 0.5)
    speed = (
        pd.to_numeric(df["Speed"], errors="coerce").values.astype(float)
        if "Speed" in df.columns
        else np.full(len(df), np.nan)
    )
    gear = (
        pd.to_numeric(df["nGear"], errors="coerce").values.astype(float)
        if "nGear" in df.columns
        else np.full(len(df), np.nan)
    )

    n = len(df)
    events: List[Dict[str, Any]] = []
    total_dur = 0.0
    total_dist = 0.0

    i = 0
    event_no = 0
    while i < n:
        if not brake_active[i]:
            i += 1
            continue

        # Fren olayı başlangıcı: i
        start_idx = i
        while i < n and brake_active[i]:
            i += 1
        end_active_idx = i - 1
        # Frenin bırakıldığı ilk örnek (varsa) veya son aktif örnek
        release_idx = min(i, n - 1)

        d_start = float(dist[start_idx])
        # Eğer birden fazla örnek sürdüyse son aktif örnek ile bırakma örneği ortası, tek örnekse bir sonraki örneğe kadar olan adım
        if end_active_idx > start_idx:
            d_end = float(dist[end_active_idx])
            if release_idx > end_active_idx:
                d_end = float(0.5 * (dist[end_active_idx] + dist[release_idx]))
        else:
            if release_idx > start_idx:
                d_end = float(0.5 * (dist[start_idx] + dist[release_idx]))
            elif start_idx > 0:
                d_end = float(d_start + 0.5 * (d_start - dist[start_idx - 1]))
            else:
                d_end = d_start

        b_dist = max(0.0, d_end - d_start)

        # Süre hesabı
        t_start = time_sec[start_idx]
        if end_active_idx > start_idx:
            t_end = time_sec[end_active_idx]
            if release_idx > end_active_idx and pd.notna(time_sec[release_idx]) and pd.notna(t_end):
                t_end = 0.5 * (t_end + time_sec[release_idx])
        else:
            if release_idx > start_idx and pd.notna(time_sec[release_idx]) and pd.notna(t_start):
                t_end = 0.5 * (t_start + time_sec[release_idx])
            else:
                t_end = t_start

        b_dur = float(max(0.0, t_end - t_start)) if (pd.notna(t_start) and pd.notna(t_end)) else np.nan

        seg_speeds = speed[start_idx : release_idx + 1]
        valid_speeds = seg_speeds[~np.isnan(seg_speeds)]
        entry_spd = float(valid_speeds[0]) if len(valid_speeds) > 0 else np.nan
        min_spd = float(np.min(valid_speeds)) if len(valid_speeds) > 0 else np.nan
        spd_drop = float(entry_spd - min_spd) if (pd.notna(entry_spd) and pd.notna(min_spd)) else np.nan

        seg_gears = gear[start_idx : release_idx + 1]
        valid_gears = seg_gears[~np.isnan(seg_gears)]
        min_gear_val = int(np.min(valid_gears)) if len(valid_gears) > 0 else None

        event_no += 1
        total_dist += b_dist
        if pd.notna(b_dur):
            total_dur += b_dur

        events.append(
            {
                "EventNo": event_no,
                "StartDistance_m": round(d_start, 1),
                "EndDistance_m": round(d_end, 1),
                "BrakingDistanceRange": f"{d_start:.0f}m – {d_end:.0f}m",
                "BrakingDistance_m": round(b_dist, 1),
                "BrakeDuration_sec": round(b_dur, 3) if pd.notna(b_dur) else np.nan,
                "EntrySpeed_kmh": round(entry_spd, 1) if pd.notna(entry_spd) else np.nan,
                "MinSpeed_kmh": round(min_spd, 1) if pd.notna(min_spd) else np.nan,
                "SpeedDrop_kmh": round(spd_drop, 1) if pd.notna(spd_drop) else np.nan,
                "MinGear": min_gear_val,
                "SampleCount": int(end_active_idx - start_idx + 1),
            }
        )

    events_df = pd.DataFrame(events, columns=event_cols)
    return events_df, round(float(total_dur), 3), round(float(total_dist), 1)


# ==============================================================================
# 3. ORTAK MESAFE EKSENİNDE TELEMETRİ HİZALAMA VE KARŞILAŞTIRMA (BÖLÜM 4)
# ==============================================================================

def _interpolate_continuous(
    target_dist: np.ndarray,
    src_dist: np.ndarray,
    src_vals: np.ndarray,
) -> np.ndarray:
    """Sürekli (continuous) telemetri kanalları için NaN-güvenli doğrusal interpolasyon."""
    mask = (~np.isnan(src_dist)) & (~np.isnan(src_vals))
    if np.sum(mask) < 2:
        return np.full_like(target_dist, np.nan, dtype=float)
    d_clean = src_dist[mask]
    v_clean = src_vals[mask]
    # Tekrarlı mesafe noktalarını tekilleştir
    unique_d, unique_idx = np.unique(d_clean, return_index=True)
    if len(unique_d) < 2:
        return np.full_like(target_dist, np.nan, dtype=float)
    return np.interp(target_dist, unique_d, v_clean[unique_idx], left=np.nan, right=np.nan)


def _resample_discrete(
    target_dist: np.ndarray,
    src_dist: np.ndarray,
    src_vals: np.ndarray,
) -> np.ndarray:
    """
    Kesikli (discrete) telemetri kanalları (BrakeRaw, nGear, DRS) için
    ara değer uydurmadan (sürekli interpolasyon yapmadan) basamaklı / önceki durum
    (previous-step / zero-order hold) örneklemesi yapar.
    """
    mask = (~np.isnan(src_dist)) & (~np.isnan(src_vals))
    if np.sum(mask) == 0:
        return np.full_like(target_dist, np.nan, dtype=float)
    d_clean = src_dist[mask]
    v_clean = src_vals[mask]
    unique_d, unique_idx = np.unique(d_clean, return_index=True)
    u_vals = v_clean[unique_idx]

    # searchsorted ile her target_dist için kendisinden küçük veya eşit en son örneği al (zero-order hold)
    idx = np.searchsorted(unique_d, target_dist, side="right") - 1
    out = np.full_like(target_dist, np.nan, dtype=float)
    valid = (idx >= 0) & (target_dist <= unique_d[-1])
    out[valid] = u_vals[idx[valid]]
    return out


def _decode_drs_active(drs_series: pd.Series) -> pd.Series:
    """
    FastF1 DRS kanal kodlarından DRS'nin açık (aktif) olup olmadığını (0 veya 1)
    kesikli olarak belirler. (FastF1'de 10, 12, 14 = DRS Açık; veya 0/1 ise 1 = Açık).
    """
    num = pd.to_numeric(drs_series, errors="coerce")
    out = pd.Series(np.nan, index=drs_series.index, dtype=float)
    valid = num.notna()
    if not valid.any():
        return out
    vals = num.loc[valid]
    if vals.max() <= 1.0:
        out.loc[valid] = (vals > 0).astype(int)
    else:
        out.loc[valid] = vals.isin([10, 12, 14]).astype(int)
    return out


def align_and_compare_telemetry(
    raw_tel_d1: pd.DataFrame,
    raw_tel_d2: pd.DataFrame,
    driver1_name: str = "Driver 1",
    driver2_name: str = "Driver 2",
    window_m: float = 20.0,
    distance_step_m: float = 2.0,
) -> Dict[str, Any]:
    """
    İki sürücünün telemetri verilerini işler, ham verileri ayrı tutar ve
    tüm kanalları ortak bir Distance (m) ekseninde hizalar.

    Sürekli Kanallar (Doğrusal İnterpolasyon):
    - Speed (km/h)
    - Throttle (%)
    - RPM
    - BrakeActivity (%)
    - SmoothedBrakeActivity (%)

    Kesikli Kanallar (Zero-Order Hold / Discrete Step — Sürekli İnterpolasyon Yapılmaz):
    - BrakeRaw (0/1 Boolean)
    - nGear (Vites 1..8)
    - DRS (Ham durum kodu) ve DRS_Active (0/1)
    """
    b1 = process_brake_telemetry(raw_tel_d1, window_m=window_m)
    b2 = process_brake_telemetry(raw_tel_d2, window_m=window_m)

    df1 = b1["processed_df"]
    df2 = b2["processed_df"]

    if df1.empty or df2.empty:
        return {
            "aligned_df": pd.DataFrame(),
            "driver1_processed_df": df1,
            "driver2_processed_df": df2,
            "driver1_brake": b1,
            "driver2_brake": b2,
            "zone_comparison_df": pd.DataFrame(),
            "summary_metrics": {},
            "available_channels": [],
            "disclaimer": BRAKE_ACTIVITY_DISCLAIMER,
        }

    if "DRS" in df1.columns:
        df1["DRS_Active"] = _decode_drs_active(df1["DRS"])
    if "DRS" in df2.columns:
        df2["DRS_Active"] = _decode_drs_active(df2["DRS"])

    min_d = max(float(df1["Distance"].min()), float(df2["Distance"].min()), 0.0)
    max_d = min(float(df1["Distance"].max()), float(df2["Distance"].max()))

    if max_d <= min_d or not np.isfinite(max_d):
        return {
            "aligned_df": pd.DataFrame(),
            "driver1_processed_df": df1,
            "driver2_processed_df": df2,
            "driver1_brake": b1,
            "driver2_brake": b2,
            "zone_comparison_df": pd.DataFrame(),
            "summary_metrics": {},
            "available_channels": [],
            "disclaimer": BRAKE_ACTIVITY_DISCLAIMER,
        }

    step = max(0.5, float(distance_step_m))
    common_dist = np.arange(min_d, max_d + step * 0.5, step, dtype=float)
    aligned = pd.DataFrame({"Distance": common_dist})

    d1_dist = df1["Distance"].values.astype(float)
    d2_dist = df2["Distance"].values.astype(float)

    available_channels: List[str] = []

    # 1. Sürekli kanalları hizala
    for ch in CONTINUOUS_TELEMETRY_CHANNELS:
        has_c1 = ch in df1.columns and df1[ch].notna().any()
        has_c2 = ch in df2.columns and df2[ch].notna().any()
        if has_c1 or has_c2:
            available_channels.append(ch)
            aligned[f"{ch}_D1"] = (
                _interpolate_continuous(common_dist, d1_dist, pd.to_numeric(df1[ch], errors="coerce").values.astype(float))
                if has_c1
                else np.nan
            )
            aligned[f"{ch}_D2"] = (
                _interpolate_continuous(common_dist, d2_dist, pd.to_numeric(df2[ch], errors="coerce").values.astype(float))
                if has_c2
                else np.nan
            )

    # X ve Y pist koordinatlarını ortak pist çizgisi olarak birleştir
    if "X_D1" in aligned.columns or "X_D2" in aligned.columns:
        if "X_D1" in aligned.columns and "X_D2" in aligned.columns:
            aligned["X"] = aligned["X_D1"].combine_first(aligned["X_D2"])
        elif "X_D1" in aligned.columns:
            aligned["X"] = aligned["X_D1"]
        else:
            aligned["X"] = aligned["X_D2"]

    if "Y_D1" in aligned.columns or "Y_D2" in aligned.columns:
        if "Y_D1" in aligned.columns and "Y_D2" in aligned.columns:
            aligned["Y"] = aligned["Y_D1"].combine_first(aligned["Y_D2"])
        elif "Y_D1" in aligned.columns:
            aligned["Y"] = aligned["Y_D1"]
        else:
            aligned["Y"] = aligned["Y_D2"]

    # 2. Kesikli (Discrete) kanalları sürekli interpolasyon YAPMADAN hizala
    for ch in ("BrakeRaw", "nGear", "DRS", "DRS_Active"):
        has_c1 = ch in df1.columns and df1[ch].notna().any()
        has_c2 = ch in df2.columns and df2[ch].notna().any()
        if has_c1 or has_c2:
            if ch not in available_channels:
                available_channels.append(ch)
            aligned[f"{ch}_D1"] = (
                _resample_discrete(common_dist, d1_dist, pd.to_numeric(df1[ch], errors="coerce").values.astype(float))
                if has_c1
                else np.nan
            )
            aligned[f"{ch}_D2"] = (
                _resample_discrete(common_dist, d2_dist, pd.to_numeric(df2[ch], errors="coerce").values.astype(float))
                if has_c2
                else np.nan
            )

    # Hız farkı (SpeedDelta = Speed_D1 - Speed_D2)
    if "Speed_D1" in aligned.columns and "Speed_D2" in aligned.columns:
        aligned["SpeedDelta_kmh"] = aligned["Speed_D1"] - aligned["Speed_D2"]
        aligned["AbsSpeedDelta_kmh"] = aligned["SpeedDelta_kmh"].abs()
    else:
        aligned["SpeedDelta_kmh"] = np.nan
        aligned["AbsSpeedDelta_kmh"] = np.nan

    # Gaz / Fren kullanım bölgeleri sınıflandırması (Her mesafe adımında)
    for suffix, drv_label in (("D1", driver1_name), ("D2", driver2_name)):
        thr_col = f"Throttle_{suffix}"
        brk_col = f"BrakeRaw_{suffix}"
        if thr_col in aligned.columns and brk_col in aligned.columns:
            thr = aligned[thr_col]
            brk = aligned[brk_col]
            zone = np.where(
                brk > 0.5,
                "Frenleme (Brake Aktif)",
                np.where(
                    thr >= 95.0,
                    "Tam Gaz Pedalı (Throttle ≥ %95)",
                    np.where(
                        thr <= 5.0,
                        "Süzülme (Gaz Pedalı ≤ %5 ve Fren Kapalı)",
                        "Kısmi Gaz Pedalı (%5 < Throttle < %95)",
                    ),
                ),
            )
            aligned[f"DrivingZone_{suffix}"] = zone

    zone_comp_df = _build_driving_zone_summary(aligned, driver1_name, driver2_name)
    summary_metrics = _build_telemetry_summary_metrics(aligned, b1, b2, driver1_name, driver2_name)

    return {
        "aligned_df": aligned,
        "driver1_processed_df": df1,
        "driver2_processed_df": df2,
        "driver1_brake": b1,
        "driver2_brake": b2,
        "zone_comparison_df": zone_comp_df,
        "summary_metrics": summary_metrics,
        "available_channels": available_channels,
        "disclaimer": BRAKE_ACTIVITY_DISCLAIMER,
    }


def _build_driving_zone_summary(
    aligned: pd.DataFrame,
    driver1_name: str,
    driver2_name: str,
) -> pd.DataFrame:
    """
    Ortak mesafe ekseni üzerinde iki sürücünün Tam Gaz, Kısmi Gaz, Frenleme ve
    Süzülme (Coasting) bölgelerinde geçirdiği mesafe oranlarını (%) ve metrelerini karşılaştırır.
    """
    if "DrivingZone_D1" not in aligned.columns or "DrivingZone_D2" not in aligned.columns or aligned.empty:
        return pd.DataFrame()

    categories = [
        "Tam Gaz Pedalı (Throttle ≥ %95)",
        "Kısmi Gaz Pedalı (%5 < Throttle < %95)",
        "Frenleme (Brake Aktif)",
        "Süzülme (Gaz Pedalı ≤ %5 ve Fren Kapalı)",
    ]
    total_pts = len(aligned)
    dist_span = float(aligned["Distance"].max() - aligned["Distance"].min()) if total_pts > 1 else 0.0

    rows: List[Dict[str, Any]] = []
    for cat in categories:
        pct1 = float((aligned["DrivingZone_D1"] == cat).mean() * 100.0)
        pct2 = float((aligned["DrivingZone_D2"] == cat).mean() * 100.0)
        m1 = (pct1 / 100.0) * dist_span
        m2 = (pct2 / 100.0) * dist_span
        rows.append(
            {
                "Sürüş Bölgesi": cat,
                f"{driver1_name} Oran (%)": round(pct1, 1),
                f"{driver2_name} Oran (%)": round(pct2, 1),
                f"Oran Farkı ({driver1_name} - {driver2_name}) (%)": round(pct1 - pct2, 1),
                f"{driver1_name} Mesafe (m)": round(m1, 0),
                f"{driver2_name} Mesafe (m)": round(m2, 0),
                f"Mesafe Farkı (m)": round(m1 - m2, 0),
            }
        )
    return pd.DataFrame(rows)


def _build_telemetry_summary_metrics(
    aligned: pd.DataFrame,
    b1: Dict[str, Any],
    b2: Dict[str, Any],
    driver1_name: str,
    driver2_name: str,
) -> Dict[str, Any]:
    """İki sürücünün telemetri ve fren özet istatistiklerini döndürür."""
    summary: Dict[str, Any] = {
        "grid_points_n": int(len(aligned)),
        "track_distance_m": float(aligned["Distance"].max()) if not aligned.empty else np.nan,
        "max_speed_d1_kmh": float(aligned["Speed_D1"].max()) if "Speed_D1" in aligned.columns else np.nan,
        "max_speed_d2_kmh": float(aligned["Speed_D2"].max()) if "Speed_D2" in aligned.columns else np.nan,
        "mean_speed_d1_kmh": float(aligned["Speed_D1"].mean()) if "Speed_D1" in aligned.columns else np.nan,
        "mean_speed_d2_kmh": float(aligned["Speed_D2"].mean()) if "Speed_D2" in aligned.columns else np.nan,
        "max_abs_speed_diff_kmh": np.nan,
        "max_abs_speed_diff_distance_m": np.nan,
        "max_abs_speed_diff_faster_driver": "",
        "d1_brake_duration_sec": b1.get("brake_duration_sec", np.nan),
        "d2_brake_duration_sec": b2.get("brake_duration_sec", np.nan),
        "d1_braking_distance_m": b1.get("total_braking_distance_m", np.nan),
        "d2_braking_distance_m": b2.get("total_braking_distance_m", np.nan),
        "d1_brake_events_count": b1.get("brake_events_count", 0),
        "d2_brake_events_count": b2.get("brake_events_count", 0),
        "d1_raw_samples_n": b1.get("sample_size", 0),
        "d2_raw_samples_n": b2.get("sample_size", 0),
    }

    if "AbsSpeedDelta_kmh" in aligned.columns and aligned["AbsSpeedDelta_kmh"].notna().any():
        idx_max = aligned["AbsSpeedDelta_kmh"].idxmax()
        row = aligned.loc[idx_max]
        summary["max_abs_speed_diff_kmh"] = float(row["AbsSpeedDelta_kmh"])
        summary["max_abs_speed_diff_distance_m"] = float(row["Distance"])
        signed_diff = float(row["SpeedDelta_kmh"])
        summary["max_abs_speed_diff_faster_driver"] = driver1_name if signed_diff > 0 else driver2_name

    return summary

