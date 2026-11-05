"""
F1 Performance Analysis & Telemetry System — HTML Dashboard & Localhost Sunucu Modülü (f1_html_dashboard.py)
=============================================================================================================
Mevcut f1_performance.py, f1_telemetry.py ve f1_data_loader.py modüllerindeki tüm analizleri
bağımsız (self-contained, çevrimdışı Plotly.js gömülü) interaktif bir HTML sayfasına dönüştürür
ve dahili Python HTTP sunucusu üzerinden http://localhost:8050 adresinde yayınlar.

Kullanıcı tarayıcı üzerinden:
- Season, Grand Prix (2026 ve Meksika GP dahil), Session Type,
- Driver 1, Driver 2, Reference Driver,
- Lap Number, Stint, Tyre Compound, Brake Smoothing Window (10m/20m/30m)
seçimlerini değiştirdiğinde localhost sunucusu analizleri anında yeniden hesaplayıp HTML olarak sunar.
"""

from __future__ import annotations

import html
import json
import os
import socket
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    calculate_theoretical_optimal,
    compare_drivers_ttest,
    compare_matching_stints,
    filter_clean_laps,
    format_laptime_str,
    get_fastest_lap_telemetry_comparison,
    run_full_race_analysis,
    to_seconds_series,
)
from f1_telemetry import (
    BRAKE_ACTIVITY_DISCLAIMER,
    align_and_compare_telemetry,
)

COMPOUND_STYLES: Dict[str, Dict[str, str]] = {
    "SOFT": {"color": "#ff3333", "symbol": "circle"},
    "MEDIUM": {"color": "#ffe600", "symbol": "diamond"},
    "HARD": {"color": "#f0f0f4", "symbol": "square"},
    "INTERMEDIATE": {"color": "#39b54a", "symbol": "triangle-up"},
    "WET": {"color": "#00a0de", "symbol": "cross"},
    "UNKNOWN": {"color": "#a855f7", "symbol": "x"},
}

DRIVER_COLORS = ("#38bdf8", "#fb923c")

# Sunucu düzeyinde yüklü seans önbelleği (aynı yarış/seans için tekrar disk okumasını hızlandırır)
_SESSION_MEMORY_CACHE: Dict[Tuple[int, str, str], Tuple[Optional[Any], Optional[str]]] = {}


def _get_cached_session(
    year: int,
    event_name: str,
    session_type: str,
    force_reload: bool = False,
) -> Tuple[Optional[Any], Optional[str]]:
    norm_sess = normalize_session_code(session_type, year=year)
    key = (int(year), str(event_name), str(norm_sess))
    if force_reload and key in _SESSION_MEMORY_CACHE:
        del _SESSION_MEMORY_CACHE[key]
    if key in _SESSION_MEMORY_CACHE:
        cached_sess, cached_err = _SESSION_MEMORY_CACHE[key]
        if cached_sess is not None and cached_err is None:
            return cached_sess, None
        del _SESSION_MEMORY_CACHE[key]

    session_obj, load_err = load_f1_session(
        year=year,
        event_name=event_name,
        session_type=norm_sess,
        load_telemetry=True,
        load_weather=False,
        load_messages=True,
        cache_dir=DEFAULT_CACHE_DIR,
    )
    # Sadece başarılı yüklemeleri bellekte tut; hatalı yüklemeleri önbelleğe alma
    if session_obj is not None and load_err is None:
        _SESSION_MEMORY_CACHE[key] = (session_obj, None)
    return session_obj, load_err


def _apply_lap_filters(
    laps_df: pd.DataFrame,
    selected_stint: str,
    selected_compound: str,
) -> pd.DataFrame:
    if laps_df is None or laps_df.empty:
        return pd.DataFrame() if laps_df is None else laps_df.copy()
    out = laps_df.copy()
    if selected_stint != "Tümü" and "Stint" in out.columns:
        try:
            out = out[out["Stint"] == int(selected_stint)].copy()
        except ValueError:
            pass
    if selected_compound != "Tümü" and "Compound" in out.columns:
        out = out[out["Compound"].astype(str).str.upper() == selected_compound.upper()].copy()
    return out


def _extract_lap_telemetry(
    session_obj: Any,
    driver_laps: pd.DataFrame,
    lap_selection: str,
) -> Tuple[pd.DataFrame, Optional[int], Optional[float], Optional[str]]:
    return extract_driver_lap_telemetry(
        session=session_obj,
        driver_laps=driver_laps,
        lap_selection=lap_selection,
        cache_dir=DEFAULT_CACHE_DIR,
    )


def _df_to_styled_html(df: pd.DataFrame, float_precision: int = 3) -> str:
    """DataFrame'i temiz HTML tablosuna dönüştürür."""
    if df is None or df.empty:
        return "<div class='empty-box'>Bu görünüm için veri bulunmuyor.</div>"
    disp = df.copy()
    for col in disp.columns:
        if pd.api.types.is_float_dtype(disp[col]):
            disp[col] = disp[col].apply(
                lambda v: f"{v:.{float_precision}f}" if (pd.notna(v) and np.isfinite(v)) else "-"
            )
        else:
            disp[col] = disp[col].apply(lambda v: "-" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
    return disp.to_html(index=False, classes="data-table", border=0, escape=True)


# ==============================================================================
# PLOTLY GRAFİK ÜRETİCİLERİ
# ==============================================================================

def _build_stint_figure(
    raw_d1: pd.DataFrame,
    clean_d1: pd.DataFrame,
    raw_d2: pd.DataFrame,
    clean_d2: pd.DataFrame,
    rep_d1: Dict[str, Any],
    rep_d2: Dict[str, Any],
    driver1: str,
    driver2: str,
    fuel_effect_sec_per_lap: float = 0.035,
) -> go.Figure:
    """
    Görsel 1: Stint ve Hamur Renkli Tur Süresi Dağılımı (Scatter) + OLS Regresyon Trendi
    ve Filtrelenen Turların Nedenleriyle Birlikte Gösterimi.
    """
    fig = go.Figure()
    dash_styles = ["solid", "dash", "dot", "dashdot", "longdash"]

    for drv_idx, (drv_code, raw_df_drv, clean_df_drv, rep_drv, base_color) in enumerate(
        [
            (driver1, raw_d1, clean_d1, rep_d1 or {}, DRIVER_COLORS[0]),
            (driver2, raw_d2, clean_d2, rep_d2 or {}, DRIVER_COLORS[1]),
        ]
    ):
        if clean_df_drv is None or clean_df_drv.empty or "LapNumber" not in clean_df_drv.columns:
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
            c_style = COMPOUND_STYLES.get(comp, COMPOUND_STYLES["UNKNOWN"])
            dash_s = dash_styles[(st_int - 1) % len(dash_styles)]
            tyre_life = grp["TyreLife"].fillna(-1).astype(int) if "TyreLife" in grp.columns else pd.Series(-1, index=grp.index)

            # 1. Temiz turlar (Scatter noktaları)
            fig.add_trace(
                go.Scatter(
                    x=grp["LapNumber"],
                    y=grp["LapTime_Sec"],
                    mode="lines+markers",
                    name=f"{drv_code} S{st_int} ({comp})",
                    legendgroup=f"{drv_code}_S{st_int}",
                    line=dict(color=base_color, width=2.2, dash=dash_s),
                    marker=dict(
                        color=c_style["color"],
                        symbol=c_style["symbol"],
                        size=9,
                        line=dict(color=base_color, width=1.8),
                    ),
                    customdata=np.stack([tyre_life.values, np.full(len(grp), comp)], axis=-1),
                    hovertemplate=(
                        f"<b>{drv_code} — Stint {st_int}</b><br>"
                        "Tur: %{x}<br>Tur Süresi: %{y:.3f} sn<br>"
                        "Lastik Hamuru: %{customdata[1]} (Lastik Yaşı: %{customdata[0]})<extra></extra>"
                    ),
                )
            )

            # 2. OLS Doğrusal Regresyon Trend Çizgisi (Stint Aşınma Eğimi)
            if len(grp) >= 3:
                x_vals = grp["LapNumber"].values.astype(float)
                y_vals = grp["LapTime_Sec"].values.astype(float)
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                fuel_corrected_slope = slope + fuel_effect_sec_per_lap
                med_v = float(grp["LapTime_Sec"].median())
                y_fit = slope * x_vals + intercept

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_fit,
                        mode="lines",
                        name=f"{drv_code} S{st_int} Trend",
                        legendgroup=f"{drv_code}_S{st_int}",
                        showlegend=False,
                        line=dict(color=base_color, width=1.6, dash="dot"),
                        hovertemplate=(
                            f"<b>{drv_code} Stint {st_int} ({comp}) Aşınma Eğimi</b><br>"
                            f"Gözlenen Aşınma Eğimi: <b>{slope:+.3f} sn/tur</b><br>"
                            f"Yakıt Düzeltilmiş Aşınma Eğimi: <b>{fuel_corrected_slope:+.3f} sn/tur</b><br>"
                            f"Medyan Tur Süresi: {med_v:.3f} sn<extra></extra>"
                        ),
                    )
                )

        # 3. Filtrelenen Turlar (Elenen turların saydam x işaretçileri ve hover nedenleri)
        reasons_list = rep_drv.get("reasons_by_lap", [])
        if reasons_list:
            r_df = pd.DataFrame(reasons_list)
            if not r_df.empty and "LapNumber" in r_df.columns and "LapTime_Sec" in r_df.columns:
                r_valid = r_df.dropna(subset=["LapNumber", "LapTime_Sec"]).sort_values("LapNumber")
                if not r_valid.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=r_valid["LapNumber"],
                            y=r_valid["LapTime_Sec"],
                            mode="markers",
                            name=f"{drv_code} Filtrelenen Turlar",
                            marker=dict(
                                symbol="x",
                                size=8,
                                color="rgba(148, 163, 184, 0.6)",
                                line=dict(color=base_color, width=1.4),
                            ),
                            customdata=np.stack([
                                r_valid["Reason"].astype(str).values,
                                r_valid["Compound"].fillna("-").astype(str).values if "Compound" in r_valid.columns else np.full(len(r_valid), "-"),
                                r_valid["Stint"].fillna(1).astype(str).values if "Stint" in r_valid.columns else np.full(len(r_valid), "1"),
                            ], axis=-1),
                            hovertemplate=(
                                f"<b>{drv_code} — Filtrelenen Tur %{{x}}</b><br>"
                                "Tur Süresi: %{y:.3f} sn<br>"
                                "Filtreleme Nedeni: <b>%{customdata[0]}</b><br>"
                                "Stint: %{customdata[2]} | Hamur: %{customdata[1]}<extra></extra>"
                            ),
                        )
                    )

        # 4. Pit-Stop Düşey Çizgileri
        if raw_df_drv is not None and "PitInTime" in raw_df_drv.columns and "LapNumber" in raw_df_drv.columns:
            pit_laps = raw_df_drv[raw_df_drv["PitInTime"].notna()]["LapNumber"].dropna().astype(int).unique()
            for p_lap in pit_laps:
                fig.add_vline(
                    x=float(p_lap),
                    line_width=1.4,
                    line_dash="dashdot",
                    line_color=base_color,
                    annotation_text=f"PIT {drv_code} (T{p_lap})",
                    annotation_position="top left" if drv_idx == 0 else "bottom right",
                )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        title="Tur Süresi ve Tur Numarası Dağılımı (Stint, Hamur, OLS Regresyon Trendi ve Filtrelenen Turlar)",
        xaxis_title="Tur Numarası (LapNumber)",
        yaxis_title="Tur Süresi (sn)",
        hovermode="closest",
        height=520,
        margin=dict(l=50, r=30, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _build_delta_figure(
    lap_deltas_df: pd.DataFrame,
    driver1: str,
    driver2: str,
) -> go.Figure:
    """
    Görsel 2: Kümülatif Zaman Farkı (Worm Grafiği) ve Tur Başına Zaman Farkı Bar Grafiği.
    """
    plot_df = lap_deltas_df[lap_deltas_df["Both_Clean"]].copy() if not lap_deltas_df.empty and "Both_Clean" in lap_deltas_df.columns else lap_deltas_df.copy()
    if plot_df.empty:
        plot_df = lap_deltas_df.copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"Kümülatif Zaman Farkı — Worm Grafiği (sn)  [Aşağı Eğim: {driver1} Lehine | Yukarı Eğim: {driver2} Lehine]",
            f"Tur Başına Zaman Farkı: Δt = {driver1} - {driver2} (sn)  [Negatif (<0): {driver1} Daha Hızlı | Pozitif (>0): {driver2} Daha Hızlı]",
        ),
    )

    if not plot_df.empty:
        # Üst Panel: Kümülatif Delta Worm Chart
        fig.add_trace(
            go.Scatter(
                x=plot_df["LapNumber"],
                y=plot_df["Cumulative_Delta_Clean_Sec"],
                mode="lines+markers",
                name=f"Kümülatif Temiz Zaman Farkı ({driver1} - {driver2})",
                line=dict(color="#a78bfa", width=2.6),
                marker=dict(size=5, color="#a78bfa"),
                hovertemplate="Tur: %{x}<br>Kümülatif Temiz Fark: %{y:+.3f} sn<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if "Cumulative_Delta_All_Sec" in plot_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["LapNumber"],
                    y=plot_df["Cumulative_Delta_All_Sec"],
                    mode="lines",
                    name="Toplam Seans Kümülatif Farkı (Pit/SC Dahil)",
                    line=dict(color="#64748b", width=1.5, dash="dash"),
                    hovertemplate="Tur: %{x}<br>Toplam Kümülatif Fark: %{y:+.3f} sn<extra></extra>",
                ),
                row=1,
                col=1,
            )
        fig.add_hline(y=0.0, line_dash="solid", line_color="#94a3b8", line_width=1.2, row=1, col=1)

        # Alt Panel: Tur Başına Delta Bar Grafiği (İki sürücünün renkleriyle ayrılmış)
        colors = np.where(plot_df["Delta_Sec"] < 0, DRIVER_COLORS[0], DRIVER_COLORS[1])
        fig.add_trace(
            go.Bar(
                x=plot_df["LapNumber"],
                y=plot_df["Delta_Sec"],
                name=f"Tur Süresi Farkı ({driver1} - {driver2})",
                marker_color=colors,
                customdata=plot_df["Faster_Driver"],
                hovertemplate="Tur: %{x}<br>Tur Süresi Farkı (Δt): %{y:+.3f} sn<br>Daha Hızlı Sürücü: <b>%{customdata}</b><extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0.0, line_color="#94a3b8", line_width=1.5, row=2, col=1)

        fig.add_annotation(
            xref="paper",
            yref="y2",
            x=0.01,
            y=float(plot_df["Delta_Sec"].min()) * 0.85 if plot_df["Delta_Sec"].min() < 0 else -0.3,
            text=f"⬇️ Negatif (< 0 sn): {driver1} daha hızlı",
            showarrow=False,
            bgcolor="rgba(56, 189, 248, 0.2)",
            bordercolor=DRIVER_COLORS[0],
        )
        fig.add_annotation(
            xref="paper",
            yref="y2",
            x=0.01,
            y=float(plot_df["Delta_Sec"].max()) * 0.85 if plot_df["Delta_Sec"].max() > 0 else 0.3,
            text=f"⬆️ Pozitif (> 0 sn): {driver2} daha hızlı",
            showarrow=False,
            bgcolor="rgba(251, 146, 60, 0.2)",
            bordercolor=DRIVER_COLORS[1],
        )

    fig.update_yaxes(title_text="Kümülatif Δ (sn)", row=1, col=1)
    fig.update_yaxes(title_text="Δ Tur Süresi (sn)", row=2, col=1)
    fig.update_xaxes(title_text="Tur Numarası (LapNumber)", row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=580,
        margin=dict(l=50, r=30, t=60, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def _build_box_dist_fig(
    clean_d1: pd.DataFrame,
    clean_d2: pd.DataFrame,
    driver1: str,
    driver2: str,
) -> go.Figure:
    """
    Görsel 3: Tutarlılık ve Tur Temposu Dağılımı (Keman + Kutu Grafiği).
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Genel Temiz Tur Süresi Dağılımı (Keman ve Kutu Grafiği)",
            "Stint Bazında Tur Süresi Dağılımı (Kutu Grafiği)",
        ),
    )

    t1 = to_seconds_series(clean_d1["LapTime"]).dropna() if (clean_d1 is not None and not clean_d1.empty and "LapTime" in clean_d1.columns) else pd.Series(dtype=float)
    t2 = to_seconds_series(clean_d2["LapTime"]).dropna() if (clean_d2 is not None and not clean_d2.empty and "LapTime" in clean_d2.columns) else pd.Series(dtype=float)

    if not t1.empty:
        fig.add_trace(
            go.Violin(
                y=t1,
                name=driver1,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.25,
                line_color=DRIVER_COLORS[0],
                fillcolor="rgba(56, 189, 248, 0.25)",
                hovertemplate=f"<b>{driver1}</b><br>Tur Süresi: %{{y:.3f}} sn<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if not t2.empty:
        fig.add_trace(
            go.Violin(
                y=t2,
                name=driver2,
                box_visible=True,
                meanline_visible=True,
                points="all",
                jitter=0.25,
                line_color=DRIVER_COLORS[1],
                fillcolor="rgba(251, 146, 60, 0.25)",
                hovertemplate=f"<b>{driver2}</b><br>Tur Süresi: %{{y:.3f}} sn<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Stint bazında kutu grafikleri
    for drv_code, c_df, base_color in [
        (driver1, clean_d1, DRIVER_COLORS[0]),
        (driver2, clean_d2, DRIVER_COLORS[1]),
    ]:
        if c_df is not None and not c_df.empty and "Stint" in c_df.columns and "LapTime" in c_df.columns:
            cdf = c_df.copy()
            cdf["_sec"] = to_seconds_series(cdf["LapTime"])
            cdf = cdf.dropna(subset=["_sec", "Stint"])
            for st_val, grp in cdf.groupby("Stint"):
                comp = str(grp["Compound"].dropna().iloc[0]).upper() if "Compound" in grp.columns and grp["Compound"].notna().any() else ""
                fig.add_trace(
                    go.Box(
                        y=grp["_sec"],
                        x=[f"{drv_code} S{int(st_val)} ({comp})"] * len(grp),
                        name=f"{drv_code} S{int(st_val)}",
                        marker_color=base_color,
                        boxmean=True,
                        showlegend=False,
                        hovertemplate=f"<b>{drv_code} Stint {int(st_val)} ({comp})</b><br>Tur Süresi: %{{y:.3f}} sn<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )

    fig.update_yaxes(title_text="Temiz Tur Süresi (sn)", row=1, col=1)
    fig.update_yaxes(title_text="Temiz Tur Süresi (sn)", row=1, col=2)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=450,
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _build_sector_figure(
    sector_comp_df: pd.DataFrame,
    opt_d1: Dict[str, Any],
    opt_d2: Dict[str, Any],
    driver1: str,
    driver2: str,
    ref_driver: str,
) -> go.Figure:
    """
    Görsel 4: Sektör Üstünlüğü (S1/S2/S3 Ortalama, Medyan, En Hızlı) ve Teorik Optimal Tur Karşılaştırması.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Sektör Zaman Farkları ({driver1} - {driver2}) [sn]",
            f"Gerçek En Hızlı Tur vs Teorik Optimal Tur [sn]",
        ),
    )

    if not sector_comp_df.empty:
        mean_diff_col = f"Ort. Fark ({driver1} - {driver2}) (sn)"
        med_diff_col = f"Medyan Fark ({driver1} - {driver2}) (sn)"
        best_diff_col = f"En İyi Fark ({driver1} - {driver2}) (sn)"

        fig.add_trace(
            go.Bar(
                y=sector_comp_df["Sektör"],
                x=sector_comp_df[mean_diff_col],
                orientation="h",
                name="Ortalama Sektör Farkı (sn)",
                marker_color="#38bdf8",
                hovertemplate="<b>%{y}</b><br>Ortalama Fark: %{x:+.3f} sn<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                y=sector_comp_df["Sektör"],
                x=sector_comp_df[med_diff_col],
                orientation="h",
                name="Medyan Sektör Farkı (sn)",
                marker_color="#f59e0b",
                hovertemplate="<b>%{y}</b><br>Medyan Fark: %{x:+.3f} sn<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                y=sector_comp_df["Sektör"],
                x=sector_comp_df[best_diff_col],
                orientation="h",
                name="En Hızlı Sektör Farkı (sn)",
                marker_color="#a78bfa",
                hovertemplate="<b>%{y}</b><br>En Hızlı Fark: %{x:+.3f} sn<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="#94a3b8", row=1, col=1)

    # Sağ Panel: Gerçek En Hızlı vs Teorik Optimal Tur
    d1_actual = opt_d1.get("actual_best_sec", np.nan)
    d1_theo = opt_d1.get("theoretical_optimal_sec", np.nan)
    d2_actual = opt_d2.get("actual_best_sec", np.nan)
    d2_theo = opt_d2.get("theoretical_optimal_sec", np.nan)

    x_labels = [f"{driver1} En Hızlı", f"{driver1} Optimal", f"{driver2} En Hızlı", f"{driver2} Optimal"]
    y_vals = [d1_actual, d1_theo, d2_actual, d2_theo]
    bar_colors = [DRIVER_COLORS[0], "#22c55e", DRIVER_COLORS[1], "#22c55e"]

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_vals,
            marker_color=bar_colors,
            name="Tur Süresi (sn)",
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Süre: %{y:.3f} sn<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Fark (sn) [Negatif: D1 Daha Hızlı | Pozitif: D2 Daha Hızlı]", row=1, col=1)
    fig.update_yaxes(title_text="Sektör", row=1, col=1)
    fig.update_yaxes(title_text="Tur Süresi (sn)", row=1, col=2)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        barmode="group",
        height=420,
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _build_track_dominance_fig(
    aligned_df: pd.DataFrame,
    driver1: str,
    driver2: str,
    dominance_summary: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    Görsel 5B: 2D Pist Dominansı Haritası (Track Dominance Map) — Pist mikro-sektörlerinin
    hızlı olan sürücünün rengiyle boyanması (X, Y koordinatları olmadığında 1D mesafe şeridine zarif fallback).
    """
    fig = go.Figure()
    if aligned_df is None or aligned_df.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            title="Pist Dominansı Haritası (Telemetri Verisi Bulunamadı)",
            height=380,
        )
        return fig

    has_xy = (
        "X" in aligned_df.columns and "Y" in aligned_df.columns
        and aligned_df["X"].notna().any() and aligned_df["Y"].notna().any()
    )

    d_sum = dominance_summary or {}
    d1_pct = d_sum.get("driver1_faster_pct", 0.0)
    d2_pct = d_sum.get("driver2_faster_pct", 0.0)

    if has_xy:
        plot_df = aligned_df.dropna(subset=["X", "Y"]).copy()
        if "SpeedDelta_kmh" not in plot_df.columns:
            plot_df["SpeedDelta_kmh"] = 0.0

        step_window_m = 50.0
        plot_df["Segment"] = (plot_df["Distance"] // step_window_m).astype(int)

        for _, seg_grp in plot_df.groupby("Segment", sort=True):
            avg_delta = float(seg_grp["SpeedDelta_kmh"].mean())
            seg_dist = float(seg_grp["Distance"].mean())
            seg_spd1 = float(seg_grp["Speed_D1"].mean()) if "Speed_D1" in seg_grp.columns else 0.0
            seg_spd2 = float(seg_grp["Speed_D2"].mean()) if "Speed_D2" in seg_grp.columns else 0.0

            if avg_delta > 0:
                seg_color = DRIVER_COLORS[0]
                faster_name = driver1
            elif avg_delta < 0:
                seg_color = DRIVER_COLORS[1]
                faster_name = driver2
            else:
                seg_color = "#94a3b8"
                faster_name = "Eşit"

            fig.add_trace(
                go.Scatter(
                    x=seg_grp["X"],
                    y=seg_grp["Y"],
                    mode="lines",
                    line=dict(color=seg_color, width=5.5),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{faster_name} Daha Hızlı</b><br>"
                        f"Mesafe: {seg_dist:.0f} m<br>"
                        f"Hız Farkı ({driver1} - {driver2}): {avg_delta:+.1f} km/h<br>"
                        f"{driver1} Hız: {seg_spd1:.1f} km/h | {driver2} Hız: {seg_spd2:.1f} km/h<extra></extra>"
                    ),
                )
            )

        start_row = plot_df.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[start_row["X"]],
                y=[start_row["Y"]],
                mode="markers+text",
                marker=dict(symbol="star", size=14, color="#ffffff", line=dict(color="#eab308", width=2)),
                text=["Start / Finish"],
                textposition="top center",
                textfont=dict(color="#f8fafc", size=11),
                name="Start / Finish",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color=DRIVER_COLORS[0], width=5),
                name=f"{driver1} Daha Hızlı (%{d1_pct:.1f} pist mesafesi)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color=DRIVER_COLORS[1], width=5),
                name=f"{driver2} Daha Hızlı (%{d2_pct:.1f} pist mesafesi)",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            title=f"2D Pist Dominansı Haritası — Track Dominance ({driver1} vs {driver2})",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
            height=540,
            margin=dict(l=30, r=30, t=60, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
    else:
        valid_df = aligned_df.dropna(subset=["Distance", "SpeedDelta_kmh"]).copy()
        colors = np.where(valid_df["SpeedDelta_kmh"] > 0, DRIVER_COLORS[0], DRIVER_COLORS[1])
        fig.add_trace(
            go.Bar(
                x=valid_df["Distance"],
                y=valid_df["SpeedDelta_kmh"],
                marker_color=colors,
                name="Hız Farkı (km/h)",
                hovertemplate="Mesafe: %{x:.0f} m<br>Hız Farkı: %{y:+.1f} km/h<extra></extra>",
            )
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            title=f"1D Pist Mesafe Dominansı — Hız Farkı ({driver1} vs {driver2}) [X, Y Koordinatları Mevcut Değil]",
            xaxis_title="Ortak Pist Mesafesi (metre)",
            yaxis_title="Δ Hız (km/h)",
            height=380,
            margin=dict(l=50, r=30, t=60, b=40),
        )

    return fig


def _build_telemetry_figure(
    aligned_df: pd.DataFrame,
    driver1: str,
    driver2: str,
    lap_no_1: Optional[int],
    lap_no_2: Optional[int],
    brake_window_m: int,
    selected_channels: List[str],
) -> go.Figure:
    """
    Görsel 5A: Ortak Mesafe (Distance) Ekseninde Senkronize Çok Kanallı Telemetri Eğrileri.
    """
    channel_meta = {
        "Speed": {"title": "Hız — Speed (km/h)", "yaxis": "Hız (km/h)", "shape": "linear", "range": None, "unit": "km/h"},
        "SpeedDelta": {"title": f"Hız Farkı — ΔSpeed ({driver1} - {driver2}) (km/h)", "yaxis": "Δ Hız (km/h)", "shape": "linear", "range": None, "unit": "km/h"},
        "Throttle": {"title": "Gaz Pedalı — Throttle (%)", "yaxis": "Gaz Pedalı (%)", "shape": "linear", "range": [-5, 105], "unit": "%"},
        "SmoothedBrakeActivity": {
            "title": f"Yumuşatılmış Fren Aktivitesi — SmoothedBrakeActivity (%) [Pencere: {brake_window_m}m, Fren Basıncı Değildir]",
            "yaxis": "Fren Aktivitesi (%)",
            "shape": "linear",
            "range": [-5, 105],
            "unit": "%",
        },
        "BrakeActivity": {
            "title": f"Fren Aktivitesi — BrakeActivity (%) [Pencere: {brake_window_m}m, Fren Basıncı Değildir]",
            "yaxis": "Fren Aktivitesi (%)",
            "shape": "linear",
            "range": [-5, 105],
            "unit": "%",
        },
        "BrakeRaw": {"title": "Ham Fren Durumu — Brake (İkili Kanal: 0=Kapalı, 1=Açık) [Kesikli]", "yaxis": "0 / 1", "shape": "hv", "range": [-0.15, 1.15], "unit": ""},
        "nGear": {"title": "Vites — nGear (1-8) [Kesikli]", "yaxis": "Vites", "shape": "hv", "range": [0.5, 8.5], "unit": ""},
        "RPM": {"title": "Motor Devri — RPM (devir/dk)", "yaxis": "RPM", "shape": "linear", "range": None, "unit": " RPM"},
        "DRS": {"title": "DRS Durumu — DRS [Kesikli]", "yaxis": "DRS", "shape": "hv", "range": None, "unit": ""},
    }

    channels = [c for c in selected_channels if c in channel_meta]
    if not channels:
        channels = ["Speed", "Throttle", "SmoothedBrakeActivity", "BrakeRaw", "nGear"]

    n_rows = len(channels)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=max(0.02, 0.16 / n_rows),
        subplot_titles=[channel_meta[c]["title"] for c in channels],
    )

    if not aligned_df.empty:
        for r_idx, ch in enumerate(channels, start=1):
            meta = channel_meta[ch]
            show_leg = r_idx == 1
            if ch == "SpeedDelta":
                fig.add_trace(
                    go.Scatter(
                        x=aligned_df["Distance"],
                        y=aligned_df["SpeedDelta_kmh"],
                        mode="lines",
                        name=f"Hız Farkı ({driver1} - {driver2})",
                        line=dict(color="#a78bfa", width=1.8),
                        showlegend=False,
                        hovertemplate=f"Mesafe: %{{x:.0f}} m<br>Hız Farkı ({driver1} - {driver2}): %{{y:+.1f}} km/h<extra></extra>",
                    ),
                    row=r_idx,
                    col=1,
                )
                fig.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8", row=r_idx, col=1)
            else:
                c1, c2 = f"{ch}_D1", f"{ch}_D2"
                u_str = meta["unit"]
                if c1 in aligned_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=aligned_df["Distance"],
                            y=aligned_df[c1],
                            mode="lines",
                            name=f"{driver1} (Tur {lap_no_1})",
                            legendgroup=driver1,
                            showlegend=show_leg,
                            line=dict(color=DRIVER_COLORS[0], width=2.0, shape=meta["shape"]),
                            hovertemplate=f"<b>{driver1}</b> (Tur {lap_no_1})<br>Mesafe: %{{x:.0f}} m<br>Değer: %{{y:.1f}}{u_str}<extra></extra>",
                        ),
                        row=r_idx,
                        col=1,
                    )
                if c2 in aligned_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=aligned_df["Distance"],
                            y=aligned_df[c2],
                            mode="lines",
                            name=f"{driver2} (Tur {lap_no_2})",
                            legendgroup=driver2,
                            showlegend=show_leg,
                            line=dict(color=DRIVER_COLORS[1], width=2.0, shape=meta["shape"]),
                            hovertemplate=f"<b>{driver2}</b> (Tur {lap_no_2})<br>Mesafe: %{{x:.0f}} m<br>Değer: %{{y:.1f}}{u_str}<extra></extra>",
                        ),
                        row=r_idx,
                        col=1,
                    )
            fig.update_yaxes(title_text=meta["yaxis"], range=meta["range"], row=r_idx, col=1)

    fig.update_xaxes(title_text="Ortak Pist Mesafesi — Distance (metre)", row=n_rows, col=1)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=max(440, 210 * n_rows),
        margin=dict(l=55, r=30, t=60, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ==============================================================================
# TAM SAYFA HTML DASHBOARD ÜRETİCİSİ
# ==============================================================================

def generate_dashboard_html(
    year: int = 2025,
    event_name: str = "Australian Grand Prix",
    session_type: str = "R",
    driver1: str = "NOR",
    driver2: str = "VER",
    ref_driver: Optional[str] = None,
    lap_number: str = "Fastest",
    stint_filter: str = "Tümü",
    compound_filter: str = "Tümü",
    brake_window_m: int = 20,
    by_stint_iqr: bool = True,
    selected_channels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    force_reload: bool = False,
) -> str:
    """
    Tüm F1 performans, stint, sektör, fark ve telemetri analizlerini çalıştırarak
    bağımsız (self-contained) ve localhost üzerinde etkileşimli çalışan bir HTML
    dokümanı üretir.
    """
    setup_fastf1_cache(DEFAULT_CACHE_DIR)
    if selected_channels is None or len(selected_channels) == 0:
        selected_channels = [
            "Speed",
            "SpeedDelta",
            "Throttle",
            "SmoothedBrakeActivity",
            "BrakeActivity",
            "BrakeRaw",
            "nGear",
            "RPM",
            "DRS",
        ]

    schedule_df = get_season_schedule(year, include_testing=False, cache_dir=DEFAULT_CACHE_DIR)
    events_list = (
        schedule_df["EventName"].dropna().unique().tolist()
        if not schedule_df.empty and "EventName" in schedule_df.columns
        else ["Australian Grand Prix", "Bahrain Grand Prix", "United States Grand Prix", "Mexico City Grand Prix"]
    )
    for must_have in ("United States Grand Prix", "Mexico City Grand Prix"):
        if must_have not in events_list:
            events_list.append(must_have)
    if event_name not in events_list:
        events_list.insert(0, event_name)

    session_type = normalize_session_code(session_type, year=year)
    available_sessions = get_event_available_sessions(year, event_name, cache_dir=DEFAULT_CACHE_DIR)
    session_notice: Optional[str] = None
    if session_type not in available_sessions:
        requested_sess = session_type
        fallback_code = "R" if "R" in available_sessions else next(iter(available_sessions.keys()))
        session_notice = (
            f"'{requested_sess}' seansı {year} {event_name} hafta sonu programında bulunmamaktadır. "
            f"Seans otomatik olarak '{fallback_code}' ({available_sessions[fallback_code]['label']}) olarak ayarlandı."
        )
        session_type = fallback_code

    session_obj, load_err = _get_cached_session(year, event_name, session_type, force_reload=force_reload)
    # Başarılı indirme sonrası yerel önbellekteki yarış listesini anında güncelle
    cached_list = list_cached_sessions(DEFAULT_CACHE_DIR)
    cached_badges = " | ".join(
        [f"{c['year']} {c['event_name']} ({c['session_name']})" for c in cached_list]
    ) or "Önbellek boş"
    diag_info = get_last_load_diagnostics(year, event_name, session_type)

    year_opts_err = "".join(
        f'<option value="{y}" {"selected" if y == year else ""}>{y}</option>' for y in range(2026, 2017, -1)
    )
    event_opts_err = "".join(
        f'<option value="{html.escape(str(ev))}" {"selected" if str(ev) == event_name else ""}>{html.escape(str(ev))}</option>'
        for ev in events_list
    )
    sess_opts_err = "".join(
        f'<option value="{k}" {"selected" if k == session_type else ""}>{meta["full_label"]}</option>'
        for k, meta in available_sessions.items()
    )

    if session_obj is None or load_err:
        exc_type_str = diag_info.get("exception_type") or "SessionLoadError"
        cat_code_str = diag_info.get("error_category") or "UNKNOWN"
        cat_label_str = diag_info.get("error_category_label") or "Seans Yükleme Hatası"
        exc_raw_str = diag_info.get("exception_message") or str(load_err)
        base_url_str = diag_info.get("livetiming_base_url") or "https://livetiming.formula1.com"
        ff1_ver_str = diag_info.get("fastf1_version") or "3.9.0"
        dns_ok = diag_info.get("dns_resolved")
        dns_status_str = "Çözümlendi (OK)" if dns_ok else "Çözümlenemedi (DNS Hatası)"
        auth_status_str = diag_info.get("auth_status") or "Not authenticated"
        logs_list = diag_info.get("captured_logs") or []
        logs_html = (
            "<ul>" + "".join(f"<li><code>{html.escape(str(lg))}</code></li>" for lg in logs_list) + "</ul>"
            if logs_list
            else "<i>Ek FastF1 uyarı kaydı yok.</i>"
        )
        retry_url = (
            f"/?year={year}&event={urllib.parse.quote_plus(event_name)}"
            f"&session={urllib.parse.quote_plus(session_type)}&d1={urllib.parse.quote_plus(driver1)}"
            f"&d2={urllib.parse.quote_plus(driver2)}&retry=1"
        )
        err_html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><title>F1 Dashboard — Veri Tanı Raporu</title>
<style>
body{{background:#0b1120;color:#f8fafc;font-family:system-ui,sans-serif;padding:32px;line-height:1.5;}}
.card{{background:#1e293b;border:1px solid #ef4444;border-radius:12px;padding:26px;max-width:920px;margin:auto;}}
.diag-box{{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:14px 16px;margin:14px 0;font-size:0.88rem;}}
.badge-err{{display:inline-block;background:#7f1d1d;color:#fecaca;padding:3px 10px;border-radius:6px;font-weight:700;font-size:0.8rem;}}
.btn-retry{{display:inline-block;background:linear-gradient(135deg,#2563eb,#4f46e5);color:#fff;padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:700;margin-right:10px;}}
a{{color:#38bdf8;}}
select,button{{padding:8px 10px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#f8fafc;}}
#loading-overlay{{display:none;position:fixed;inset:0;background:rgba(11,17,32,0.88);z-index:9999;align-items:center;justify-content:center;flex-direction:column;}}
.spinner{{width:48px;height:48px;border:5px solid #334155;border-top-color:#38bdf8;border-radius:50%;animation:spin 0.9s linear infinite;margin-bottom:14px;}}
@keyframes spin{{to{{transform:rotate(360deg);}}}}
</style></head>
<body>
<div id="loading-overlay">
    <div class="spinner"></div>
    <div style="font-size:1.05rem;font-weight:700;color:#f8fafc;">⏳ Seans verisi çevrimiçi sağlayıcıdan indiriliyor, lütfen bekleyin...</div>
</div>
<div class="card">
    <h2>⚠️ Seans Verisi Yüklenemedi — Teknik Tanı Raporu</h2>
    {f'''<div style="background: rgba(56, 189, 248, 0.15); border-left: 4px solid #38bdf8; padding: 12px 14px; border-radius: 6px; margin: 12px 0; color: #bae6fd;">
        <b>ℹ️ Seans Takvim Uyarısı:</b> {html.escape(session_notice)}
    </div>''' if session_notice else ''}
    <p><span class="badge-err">{html.escape(cat_code_str)}: {html.escape(cat_label_str)}</span> &nbsp; <b>İstisna Tipi:</b> <code>{html.escape(exc_type_str)}</code></p>
    <p>{html.escape(str(load_err))}</p>
    <div class="diag-box">
        <div><b>Seçilen Seans:</b> {year} {html.escape(event_name)} ({html.escape(session_type)})</div>
        <div><b>FastF1 Sürümü:</b> <code>{html.escape(ff1_ver_str)}</code> | <b>Resmi Sunucu:</b> <code>{html.escape(base_url_str)}</code></div>
        <div><b>DNS Çözümleme Durumu:</b> <code>{html.escape(dns_status_str)}</code> | <b>F1TV Kimlik Doğrulama Durumu:</b> <code>{html.escape(auth_status_str)}</code></div>
        <div><b>Gerçek İstisna Mesajı:</b> <code>{html.escape(exc_raw_str)}</code></div>
        <div style="margin-top:8px;"><b>Yakalanan FastF1 Günlükleri:</b> {logs_html}</div>
    </div>
    <p style="margin:18px 0;">
        <a class="btn-retry" href="{retry_url}" onclick="document.getElementById('loading-overlay').style.display='flex';">🔄 Seansı Çevrimiçi İndir / Yeniden Dene</a>
    </p>
    <form method="GET" action="/" onsubmit="document.getElementById('loading-overlay').style.display='flex';" style="background:#0f172a;padding:14px;border-radius:8px;margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
        <label>Sezon: <select name="year">{year_opts_err}</select></label>
        <label>Grand Prix: <select name="event">{event_opts_err}</select></label>
        <label>Seans: <select name="session">{sess_opts_err}</select></label>
        <input type="hidden" name="retry" value="1">
        <button type="submit" style="background:#2563eb;border:none;font-weight:700;cursor:pointer;">Seçili Yarışı İndir ve Aç</button>
    </form>
    <p style="margin-top:16px;"><b>Yerel Önbellekte Hazır Yarışlar:</b> {html.escape(cached_badges)}</p>
    <p>
        <a href="/?year=2025&event=Australian+Grand+Prix&session=R&d1=NOR&d2=VER">👉 2025 Australian GP (NOR vs VER)</a> |
        <a href="/?year=2026&event=Australian+Grand+Prix&session=R&d1=RUS&d2=LEC">👉 2026 Australian GP (RUS vs LEC)</a> |
        <a href="/?year=2026&event=United+States+Grand+Prix&session=R&d1=NOR&d2=VER">👉 2026 United States GP (COTA)</a>
    </p>
</div>
<script>
async function updateAvailableSessions() {{
    var yearSelect = document.querySelector('select[name="year"]');
    var eventSelect = document.querySelector('select[name="event"]');
    var sessionSelect = document.querySelector('select[name="session"]');
    if (!yearSelect || !eventSelect || !sessionSelect) return;

    var y = yearSelect.value;
    var ev = eventSelect.value;
    var prevSession = sessionSelect.value;

    try {{
        var resp = await fetch('/api/sessions?year=' + encodeURIComponent(y) + '&event=' + encodeURIComponent(ev));
        if (!resp.ok) return;
        var sessions = await resp.json();
        if (!Array.isArray(sessions) || sessions.length === 0) return;

        sessionSelect.innerHTML = '';
        var foundPrev = false;
        sessions.forEach(function(s) {{
            var opt = document.createElement('option');
            opt.value = s.code;
            opt.textContent = s.full_label || (s.code + ' — ' + s.label);
            if (s.code === prevSession) {{
                opt.selected = true;
                foundPrev = true;
            }}
            sessionSelect.appendChild(opt);
        }});

        if (!foundPrev) {{
            var hasR = sessions.some(function(s) {{ return s.code === 'R'; }});
            sessionSelect.value = hasR ? 'R' : sessions[0].code;
        }}
    }} catch (e) {{
        console.warn('Seans listesi güncellenemedi:', e);
    }}
}}
document.addEventListener('DOMContentLoaded', function() {{
    var eventSelect = document.querySelector('select[name="event"]');
    var yearSelect = document.querySelector('select[name="year"]');
    if (eventSelect) eventSelect.addEventListener('change', updateAvailableSessions);
    if (yearSelect) yearSelect.addEventListener('change', updateAvailableSessions);
}});
</script>
</body></html>"""
        return err_html

    drivers_info = get_session_drivers_info(session_obj)
    driver_codes = [d["abbreviation"] for d in drivers_info]
    driver_labels = {
        d["abbreviation"]: f"{d['abbreviation']} — {d['full_name']}" if d.get("full_name") else d["abbreviation"]
        for d in drivers_info
    }

    if driver1 not in driver_codes and driver_codes:
        driver1 = driver_codes[0]
    if driver2 not in driver_codes and len(driver_codes) > 1:
        driver2 = driver_codes[1] if driver_codes[1] != driver1 else driver_codes[0]
    if not ref_driver or ref_driver not in driver_codes:
        ref_driver = driver1

    raw_d1_all = get_raw_driver_laps(session_obj, driver1)
    raw_d2_all = get_raw_driver_laps(session_obj, driver2)
    raw_ref_all = get_raw_driver_laps(session_obj, ref_driver)

    all_stints = sorted(
        set(
            (raw_d1_all["Stint"].dropna().astype(int).tolist() if "Stint" in raw_d1_all.columns else [])
            + (raw_d2_all["Stint"].dropna().astype(int).tolist() if "Stint" in raw_d2_all.columns else [])
        )
    )
    all_compounds = sorted(
        {
            str(c).upper()
            for c in (
                (raw_d1_all["Compound"].dropna().tolist() if "Compound" in raw_d1_all.columns else [])
                + (raw_d2_all["Compound"].dropna().tolist() if "Compound" in raw_d2_all.columns else [])
            )
            if str(c).upper() not in ("", "NAN", "NONE")
        }
    )
    all_laps = sorted(
        set(
            (raw_d1_all["LapNumber"].dropna().astype(int).tolist() if "LapNumber" in raw_d1_all.columns else [])
            + (raw_d2_all["LapNumber"].dropna().astype(int).tolist() if "LapNumber" in raw_d2_all.columns else [])
        )
    )

    raw_d1 = _apply_lap_filters(raw_d1_all, stint_filter, compound_filter)
    raw_d2 = _apply_lap_filters(raw_d2_all, stint_filter, compound_filter)
    raw_ref = _apply_lap_filters(raw_ref_all, stint_filter, compound_filter)

    # Merkezi Analiz Boru Hattı (Tekil Veri Akışı / Unified Pipeline)
    full_analysis = run_full_race_analysis(
        session=session_obj,
        driver1=driver1,
        driver2=driver2,
        ref_driver=ref_driver,
        by_stint=by_stint_iqr,
        fuel_effect_sec_per_lap=0.035,
        load_telemetry=False,
        raw_d1_override=raw_d1,
        raw_d2_override=raw_d2,
        raw_ref_override=raw_ref,
    )

    clean_d1 = full_analysis["clean_d1"]
    clean_d2 = full_analysis["clean_d2"]
    clean_ref = full_analysis["clean_ref"]
    rep_d1 = full_analysis["report_d1"]
    rep_d2 = full_analysis["report_d2"]
    pace_d1 = full_analysis["pace_d1"]
    pace_d2 = full_analysis["pace_d2"]
    cons_d1 = full_analysis["consistency_d1"]
    cons_d2 = full_analysis["consistency_d2"]
    ttest_det = full_analysis["ttest"]
    stints_d1 = full_analysis["stints_d1"]
    stints_d2 = full_analysis["stints_d2"]
    stints_ref = full_analysis["stints_ref"]
    matching_stints = full_analysis["matching_stints"]
    lap_deltas = full_analysis["lap_deltas"]
    sector_comp = full_analysis["sector_comparison"]
    opt_d1 = full_analysis["optimal_d1"]
    opt_d2 = full_analysis["optimal_d2"]
    fastest_lap_sec = full_analysis["fastest_lap_seconds"]
    session_cond = summarize_session_conditions(session_obj)

    # Telemetri hesaplama (Fastest lap helper veya seçili tur ile)
    dominance_summary = {}
    if lap_number in ("Fastest", "En Hızlı Tur", "En Hızlı Tur (Fastest)", ""):
        tel_data = get_fastest_lap_telemetry_comparison(
            session=session_obj,
            driver1=driver1,
            driver2=driver2,
            clean_laps_d1=clean_d1,
            clean_laps_d2=clean_d2,
            cache_dir=DEFAULT_CACHE_DIR,
            brake_window_m=float(brake_window_m),
        )
        aligned_df = tel_data.get("aligned_df", pd.DataFrame())
        tel_comp = tel_data.get("tel_comp", {})
        lap_no_1 = tel_data.get("lap_number_1")
        lap_no_2 = tel_data.get("lap_number_2")
        lap_sec_1 = tel_data.get("lap_sec_1")
        lap_sec_2 = tel_data.get("lap_sec_2")
        dominance_summary = tel_data.get("dominance_summary", {})
    else:
        raw_tel_1, lap_no_1, lap_sec_1, _ = _extract_lap_telemetry(session_obj, raw_d1_all, lap_number)
        raw_tel_2, lap_no_2, lap_sec_2, _ = _extract_lap_telemetry(session_obj, raw_d2_all, lap_number)
        tel_comp = align_and_compare_telemetry(
            raw_tel_1, raw_tel_2, driver1_name=driver1, driver2_name=driver2, window_m=float(brake_window_m), distance_step_m=2.0
        )
        aligned_df = tel_comp.get("aligned_df", pd.DataFrame())
        if not aligned_df.empty and "SpeedDelta_kmh" in aligned_df.columns:
            valid_spds = aligned_df.dropna(subset=["SpeedDelta_kmh"])
            n_pts = len(valid_spds)
            if n_pts > 0:
                d1_f = int((valid_spds["SpeedDelta_kmh"] > 0).sum())
                d2_f = int((valid_spds["SpeedDelta_kmh"] < 0).sum())
                dominance_summary = {
                    "driver1_faster_pct": float((d1_f / n_pts) * 100.0),
                    "driver2_faster_pct": float((d2_f / n_pts) * 100.0),
                }

    # Plotly grafiklerini HTML parçalarına dönüştür (İlk grafikte plotly.js gömülüdür, tamamen çevrimdışı çalışır)
    fig_stint = _build_stint_figure(raw_d1, clean_d1, raw_d2, clean_d2, rep_d1, rep_d2, driver1, driver2)
    fig_delta = _build_delta_figure(lap_deltas, driver1, driver2)
    fig_dist = _build_box_dist_fig(clean_d1, clean_d2, driver1, driver2)
    fig_sec = _build_sector_figure(sector_comp, opt_d1, opt_d2, driver1, driver2, ref_driver)
    fig_tel = _build_telemetry_figure(
        aligned_df, driver1, driver2, lap_no_1, lap_no_2, int(brake_window_m), selected_channels
    )
    fig_track = _build_track_dominance_fig(aligned_df, driver1, driver2, dominance_summary)

    chart_stint_html = fig_stint.to_html(full_html=False, include_plotlyjs=True)
    chart_delta_html = fig_delta.to_html(full_html=False, include_plotlyjs=False)
    chart_dist_html = fig_dist.to_html(full_html=False, include_plotlyjs=False)
    chart_sec_html = fig_sec.to_html(full_html=False, include_plotlyjs=False)
    chart_tel_html = fig_tel.to_html(full_html=False, include_plotlyjs=False)
    chart_track_html = fig_track.to_html(full_html=False, include_plotlyjs=False)

    # Optimal tur ve kaynak tur tablosu
    opt_rows = []
    for drv_code, o_det in ((driver1, opt_d1), (driver2, opt_d2)):
        secs = o_det.get("sectors", {})
        s1_d, s2_d, s3_d = secs.get("Sector1Time", {}), secs.get("Sector2Time", {}), secs.get("Sector3Time", {})
        opt_rows.append(
            {
                "Sürücü": drv_code,
                "Örneklem (n)": o_det.get("sample_size", 0),
                "En Hızlı Tur (sn)": o_det.get("actual_best_sec", np.nan),
                "En Hızlı Tur No (Lastik Hamuru)": f"Tur {o_det.get('actual_best_lap_number')} ({o_det.get('actual_best_compound') or '-'})",
                "Teorik Optimal Tur (sn)": o_det.get("theoretical_optimal_sec", np.nan),
                "Potansiyel Kazanç (En Hızlı Tur - Teorik Optimal Tur) (sn)": o_det.get("development_potential_sec", np.nan),
                "En Hızlı S1 (Kaynak Tur)": f"{s1_d.get('best_sector_sec', np.nan):.3f} sn (Tur {s1_d.get('from_lap_number')}, Stint {s1_d.get('from_stint')}, {s1_d.get('from_compound')})"
                if pd.notna(s1_d.get("best_sector_sec"))
                else "-",
                "En Hızlı S2 (Kaynak Tur)": f"{s2_d.get('best_sector_sec', np.nan):.3f} sn (Tur {s2_d.get('from_lap_number')}, Stint {s2_d.get('from_stint')}, {s2_d.get('from_compound')})"
                if pd.notna(s2_d.get("best_sector_sec"))
                else "-",
                "En Hızlı S3 (Kaynak Tur)": f"{s3_d.get('best_sector_sec', np.nan):.3f} sn (Tur {s3_d.get('from_lap_number')}, Stint {s3_d.get('from_stint')}, {s3_d.get('from_compound')})"
                if pd.notna(s3_d.get("best_sector_sec"))
                else "-",
            }
        )
    opt_table_html = _df_to_styled_html(pd.DataFrame(opt_rows))

    # Görüntüleme amaçlı Türkçe sütun eşlemeleri (dahili DataFrame sütunları değiştirilmez)
    stint_col_map = {
        "Driver": "Sürücü",
        "Stint": "Stint",
        "Compound": "Lastik Hamuru",
        "StartLap": "Başlangıç Turu",
        "EndLap": "Bitiş Turu",
        "TotalLaps": "Toplam Tur",
        "ValidLapCount": "Geçerli Tur (n)",
        "TyreLifeStart": "Başlangıç Lastik Yaşı",
        "TyreLifeEnd": "Bitiş Lastik Yaşı",
        "MeanLapTime_Sec": "Ortalama Tur Süresi (sn)",
        "MedianLapTime_Sec": "Medyan Tur Süresi (sn)",
        "BestLapTime_Sec": "En Hızlı Tur (sn)",
        "BestLapNumber": "En Hızlı Tur No",
        "WorstLapTime_Sec": "En Yavaş Tur (sn)",
        "WorstLapNumber": "En Yavaş Tur No",
        "StdLapTime_Sec": "Standart Sapma (sn)",
        "CV_Pct": "Değişim Katsayısı — CV (%)",
        "First3Mean_Sec": "İlk 3 Tur Ortalama (sn)",
        "Last3Mean_Sec": "Son 3 Tur Ortalama (sn)",
        "Last3_Minus_First3_Sec": "Performans Kaybı: Son 3 - İlk 3 (sn)",
        "LinearTrend_SecPerLap": "Gözlenen Aşınma Eğimi (sn/tur)",
        "FuelCorrectedTrend_SecPerLap": "Yakıt Düzeltilmiş Aşınma Eğimi (sn/tur)",
        "DiffToRefMedian_Sec": f"Referans Sürücü ({ref_driver}) Medyan Farkı (sn)",
        "SC_VSC_LapCount": "SC/VSC Tur Sayısı",
        "SampleNote": "Örneklem ve Metodolojik Not",
    }
    stints_disp_df = pd.concat([stints_d1, stints_d2], ignore_index=True).rename(columns=stint_col_map)

    sector_col_map = {
        f"{driver1} Ort. (sn)": f"{driver1} Ortalama (sn)",
        f"{driver2} Ort. (sn)": f"{driver2} Ortalama (sn)",
        f"Ort. Fark ({driver1} - {driver2}) (sn)": f"Ortalama Sektör Zaman Farkı ({driver1} - {driver2}) (sn)",
        f"Medyan Fark ({driver1} - {driver2}) (sn)": f"Medyan Sektör Zaman Farkı ({driver1} - {driver2}) (sn)",
        f"{driver1} En İyi (sn)": f"{driver1} En Hızlı (sn)",
        f"{driver2} En İyi (sn)": f"{driver2} En Hızlı (sn)",
        f"En İyi Fark ({driver1} - {driver2}) (sn)": f"En Hızlı Sektör Zaman Farkı ({driver1} - {driver2}) (sn)",
        f"{driver1} Z-Skor (Ref: {ref_driver})": f"{driver1} Z-Skoru (Referans Sürücü: {ref_driver})",
        f"{driver2} Z-Skor (Ref: {ref_driver})": f"{driver2} Z-Skoru (Referans Sürücü: {ref_driver})",
    }
    sector_disp_df = sector_comp.rename(columns=sector_col_map)

    brake_col_map = {
        "EventNo": "Fren Olayı No",
        "StartDistance_m": "Başlangıç Mesafesi (m)",
        "EndDistance_m": "Bitiş Mesafesi (m)",
        "BrakingDistanceRange": "Frenleme Mesafe Aralığı",
        "BrakingDistance_m": "Frenleme Mesafesi (m)",
        "BrakeDuration_sec": "Fren Süresi (sn)",
        "EntrySpeed_kmh": "Giriş Hızı (km/h)",
        "MinSpeed_kmh": "Minimum Hız (km/h)",
        "SpeedDrop_kmh": "Hız Düşüşü / Yavaşlama (km/h)",
        "MinGear": "Minimum Vites",
        "SampleCount": "Örneklem (n)",
    }
    brake_ev_d1_disp = tel_comp["driver1_brake"]["brake_events_df"].rename(columns=brake_col_map)
    brake_ev_d2_disp = tel_comp["driver2_brake"]["brake_events_df"].rename(columns=brake_col_map)

    reasons_col_map = {
        "LapNumber": "Tur No",
        "Stint": "Stint",
        "Compound": "Lastik Hamuru",
        "LapTime_Sec": "Tur Süresi (sn)",
        "Reason": "Filtreleme Nedeni",
    }
    reasons_d1_disp = pd.DataFrame(rep_d1["reasons_by_lap"]).rename(columns=reasons_col_map)
    reasons_d2_disp = pd.DataFrame(rep_d2["reasons_by_lap"]).rename(columns=reasons_col_map)

    # Filtreleme özet tablosu
    filter_summary_df = pd.DataFrame(
        [
            {
                "Sürücü": driver1,
                "Toplam Ham Tur": rep_d1["total_raw_laps"],
                "Geçerli Temiz Tur (n)": rep_d1["clean_laps_count"],
                "Eksik Tur Süresi (NaN)": rep_d1["missing_or_invalid_laptime"],
                "Pit Giriş / Çıkış (Pit-In / Pit-Out)": rep_d1["pit_in_out_laps"],
                "Silinen Tur (Deleted)": rep_d1["deleted_laps"],
                "SC / VSC / Kırmızı Bayrak": rep_d1["sc_vsc_red_laps"],
                "Çeyrekler Açıklığı (IQR) Aykırı Değer": rep_d1["iqr_outlier_laps"],
            },
            {
                "Sürücü": driver2,
                "Toplam Ham Tur": rep_d2["total_raw_laps"],
                "Geçerli Temiz Tur (n)": rep_d2["clean_laps_count"],
                "Eksik Tur Süresi (NaN)": rep_d2["missing_or_invalid_laptime"],
                "Pit Giriş / Çıkış (Pit-In / Pit-Out)": rep_d2["pit_in_out_laps"],
                "Silinen Tur (Deleted)": rep_d2["deleted_laps"],
                "SC / VSC / Kırmızı Bayrak": rep_d2["sc_vsc_red_laps"],
                "Çeyrekler Açıklığı (IQR) Aykırı Değer": rep_d2["iqr_outlier_laps"],
            },
        ]
    )

    # Seçim kutuları (HTML option listeleri)
    year_opts = "".join(
        f'<option value="{y}" {"selected" if y == year else ""}>{y}</option>' for y in range(2026, 2017, -1)
    )
    event_opts = "".join(
        f'<option value="{html.escape(str(ev))}" {"selected" if str(ev) == event_name else ""}>{html.escape(str(ev))}</option>'
        for ev in events_list
    )
    sess_opts = "".join(
        f'<option value="{k}" {"selected" if k == session_type else ""}>{meta["full_label"]}</option>'
        for k, meta in available_sessions.items()
    )
    d1_opts = "".join(
        f'<option value="{c}" {"selected" if c == driver1 else ""}>{html.escape(driver_labels.get(c, c))}</option>'
        for c in driver_codes
    )
    d2_opts = "".join(
        f'<option value="{c}" {"selected" if c == driver2 else ""}>{html.escape(driver_labels.get(c, c))}</option>'
        for c in driver_codes
    )
    ref_opts = "".join(
        f'<option value="{c}" {"selected" if c == ref_driver else ""}>{html.escape(driver_labels.get(c, c))}</option>'
        for c in driver_codes
    )
    stint_opts = '<option value="Tümü">Tümü</option>' + "".join(
        f'<option value="{s}" {"selected" if str(s) == str(stint_filter) else ""}>Stint {s}</option>'
        for s in all_stints
    )
    comp_opts = '<option value="Tümü">Tümü</option>' + "".join(
        f'<option value="{c}" {"selected" if c == str(compound_filter).upper() else ""}>{c}</option>'
        for c in all_compounds
    )
    lap_opts = f'<option value="Fastest" {"selected" if lap_number in ("Fastest", "En Hızlı Tur", "En Hızlı Tur (Fastest)") else ""}>En Hızlı Tur</option>' + "".join(
        f'<option value="{lno}" {"selected" if str(lno) == str(lap_number) else ""}>Tur {lno}</option>'
        for lno in all_laps
    )
    bw_opts = "".join(
        f'<option value="{w}" {"selected" if int(w) == int(brake_window_m) else ""}>{w} metre</option>'
        for w in (10, 20, 30)
    )

    t_sum = tel_comp["summary_metrics"]
    p_val_str = f"{ttest_det['p_value']:.5f}" if pd.notna(ttest_det.get("p_value")) else "Hesaplanamadı"
    t_stat_str = f"{ttest_det['t_stat']:.4f}" if pd.notna(ttest_det.get("t_stat")) else "-"

    html_doc = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>F1 Performans ve Telemetri Analiz Paneli — {year} {html.escape(event_name)}</title>
    <style>
        :root {{
            --bg-main: #0b1120;
            --bg-card: #1e293b;
            --bg-header: #0f172a;
            --border-color: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --d1-color: #38bdf8;
            --d2-color: #fb923c;
            --accent: #6366f1;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: var(--bg-main);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.5;
        }}
        .topbar {{
            background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 18px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .topbar h1 {{ margin: 0; font-size: 1.45rem; letter-spacing: -0.02em; }}
        .topbar .subtitle {{ color: var(--text-muted); font-size: 0.86rem; margin-top: 4px; }}
        .quick-links a {{
            display: inline-block;
            background: #1e293b;
            color: #38bdf8;
            border: 1px solid #38bdf8;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 0.8rem;
            text-decoration: none;
            margin-left: 6px;
        }}
        .quick-links a:hover {{ background: #38bdf8; color: #0f172a; }}
        .container {{ max-width: 1560px; margin: 0 auto; padding: 20px 24px 60px; }}
        .controls-panel {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 22px;
        }}
        .controls-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            align-items: end;
        }}
        .control-item label {{
            display: block;
            font-size: 0.75rem;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 4px;
            font-weight: 600;
        }}
        .control-item select, .control-item button {{
            width: 100%;
            padding: 8px 10px;
            border-radius: 6px;
            border: 1px solid #475569;
            background: #0f172a;
            color: var(--text-main);
            font-size: 0.86rem;
        }}
        .btn-submit {{
            background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
            border: none !important;
            font-weight: 700;
            cursor: pointer;
            color: #ffffff !important;
            padding: 9px 14px !important;
        }}
        .btn-submit:hover {{ opacity: 0.92; }}
        .drivers-banner {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
            gap: 18px;
            margin-bottom: 22px;
        }}
        .driver-card {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 18px 20px;
            border: 1px solid var(--border-color);
        }}
        .driver-card.d1 {{ border-left: 5px solid var(--d1-color); }}
        .driver-card.d2 {{ border-left: 5px solid var(--d2-color); }}
        .driver-card h2 {{ margin: 0 0 4px 0; font-size: 1.2rem; }}
        .driver-meta {{ color: var(--text-muted); font-size: 0.82rem; margin-bottom: 14px; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }}
        .kpi-box {{
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 12px;
        }}
        .kpi-label {{ font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; }}
        .kpi-val {{ font-size: 1.15rem; font-weight: 700; margin-top: 2px; }}
        .kpi-sub {{ font-size: 0.74rem; color: #38bdf8; margin-top: 2px; }}
        .tabs-nav {{
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .tab-btn {{
            background: #1e293b;
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            border-bottom: none;
            padding: 10px 18px;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        .tab-btn.active {{
            background: #38bdf8;
            color: #0f172a;
            border-color: #38bdf8;
        }}
        .tab-section {{ display: none; }}
        .tab-section.active {{ display: block; }}
        .panel {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 20px;
        }}
        .panel h3 {{ margin-top: 0; font-size: 1.08rem; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .methodology-note {{
            background: rgba(245, 158, 11, 0.12);
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-bottom: 16px;
            color: #fde68a;
        }}
        .info-note {{
            background: rgba(56, 189, 248, 0.12);
            border-left: 4px solid #38bdf8;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-bottom: 16px;
            color: #bae6fd;
        }}
        .table-wrap {{ overflow-x: auto; }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.83rem;
        }}
        table.data-table th {{
            background: #0f172a;
            color: #38bdf8;
            text-align: left;
            padding: 9px 10px;
            border-bottom: 2px solid #334155;
            white-space: nowrap;
        }}
        table.data-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #334155;
            white-space: nowrap;
        }}
        table.data-table tr:hover td {{ background: rgba(255, 255, 255, 0.04); }}
        .two-col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
            gap: 18px;
        }}
        #loading-overlay {{
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(11, 17, 32, 0.88);
            z-index: 9999;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #f8fafc;
            text-align: center;
            padding: 24px;
        }}
        .spinner {{
            width: 52px;
            height: 52px;
            border: 5px solid #334155;
            border-top-color: #38bdf8;
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
            margin-bottom: 16px;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>

<div id="loading-overlay">
    <div class="spinner"></div>
    <h3 style="margin:0 0 8px 0;">⏳ Seans ve Telemetri Verileri Yükleniyor...</h3>
    <p style="margin:0;color:#94a3b8;max-width:540px;font-size:0.9rem;">
        Önbellekte bulunmayan seanslar FastF1 / OpenF1 API üzerinden indirilip yerel önbelleğe kaydediliyor. Lütfen bekleyin...
    </p>
</div>

<div class="topbar">
    <div>
        <h1>🏎️ F1 Performans ve Telemetri Analiz Paneli</h1>
        <div class="subtitle">
            Aktif Seans: <b>{year} {html.escape(event_name)} ({available_sessions.get(session_type, {}).get('label') or SESSION_TYPE_OPTIONS.get(session_type, session_type)})</b> |
            Karşılaştırma: <b>{driver1} vs {driver2}</b> (Referans Sürücü: <b>{ref_driver}</b>) |
            Önbellekte Hazır: {html.escape(cached_badges)}
        </div>
    </div>
    <div class="quick-links">
        <a href="/?year=2025&event=Australian+Grand+Prix&session=R&d1=NOR&d2=VER&ref=NOR&bw=20" onclick="showLoadingOverlay()">⚡ 2025 Avustralya GP (NOR vs VER)</a>
        <a href="/?year=2026&event=Australian+Grand+Prix&session=R&d1=RUS&d2=LEC&ref=RUS&bw=20" onclick="showLoadingOverlay()">⚡ 2026 Avustralya GP (RUS vs LEC)</a>
        <a href="/?year=2026&event=United+States+Grand+Prix&session=R&d1=VER&d2=NOR&ref=VER&bw=20" onclick="showLoadingOverlay()">🇺🇸 2026 ABD GP / COTA (VER vs NOR)</a>
    </div>
</div>

<div class="container">
    {f'''<div class="methodology-note" style="border-left-color: #38bdf8; background: rgba(56, 189, 248, 0.15); color: #bae6fd; margin-bottom: 20px;">
        <b>ℹ️ Seans Takvim Bildirimi:</b> {html.escape(session_notice)}
    </div>''' if session_notice else ''}
    <!-- KULLANICI KONTROLLERİ (BÖLÜM 6) -->
    <form class="controls-panel" method="GET" action="/" onsubmit="showLoadingOverlay()">
        <div class="controls-grid">
            <div class="control-item">
                <label>1. Sezon</label>
                <select name="year">{year_opts}</select>
            </div>
            <div class="control-item">
                <label>2. Grand Prix (Yarış)</label>
                <select name="event">{event_opts}</select>
            </div>
            <div class="control-item">
                <label>3. Seans Tipi</label>
                <select name="session">{sess_opts}</select>
            </div>
            <div class="control-item">
                <label>4. 1. Sürücü</label>
                <select name="d1">{d1_opts}</select>
            </div>
            <div class="control-item">
                <label>5. 2. Sürücü</label>
                <select name="d2">{d2_opts}</select>
            </div>
            <div class="control-item">
                <label>6. Referans Sürücü</label>
                <select name="ref">{ref_opts}</select>
            </div>
            <div class="control-item">
                <label>7. Stint</label>
                <select name="stint">{stint_opts}</select>
            </div>
            <div class="control-item">
                <label>8. Lastik Hamuru</label>
                <select name="compound">{comp_opts}</select>
            </div>
            <div class="control-item">
                <label>9. Telemetri Turu</label>
                <select name="lap">{lap_opts}</select>
            </div>
            <div class="control-item">
                <label>10. Fren Aktivitesi Penceresi</label>
                <select name="bw">{bw_opts}</select>
            </div>
            <div class="control-item">
                <button type="submit" class="btn-submit">🔄 Analizi Güncelle</button>
            </div>
        </div>
    </form>

    <!-- BÖLÜM A: GENEL PERFORMANS KARTLARI -->
    <div class="drivers-banner">
        <div class="driver-card d1">
            <h2>🟦 {html.escape(driver_labels.get(driver1, driver1))}</h2>
            <div class="driver-meta">
                Geçerli Temiz Tur: <b>n = {rep_d1['clean_laps_count']}</b> / Toplam {rep_d1['total_raw_laps']} Tur
                ({rep_d1['total_raw_laps'] - rep_d1['clean_laps_count']} tur filtrelendi)
            </div>
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-label">Tur Temposu (En Hızlı Tura Göre)</div>
                    <div class="kpi-val">{f"%{pace_d1['pace_score_vs_fastest_pct']:.3f}" if pd.notna(pace_d1['pace_score_vs_fastest_pct']) else "-"}</div>
                    <div class="kpi-sub">En Hızlı Tur Farkı: {f"{pace_d1['delta_vs_fastest_sec']:+.3f} sn" if pd.notna(pace_d1['delta_vs_fastest_sec']) else "-"}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Medyan Tur Süresi</div>
                    <div class="kpi-val">{format_laptime_str(pace_d1['median_lap_sec'])}</div>
                    <div class="kpi-sub">Referans Sürücü ({ref_driver}): {f"{pace_d1['delta_vs_ref_sec']:+.3f} sn (%{pace_d1['pace_score_vs_ref_pct']:+.2f})" if pd.notna(pace_d1['delta_vs_ref_sec']) else "-"}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">En Hızlı Tur</div>
                    <div class="kpi-val">{format_laptime_str(pace_d1['best_lap_sec'])}</div>
                    <div class="kpi-sub">Teorik Optimal Tur: {format_laptime_str(opt_d1['theoretical_optimal_sec'])}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Tur Tutarlılığı — Değişim Katsayısı (CV)</div>
                    <div class="kpi-val">{f"CV (%): %{cons_d1['cv_pct']:.3f}" if pd.notna(cons_d1['cv_pct']) else "-"}</div>
                    <div class="kpi-sub">Stint CV (%): {f"%{cons_d1['stint_weighted_cv_pct']:.3f}" if pd.notna(cons_d1['stint_weighted_cv_pct']) else "-"} | Standart Sapma: {f"{cons_d1['std_sec']:.3f} sn" if pd.notna(cons_d1['std_sec']) else "-"}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Geçerli Tur Sayısı (n)</div>
                    <div class="kpi-val">{rep_d1['clean_laps_count']} / {rep_d1['total_raw_laps']}</div>
                    <div class="kpi-sub">Ortalama Tur Süresi: {format_laptime_str(pace_d1['mean_lap_sec'])}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Stint Sayısı</div>
                    <div class="kpi-val">{len(stints_d1)}</div>
                    <div class="kpi-sub">Çeyrekler Açıklığı (IQR): {f"{cons_d1['iqr_sec']:.3f} sn" if pd.notna(cons_d1['iqr_sec']) else "-"}</div>
                </div>
            </div>
        </div>

        <div class="driver-card d2">
            <h2>🟧 {html.escape(driver_labels.get(driver2, driver2))}</h2>
            <div class="driver-meta">
                Geçerli Temiz Tur: <b>n = {rep_d2['clean_laps_count']}</b> / Toplam {rep_d2['total_raw_laps']} Tur
                ({rep_d2['total_raw_laps'] - rep_d2['clean_laps_count']} tur filtrelendi)
            </div>
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-label">Tur Temposu (En Hızlı Tura Göre)</div>
                    <div class="kpi-val">{f"%{pace_d2['pace_score_vs_fastest_pct']:.3f}" if pd.notna(pace_d2['pace_score_vs_fastest_pct']) else "-"}</div>
                    <div class="kpi-sub">En Hızlı Tur Farkı: {f"{pace_d2['delta_vs_fastest_sec']:+.3f} sn" if pd.notna(pace_d2['delta_vs_fastest_sec']) else "-"}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Medyan Tur Süresi</div>
                    <div class="kpi-val">{format_laptime_str(pace_d2['median_lap_sec'])}</div>
                    <div class="kpi-sub">Referans Sürücü ({ref_driver}): {f"{pace_d2['delta_vs_ref_sec']:+.3f} sn (%{pace_d2['pace_score_vs_ref_pct']:+.2f})" if pd.notna(pace_d2['delta_vs_ref_sec']) else "-"}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">En Hızlı Tur</div>
                    <div class="kpi-val">{format_laptime_str(pace_d2['best_lap_sec'])}</div>
                    <div class="kpi-sub">Teorik Optimal Tur: {format_laptime_str(opt_d2['theoretical_optimal_sec'])}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Tur Tutarlılığı — Değişim Katsayısı (CV)</div>
                    <div class="kpi-val">{f"CV (%): %{cons_d2['cv_pct']:.3f}" if pd.notna(cons_d2['cv_pct']) else "-"}</div>
                    <div class="kpi-sub">Stint CV (%): {f"%{cons_d2['stint_weighted_cv_pct']:.3f}" if pd.notna(cons_d2['stint_weighted_cv_pct']) else "-"} | Standart Sapma: {f"{cons_d2['std_sec']:.3f} sn" if pd.notna(cons_d2['std_sec']) else "-"}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Geçerli Tur Sayısı (n)</div>
                    <div class="kpi-val">{rep_d2['clean_laps_count']} / {rep_d2['total_raw_laps']}</div>
                    <div class="kpi-sub">Ortalama Tur Süresi: {format_laptime_str(pace_d2['mean_lap_sec'])}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Stint Sayısı</div>
                    <div class="kpi-val">{len(stints_d2)}</div>
                    <div class="kpi-sub">Çeyrekler Açıklığı (IQR): {f"{cons_d2['iqr_sec']:.3f} sn" if pd.notna(cons_d2['iqr_sec']) else "-"}</div>
                </div>
            </div>
        </div>
    </div>

    <!-- SEKME MENÜSÜ -->
    <div class="tabs-nav">
        <button class="tab-btn active" onclick="switchTab('tab-stint', this)">🛞 B. Stint ve Lastik Analizi</button>
        <button class="tab-btn" onclick="switchTab('tab-diff', this)">⚖️ C. Sürücü Fark Analizi</button>
        <button class="tab-btn" onclick="switchTab('tab-tel', this)">📡 D. Telemetri ve Fren Aktivitesi Analizi</button>
        <button class="tab-btn" onclick="switchTab('tab-sec', this)">⏱️ E. Sektör Zaman Farkı ve Teorik Optimal Tur</button>
        <button class="tab-btn" onclick="switchTab('tab-qual', this)">🛡️ F. Veri Doğrulama ve Güvenilirlik</button>
    </div>

    <!-- B. STINT ANALİZİ -->
    <div id="tab-stint" class="tab-section active">
        <div class="methodology-note">⚠️ {html.escape(STINT_DEGRADATION_METHODOLOGY_WARNING)}</div>
        <div class="panel">
            <h3>📈 Stint Gelişimi, Lastik Hamuru ve Pit-Stop Noktaları</h3>
            {chart_stint_html}
        </div>
        <div class="panel">
            <h3>🎻 Tur Tutarlılığı ve Temiz Tur Süresi Dağılımı (Keman ve Kutu Grafikleri)</h3>
            {chart_dist_html}
        </div>
        <div class="panel">
            <h3>📊 Stint Metrikleri Tablosu (Ortalama, Medyan, En Hızlı Tur, Standart Sapma, Değişim Katsayısı — CV (%), Gözlenen ve Yakıt Düzeltilmiş Aşınma Eğimi, Lastik Yaşı)</h3>
            <div class="table-wrap">{_df_to_styled_html(stints_disp_df)}</div>
        </div>
        <div class="panel">
            <h3>🔄 Aynı Lastik Hamuru ve Benzer Lastik Yaşındaki Stintlerin Karşılaştırması</h3>
            <div class="table-wrap">{_df_to_styled_html(matching_stints)}</div>
        </div>
    </div>

    <!-- C. SÜRÜCÜ FARK ANALİZİ -->
    <div id="tab-diff" class="tab-section">
        <div class="info-note">
            ℹ️ <b>Fark Yönü Tanımı (Δ = {driver1} - {driver2}):</b>
            Negatif değerler (<b>&lt; 0 sn</b>) <b>{driver1}</b> sürücüsünün daha hızlı olduğunu;
            Pozitif değerler (<b>&gt; 0 sn</b>) <b>{driver2}</b> sürücüsünün daha hızlı olduğunu gösterir.
        </div>
        <div class="panel">
            <h3>📉 Tur Bazlı ve Kümülatif Tur Süresi Farkı Grafiği</h3>
            {chart_delta_html}
        </div>
        <div class="panel">
            <h3>🔬 Yardımcı İstatistiksel Analiz: Welch t-Testi</h3>
            <p>
                <b>Welch t-İstatistiği:</b> <code>{t_stat_str}</code> |
                <b>p-Değeri:</b> <code>{p_val_str}</code> |
                <b>Örneklem Sayısı:</b> <code>n1 = {ttest_det['n_driver1']}, n2 = {ttest_det['n_driver2']}</code> |
                <b>Ortalama Tur Süresi Farkı ({driver1} - {driver2}):</b> <code>{f"{ttest_det['mean_diff_sec']:+.3f} sn" if pd.notna(ttest_det.get("mean_diff_sec")) else "-"}</code>
            </p>
            <div class="methodology-note">⚠️ {html.escape(TTEST_METHODOLOGY_WARNING)}</div>
        </div>
    </div>

    <!-- D. TELEMETRİ ANALİZİ -->
    <div id="tab-tel" class="tab-section">
        <div class="info-note">
            ℹ️ {html.escape(BRAKE_ACTIVITY_DISCLAIMER)}
            <br><b>Seçilen Tur Süresi:</b> {driver1} Tur {lap_no_1} ({format_laptime_str(lap_sec_1)}, n={t_sum.get('d1_raw_samples_n', 0)}) —
            {driver2} Tur {lap_no_2} ({format_laptime_str(lap_sec_2)}, n={t_sum.get('d2_raw_samples_n', 0)}) |
            <b>Toplam Fren Süresi (BrakeDuration):</b> {driver1}={t_sum.get('d1_brake_duration_sec', 0):.2f} sn, {driver2}={t_sum.get('d2_brake_duration_sec', 0):.2f} sn |
            <b>Toplam Frenleme Mesafesi (BrakingDistance):</b> {driver1}={t_sum.get('d1_braking_distance_m', 0):.1f} m, {driver2}={t_sum.get('d2_braking_distance_m', 0):.1f} m
        </div>
        <div class="panel">
            <h3>🗺️ 2D Pist Hız Dominansı Haritası (Track Dominance Map)</h3>
            {chart_track_html}
        </div>
        <div class="panel">
            <h3>📡 Ortak Mesafe (Distance) Ekseninde Hizalanmış Telemetri Panelleri</h3>
            {chart_tel_html}
        </div>
        <div class="two-col">
            <div class="panel">
                <h3>🚦 Gaz Pedalı / Fren / Süzülme Sürüş Bölgesi Karşılaştırması</h3>
                <div class="table-wrap">{_df_to_styled_html(tel_comp["zone_comparison_df"], float_precision=1)}</div>
            </div>
            <div class="panel">
                <h3>🛑 {driver1} Fren Olayları ve Yavaşlama Analizi (BrakeEvents & BrakingDistance)</h3>
                <div class="table-wrap">{_df_to_styled_html(brake_ev_d1_disp, float_precision=2)}</div>
            </div>
        </div>
        <div class="panel">
            <h3>🛑 {driver2} Fren Olayları ve Yavaşlama Analizi (BrakeEvents & BrakingDistance)</h3>
            <div class="table-wrap">{_df_to_styled_html(brake_ev_d2_disp, float_precision=2)}</div>
        </div>
    </div>

    <!-- E. SEKTÖR VE OPTİMAL TUR ANALİZİ -->
    <div id="tab-sec" class="tab-section">
        <div class="panel">
            <h3>⏱️ Sektör Z-Skoru ve Sektör Zaman Farkı Analizi (Referans Sürücü: {ref_driver})</h3>
            {chart_sec_html}
            <div class="table-wrap" style="margin-top:16px;">{_df_to_styled_html(sector_disp_df)}</div>
        </div>
        <div class="panel">
            <h3>🎯 Teorik Optimal Tur ve En Hızlı Sektörlerin Geldiği Kaynak Turlar</h3>
            <div class="table-wrap">{opt_table_html}</div>
        </div>
    </div>

    <!-- F. VERİ DOĞRULAMA VE GÜVENİLİRLİK RAPORU -->
    <div id="tab-qual" class="tab-section">
        <div class="panel">
            <h3>🛡️ Tur Filtreleme Dökümü ve Örneklem Sayıları (n)</h3>
            <div class="table-wrap">{_df_to_styled_html(filter_summary_df)}</div>
        </div>
        <div class="two-col">
            <div class="panel">
                <h3>🔍 {driver1} — Filtrelenen Turlar ve Nedenleri</h3>
                <div class="table-wrap">{_df_to_styled_html(reasons_d1_disp)}</div>
            </div>
            <div class="panel">
                <h3>🔍 {driver2} — Filtrelenen Turlar ve Nedenleri</h3>
                <div class="table-wrap">{_df_to_styled_html(reasons_d2_disp)}</div>
            </div>
        </div>
        <div class="panel">
            <h3>🚨 Seans İçi Güvenlik Aracı (SC / VSC / Kırmızı Bayrak) Kayıtları</h3>
            <p>
                <b>Toplam Seans Turu:</b> {session_cond['total_session_laps']} |
                <b>SC Turları:</b> {session_cond['sc_laps'] or 'Yok'} |
                <b>VSC Turları:</b> {session_cond['vsc_laps'] or 'Yok'} |
                <b>Kırmızı Bayrak Turları:</b> {session_cond['red_flag_laps'] or 'Yok'}
            </p>
        </div>
    </div>
</div>

<script>
function showLoadingOverlay() {{
    var el = document.getElementById('loading-overlay');
    if (el) {{ el.style.display = 'flex'; }}
}}
function switchTab(tabId, btnElem) {{
    document.querySelectorAll('.tab-section').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btnElem.classList.add('active');
    // Sekme görünür olduğunda Plotly grafiklerinin genişliğini yeniden boyutlandır
    window.dispatchEvent(new Event('resize'));
}}
async function updateAvailableSessions() {{
    var yearSelect = document.querySelector('select[name="year"]');
    var eventSelect = document.querySelector('select[name="event"]');
    var sessionSelect = document.querySelector('select[name="session"]');
    if (!yearSelect || !eventSelect || !sessionSelect) return;

    var y = yearSelect.value;
    var ev = eventSelect.value;
    var prevSession = sessionSelect.value;

    try {{
        var resp = await fetch('/api/sessions?year=' + encodeURIComponent(y) + '&event=' + encodeURIComponent(ev));
        if (!resp.ok) return;
        var sessions = await resp.json();
        if (!Array.isArray(sessions) || sessions.length === 0) return;

        sessionSelect.innerHTML = '';
        var foundPrev = false;
        sessions.forEach(function(s) {{
            var opt = document.createElement('option');
            opt.value = s.code;
            opt.textContent = s.full_label || (s.code + ' — ' + s.label);
            if (s.code === prevSession) {{
                opt.selected = true;
                foundPrev = true;
            }}
            sessionSelect.appendChild(opt);
        }});

        if (!foundPrev) {{
            var hasR = sessions.some(function(s) {{ return s.code === 'R'; }});
            sessionSelect.value = hasR ? 'R' : sessions[0].code;
        }}
    }} catch (e) {{
        console.warn('Seans listesi güncellenemedi:', e);
    }}
}}
document.addEventListener('DOMContentLoaded', function() {{
    var eventSelect = document.querySelector('select[name="event"]');
    var yearSelect = document.querySelector('select[name="year"]');
    if (eventSelect) eventSelect.addEventListener('change', updateAvailableSessions);
    if (yearSelect) yearSelect.addEventListener('change', updateAvailableSessions);
}});
</script>
</body>
</html>"""

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_doc)

    return html_doc


# ==============================================================================
# LOCALHOST HTTP SUNUCUSU
# ==============================================================================

class F1DashboardRequestHandler(BaseHTTPRequestHandler):
    """Tarayıcıdan gelen GET isteklerini okur, ilgili F1 analizini HTML olarak döndürür."""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        # Dinamik seans listesi API uç noktası
        if parsed.path == "/api/sessions":
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                y = int(qs.get("year", ["2026"])[0])
            except (ValueError, TypeError):
                y = 2026
            ev = qs.get("event", ["Australian Grand Prix"])[0]
            avail = get_event_available_sessions(y, ev)
            clean_list = [
                {
                    "code": m["code"],
                    "name": m["name"],
                    "label": m["label"],
                    "full_label": m["full_label"],
                    "session_number": m.get("session_number", 1),
                }
                for m in avail.values()
            ]
            resp_bytes = json.dumps(clean_list, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return

        if parsed.path not in ("/", "/index.html", "/dashboard"):
            self.send_response(404)
            self.end_headers()
            return

        qs = urllib.parse.parse_qs(parsed.query)
        year = int(qs.get("year", ["2025"])[0])
        event = qs.get("event", ["Australian Grand Prix"])[0]
        session_type = qs.get("session", ["R"])[0]
        d1 = qs.get("d1", ["NOR"])[0]
        d2 = qs.get("d2", ["VER"])[0]
        ref = qs.get("ref", [d1])[0]
        stint = qs.get("stint", ["Tümü"])[0]
        compound = qs.get("compound", ["Tümü"])[0]
        lap = qs.get("lap", ["Fastest"])[0]
        bw = int(qs.get("bw", ["20"])[0])
        retry = qs.get("retry", ["0"])[0] == "1"

        try:
            body_str = generate_dashboard_html(
                year=year,
                event_name=event,
                session_type=session_type,
                driver1=d1,
                driver2=d2,
                ref_driver=ref,
                lap_number=lap,
                stint_filter=stint,
                compound_filter=compound,
                brake_window_m=bw,
                by_stint_iqr=True,
                force_reload=retry,
            )
            body_bytes = body_str.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)
        except Exception as exc:
            err_bytes = f"<h1>Analiz Hatası</h1><pre>{html.escape(str(exc))}</pre>".encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(err_bytes)

    def log_message(self, format: str, *args: Any) -> None:
        # Terminal çıktısını kirletmemek için sessiz loglama
        return


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def serve_html_on_localhost(
    host: str = "127.0.0.1",
    port: int = 8050,
    open_browser: bool = True,
    block: bool = True,
    initial_year: int = 2025,
    initial_event: str = "Australian Grand Prix",
    initial_driver1: str = "NOR",
    initial_driver2: str = "VER",
) -> str:
    """
    HTML analiz raporunu hem reports/f1_dashboard.html dosyasına kaydeder hem de
    http://localhost:8050 adresinde canlı olarak yayınlar.
    """
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    html_file_path = os.path.join(reports_dir, "f1_dashboard.html")

    generate_dashboard_html(
        year=initial_year,
        event_name=initial_event,
        session_type="R",
        driver1=initial_driver1,
        driver2=initial_driver2,
        ref_driver=initial_driver1,
        save_path=html_file_path,
    )

    url = f"http://localhost:{port}"
    print(f"\n[HTML RAPOR] Statik HTML dosyası kaydedildi: {html_file_path}")

    if _is_port_in_use(host, port):
        print(f"[LOCALHOST] HTML Dashboard sunucusu zaten çalışıyor: {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return url

    server = ThreadingHTTPServer((host, port), F1DashboardRequestHandler)
    print(f"[LOCALHOST] İnteraktif HTML Analiz Dashboard yayında: {url}")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if block:
        print("Sunucuyu durdurmak için Ctrl+C tuşlarına basın...")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nLocalhost sunucusu kapatıldı.")
            server.server_close()
    else:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

    return url


if __name__ == "__main__":
    serve_html_on_localhost(port=8050, open_browser=False, block=True)

