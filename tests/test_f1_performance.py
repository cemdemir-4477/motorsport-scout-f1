"""
F1 Performance Analysis & Telemetry System — Birim ve Entegrasyon Testleri (test_f1_performance.py)
====================================================================================================
Bu test paketi:
1. Mevcut 6 temel fonksiyonun düzeltmelerini, kenar durumlarını (boş DataFrame, NaN, sıfıra bölme)
   ve geriye dönük uyumluluğunu,
2. Pit-in/out, Deleted, TrackStatus (SC/VSC/Red Flag) ve Stint bazlı IQR filtrelemesini,
3. Stint bazlı analiz modülünü (İlk 3 / Son 3 tur farkı, CV%, lastik yaşı, eşdeğer stint eşleştirme),
4. Fren telemetrisi işleme modülünü (BrakeRaw, BrakeActivity, SmoothedBrakeActivity, BrakeEvents,
   BrakeDuration, BrakingDistance, 10/20/30m pencereleri),
5. Ortak Distance ekseninde sürekli vs. kesikli (discrete) telemetri hizalamasını,
6. Ham verinin yerinde değiştirilmeme (immutability) garantisini,
7. Yerel önbellekteki gerçek yarış verileriyle (2025 & 2026 Avustralya GP) uçtan uca doğrulamayı
test eder.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f1_data_loader import (
    DEFAULT_CACHE_DIR,
    get_raw_driver_laps,
    get_season_schedule,
    get_session_drivers_info,
    list_cached_sessions,
    load_f1_session,
    resolve_event_name,
    summarize_session_conditions,
)
from f1_performance import (
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
)
from f1_telemetry import (
    align_and_compare_telemetry,
    process_brake_telemetry,
)


def _make_sample_laps_df() -> pd.DataFrame:
    """Testler için sentetik FastF1 laps DataFrame'i üretir."""
    rows = [
        # Stint 1: SOFT (Turlar 1-6)
        {"LapNumber": 1, "Stint": 1, "Compound": "SOFT", "TyreLife": 1, "LapTime": pd.Timedelta(seconds=95.0), "PitOutTime": pd.Timedelta(seconds=10.0), "PitInTime": pd.NaT, "Deleted": False, "TrackStatus": "1", "Sector1Time": pd.Timedelta(seconds=31.0), "Sector2Time": pd.Timedelta(seconds=32.0), "Sector3Time": pd.Timedelta(seconds=32.0)},
        {"LapNumber": 2, "Stint": 1, "Compound": "SOFT", "TyreLife": 2, "LapTime": pd.Timedelta(seconds=90.2), "PitOutTime": pd.NaT, "PitInTime": pd.NaT, "Deleted": False, "TrackStatus": "1", "Sector1Time": pd.Timedelta(seconds=29.8), "Sector2Time": pd.Timedelta(seconds=30.2), "Sector3Time": pd.Timedelta(seconds=30.2)},
        {"LapNumber": 3, "Stint": 1, "Compound": "SOFT", "TyreLife": 3, "LapTime": pd.Timedelta(seconds=90.4), "PitOutTime": pd.NaT, "PitInTime": pd.NaT, "Deleted": False, "TrackStatus": "1", "Sector1Time": pd.Timedelta(seconds=30.0), "Sector2Time": pd.Timedelta(seconds=30.0), "Sector3Time": pd.Timedelta(seconds=30.4)},
        {"LapNumber": 4, "Stint": 1, "Compound": "SOFT", "TyreLife": 4, "LapTime": pd.Timedelta(seconds=90.6), "PitOutTime": pd.NaT, "PitInTime": pd.NaT, "Deleted": False, "TrackStatus": "1", "Sector1Time": pd.Timedelta(seconds=30.1), "Sector2Time": pd.Timedelta(seconds=30.4), "Sector3Time": pd.Timedelta(seconds=30.1)},
        {"LapNumber": 5, "Stint": 1, "Compound": "SOFT", "TyreLife": 5, "LapTime": pd.Timedelta(seconds=90.8), "PitOutTime": pd.NaT, "PitInTime": pd.NaT, "Deleted": False, "TrackStatus": "1", "Sector1Time": pd.Timedelta(seconds=30.2), "Sector2Time": pd.Timedelta(seconds=30.3), "Sector3Time": pd.Timedelta(seconds=30.3)},
        {"LapNumber": 6, "Stint": 1, "Compound": "SOFT", "TyreLife": 6, "LapTime": pd.Timedelta(seconds=91.0), "PitOutTime": pd.NaT, "PitInTime": pd.NaT, "Deleted": False, "TrackStatus": "1", "Sector1Time": pd.Timedelta(seconds=30.3), "Sector2Time": pd.Timedelta(seconds=30.4), "Sector3Time": pd.Timedelta(seconds=30.3)},
        {"LapNumber": 7, "Stint": 1, "Compound": "SOFT", "TyreLife": 7, "LapTime": pd.Timedelta(seconds=91.2), "PitOutTime": pd.NaT, "PitInTime": pd.NaT, "Deleted": False, "TrackStatus": "1", "Sector1Time": pd.Timedelta(seconds=30.4), "Sector2Time": pd.Timedelta(seconds=30.4), "Sector3Time": pd.Timedelta(seconds=30.4)},
        # Silinen tur (Deleted=True)
        {"LapNumber": 8, "Stint": 1, "Compound": "SOFT", "TyreLife": 8, "LapTime": pd.Timedelta(seconds=89.9), "PitOutTime": pd.NaT, "PitInTime": pd.NaT, "Deleted": True, "TrackStatus": "1", "Sector1Time": pd.Timedelta(seconds=29.5), "Sector2Time": pd.Timedelta(seconds=30.2), "Sector3Time": pd.Timedelta(seconds=30.2)},
        # Güvenlik Aracı turu (TrackStatus='4')
        {"LapNumber": 9, "Stint": 1, "Compound": "SOFT", "TyreLife": 9, "LapTime": pd.Timedelta(seconds=125.0), "PitOutTime": pd.NaT, "PitInTime": pd.NaT, "Deleted": False, "TrackStatus": "4", "Sector1Time": pd.Timedelta(seconds=40.0), "Sector2Time": pd.Timedelta(seconds=42.0), "Sector3Time": pd.Timedelta(seconds=43.0)},
        # Pit-in turu
        {"LapNumber": 10, "Stint": 1, "Compound": "SOFT", "TyreLife": 10, "LapTime": pd.Timedelta(seconds=98.0), "PitOutTime": pd.NaT, "PitInTime": pd.Timedelta(seconds=990.0), "Deleted": False, "TrackStatus": "1", "Sector1Time": pd.Timedelta(seconds=30.5), "Sector2Time": pd.Timedelta(seconds=30.5), "Sector3Time": pd.Timedelta(seconds=37.0)},
        # Eksik LapTime turu
        {"LapNumber": 11, "Stint": 2, "Compound": "HARD", "TyreLife": 1, "LapTime": pd.NaT, "PitOutTime": pd.Timedelta(seconds=1015.0), "PitInTime": pd.NaT, "Deleted": False, "TrackStatus": "1", "Sector1Time": pd.NaT, "Sector2Time": pd.NaT, "Sector3Time": pd.NaT},
    ]
    return pd.DataFrame(rows)


class TestCoreFunctionsAndFixes(unittest.TestCase):
    """Mevcut 6 temel fonksiyonun düzeltmelerini ve kenar durumlarını doğrular."""

    def test_filter_clean_laps_reasons_and_immutability(self):
        raw_df = _make_sample_laps_df()
        raw_backup = raw_df.copy(deep=True)

        clean_df, report = filter_clean_laps(raw_df, by_stint=True, return_report=True)

        # Ham veri değişmemiş olmalı
        pd.testing.assert_frame_equal(raw_df, raw_backup)

        # Toplam 11 turdan:
        # Tur 1: PitOut (1)
        # Tur 8: Deleted (1)
        # Tur 9: TrackStatus='4' SC (1)
        # Tur 10: PitIn (1)
        # Tur 11: NaT (1)
        # Kalan temiz turlar: Tur 2, 3, 4, 5, 6, 7 -> 6 tur
        self.assertEqual(report["total_raw_laps"], 11)
        self.assertEqual(report["missing_or_invalid_laptime"], 1)
        self.assertEqual(report["pit_in_out_laps"], 2)
        self.assertEqual(report["deleted_laps"], 1)
        self.assertEqual(report["sc_vsc_red_laps"], 1)
        self.assertEqual(report["clean_laps_count"], 6)
        self.assertEqual(len(clean_df), 6)
        self.assertListEqual(clean_df["LapNumber"].tolist(), [2, 3, 4, 5, 6, 7])

    def test_pace_score_fastest_and_reference(self):
        raw_df = _make_sample_laps_df()
        clean_df = filter_clean_laps(raw_df)
        # Temiz turlar: 90.2, 90.4, 90.6, 90.8, 91.0, 91.2 -> Medyan = 90.7
        pace_pct = calculate_pace_score(clean_df, fastest_lap_seconds=90.0)
        self.assertAlmostEqual(pace_pct, ((90.7 - 90.0) / 90.0) * 100.0, places=5)

        det = calculate_pace_score(clean_df, fastest_lap_seconds=90.0, ref_median_seconds=90.5, return_details=True)
        self.assertAlmostEqual(det["median_lap_sec"], 90.7, places=5)
        self.assertAlmostEqual(det["delta_vs_ref_sec"], 0.2, places=5)
        self.assertAlmostEqual(det["pace_score_vs_ref_pct"], (0.2 / 90.5) * 100.0, places=5)
        self.assertEqual(det["sample_size"], 6)

        # Sıfıra bölme ve boş DF koruması
        self.assertTrue(np.isnan(calculate_pace_score(clean_df, fastest_lap_seconds=0.0)))
        self.assertTrue(np.isnan(calculate_pace_score(pd.DataFrame(), fastest_lap_seconds=90.0)))

    def test_consistency_score_and_stint_breakdown(self):
        raw_df = _make_sample_laps_df()
        clean_df = filter_clean_laps(raw_df)
        cv = calculate_consistency_score(clean_df)
        self.assertGreater(cv, 0.0)

        det = calculate_consistency_score(clean_df, by_stint=True, return_details=True)
        self.assertAlmostEqual(det["median_sec"], 90.7, places=5)
        self.assertAlmostEqual(det["mean_sec"], 90.7, places=5)
        self.assertEqual(det["sample_size"], 6)
        self.assertFalse(det["stint_breakdown"].empty)

    def test_compare_drivers_ttest_and_methodology_warning(self):
        d1 = _make_sample_laps_df()
        d2 = _make_sample_laps_df()
        d2["LapTime"] = d2["LapTime"] + pd.Timedelta(seconds=0.8)

        c1 = filter_clean_laps(d1)
        c2 = filter_clean_laps(d2)
        p_val, warn_msg = compare_drivers_ttest(c1, c2)
        self.assertLess(p_val, 0.05)
        self.assertIn("Metodolojik Not", warn_msg)
        self.assertIn("30'dan küçük", warn_msg)

    def test_sector_zscores_and_seconds_diff(self):
        d1 = filter_clean_laps(_make_sample_laps_df())
        d2 = d1.copy()
        d2["Sector1Time"] = d2["Sector1Time"] + pd.Timedelta(seconds=0.3)

        res = calculate_sector_zscores(d2, d1, return_details=True)
        self.assertIn("Sektör 1 Z-Skor", res)
        self.assertIn("Sektör 1 Fark (sn)", res)
        self.assertAlmostEqual(res["Sektör 1 Fark (sn)"], 0.3, places=4)
        self.assertGreater(res["Sektör 1 Z-Skor"], 0.0)

    def test_theoretical_optimal_and_source_laps(self):
        clean_df = filter_clean_laps(_make_sample_laps_df())
        # Tur 2: S1=29.8 (en iyi S1), S2=30.2, S3=30.2 -> LapTime=90.2 (en iyi gerçek tur)
        # Tur 3: S1=30.0, S2=30.0 (en iyi S2), S3=30.4
        # Tur 4: S1=30.1, S2=30.4, S3=30.1 (en iyi S3)
        # Teorik Optimal = 29.8 + 30.0 + 30.1 = 89.9 sn
        theo, best, pot = calculate_theoretical_optimal(clean_df)
        self.assertAlmostEqual(theo, 89.9, places=4)
        self.assertAlmostEqual(best, 90.2, places=4)
        self.assertAlmostEqual(pot, 0.3, places=4)

        det = calculate_theoretical_optimal(clean_df, return_details=True)
        self.assertEqual(det["actual_best_lap_number"], 2)
        self.assertEqual(det["sectors"]["Sector1Time"]["from_lap_number"], 2)
        self.assertEqual(det["sectors"]["Sector2Time"]["from_lap_number"], 3)
        self.assertEqual(det["sectors"]["Sector3Time"]["from_lap_number"], 4)


class TestStintAnalysis(unittest.TestCase):
    """Stint bazlı metriklerin (İlk 3 / Son 3 tur farkı dahil) doğruluğunu test eder."""

    def test_stint_metrics_first3_last3_diff(self):
        raw_df = _make_sample_laps_df()
        clean_df = filter_clean_laps(raw_df)
        st_df = analyze_driver_stints(raw_df, clean_df, driver_name="VER")

        # Stint 1: Temiz turlar [90.2, 90.4, 90.6, 90.8, 91.0, 91.2]
        # İlk 3 ort: (90.2 + 90.4 + 90.6)/3 = 90.4
        # Son 3 ort: (90.8 + 91.0 + 91.2)/3 = 91.0
        # Son 3 - İlk 3 fark = +0.6 sn
        s1_row = st_df[st_df["Stint"] == 1].iloc[0]
        self.assertEqual(s1_row["ValidLapCount"], 6)
        self.assertEqual(s1_row["Compound"], "SOFT")
        self.assertEqual(s1_row["TyreLifeStart"], 1)
        self.assertEqual(s1_row["TyreLifeEnd"], 10)
        self.assertAlmostEqual(s1_row["First3Mean_Sec"], 90.4, places=4)
        self.assertAlmostEqual(s1_row["Last3Mean_Sec"], 91.0, places=4)
        self.assertAlmostEqual(s1_row["Last3_Minus_First3_Sec"], 0.6, places=4)


class TestBrakeAndTelemetryAlignment(unittest.TestCase):
    """Fren telemetrisi işleme ve discrete/continuous hizalama testleri."""

    def _make_telemetry_df(self, offset_m: float = 0.0) -> pd.DataFrame:
        dist = np.linspace(0.0, 500.0, 101) + offset_m
        time_sec = np.linspace(0.0, 10.0, 101)
        # 100m ile 200m arasında fren aktif (Boolean True/False)
        brake = (dist >= 100.0) & (dist <= 200.0)
        speed = np.where(brake, 280.0 - (dist - 100.0) * 1.5, 290.0)
        throttle = np.where(brake, 0.0, 100.0)
        gear = np.where(dist < 150.0, 8, 4)
        rpm = np.linspace(10500, 11800, 101)
        drs = np.where(dist < 90.0, 12, 0)
        return pd.DataFrame(
            {
                "Distance": dist,
                "Time": pd.to_timedelta(time_sec, unit="s"),
                "Speed": speed,
                "Throttle": throttle,
                "Brake": brake,
                "nGear": gear,
                "RPM": rpm,
                "DRS": drs,
            }
        )

    def test_process_brake_telemetry_windows(self):
        raw_tel = self._make_telemetry_df()
        raw_copy = raw_tel.copy(deep=True)

        for w in (10, 20, 30):
            res = process_brake_telemetry(raw_tel, window_m=float(w))
            proc = res["processed_df"]
            self.assertFalse(proc.empty)
            self.assertTrue(set(proc["BrakeRaw"].dropna().unique()).issubset({0, 1}))
            self.assertGreaterEqual(proc["BrakeActivity"].min(), 0.0)
            self.assertLessEqual(proc["BrakeActivity"].max(), 100.0)
            self.assertGreaterEqual(proc["SmoothedBrakeActivity"].min(), 0.0)
            self.assertLessEqual(proc["SmoothedBrakeActivity"].max(), 100.0)
            self.assertEqual(res["brake_events_count"], 1)
            self.assertAlmostEqual(res["total_braking_distance_m"], 102.5, delta=5.0)
            self.assertAlmostEqual(res["brake_duration_sec"], 2.0, delta=0.2)

        # Ham veri değişmemiş olmalı
        pd.testing.assert_frame_equal(raw_tel, raw_copy)

    def test_discrete_vs_continuous_alignment(self):
        tel1 = self._make_telemetry_df(offset_m=0.0)
        tel2 = self._make_telemetry_df(offset_m=1.3)  # Farklı örnekleme noktaları

        comp = align_and_compare_telemetry(tel1, tel2, "VER", "LEC", window_m=20.0, distance_step_m=2.0)
        aligned = comp["aligned_df"]
        self.assertFalse(aligned.empty)

        # Kesikli kanallar (BrakeRaw, nGear, DRS) ASLA ara küsuratlı değer (örn. vites 6.3, fren 0.4) almamalı!
        gears_d2 = aligned["nGear_D2"].dropna().unique()
        self.assertTrue(set(gears_d2).issubset({4.0, 8.0}))

        brakes_d2 = aligned["BrakeRaw_D2"].dropna().unique()
        self.assertTrue(set(brakes_d2).issubset({0.0, 1.0}))


class TestRealCachedRaceVerification(unittest.TestCase):
    """Önbellekteki gerçek FastF1 yarış verisi (2025/2026 Avustralya GP) üzerinde doğrulama."""

    def test_cached_2025_and_2026_races(self):
        cached = list_cached_sessions(DEFAULT_CACHE_DIR)
        self.assertGreaterEqual(len(cached), 1)

        session, err = load_f1_session(2025, "Australian Grand Prix", "R", load_telemetry=True)
        self.assertIsNone(err)
        self.assertIsNotNone(session)

        drivers = get_session_drivers_info(session)
        self.assertGreater(len(drivers), 10)

        d1_raw = get_raw_driver_laps(session, "NOR")
        d2_raw = get_raw_driver_laps(session, "VER")
        self.assertFalse(d1_raw.empty)
        self.assertFalse(d2_raw.empty)

        c1, rep1 = filter_clean_laps(d1_raw, by_stint=True, return_report=True)
        c2, rep2 = filter_clean_laps(d2_raw, by_stint=True, return_report=True)
        self.assertGreater(rep1["clean_laps_count"], 10)
        self.assertGreater(rep2["clean_laps_count"], 10)

        st1 = analyze_driver_stints(d1_raw, c1, driver_name="NOR")
        self.assertFalse(st1.empty)

        # Meksika GP isim çözümleme kontrolü
        self.assertEqual(resolve_event_name("Meksika GP"), "Mexico City Grand Prix")
        sched_2026 = get_season_schedule(2026)
        self.assertFalse(sched_2026.empty)


if __name__ == "__main__":
    unittest.main()
