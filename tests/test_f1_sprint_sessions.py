"""
Unit Tests for Sprint Sessions and Dynamic Session Resolution
(tests/test_f1_sprint_sessions.py)
"""

import unittest
from f1_data_loader import (
    SESSION_TYPE_OPTIONS,
    SESSION_NAME_TO_CODE,
    CODE_TO_SESSION_NAME,
    normalize_session_code,
    get_event_available_sessions,
    load_f1_session,
    get_last_load_diagnostics,
    DEFAULT_CACHE_DIR,
)


class TestSprintSessionMappingsAndResolution(unittest.TestCase):
    def test_session_type_options_and_mappings(self):
        """Resmî oturum kodlarının ve Türkçe etiketlerinin doğrulanması."""
        self.assertIn("SQ", SESSION_TYPE_OPTIONS)
        self.assertIn("SS", SESSION_TYPE_OPTIONS)
        self.assertIn("S", SESSION_TYPE_OPTIONS)
        self.assertIn("Q", SESSION_TYPE_OPTIONS)
        self.assertIn("R", SESSION_TYPE_OPTIONS)
        self.assertIn("FP1", SESSION_TYPE_OPTIONS)
        self.assertIn("FP2", SESSION_TYPE_OPTIONS)
        self.assertIn("FP3", SESSION_TYPE_OPTIONS)

        self.assertEqual(SESSION_TYPE_OPTIONS["SQ"], "Sprint Sıralama")
        self.assertEqual(SESSION_TYPE_OPTIONS["SS"], "Sprint Sıralama (Shootout)")
        self.assertEqual(SESSION_TYPE_OPTIONS["S"], "Sprint Yarışı")
        self.assertEqual(SESSION_TYPE_OPTIONS["Q"], "Sıralama Turları")
        self.assertEqual(SESSION_TYPE_OPTIONS["R"], "Ana Yarış")

        self.assertEqual(SESSION_NAME_TO_CODE["Sprint Qualifying"], "SQ")
        self.assertEqual(SESSION_NAME_TO_CODE["Sprint Shootout"], "SS")
        self.assertEqual(SESSION_NAME_TO_CODE["Sprint"], "S")
        self.assertEqual(SESSION_NAME_TO_CODE["Qualifying"], "Q")
        self.assertEqual(SESSION_NAME_TO_CODE["Race"], "R")

    def test_normalize_session_code(self):
        """Oturum kodu normalizasyon fonksiyonunun farklı girdilerle testi."""
        self.assertEqual(normalize_session_code("SQ"), "SQ")
        self.assertEqual(normalize_session_code("sprint qualifying"), "SQ")
        self.assertEqual(normalize_session_code("SS"), "SS")
        self.assertEqual(normalize_session_code("sprint shootout"), "SS")
        self.assertEqual(normalize_session_code("S"), "S")
        self.assertEqual(normalize_session_code("sprint"), "S")
        self.assertEqual(normalize_session_code("sprint yarışı"), "S")
        self.assertEqual(normalize_session_code("Q"), "Q")
        self.assertEqual(normalize_session_code("qualifying"), "Q")
        self.assertEqual(normalize_session_code("sıralama turları"), "Q")
        self.assertEqual(normalize_session_code("R"), "R")
        self.assertEqual(normalize_session_code("race"), "R")
        self.assertEqual(normalize_session_code("ana yarış"), "R")
        self.assertEqual(normalize_session_code("FP1"), "FP1")
        self.assertEqual(normalize_session_code("practice 1"), "FP1")

        # 2021-2022 sezonlarında 'Sprint Qualifying' sprint yarışını ifade eder
        self.assertEqual(normalize_session_code("SQ", year=2021), "S")
        self.assertEqual(normalize_session_code("Sprint Qualifying", year=2022), "S")

        # 2023 sezonunda sprint sıralaması Sprint Shootout'tur (SS)
        self.assertEqual(normalize_session_code("sprint qualifying", year=2023), "SS")

    def test_dynamic_session_detection_by_era(self):
        """Hafta sonu formatına ve yıllara göre geçerli seansların tespiti."""
        # 1. 2024 Sprint Qualifying formatı (örn. 2024 Chinese GP)
        avail_24 = get_event_available_sessions(2024, "Chinese Grand Prix")
        self.assertIn("FP1", avail_24)
        self.assertIn("SQ", avail_24)
        self.assertIn("S", avail_24)
        self.assertIn("Q", avail_24)
        self.assertIn("R", avail_24)
        self.assertNotIn("FP2", avail_24)
        self.assertNotIn("FP3", avail_24)
        self.assertNotIn("SS", avail_24)

        # 2. 2023 Sprint Shootout formatı (örn. 2023 Austrian GP)
        avail_23 = get_event_available_sessions(2023, "Austrian Grand Prix")
        self.assertIn("FP1", avail_23)
        self.assertIn("Q", avail_23)
        self.assertIn("SS", avail_23)
        self.assertIn("S", avail_23)
        self.assertIn("R", avail_23)
        self.assertNotIn("FP2", avail_23)
        self.assertNotIn("FP3", avail_23)
        self.assertNotIn("SQ", avail_23)

        # 3. 2021-2022 Sprint formatı (örn. 2021 British GP)
        avail_21 = get_event_available_sessions(2021, "British Grand Prix")
        self.assertIn("FP1", avail_21)
        self.assertIn("Q", avail_21)
        self.assertIn("FP2", avail_21)
        self.assertIn("S", avail_21)
        self.assertIn("R", avail_21)
        self.assertNotIn("FP3", avail_21)
        self.assertNotIn("SQ", avail_21)
        self.assertNotIn("SS", avail_21)

        # 4. Klasik (Conventional) format (örn. 2025/2026 Australian GP)
        avail_conv = get_event_available_sessions(2026, "Australian Grand Prix")
        self.assertIn("FP1", avail_conv)
        self.assertIn("FP2", avail_conv)
        self.assertIn("FP3", avail_conv)
        self.assertIn("Q", avail_conv)
        self.assertIn("R", avail_conv)
        self.assertNotIn("SQ", avail_conv)
        self.assertNotIn("SS", avail_conv)
        self.assertNotIn("S", avail_conv)

    def test_invalid_session_handling_on_sprint_weekend(self):
        """Sprint hafta sonunda FP3 gibi bulunmayan bir seans seçildiğinde tutarlı hata yönetimi."""
        sess, err = load_f1_session(2024, "Chinese Grand Prix", "FP3")
        self.assertIsNone(sess)
        self.assertIsNotNone(err)
        self.assertIn("programında bulunmamaktadır", err)

        diag = get_last_load_diagnostics(2024, "Chinese Grand Prix", "FP3")
        self.assertEqual(diag.get("error_category"), "INVALID_SESSION_FOR_EVENT")
        self.assertIn("Geçersiz Seans Tipi", diag.get("error_category_label", ""))

    def test_cached_sprint_sessions_load_cleanly(self):
        """Önbelleğe alınmış 2023 ve 2024 sprint oturumlarının başarıyla yüklenmesi."""
        # 2024 Çin GP SQ
        sess_sq, err_sq = load_f1_session(2024, "Chinese Grand Prix", "SQ")
        self.assertIsNotNone(sess_sq)
        self.assertIsNone(err_sq)
        self.assertEqual(sess_sq.name, "Sprint Qualifying")

        # 2024 Çin GP S (Sprint Race)
        sess_s24, err_s24 = load_f1_session(2024, "Chinese Grand Prix", "S")
        self.assertIsNotNone(sess_s24)
        self.assertIsNone(err_s24)
        self.assertEqual(sess_s24.name, "Sprint")

        # 2023 Avusturya GP SS (Sprint Shootout)
        sess_ss, err_ss = load_f1_session(2023, "Austrian Grand Prix", "SS")
        self.assertIsNotNone(sess_ss)
        self.assertIsNone(err_ss)
        self.assertEqual(sess_ss.name, "Sprint Shootout")

        # 2023 Avusturya GP S (Sprint Race)
        sess_s23, err_s23 = load_f1_session(2023, "Austrian Grand Prix", "S")
        self.assertIsNotNone(sess_s23)
        self.assertIsNone(err_s23)
        self.assertEqual(sess_s23.name, "Sprint")


if __name__ == "__main__":
    unittest.main()

