import streamlit as st
import fastf1
import pandas as pd
import os

# Mevcut f1_performance.py dosyasındaki hesaplama fonksiyonlarını içe aktarıyoruz
from f1_performance import (
    filter_clean_laps,
    calculate_pace_score,
    calculate_consistency_score,
    compare_drivers_ttest,
    calculate_sector_zscores
)

# Streamlit sayfası temel ayarları
st.set_page_config(page_title="Sürücü Performans Skoru", layout="wide")

# Cache dizini ayarı (FastF1 verilerini tekrar tekrar indirmemek için)
os.makedirs("scratch/fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache('scratch/fastf1_cache')

st.title("🏎️ Sürücü Performans Skoru")
st.markdown("Bu uygulama, seçtiğiniz yarış ve iki pilot üzerinden temiz turları (medyan ± 1.5×IQR) filtreleyerek pace, tutarlılık ve sektör performanslarını istatistiksel olarak karşılaştırır.")

# --- Cache'li Veri Yükleme Fonksiyonları ---

@st.cache_data(show_spinner=False)
def get_season_schedule(year):
    """Belirtilen yılın yarış takvimini çeker."""
    try:
        schedule = fastf1.get_event_schedule(year)
        # Sadece test (testing) olmayan gerçek hafta sonlarını alalım
        schedule = schedule[schedule['EventFormat'] != 'testing']
        return schedule
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_session(year, event_name):
    """Belirtilen yarışın (R = Race) seans verisini çeker."""
    try:
        session = fastf1.get_session(year, event_name, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        return session
    except Exception as e:
        return None

# --- YAN MENÜ (SIDEBAR) ---
st.sidebar.header("⚙️ Ayarlar")

# 1. Sezon Yılı Seçimi
selected_year = st.sidebar.selectbox(
    "Sezon Seçin", 
    list(range(2024, 2017, -1)), 
    index=0
)

# 2. Yarış Seçimi
schedule = get_season_schedule(selected_year)

if schedule.empty:
    st.sidebar.error("Takvim verisi çekilemedi.")
    events = []
else:
    events = schedule['EventName'].tolist()

selected_event = st.sidebar.selectbox("Yarış Seçin", events)

# Pilot Listesini Çekme (Yan menüde veriyi önceden yüklemek zorundayız)
driver_options = []
session = None

if selected_event:
    with st.sidebar:
        with st.spinner("Yarış verisi yükleniyor (ilk seferde zaman alabilir)..."):
            session = load_session(selected_year, selected_event)
            
            if session is not None and len(session.drivers) > 0:
                for drv in session.drivers:
                    try:
                        info = session.get_driver(drv)
                        if 'Abbreviation' in info:
                            driver_options.append(info['Abbreviation'])
                    except Exception:
                        continue

# 3. Pilot Seçimleri
if driver_options:
    driver1 = st.sidebar.selectbox("Pilot 1", driver_options, index=0)
    # Eğer birden fazla pilot varsa ikinciyi otomatik farklı seç
    driver2_default_index = 1 if len(driver_options) > 1 else 0
    driver2 = st.sidebar.selectbox("Pilot 2", driver_options, index=driver2_default_index)
else:
    driver1, driver2 = None, None
    if selected_event:
        st.sidebar.error("Bu yarış için pilot listesi bulunamadı.")

# 4. Analiz Butonu
analyze_btn = st.sidebar.button("Analiz Et", type="primary")


# --- ANA EKRAN ANALİZ BÖLÜMÜ ---

if analyze_btn:
    if not driver1 or not driver2:
        st.warning("Lütfen iki pilot seçtiğinizden emin olun.")
    elif driver1 == driver2:
        st.warning("Lütfen karşılaştırma için iki farklı pilot seçin.")
    elif session is None:
        st.error("Yarış verisi (Session) yüklenemedi. Belki seans iptal edilmiş veya veri henüz hazır olmayabilir.")
    else:
        with st.spinner(f"**{driver1}** ve **{driver2}** için performans skorları hesaplanıyor..."):
            try:
                laps = session.laps
                
                # Pilotların tur verilerini al
                d1_laps = laps.pick_drivers(driver1)
                d2_laps = laps.pick_drivers(driver2)
                
                if d1_laps.empty or d2_laps.empty:
                    st.error("Seçilen pilotların birinin veya ikisinin de bu yarışta tur verisi bulunamadı (Yarışa başlamamış olabilirler).")
                else:
                    # 1. Temiz Tur Filtreleme
                    clean_d1 = filter_clean_laps(d1_laps)
                    clean_d2 = filter_clean_laps(d2_laps)
                    
                    # 2. Tüm yarışın en hızlı turunu bul
                    fastest_lap_seconds = laps['LapTime'].dt.total_seconds().min()
                    
                    # 3. Skorları Hesapla
                    pace_d1 = calculate_pace_score(clean_d1, fastest_lap_seconds)
                    pace_d2 = calculate_pace_score(clean_d2, fastest_lap_seconds)
                    
                    cons_d1 = calculate_consistency_score(clean_d1)
                    cons_d2 = calculate_consistency_score(clean_d2)
                    
                    # 4. İstatistiksel Karşılaştırma
                    p_val, warning_msg = compare_drivers_ttest(clean_d1, clean_d2)
                    
                    # 5. Sektör Ayrıştırma (Referans: Pilot 1)
                    sec_z_d1 = calculate_sector_zscores(clean_d1, clean_d1)
                    sec_z_d2 = calculate_sector_zscores(clean_d2, clean_d1)
                    
                    
                    # --- SONUÇLARI GÖSTER ---
                    
                    st.subheader("📊 Performans Tablosu")
                    
                    # Verileri Tablolaştırma
                    results = [
                        {
                            'Pilot': driver1,
                            'Temiz Tur Sayısı': len(clean_d1),
                            'Pace Skoru (%)': pace_d1,
                            'Tutarlılık Skoru (CV %)': cons_d1,
                            f'S1 Z-Skor (Ref: {driver1})': sec_z_d1.get('Sektör 1 Z-Skor', float('nan')),
                            f'S2 Z-Skor (Ref: {driver1})': sec_z_d1.get('Sektör 2 Z-Skor', float('nan')),
                            f'S3 Z-Skor (Ref: {driver1})': sec_z_d1.get('Sektör 3 Z-Skor', float('nan'))
                        },
                        {
                            'Pilot': driver2,
                            'Temiz Tur Sayısı': len(clean_d2),
                            'Pace Skoru (%)': pace_d2,
                            'Tutarlılık Skoru (CV %)': cons_d2,
                            f'S1 Z-Skor (Ref: {driver1})': sec_z_d2.get('Sektör 1 Z-Skor', float('nan')),
                            f'S2 Z-Skor (Ref: {driver1})': sec_z_d2.get('Sektör 2 Z-Skor', float('nan')),
                            f'S3 Z-Skor (Ref: {driver1})': sec_z_d2.get('Sektör 3 Z-Skor', float('nan'))
                        }
                    ]
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results.style.format(precision=3), use_container_width=True)
                    
                    # Örneklem sayısı uyarısı
                    if len(clean_d1) < 30 or len(clean_d2) < 30:
                        st.warning("⚠️ **Örneklem Uyarısı:** Pilotlardan en az birinin temiz tur sayısı 30'dan az. Bu durum t-testi sonucunun ve performans skorlarının güvenilirliğini düşürebilir.")
                    if warning_msg:
                        st.warning(warning_msg)
                    
                    # P-Değeri Gösterimi
                    st.subheader("🔬 İstatistiksel Karşılaştırma (Welch's t-test)")
                    if pd.isna(p_val):
                         st.error("Hesaplama yapılamadı (Yeterli tur verisi yok).")
                    elif p_val < 0.05:
                        st.success(f"**Anlamlı fark var!** (p = {p_val:.5f})\n\n{driver1} ve {driver2} pilotlarının temiz tur zamanları arasında istatistiksel olarak anlamlı bir fark bulunmaktadır.")
                    else:
                        st.info(f"**Fark istatistiksel olarak anlamlı değil.** (p = {p_val:.5f})\n\nİki pilotun yarış temposu istatistiksel olarak birbirinden ayırt edilemiyor.")
                        
                    
                    # Bar Chart'lar ile Görselleştirme
                    st.subheader("📈 Görsel Karşılaştırma")
                    col1, col2 = st.columns(2)
                    
                    # Grafikler için DF hazırlığı
                    df_charts = pd.DataFrame({
                        'Pilot': [driver1, driver2],
                        'Pace Skoru': [pace_d1, pace_d2],
                        'Tutarlılık (CV)': [cons_d1, cons_d2]
                    }).set_index('Pilot')
                    
                    with col1:
                        st.markdown("**Pace Skoru (%)**")
                        st.caption("Daha düşük = Daha hızlı (En hızlı tura yakın)")
                        st.bar_chart(df_charts['Pace Skoru'], use_container_width=True)
                        
                    with col2:
                        st.markdown("**Tutarlılık Skoru (CV %)**")
                        st.caption("Daha düşük = Daha istikrarlı sürüş")
                        st.bar_chart(df_charts['Tutarlılık (CV)'], use_container_width=True)
            
            except Exception as e:
                st.error(f"Analiz sırasında beklenmeyen bir hata oluştu: {str(e)}")
