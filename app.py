import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Mevcut f1_performance.py dosyasındaki hesaplama fonksiyonlarını içe aktarıyoruz
from f1_performance import (
    filter_clean_laps,
    calculate_pace_score,
    calculate_consistency_score,
    compare_drivers_ttest,
    calculate_sector_zscores,
    calculate_theoretical_optimal
)

# Streamlit sayfası temel ayarları
st.set_page_config(page_title="Sürücü Performans Skoru", layout="wide")

# Cache dizini ayarı (FastF1 verilerini tekrar tekrar indirmemek için)
os.makedirs("scratch/fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache('scratch/fastf1_cache')

st.title("🏎️ Sürücü Performans Skoru")
st.markdown("Bu uygulama, seçtiğiniz yarış ve iki pilot üzerinden temiz turları (medyan ± 1.5×IQR) filtreleyerek pace, tutarlılık ve sektör performanslarını istatistiksel olarak karşılaştırır.")

# --- Yardımcı Fonksiyon: Başlık ve Bilgi İkonu ---
def header_with_info(title, info_text, level="h3"):
    """Belirtilen başlığın yanına bir info butonu (popover) ekler."""
    # Sütunları başlığa göre boyutlandırmak için auto-width (layout) kullanımı
    col1, col2 = st.columns([0.95, 0.05])
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

# --- Cache'li Veri Yükleme Fonksiyonları ---
@st.cache_data(show_spinner=False)
def get_season_schedule(year):
    """Belirtilen yılın yarış takvimini çeker."""
    try:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule['EventFormat'] != 'testing']
        return schedule
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_session(year, event_name):
    """Belirtilen yarışın (R = Race) seans verisini çeker."""
    try:
        session = fastf1.get_session(year, event_name, 'R')
        session.load(telemetry=True, laps=True, weather=False, messages=False)
        return session
    except Exception as e:
        return None

# --- YAN MENÜ (SIDEBAR) ---
st.sidebar.header("⚙️ Ayarlar")

selected_year = st.sidebar.selectbox("Sezon Seçin", list(range(2026, 2017, -1)), index=1)

schedule = get_season_schedule(selected_year)
if schedule.empty:
    st.sidebar.error("Takvim verisi çekilemedi.")
    events = []
else:
    events = schedule['EventName'].tolist()

selected_event = st.sidebar.selectbox("Yarış Seçin", events)

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

if driver_options:
    driver1 = st.sidebar.selectbox("Pilot 1", driver_options, index=0)
    driver2_default_index = 1 if len(driver_options) > 1 else 0
    driver2 = st.sidebar.selectbox("Pilot 2", driver_options, index=driver2_default_index)
else:
    driver1, driver2 = None, None
    if selected_event:
        st.sidebar.error("Pilot verisi çekilemedi. Bu yarış henüz gerçekleşmemiş olabilir.")

analyze_btn = st.sidebar.button("Analiz Et", type="primary")

# --- ANA EKRAN ANALİZ BÖLÜMÜ ---
if analyze_btn:
    if not driver1 or not driver2:
        st.warning("Lütfen iki pilot seçtiğinizden emin olun.")
    elif driver1 == driver2:
        st.warning("Lütfen karşılaştırma için iki farklı pilot seçin.")
    elif session is None:
        st.error("Yarış verisi yüklenemedi. Seçilen yarış henüz gerçekleşmemiş olabilir veya veri sistemi tarafından henüz sağlanmamış olabilir.")
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
                    
                    # 6. Teorik Optimal Tur Analizi
                    sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
                    theo_opt_d1, act_best_d1, pot_d1 = calculate_theoretical_optimal(clean_d1, sector_cols)
                    theo_opt_d2, act_best_d2, pot_d2 = calculate_theoretical_optimal(clean_d2, sector_cols)
                    
                    
                    # Sekmeleri oluştur
                    tab1, tab2 = st.tabs(["Genel Analiz", "Telemetri Analizi"])
                    
                    with tab1:
                        # --- SONUÇLARI GÖSTER ---
                        
                        header_with_info(
                            "📊 Performans Tablosu", 
                            "Sektör z-skoru: Kaç standart sapma uzakta olduğunu gösterir, referans pilota (Pilot 1) göre normalize eder.",
                            level="h2"
                        )
                        
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
                        st.write("---")
                        header_with_info(
                            "🔬 İstatistiksel Karşılaştırma (Welch's t-test)",
                            "0.05'in altı, farkın şans eseri olma ihtimalinin düşük olduğunu gösterir - ama küçük örneklemde temkinli yorumlanmalı.",
                            level="h2"
                        )
                        
                        if pd.isna(p_val):
                             st.error("Hesaplama yapılamadı (Yeterli tur verisi yok).")
                        elif p_val < 0.05:
                            st.success(f"**Anlamlı fark var!** (p = {p_val:.5f})\n\n{driver1} ve {driver2} pilotlarının temiz tur zamanları arasında istatistiksel olarak anlamlı bir fark bulunmaktadır.")
                        else:
                            st.info(f"**Fark istatistiksel olarak anlamlı değil.** (p = {p_val:.5f})\n\nİki pilotun yarış temposu istatistiksel olarak birbirinden ayırt edilemiyor.")
                            
                        
                        # Bar Chart'lar ile Görselleştirme
                        st.write("---")
                        st.subheader("📈 Görsel Karşılaştırma")
                        col1, col2 = st.columns(2)
                        
                        df_charts = pd.DataFrame({
                            'Pilot': [driver1, driver2],
                            'Pace Skoru': [pace_d1, pace_d2],
                            'Tutarlılık (CV)': [cons_d1, cons_d2]
                        }).set_index('Pilot')
                        
                        with col1:
                            header_with_info("Pace Skoru (%)", "Medyan kullanıyoruz çünkü aykırı değerlere (pit, hata) karşı dayanıklı, ortalamadan daha güvenilir.", level="h3")
                            st.caption("Daha düşük = Daha hızlı (En hızlı tura yakın)")
                            st.bar_chart(df_charts['Pace Skoru'], use_container_width=True)
                            
                        with col2:
                            header_with_info("Tutarlılık Skoru (CV %)", "Varyasyon katsayısı, standart sapmayı ortalamaya bölerek farklı ölçeklerdeki tutarlılığı karşılaştırılabilir yapar.", level="h3")
                            st.caption("Daha düşük = Daha istikrarlı sürüş")
                            st.bar_chart(df_charts['Tutarlılık (CV)'], use_container_width=True)
                            
                        # 6. Teorik Optimal Tur Analizi Bölümü
                        st.write("---")
                        header_with_info(
                            "⏱️ Teorik Optimal Tur Analizi", 
                            "Her sektördeki en iyi performansın toplamı - pilotun ulaşabileceği teorik alt sınır.",
                            level="h2"
                        )
                        
                        df_optimal = pd.DataFrame({
                            'Pilot': [driver1, driver2],
                            'Gerçek En Hızlı Tur (Sn)': [act_best_d1, act_best_d2],
                            'Teorik Optimal Tur (Sn)': [theo_opt_d1, theo_opt_d2],
                            'Gelişim Potansiyeli (Sn)': [pot_d1, pot_d2] # Gerçek - Teorik
                        }).set_index('Pilot')
                        
                        col_opt1, col_opt2 = st.columns([0.6, 0.4])
                        with col_opt1:
                            st.markdown("**Tur Zamanları (Saniye)**")
                            st.dataframe(df_optimal.style.format(precision=3), use_container_width=True)
                            
                        with col_opt2:
                            st.markdown("**Gelişim Potansiyeli (Saniye)**")
                            st.caption("Sektörleri birleştirmede yaşanan kayıp (Daha düşük = Tek turda daha verimli)")
                            st.bar_chart(df_optimal['Gelişim Potansiyeli (Sn)'], use_container_width=True)

                    with tab2:
                        st.subheader("📡 Telemetri Karşılaştırması (En Hızlı Turlar)")
                        st.markdown(f"**{driver1}** ve **{driver2}** pilotlarının bu yarıştaki en hızlı turlarına ait telemetri verileri.")
                        
                        try:
                            # En hızlı turları al
                            fastest_d1 = d1_laps.pick_fastest()
                            fastest_d2 = d2_laps.pick_fastest()
                            
                            # Telemetriyi al ve mesafeyi ekle
                            tel_d1 = fastest_d1.get_telemetry().add_distance()
                            tel_d2 = fastest_d2.get_telemetry().add_distance()
                            # Fren (Brake) verisini düzenle
                            # Eğer bool veya sadece 0/1 ise, 0-100 aralığına çek ve step (hv) yap
                            is_brake_bool = (
                                tel_d1['Brake'].dtype == bool or 
                                set(tel_d1['Brake'].dropna().unique()).issubset({0, 1, True, False, 0.0, 1.0})
                            )
                            
                            brake_shape = 'hv' if is_brake_bool else 'linear'
                            
                            for tel in [tel_d1, tel_d2]:
                                tel['Brake'] = tel['Brake'].astype(float)
                                if is_brake_bool or tel['Brake'].max() <= 1.0:
                                    tel['Brake'] = tel['Brake'] * 100
                            
                            # Plotly ile Grafik çizimi
                            fig = make_subplots(
                                rows=3, cols=1, 
                                shared_xaxes=True, 
                                vertical_spacing=0.05,
                                subplot_titles=("Hız (km/h)", "Gaz (%)", "Fren (%)")
                            )
                            
                            # Renkler
                            color_d1 = '#1f77b4'
                            color_d2 = '#ff7f0e'
                            
                            # Hız (Speed) - Row 1
                            fig.add_trace(go.Scatter(x=tel_d1['Distance'], y=tel_d1['Speed'], mode='lines', name=driver1, line=dict(color=color_d1), legendgroup=driver1), row=1, col=1)
                            fig.add_trace(go.Scatter(x=tel_d2['Distance'], y=tel_d2['Speed'], mode='lines', name=driver2, line=dict(color=color_d2), legendgroup=driver2), row=1, col=1)
                            
                            # Gaz (Throttle) - Row 2
                            fig.add_trace(go.Scatter(x=tel_d1['Distance'], y=tel_d1['Throttle'], mode='lines', name=driver1, line=dict(color=color_d1), legendgroup=driver1, showlegend=False), row=2, col=1)
                            fig.add_trace(go.Scatter(x=tel_d2['Distance'], y=tel_d2['Throttle'], mode='lines', name=driver2, line=dict(color=color_d2), legendgroup=driver2, showlegend=False), row=2, col=1)
                            
                            # Fren (Brake) - Row 3
                            fig.add_trace(go.Scatter(x=tel_d1['Distance'], y=tel_d1['Brake'], mode='lines', name=driver1, line=dict(color=color_d1, shape=brake_shape), legendgroup=driver1, showlegend=False), row=3, col=1)
                            fig.add_trace(go.Scatter(x=tel_d2['Distance'], y=tel_d2['Brake'], mode='lines', name=driver2, line=dict(color=color_d2, shape=brake_shape), legendgroup=driver2, showlegend=False), row=3, col=1)
                            
                            fig.update_layout(
                                hovermode="x unified",
                                height=800,
                                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
                            )
                            
                            # Eksen (Y-Axis) Ayarları
                            fig.update_yaxes(title_text="Hız (km/h)", row=1, col=1, dtick=50, rangemode="tozero")
                            fig.update_yaxes(title_text="Gaz (%)", row=2, col=1, range=[-5, 105], dtick=20)
                            fig.update_yaxes(title_text="Fren (%)", row=3, col=1, range=[-5, 105], dtick=20)
                            
                            fig.update_xaxes(title_text="Mesafe (m)", row=3, col=1)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Hız farkı hesaplama ve otomatik yorum
                            # Verileri mesafe (Distance) bazında hizalama (merge_asof)
                            df1 = tel_d1[['Distance', 'Speed']].dropna().sort_values('Distance')
                            df2 = tel_d2[['Distance', 'Speed']].dropna().sort_values('Distance')
                            
                            merged = pd.merge_asof(df1, df2, on='Distance', suffixes=('_d1', '_d2'), direction='nearest')
                            merged['Speed_Diff'] = abs(merged['Speed_d1'] - merged['Speed_d2'])
                            
                            max_diff_idx = merged['Speed_Diff'].idxmax()
                            max_diff_row = merged.loc[max_diff_idx]
                            
                            max_diff_val = max_diff_row['Speed_Diff']
                            max_diff_dist = max_diff_row['Distance']
                            
                            # Yaklaşık mesafe aralığı (± 50 metre)
                            dist_start = max(0, max_diff_dist - 50)
                            dist_end = max_diff_dist + 50
                            
                            st.info(f"**💡 Analiz:** İki pilot arasındaki en büyük hız farkı **{max_diff_val:.1f} km/h** ile yaklaşık **{dist_start:.0f} - {dist_end:.0f} metreleri** arasında gerçekleşmiştir.")
                            
                        except Exception as e:
                            st.warning(f"Telemetri verisi çekilemedi veya eksik. Sebebi: {str(e)}")
            
            except Exception as e:
                st.error(f"Analiz sırasında beklenmeyen bir hata oluştu: {str(e)}")
