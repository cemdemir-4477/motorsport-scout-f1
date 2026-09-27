import fastf1
import pandas as pd
import numpy as np
from scipy import stats
import warnings

# Cache ayarı, fastf1 verileri için gerekli (Daha hızlı yükleme sağlar)
fastf1.Cache.enable_cache('scratch/fastf1_cache') 

def filter_clean_laps(laps_data: pd.DataFrame) -> pd.DataFrame:
    """
    Tur zamanlarını IQR (Çeyrekler Açıklığı) yöntemine göre filtreleyerek
    temiz turları (pit, trafik, hata içermeyen) bulur.
    
    Args:
        laps_data (pd.DataFrame): Bir pilota ait tur zamanlarını içeren FastF1 verisi.
        
    Returns:
        pd.DataFrame: Sadece temiz turları içeren DataFrame.
    """
    # Tur zamanlarını saniyeye çevirip nan değerleri atalım
    lap_times = laps_data['LapTime'].dt.total_seconds().dropna()
    
    if len(lap_times) == 0:
        return laps_data.iloc[0:0]
        
    q1 = np.percentile(lap_times, 25)
    q3 = np.percentile(lap_times, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # FastF1 LapTime objeleri üzerinden filtreleme
    clean_laps = laps_data[
        (laps_data['LapTime'].dt.total_seconds() >= lower_bound) & 
        (laps_data['LapTime'].dt.total_seconds() <= upper_bound)
    ]
    
    return clean_laps

def calculate_pace_score(clean_laps: pd.DataFrame, fastest_lap_seconds: float) -> float:
    """
    Temiz turların medyan süresinin, yarıştaki en hızlı tura göre yüzde farkını hesaplar.
    Daha düşük skor, daha iyi (hızlı) bir pace anlamına gelir.
    
    Args:
        clean_laps (pd.DataFrame): Pilotun temiz turlarını içeren veri.
        fastest_lap_seconds (float): Yarıştaki en hızlı turun saniye cinsinden değeri.
        
    Returns:
        float: Yüzde cinsinden pace skoru.
    """
    if clean_laps.empty:
        return np.nan
        
    median_time = clean_laps['LapTime'].dt.total_seconds().median()
    pace_score = ((median_time - fastest_lap_seconds) / fastest_lap_seconds) * 100
    
    return pace_score

def calculate_consistency_score(clean_laps: pd.DataFrame) -> float:
    """
    Temiz turların Varyasyon Katsayısını (CV = std/mean) hesaplar.
    Düşük CV, daha istikrarlı bir sürüş anlamına gelir.
    
    Args:
        clean_laps (pd.DataFrame): Pilotun temiz turlarını içeren veri.
        
    Returns:
        float: Yüzde cinsinden tutarlılık skoru (CV).
    """
    if clean_laps.empty:
        return np.nan
        
    lap_times = clean_laps['LapTime'].dt.total_seconds()
    mean_time = lap_times.mean()
    std_time = lap_times.std()
    
    cv = (std_time / mean_time) * 100
    return cv

def compare_drivers_ttest(clean_laps_driver1: pd.DataFrame, clean_laps_driver2: pd.DataFrame) -> tuple[float, str]:
    """
    İki pilotun temiz tur zamanı dağılımlarını bağımsız örneklem t-testiyle karşılaştırır.
    
    Args:
        clean_laps_driver1 (pd.DataFrame): 1. pilotun temiz turları.
        clean_laps_driver2 (pd.DataFrame): 2. pilotun temiz turları.
        
    Returns:
        tuple[float, str]: p-değeri ve varsa uyarı mesajı.
    """
    times_d1 = clean_laps_driver1['LapTime'].dt.total_seconds().dropna().values
    times_d2 = clean_laps_driver2['LapTime'].dt.total_seconds().dropna().values
    
    warning_msg = ""
    if len(times_d1) < 30 or len(times_d2) < 30:
        warning_msg = "Uyarı: Örneklem sayısı 30'dan küçük, t-testi sonuçları güvenilir olmayabilir."
        
    # Bağımsız örneklem t-testi (equal_var=False, Welch's t-test kullanımı daha güvenlidir)
    t_stat, p_value = stats.ttest_ind(times_d1, times_d2, equal_var=False)
    
    return p_value, warning_msg

def calculate_sector_zscores(driver_laps: pd.DataFrame, ref_driver_laps: pd.DataFrame) -> dict:
    """
    Pilotun sektör sürelerini, referans pilotun sektör sürelerine göre z-skoru olarak ayrıştırır.
    
    Args:
        driver_laps (pd.DataFrame): İncelenen pilotun temiz turları.
        ref_driver_laps (pd.DataFrame): Referans pilotun temiz turları.
        
    Returns:
        dict: Her sektör (1, 2, 3) için ortalama z-skorunu içeren sözlük.
    """
    z_scores = {}
    
    for i in [1, 2, 3]:
        sector_col = f'Sector{i}Time'
        if sector_col not in driver_laps.columns or sector_col not in ref_driver_laps.columns:
            continue
            
        ref_times = ref_driver_laps[sector_col].dt.total_seconds().dropna()
        drv_times = driver_laps[sector_col].dt.total_seconds().dropna()
        
        if len(ref_times) == 0 or len(drv_times) == 0:
            z_scores[f'Sektör {i} Z-Skor'] = np.nan
            continue
            
        ref_mean = ref_times.mean()
        ref_std = ref_times.std()
        
        if ref_std == 0: # 0'a bölme hatasını önle
            z_scores[f'Sektör {i} Z-Skor'] = np.nan
            continue
            
        # Pilotun sektör zamanlarının referansa göre z-skorlarının ortalaması
        z_score_series = (drv_times - ref_mean) / ref_std
        z_scores[f'Sektör {i} Z-Skor'] = z_score_series.mean()
        
    return z_scores

def calculate_theoretical_optimal(clean_laps_df: pd.DataFrame, sector_columns: list) -> tuple:
    """
    Her sektörde pilotun attığı en iyi (minimum) zamanı bulup toplayarak teorik optimal turu hesaplar.
    Gerçek en iyi turuyla arasındaki farkı saniye cinsinden (gelişim potansiyeli) döndürür.
    
    Args:
        clean_laps_df (pd.DataFrame): Pilotun temiz turları.
        sector_columns (list): Sektör sütun isimleri (örn. ['Sector1Time', 'Sector2Time', 'Sector3Time']).
        
    Returns:
        tuple: (teorik_optimal_saniye, gercek_en_iyi_saniye, gelisim_potansiyeli_saniye)
    """
    if clean_laps_df.empty:
        return np.nan, np.nan, np.nan
        
    theoretical_optimal_time = 0.0
    for col in sector_columns:
        if col in clean_laps_df.columns:
            # Sektörlerdeki en iyi zamanı (minimum) bul ve saniyeye çevir
            best_sector_sec = clean_laps_df[col].dt.total_seconds().min()
            theoretical_optimal_time += best_sector_sec
            
    # Gerçek en iyi tur (saniye cinsinden)
    actual_best_time = clean_laps_df['LapTime'].dt.total_seconds().min()
    
    # Gelişim potansiyeli (Gerçek - Teorik Optimal)
    development_potential = actual_best_time - theoretical_optimal_time
    
    return theoretical_optimal_time, actual_best_time, development_potential

def main():
    print("Veri indiriliyor...")
    # 2024 Bahreyn Grand Prix'sini yükleyelim
    session = fastf1.get_session(2024, 'Bahrain', 'R')
    session.load(telemetry=False, weather=False, messages=False)

    print("Veri analiz ediliyor...")
    laps = session.laps
    
    driver1 = 'VER'
    driver2 = 'LEC'
    
    d1_laps = laps.pick_driver(driver1)
    d2_laps = laps.pick_driver(driver2)
    
    # 1. Temiz Tur Filtreleme
    clean_d1 = filter_clean_laps(d1_laps)
    clean_d2 = filter_clean_laps(d2_laps)
    
    # Tüm yarışın en hızlı turu (Saniye cinsinden)
    fastest_lap_seconds = laps['LapTime'].dt.total_seconds().min()
    
    # 2. Pace Skoru
    pace_d1 = calculate_pace_score(clean_d1, fastest_lap_seconds)
    pace_d2 = calculate_pace_score(clean_d2, fastest_lap_seconds)
    
    # 3. Tutarlılık Skoru
    cons_d1 = calculate_consistency_score(clean_d1)
    cons_d2 = calculate_consistency_score(clean_d2)
    
    # 4. Anlamlılık Testi (Verstappen vs Leclerc)
    p_val, warning = compare_drivers_ttest(clean_d1, clean_d2)
    
    # 5. Sektör Ayrıştırma (Referans: VER)
    sec_z_d1 = calculate_sector_zscores(clean_d1, clean_d1)
    sec_z_d2 = calculate_sector_zscores(clean_d2, clean_d1)
    
    # Sonuçları Tablolama
    results = [
        {
            'Pilot': driver1,
            'Temiz Tur Sayısı': len(clean_d1),
            'Pace Skoru (%)': round(pace_d1, 3),
            'Tutarlılık Skoru (CV %)': round(cons_d1, 3),
            'S1 Z-Skor (Ref: VER)': round(sec_z_d1.get('Sektör 1 Z-Skor', np.nan), 3),
            'S2 Z-Skor (Ref: VER)': round(sec_z_d1.get('Sektör 2 Z-Skor', np.nan), 3),
            'S3 Z-Skor (Ref: VER)': round(sec_z_d1.get('Sektör 3 Z-Skor', np.nan), 3)
        },
        {
            'Pilot': driver2,
            'Temiz Tur Sayısı': len(clean_d2),
            'Pace Skoru (%)': round(pace_d2, 3),
            'Tutarlılık Skoru (CV %)': round(cons_d2, 3),
            'S1 Z-Skor (Ref: VER)': round(sec_z_d2.get('Sektör 1 Z-Skor', np.nan), 3),
            'S2 Z-Skor (Ref: VER)': round(sec_z_d2.get('Sektör 2 Z-Skor', np.nan), 3),
            'S3 Z-Skor (Ref: VER)': round(sec_z_d2.get('Sektör 3 Z-Skor', np.nan), 3)
        }
    ]
    
    df_results = pd.DataFrame(results)
    
    print("\n--- Sürücü Performans Skorları ---")
    print(df_results.to_markdown(index=False))
    
    print("\n--- İstatistiksel Karşılaştırma ---")
    print(f"Karşılaştırılan Pilotlar: {driver1} vs {driver2}")
    print(f"P-Değeri (Welch's t-test): {p_val:.5f}")
    if p_val < 0.05:
        print("Sonuç: İki pilotun temiz tur zamanları arasında istatistiksel olarak ANLAMLI bir fark var.")
    else:
        print("Sonuç: İki pilotun temiz tur zamanları arasında istatistiksel olarak ANLAMLI BİR FARK YOK.")
        
    if warning:
        print(warning)

if __name__ == '__main__':
    main()
