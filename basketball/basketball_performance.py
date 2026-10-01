"""
Basketball Performance Analysis Module

Bu modül, nba_api kullanarak NBA oyuncularının maç verilerini çeker ve 
F1 performans analizindeki (pace, tutarlılık, z-skor) mantığını basketbola uyarlar.
"""

import pandas as pd
import numpy as np
from scipy import stats
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import time

def get_player_games(player_name: str, season: str = '2023-24') -> pd.DataFrame:
    """
    Belirtilen oyuncunun seçilen sezondaki maç (gamelog) verilerini çeker.
    """
    # Oyuncu sözlüğünden oyuncuyu bul
    nba_players = players.get_players()
    player_dict = [player for player in nba_players if player['full_name'].lower() == player_name.lower()]
    
    if not player_dict:
        raise ValueError(f"Oyuncu bulunamadı: {player_name}")
        
    player_id = player_dict[0]['id']
    
    # Maç loglarını çek
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    time.sleep(0.5) # API limitlerine takılmamak için ufak bekleme
    df = gamelog.get_data_frames()[0]
    
    return df

def filter_clean_games(games_df: pd.DataFrame, stat_column: str = 'PTS') -> pd.DataFrame:
    """
    Seçilen istatistik için IQR (Interquartile Range) yöntemini kullanarak 
    anormal maçları (sakatlık, DNP-CD, olağanüstü şanslı performanslar) filtreler.
    
    Median ± 1.5 * IQR dışındaki değerler atılır.
    """
    if games_df.empty or stat_column not in games_df.columns:
        return games_df
        
    data = games_df[stat_column]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = data.median() - 1.5 * IQR
    upper_bound = data.median() + 1.5 * IQR
    
    clean_games = games_df[(data >= lower_bound) & (data <= upper_bound)].copy()
    return clean_games

def calculate_performance_score(clean_games: pd.DataFrame, stat_column: str, league_avg: float) -> float:
    """
    Oyuncunun temiz maçlardaki istatistik medyanının, lig ortalamasına göre 
    yüzde olarak farkını hesaplar. (Pozitif ise ortalamanın üzerinde)
    """
    if clean_games.empty or league_avg == 0:
        return np.nan
        
    player_median = clean_games[stat_column].median()
    percent_diff = ((player_median - league_avg) / league_avg) * 100
    
    return percent_diff

def calculate_consistency_score(clean_games: pd.DataFrame, stat_column: str) -> float:
    """
    Seçilen istatistik için Varyasyon Katsayısı (CV) hesaplar (Standart Sapma / Ortalama).
    Daha düşük değer, maçtan maça daha tutarlı oynadığını (daha az dalgalandığını) gösterir.
    """
    if len(clean_games) < 2:
        return np.nan
        
    std_dev = clean_games[stat_column].std()
    mean_val = clean_games[stat_column].mean()
    
    if mean_val == 0:
        return np.nan
        
    cv = (std_dev / mean_val) * 100
    return cv

def compare_players_ttest(player1_games: pd.DataFrame, player2_games: pd.DataFrame, stat_column: str = 'PTS'):
    """
    İki oyuncunun aynı istatistikteki dağılımlarını Welch's t-testi ile karşılaştırır.
    """
    p1_data = player1_games[stat_column].dropna()
    p2_data = player2_games[stat_column].dropna()
    
    warning_msg = ""
    if len(p1_data) < 15 or len(p2_data) < 15:
        warning_msg = "Uyarı: Bir veya iki oyuncu için maç sayısı (örneklem) düşük (<15). p-değeri güvenilir olmayabilir."
        
    if p1_data.empty or p2_data.empty:
        return np.nan, "Veri yetersiz"
        
    t_stat, p_val = stats.ttest_ind(p1_data, p2_data, equal_var=False)
    
    return p_val, warning_msg

def calculate_category_zscores(player_games: pd.DataFrame, reference_games: pd.DataFrame) -> dict:
    """
    Temel kategorilerde (PTS, REB, AST, STL, BLK) referans oyuncuya (Pilot 1/Player 1) göre 
    ikinci oyuncunun (Player 2) z-skorlarını hesaplar.
    """
    categories = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    z_scores = {}
    
    for cat in categories:
        if cat in player_games.columns and cat in reference_games.columns:
            ref_mean = reference_games[cat].mean()
            ref_std = reference_games[cat].std()
            
            player_mean = player_games[cat].mean()
            
            if ref_std and not np.isnan(ref_std) and ref_std != 0:
                z = (player_mean - ref_mean) / ref_std
                z_scores[cat] = z
            else:
                z_scores[cat] = np.nan
                
    return z_scores


def get_valid_player_name(prompt_text: str) -> str:
    """Kullanıcıdan oyuncu adını alıp nba_api ile doğrular."""
    while True:
        name = input(prompt_text).strip()
        if not name:
            continue
            
        found_players = players.find_players_by_full_name(name)
        
        # Tam eşleşenleri bulmaya çalış
        exact_matches = [p for p in found_players if p['full_name'].lower() == name.lower()]
        
        if exact_matches:
            return exact_matches[0]['full_name']
        elif found_players:
            # Sadece bir kısmi eşleşme varsa onu öner
            matched_name = found_players[0]['full_name']
            confirm = input(f"Oyuncu tam olarak bulunamadı. '{matched_name}' demek istediniz mi? (e/h): ").strip().lower()
            if confirm == 'e':
                return matched_name
            else:
                print("Lütfen oyuncunun tam ve doğru adını girin.")
        else:
            print("Oyuncu bulunamadı, lütfen tekrar deneyin.")

def main():
    print("🏀 Basketball Performance Modülü 🏀\n")
    
    while True:
        season = input("Sezon girin (örnek: 2023-24): ").strip()
        if not season:
            season = "2023-24"
            
        p1_name = get_valid_player_name("1. Oyuncunun tam adını girin (örnek: LeBron James): ")
        p2_name = get_valid_player_name("2. Oyuncunun tam adını girin (örnek: Stephen Curry): ")
        
        stat_to_analyze = input("Hangi istatistiği karşılaştırmak istersiniz? (PTS/REB/AST/STL/BLK): ").strip().upper()
        if stat_to_analyze not in ['PTS', 'REB', 'AST', 'STL', 'BLK']:
            print("Geçersiz istatistik, varsayılan olarak PTS kullanılacak.")
            stat_to_analyze = 'PTS'
            
        print(f"\n'{p1_name}' verileri çekiliyor...")
        df_p1 = get_player_games(p1_name, season)
        print(f"'{p2_name}' verileri çekiliyor...")
        df_p2 = get_player_games(p2_name, season)
        
        clean_p1 = filter_clean_games(df_p1, stat_to_analyze)
        clean_p2 = filter_clean_games(df_p2, stat_to_analyze)
        
        print(f"\nTemiz Maç Sayıları (Aykırı Değerler Hariç, {stat_to_analyze} bazında):")
        print(f"{p1_name}: {len(clean_p1)} / {len(df_p1)}")
        print(f"{p2_name}: {len(clean_p2)} / {len(df_p2)}")
        
        # Varsayımsal lig ortalaması belirleme
        league_avg_dict = {'PTS': 20.0, 'REB': 8.0, 'AST': 5.0, 'STL': 1.5, 'BLK': 1.0}
        league_avg = league_avg_dict.get(stat_to_analyze, 10.0)
        
        perf_p1 = calculate_performance_score(clean_p1, stat_to_analyze, league_avg)
        perf_p2 = calculate_performance_score(clean_p2, stat_to_analyze, league_avg)
        
        cons_p1 = calculate_consistency_score(clean_p1, stat_to_analyze)
        cons_p2 = calculate_consistency_score(clean_p2, stat_to_analyze)
        
        z_scores_p1 = calculate_category_zscores(clean_p1, clean_p1)
        z_scores_p2 = calculate_category_zscores(clean_p2, clean_p1)
        
        p_val, warn = compare_players_ttest(clean_p1, clean_p2, stat_to_analyze)
        
        results = [
            {
                'Oyuncu': p1_name,
                'Temiz Maç': len(clean_p1),
                f'Performans (Ref: {league_avg})': f"{perf_p1:+.1f}%" if not pd.isna(perf_p1) else "N/A",
                f'Tutarlılık (CV % {stat_to_analyze})': f"{cons_p1:.1f}%" if not pd.isna(cons_p1) else "N/A",
                'Z-PTS': round(z_scores_p1.get('PTS', 0.0), 2),
                'Z-REB': round(z_scores_p1.get('REB', 0.0), 2),
                'Z-AST': round(z_scores_p1.get('AST', 0.0), 2),
                'Z-STL': round(z_scores_p1.get('STL', 0.0), 2),
                'Z-BLK': round(z_scores_p1.get('BLK', 0.0), 2)
            },
            {
                'Oyuncu': p2_name,
                'Temiz Maç': len(clean_p2),
                f'Performans (Ref: {league_avg})': f"{perf_p2:+.1f}%" if not pd.isna(perf_p2) else "N/A",
                f'Tutarlılık (CV % {stat_to_analyze})': f"{cons_p2:.1f}%" if not pd.isna(cons_p2) else "N/A",
                'Z-PTS': round(z_scores_p2.get('PTS', 0.0), 2),
                'Z-REB': round(z_scores_p2.get('REB', 0.0), 2),
                'Z-AST': round(z_scores_p2.get('AST', 0.0), 2),
                'Z-STL': round(z_scores_p2.get('STL', 0.0), 2),
                'Z-BLK': round(z_scores_p2.get('BLK', 0.0), 2)
            }
        ]
        
        df_results = pd.DataFrame(results)
        
        print("\n--- PERFORMANS TABLOSU ---")
        try:
            print(df_results.to_markdown(index=False))
        except ImportError:
            print(df_results.to_string(index=False))
        
        print(f"\n--- İSTATİSTİKSEL KARŞILAŞTIRMA (T-Test) ---")
        if warn:
            print(warn)
        if not pd.isna(p_val):
            print(f"{stat_to_analyze} İstatistiği için p-değeri: {p_val:.5f}")
            if p_val < 0.05:
                print("Anlamlı fark VAR.")
            else:
                print("Anlamlı fark YOK (Tesadüfi olabilir).")
        else:
            print("p-değeri hesaplanamadı.")
            
        cevap = input("\nBaşka bir karşılaştırma yapmak ister misiniz? (e/h): ").strip().lower()
        if cevap != 'e':
            print("Çıkış yapılıyor...")
            break

if __name__ == "__main__":
    main()
