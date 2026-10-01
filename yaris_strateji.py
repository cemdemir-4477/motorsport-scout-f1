import math

# Lift & Coast varsayımı: her 1 saniyelik pace kaybı için yaklaşık 0.08 litre 
# yakıt tasarrufu sağlanır (GT3/LMP2 sınıfları için genel bir yaklaşık değer).
# Bu, gerçek telemetri verinle zamanla kalibre edilebilir.
LC_TASARRUF_ORANI = 0.08  # litre / saniye

def zaman_cevirici(dakika, saniye):
    """Dakika ve saniye cinsinden verilen süreyi toplam saniyeye çevirir."""
    return (dakika * 60) + saniye

def pit_duvari_asistani():
    """
    Kullanıcıdan alınan anlık yarış verileriyle (süre, pace, yakıt vb.) kalan tur 
    sayısını ve yakıt gereksinimini hesaplayarak pit duvarı stratejisi önerir.
    """
    print("="*40)
    print(" 🏁 PIT DUVARI STRATEJİ EKRANI 🏁 ")
    print("="*40)
    
    try:
        # Anlık verilerin manuel olarak girilmesi
        kalan_sure_dk = float(input("Kalan Yarış Süresi (Dakika - örn: 11.5): "))
        tur_dk = int(input("Güncel Pace - Dakika (örn: 1): "))
        tur_sn = float(input("Güncel Pace - Saniye (örn: 25.7): "))
        tuketim = float(input("Tur Başına Yakıt Tüketimi (Litre): "))
        mevcut_yakit = float(input("Depodaki Mevcut Yakıt (Litre): "))
        
        # Saniye bazında hesaplamalar
        kalan_sure_sn = kalan_sure_dk * 60
        tur_suresi_sn = zaman_cevirici(tur_dk, tur_sn)
        
        # Kaç tur kaldı hesaplaması
        teorik_tur = kalan_sure_sn / tur_suresi_sn
        tamamlanacak_tur = math.ceil(teorik_tur) # Damalı bayrak için yukarı yuvarlama
        
        # Yakıt hesaplaması
        gereken_yakit = tamamlanacak_tur * tuketim
        yakit_farki = mevcut_yakit - gereken_yakit
        
        # Sonuç Raporu ve Strateji Kararı
        print("\n" + "="*40)
        print(" 📊 YARIŞ MÜHENDİSİ RAPORU")
        print("="*40)
        print(f"Damalı Bayrağa Kadar Atılacak Tam Tur : {tamamlanacak_tur} Tur")
        print(f"Yarışı Bitirmek İçin Gereken Yakıt    : {gereken_yakit:.2f} L")
        print(f"Mevcut Yakıt Durumu (Fark)            : {yakit_farki:.2f} L\n")
        
        # Duruma göre aksiyon kararı (GT3 / LMP2 dinamiklerine uygun)
        print("--- AKSİYON PLANI ---")
        if yakit_farki >= 0.5:
            print("🟢 STRATEJİ: YAKIT GÜVENDE!")
            print("Mevcut tempoda zorlamaya devam et. Piste odaklan, Push!")
        elif -0.8 <= yakit_farki < 0.5:
            print("🟡 STRATEJİ: YAKIT KRİTİK SINIRDA!")
            print("Pit stop yapmadan bitirmek için düzlük sonlarında frenajdan önce")
            print("erken gaz keserek (Lift & Coast) aracı süzdür. Tempoyu stabil tut.")
        else:
            print("🔴 STRATEJİ: YAKIT YETMEYECEK - BOX BOX!")
            print("Mevcut tempoyla ve Lift & Coast ile kurtarılamayacak kadar açık var.")
            print(f"Acilen pite gelip en az {abs(yakit_farki) + 1:.1f} Litre ekleme yapmalısın.")
            
        if yakit_farki < 0:
            kapatilacak_litre = abs(yakit_farki)
            ekstra_sn = (kapatilacak_litre / LC_TASARRUF_ORANI) / tamamlanacak_tur
            
            if ekstra_sn > (tur_suresi_sn * 0.30):
                print("\n--- 🔧 LIFT & COAST PLANI ---")
                print("⚠️ UYARI: Bu açık sadece lift&coast ile, gerçekçi bir pace kaybıyla kapatılamaz, pit stop şart!")
            else:
                yeni_pace_sn = tur_suresi_sn + ekstra_sn
                yeni_pace_dk = int(yeni_pace_sn // 60)
                yeni_pace_kalan_sn = yeni_pace_sn % 60
                yeni_pace_str = f"{yeni_pace_dk}:{yeni_pace_kalan_sn:.1f}"
                
                print(f"\n🔧 LIFT & COAST PLANI: Kalan {tamamlanacak_tur} turun TAMAMINDA yeni pace'in {yeni_pace_str} olmalı. Bu tempoyla yarışı bitirdiğinde toplam {kapatilacak_litre:.2f} litre tasarruf etmiş olacaksın ve pit yapmana gerek kalmayacak.")
            
    except ValueError:
        print("\nHata: Lütfen değerleri doğru formatta (sayı olarak) girin!")

if __name__ == "__main__":
    # Döngüye alarak yarış boyu ekranın açık kalmasını ve yeni hesaplama yapılmasını sağlıyoruz
    while True:
        pit_duvari_asistani()
        devam = input("\nYeni bir hesaplama yapmak istiyor musun? (e/h): ").lower()
        if devam != 'e':
            print("Pit duvarından çıkılıyor. İyi yarışlar!")
            break
        print("\n")