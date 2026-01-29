import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import os

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="MindTrace AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PRO CSS TASARIMI (DARK & NEON) ---
st.markdown("""
<style>
    /* Ana Arka Plan */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111;
        opacity: 0.95;
    }
    /* Buton Tasarımı */
    .stButton>button {
        background: linear-gradient(45deg, #FF512F, #DD2476);
        color: white;
        border: none;
        border-radius: 25px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(221, 36, 118, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(221, 36, 118, 0.6);
    }
    /* Metrik Kutuları */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        color: #ddd;
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; text-shadow: 2px 2px 4px #000; }
</style>
""", unsafe_allow_html=True)

# --- MODEL YÜKLEME ---
@st.cache_resource
def load_models():
    paths = ['../models', 'models']
    for path in paths:
        m_path = f"{path}/best_model.pkl"
        s_path = f"{path}/scaler.pkl"
        if os.path.exists(m_path) and os.path.exists(s_path):
            return joblib.load(m_path), joblib.load(s_path)
    return None, None

model, scaler = load_models()

# --- HEADER ---
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=90)
with col2:
    st.title("MindTrace AI: Ruh Sağlığı Analizi")
    st.markdown("#### 🚀 11 Faktörlü Gelişmiş Risk Tahmin Sistemi")

st.divider()

# --- SIDEBAR (11 GİRDİ EKSİKSİZ) ---
with st.sidebar:
    st.header("📋 Profil & Veriler")
    
    # 1. KİŞİSEL BİLGİLER
    with st.expander("👤 Kişisel Bilgiler", expanded=True):
        gender = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
        age = st.slider("Yaş", 12, 50, 21)
        relationship = st.selectbox("İlişki Durumu", ["Bekar", "İlişkisi Var", "Evli", "Boşanmış"])
        academic = st.selectbox("Eğitim Seviyesi", ["Lise", "Lisans", "Yüksek Lisans", "Doktora"])

    # 2. DİJİTAL ALIŞKANLIKLAR
    with st.expander("📱 Dijital Alışkanlıklar", expanded=True):
        platform = st.selectbox("Ana Platform", ['Instagram', 'TikTok', 'Twitter', 'YouTube', 'Facebook', 'LinkedIn'])
        usage = st.number_input("Günlük Ekran (Saat)", 0.0, 24.0, 6.0, 0.5)
        notifs = st.slider("Bildirim Sayısı", 0, 300, 50)
        switches = st.slider("App Değiştirme (Günlük)", 0, 200, 40)
        sleep = st.slider("Uyku Süresi (Saat)", 0, 12, 7)

    # 3. PSİKOLOJİK ETKİLER
    with st.expander("⚠️ Etkiler ve Çatışmalar", expanded=True):
        affects = st.radio("Derslerini/İşini Etkiliyor mu?", ["Hayır", "Evet"])
        conflicts = st.radio("Çevrenle Tartışma Yaşıyor musun?", ["Hayır", "Evet"])
    
    analyze_btn = st.button("RİSKİ HESAPLA ✨")

# --- ANA EKRAN VE HESAPLAMA ---
if analyze_btn:
    if model:
        # 1. Yükleniyor Efekti
        with st.spinner('Yapay Zeka 11 Faktörü Analiz Ediyor...'):
            time.sleep(1.2)
        
        # 2. Veri Hazırlığı (Encoding)
        # Modelin eğitiminde kullanılan LabelEncoder mantığına göre manuel eşleme:
        
        # Cinsiyet: Female=0, Male=1
        g_val = 1 if gender == "Erkek" else 0
        
        # İlişki (Alfabetik): Divorced=0, In Rel=1, Married=2, Single=3 (Türkçe sıraya dikkat)
        r_map = {"Boşanmış": 0, "İlişkisi Var": 1, "Evli": 2, "Bekar": 3}
        r_val = r_map[relationship]
        
        # Akademik: Graduate=0, High School=1, PhD=2, Undergrad=3
        a_map = {"Yüksek Lisans": 0, "Lise": 1, "Doktora": 2, "Lisans": 3}
        a_val = a_map[academic]
        
        # Platform: Facebook=0, Insta=1, Link=2, TikTok=3, Twitter=4, YouTube=5
        p_map = {'Facebook': 0, 'Instagram': 1, 'LinkedIn': 2, 'TikTok': 3, 'Twitter': 4, 'YouTube': 5}
        p_val = p_map.get(platform, 1)
        
        # Evet/Hayır: No=0, Yes=1
        aff_val = 1 if affects == "Evet" else 0
        conf_val = 1 if conflicts == "Evet" else 0

        # 3. Scaling (Sayısal Veriler İçin)
        # Sıra: Age, Usage, Switches, Notifs, Sleep
        raw_nums = np.array([[age, usage, switches, notifs, sleep]])
        scaled_nums = scaler.transform(raw_nums)
        
        # 4. Final Vektör (Eğitim Sırasına Göre Birleştirme)
        # X Sırası: [Age_s, Gender, Rel, Acad, Usage_s, Plat, Switches_s, Notifs_s, Sleep_s, Aff, Conf]
        final_input = np.array([[
            scaled_nums[0][0], # Age
            g_val,             # Gender
            r_val,             # Relationship
            a_val,             # Academic
            scaled_nums[0][1], # Usage
            p_val,             # Platform
            scaled_nums[0][2], # Switches
            scaled_nums[0][3], # Notifs
            scaled_nums[0][4], # Sleep
            aff_val,           # Affects
            conf_val           # Conflicts
        ]])
        
        # 5. Tahmin
        pred_class = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0]
        risk_score = prob[pred_class] * 100
        
        # Görsel Skor Ayarı (Düşük riskse ibre solda, yüksekse sağda olsun)
        display_score = risk_score
        if pred_class == 0: # Düşük
            display_score = 100 - risk_score # Örn: %90 eminse ibre %10'da dursun (yeşil bölge)
            label = "DÜŞÜK RİSK"
            color = "#00ff41" # Matrix Yeşili
        elif pred_class == 1: # Orta
            display_score = 50 
            label = "ORTA RİSK"
            color = "#ffa500" # Turuncu
        else: # Yüksek
            display_score = risk_score # %90 eminse ibre %90'da dursun (kırmızı bölge)
            label = "YÜKSEK RİSK"
            color = "#ff3333" # Kırmızı

        # --- DASHBOARD GÖSTERİMİ ---
        
        col_g1, col_g2 = st.columns([1, 1])
        
        with col_g1:
            # GAUGE CHART (İBRE)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = display_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Dijital Sağlık Endeksi", 'font': {'color': "white", 'size': 20}},
                number = {'suffix': "%", 'font': {'color': "white"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#444",
                    'steps': [
                        {'range': [0, 33], 'color': 'rgba(0, 255, 65, 0.3)'},
                        {'range': [33, 66], 'color': 'rgba(255, 165, 0, 0.3)'},
                        {'range': [66, 100], 'color': 'rgba(255, 51, 51, 0.3)'}],
                }
            ))
            fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Arial"})
            st.plotly_chart(fig, use_container_width=True)

        with col_g2:
            st.markdown(f"### 🤖 Sonuç: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
            st.info(f"Yapay Zeka bu sonuca %{prob.max()*100:.1f} güven oranıyla ulaştı.")
            
            # KPI Kartları
            c1, c2, c3 = st.columns(3)
            c1.metric("Uyku", f"{sleep} sa", delta="-Riskli" if sleep<6 else "İyi")
            c2.metric("Ekran", f"{usage} sa", delta="Yüksek" if usage>6 else "-İyi", delta_color="inverse")
            c3.metric("Platform", f"{platform}")

        st.markdown("---")
        
        # DETAYLI RAPOR
        st.subheader("📋 Yapay Zeka Raporu")
        
        if pred_class == 2: # Yüksek Risk
            st.error(f"""
            **Tespil Edilen Kritik Riskler:**
            1. 🔴 **Yüksek Ekran Süresi:** Günde {usage} saat kullanım, zihinsel yorgunluk sınırını aşıyor.
            2. 🔴 **Sosyal Etki:** 'Çevrenle tartışma' yaşaman, dijital bağımlılığın sosyal hayatını bozduğunu gösteriyor.
            3. **Öneri:** Acilen 'Dijital Diyet' programına başlamalısın.
            """)
        elif pred_class == 1: # Orta Risk
            st.warning("""
            **Dikkat Edilmesi Gerekenler:**
            1. 🟡 Alışkanlıklarınız sınırda. Stres seviyeniz artma eğiliminde.
            2. **Bildirimler:** Günlük {notifs} bildirim odak kaybı yaratıyor olabilir.
            3. **Öneri:** Yatmadan 1 saat önce telefonu bırakmayı deneyin.
            """)
        else:
            st.success("""
            **Durum Analizi:**
            1. 🟢 Harika! Dijital yaşamınla ruh sağlığın arasında mükemmel bir denge var.
            2. Uyku düzenin ve kullanım alışkanlıkların gayet sağlıklı.
            3. **Öneri:** Bu düzeni korumaya devam et!
            """)
            
    else:
        st.error("❌ Model dosyası bulunamadı! Lütfen önce eğitim kodunu (04) çalıştırıp modeli kaydedin.")

else:
    # Başlangıç Ekranı
    st.info("👈 Analizi başlatmak için sol menüden bilgilerinizi girin ve butona basın.")
    
    # Placeholder Kolonlar
    cols = st.columns(3)
    cols[1].markdown("<div style='text-align: center; color: #555;'>Veri Bekleniyor...</div>", unsafe_allow_html=True)