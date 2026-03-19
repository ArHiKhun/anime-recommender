import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import base64
# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="✨ Anime Recommender Pro",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================
# CUSTOM CSS - VISUAL STUNNING DENGAN EFFECTS
# ============================================================
st.markdown("""
    <style>
    /* ===== ANIMATED BACKGROUND ===== */
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(240, 147, 251, 0.5), 0 0 40px rgba(245, 87, 108, 0.3); }
        50% { box-shadow: 0 0 30px rgba(240, 147, 251, 0.8), 0 0 60px rgba(245, 87, 108, 0.5); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* ===== MAIN STYLES ===== */
    .main {
        background: linear-gradient(-45deg, #1a1a2e, #16213e, #0f3460, #1a1a2e);
        background-size: 400% 400%;
        animation: gradient-shift 15s ease infinite;
    }
    
    .block-container {
        padding: 2rem;
        max-width: 1200px;
    }
    
    /* ===== HEADER ===== */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.1), rgba(245, 87, 108, 0.1));
        border-radius: 20px;
        border: 2px solid rgba(240, 147, 251, 0.3);
        box-shadow: 0 0 30px rgba(240, 147, 251, 0.2);
        animation: glow 3s ease-in-out infinite;
    }
    
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #f093fb, #f5576c, #ff9a9e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(240, 147, 251, 0.3);
        margin-bottom: 0.5rem !important;
    }
    
    .subtitle {
        color: #a0a0a0;
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid rgba(240, 147, 251, 0.2);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem;
    }
    
    /* ===== BUTTONS ===== */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 15px 40px !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%) !important;
    }
    
    .stButton>button:active {
        transform: translateY(-1px) !important;
    }
    
    /* ===== ANIME CARDS ===== */
    .anime-card {
        background: linear-gradient(145deg, #1e1e3f, #252550);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        transition: all 0.4s ease;
        overflow: hidden;
        position: relative;
    }
    
    .anime-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .anime-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(240, 147, 251, 0.2);
        border-color: rgba(240, 147, 251, 0.5);
    }
    
    .anime-card:hover::before {
        left: 100%;
    }
    
    .anime-title {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .anime-genre {
        display: inline-block;
        background: rgba(240, 147, 251, 0.2);
        color: #f093fb;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 2px;
        border: 1px solid rgba(240, 147, 251, 0.3);
    }
    
    .anime-rating {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
    }
    
    .rating-badge {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    
    /* ===== PROGRESS BAR ===== */
    .custom-progress {
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .custom-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #f093fb, #f5576c);
        border-radius: 10px;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(240, 147, 251, 0.5);
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(145deg, #1e1e3f, #252550);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        animation: float 6s ease-in-out infinite;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
        border-color: rgba(240, 147, 251, 0.5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* ===== FILTER SECTION ===== */
    .filter-section {
        background: linear-gradient(145deg, #1e1e3f, #252550);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .filter-title {
        color: #f093fb;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* ===== SLIDER STYLING ===== */
    .stSlider > div > div {
        background: linear-gradient(90deg, #667eea, #f093fb) !important;
    }
    
    /* ===== SELECTBOX STYLING ===== */
    .stSelectbox > div > div {
        background: #1e1e3f !important;
        border-color: rgba(240, 147, 251, 0.3) !important;
        color: white !important;
    }
    
    /* ===== DATAFRAME STYLING ===== */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(145deg, #1e1e3f, #252550);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer-text {
        color: #888;
        font-size: 0.9rem;
    }
    
    .footer-heart {
        color: #f5576c;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.2); }
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #f093fb);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #f093fb, #f5576c);
    }
    
    /* ===== BADGES ===== */
    .badge-premium {
        background: linear-gradient(135deg, #ffd700, #ffaa00);
        color: #000;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 10px;
    }
    
    .badge-new {
        background: linear-gradient(135deg, #00d9ff, #00ff88);
        color: #000;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 10px;
    }
    
    .badge-classic {
        background: linear-gradient(135deg, #ff6b6b, #f5576c);
        color: #fff;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 10px;
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #f093fb, transparent);
        margin: 2rem 0;
    }
    
    /* ===== LOADING ANIMATION ===== */
    .loading-text {
        text-align: center;
        padding: 2rem;
        color: #f093fb;
        font-size: 1.2rem;
        animation: shimmer 2s linear infinite;
        background: linear-gradient(90deg, #f093fb, #f5576c, #f093fb);
        background-size: 1000px 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1e1e3f !important;
        border: 1px solid rgba(240, 147, 251, 0.3) !important;
        color: #888 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border-color: #667eea !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: #1e1e3f !important;
        border: 1px solid rgba(240, 147, 251, 0.3) !important;
        border-radius: 10px !important;
        color: #f093fb !important;
    }
    
    /* ===== TOAST ===== */
    .toast-container {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
    }
    
    </style>
    """, unsafe_allow_html=True)
# ============================================================
# DATASET ANIME - ENHANCED
# ============================================================
@st.cache_data
def load_data():
    data = {
        'judul': [
            'Attack on Titan', 'Death Note', 'One Piece', 'Naruto', 
            'Demon Slayer', 'My Hero Academia', 'Fullmetal Alchemist',
            'Spy x Family', 'Jujutsu Kaisen', 'Tokyo Ghoul',
            'Your Name', 'A Silent Voice', 'Weathering With You',
            'Horimiya', 'Toradora', 'Kaguya-sama Love is War',
            'One Punch Man', 'Mob Psycho 100', 'Hunter x Hunter',
            'Steins Gate', 'Code Geass', 'Cowboy Bebop'
        ],
        'genre': [
            'Action, Dark Fantasy, Military',
            'Supernatural, Thriller, Psychological',
            'Action, Adventure, Comedy',
            'Action, Adventure, Martial Arts',
            'Action, Dark Fantasy, Historical',
            'Action, Comedy, School, Superhero',
            'Action, Adventure, Fantasy, Military',
            'Action, Comedy, Family, Slice of Life',
            'Action, Dark Fantasy, School',
            'Action, Dark Fantasy, Horror, Psychological',
            'Romance, Drama, Fantasy',
            'Drama, Romance, School',
            'Romance, Drama, Fantasy',
            'Romance, Comedy, School, Slice of Life',
            'Romance, Comedy, Drama, School',
            'Romance, Comedy, Psychological, School',
            'Action, Comedy, Parody, Superhero',
            'Action, Comedy, Supernatural',
            'Action, Adventure, Fantasy',
            'Sci-Fi, Thriller, Drama',
            'Action, Drama, Mecha, Military',
            'Action, Adventure, Sci-Fi'
        ],
        'rating': [
            9.0, 9.0, 8.9, 8.4, 8.7, 8.4, 9.1,
            8.6, 8.7, 8.0, 8.4, 8.2, 7.9,
            8.2, 8.1, 8.4, 8.5, 8.4, 9.0,
            9.1, 8.7, 8.8
        ],
        'tahun': [
            2013, 2006, 1999, 2002, 2019, 2016, 2009,
            2022, 2020, 2014, 2016, 2016, 2019,
            2021, 2008, 2018, 2015, 2016, 2011,
            2011, 2006, 1998
        ],
        'sinopsis': [
            'Humanity fights for survival against giant humanoid Titans in a post-apocalyptic world',
            'A high school student discovers a supernatural notebook that grants him power over life and death',
            'Pirate adventure to find the ultimate treasure, One Piece',
            'Young ninja seeks recognition and dreams of becoming Hokage of his village',
            'Boy becomes demon slayer to save his sister and avenge his family',
            'Students train to become professional heroes in a world of superpowers',
            'Two brothers search for Philosopher Stone to restore their bodies',
            'Spy creates fake family for mission, unaware each has their own secrets',
            'Student fights curses in supernatural world with ancient techniques',
            'College student becomes half-ghoul after accident and must survive',
            'Two teenagers connected by fate across time and space',
            'Story about bullying, redemption, and second chances',
            'Boy meets mysterious girl who can control weather',
            'High school romance between popular and quiet students',
            'Unlikely romance between two high school students with opposite personalities',
            'Battle of wits between two student council members in love',
            'Hero who can defeat any enemy with one punch seeks a worthy challenge',
            'Boy with psychic powers tries to live normal life',
            'Young boy searches for his father and faces incredible challenges',
            'Time travel thriller with devastating consequences',
            'Exiled prince leads rebellion with supernatural power of Geass',
            'Bounty hunters travel through space in jazz-infused noir adventures'
        ],
        'episodes': [
            87, 37, 1071, 220, 26, 113, 64, 25, 24, 12,
            1, 1, 1, 13, 25, 24, 24, 25, 148, 24, 50, 26
        ],
        'studio': [
            'Wit Studio/MAPPA', 'Madhouse', 'Toei Animation', 'Pierrot',
            'ufotable', 'Bones', 'Bones', 'Wit Studio/CloverWorks', 'MAPPA',
            'Pierrot', 'CoMix Wave', 'Kyoto Animation', 'CoMix Wave',
            'CloverWorks', 'J.C.Staff', 'A-1 Pictures', 'Madhouse', 'Bones',
            'Madhouse', 'White Fox', 'Sunrise', 'Sunrise'
        ],
        'status': [
            'Completed', 'Completed', 'Ongoing', 'Completed', 'Completed',
            'Ongoing', 'Completed', 'Ongoing', 'Completed', 'Completed',
            'Movie', 'Movie', 'Movie', 'Completed', 'Completed', 'Completed',
            'Completed', 'Completed', 'Completed', 'Completed', 'Completed', 'Completed'
        ],
        'type': [
            'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV',
            'Movie', 'Movie', 'Movie', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV',
            'TV', 'TV', 'TV'
        ],
        'image_url': [
            'https://cdn.myanimelist.net/images/anime/10/47347.jpg',
            'https://cdn.myanimelist.net/images/anime/9/9453.jpg',
            'https://cdn.myanimelist.net/images/anime/6/73245.jpg',
            'https://cdn.myanimelist.net/images/anime/13/17405.jpg',
            'https://cdn.myanimelist.net/images/anime/1286/99889.jpg',
            'https://cdn.myanimelist.net/images/anime/10/78745.jpg',
            'https://cdn.myanimelist.net/images/anime/5/47421.jpg',
            'https://cdn.myanimelist.net/images/anime/1441/122795.jpg',
            'https://cdn.myanimelist.net/images/anime/1171/109222.jpg',
            'https://cdn.myanimelist.net/images/anime/1498/107326.jpg',
            'https://cdn.myanimelist.net/images/anime/5/87048.jpg',
            'https://cdn.myanimelist.net/images/anime/1122/96435.jpg',
            'https://cdn.myanimelist.net/images/anime/10/87048.jpg',
            'https://cdn.myanimelist.net/images/anime/1695/111486.jpg',
            'https://cdn.myanimelist.net/images/anime/13/22128.jpg',
            'https://cdn.myanimelist.net/images/anime/1295/106551.jpg',
            'https://cdn.myanimelist.net/images/anime/12/76049.jpg',
            'https://cdn.myanimelist.net/images/anime/8/80356.jpg',
            'https://cdn.myanimelist.net/images/anime/1337/99013.jpg',
            'https://cdn.myanimelist.net/images/anime/5/73199.jpg',
            'https://cdn.myanimelist.net/images/anime/5/50331.jpg',
            'https://cdn.myanimelist.net/images/anime/4/19646.jpg'
        ]
    }
    return pd.DataFrame(data)
# Load data
df = load_data()
# ============================================================
# HEADER SECTION
# ============================================================
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎌 Anime Recommender Pro</h1>
        <p class="subtitle">✨ Temukan Anime Sesuai Selera Kamu! ✨</p>
    </div>
""", unsafe_allow_html=True)
# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
        <h2 style="color: #f093fb; font-size: 1.5rem;">⚙️ Preferensi Kamu</h2>
        <p style="color: #888; font-size: 0.8rem;">Atur filter untuk hasil terbaik</p>
    </div>
""", unsafe_allow_html=True)
# Genre Filter
st.sidebar.markdown("""
    <div class="filter-section">
        <div class="filter-title">🎭 Genre Favorit</div>
    </div>
""", unsafe_allow_html=True)
all_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Romance', 
              'Fantasy', 'Sci-Fi', 'Thriller', 'Supernatural', 'Slice of Life']
genre_preference = st.sidebar.multiselect(
    "Pilih Genre:",
    options=all_genres,
    default=['Action'],
    label_visibility="collapsed"
)
# Rating Filter
st.sidebar.markdown("""
    <div class="filter-section">
        <div class="filter-title">⭐ Minimal Rating</div>
    </div>
""", unsafe_allow_html=True)
min_rating = st.sidebar.slider(
    "Minimal Rating:",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1,
    label_visibility="collapsed"
)
# Year Range
st.sidebar.markdown("""
    <div class="filter-section">
        <div class="filter-title">📅 Range Tahun</div>
    </div>
""", unsafe_allow_html=True)
tahun_range = st.sidebar.slider(
    "Range Tahun:",
    min_value=1990,
    max_value=2024,
    value=(2010, 2024),
    label_visibility="collapsed"
)
# Type Filter
st.sidebar.markdown("""
    <div class="filter-section">
        <div class="filter-title">📺 Tipe Anime</div>
    </div>
""", unsafe_allow_html=True)
type_options = ['All', 'TV', 'Movie', 'OVA']
selected_type = st.sidebar.selectbox(
    "Tipe:",
    options=type_options,
    label_visibility="collapsed"
)
# Sort Options
st.sidebar.markdown("""
    <div class="filter-section">
        <div class="filter-title">🔄 Urutkan</div>
    </div>
""", unsafe_allow_html=True)
sort_by = st.sidebar.selectbox(
    "Urutkan Berdasarkan:",
    options=['Rating (Tinggi ke Rendah)', 'Rating (Rendah ke Tinggi)', 'Tahun (Terbaru)', 'Tahun (Terlama)', 'Nama (A-Z)'],
    label_visibility="collapsed"
)
# Status Filter
st.sidebar.markdown("""
    <div class="filter-section">
        <div class="filter-title">📊 Status</div>
    </div>
""", unsafe_allow_html=True)
status_options = ['All', 'Completed', 'Ongoing']
selected_status = st.sidebar.selectbox(
    "Status:",
    options=status_options,
    label_visibility="collapsed"
)
# Quick Stats in Sidebar
total_anime = len(df)
avg_rating = df['rating'].mean()
st.sidebar.markdown("""
    <hr style="border-color: rgba(240, 147, 251, 0.3); margin: 2rem 0;">
    <div style="background: linear-gradient(145deg, #1e1e3f, #252550); padding: 15px; border-radius: 15px; text-align: center;">
        <h3 style="color: #f093fb; margin-bottom: 10px;">📈 Database Stats</h3>
        <div style="display: flex; justify-content: space-around;">
            <div>
                <p style="color: #667eea; font-size: 1.5rem; font-weight: bold; margin: 0;">{}</p>
                <p style="color: #888; font-size: 0.8rem; margin: 0;">Total Anime</p>
            </div>
            <div>
                <p style="color: #f5576c; font-size: 1.5rem; font-weight: bold; margin: 0;">{:.1f}</p>
                <p style="color: #888; font-size: 0.8rem; margin: 0;">Avg Rating</p>
            </div>
        </div>
    </div>
""".format(total_anime, avg_rating), unsafe_allow_html=True)
# ============================================================
# FUNGSI REKOMENDASI
# ============================================================
def get_recommendations(df, genres, min_rating, year_range, sort_method, anime_type='All', status='All'):
    # Filter berdasarkan rating dan tahun
    filtered_df = df[(df['rating'] >= min_rating) & 
                     (df['tahun'] >= year_range[0]) & 
                     (df['tahun'] <= year_range[1])].copy()
    
    # Filter berdasarkan tipe
    if anime_type != 'All':
        filtered_df = filtered_df[filtered_df['type'] == anime_type].copy()
    
    # Filter berdasarkan status
    if status != 'All':
        filtered_df = filtered_df[filtered_df['status'] == status].copy()
    
    # Filter berdasarkan genre
    if genres:
        def match_genre(anime_genres, selected_genres):
            anime_genre_list = [g.strip().lower() for g in anime_genres.split(',')]
            return any(genre.lower() in anime_genre_list for genre in selected_genres)
        
        filtered_df['match'] = filtered_df['genre'].apply(
            lambda x: match_genre(x, genres)
        )
        filtered_df = filtered_df[filtered_df['match'] == True]
    
    # Sort
    if sort_method == 'Rating (Tinggi ke Rendah)':
        filtered_df = filtered_df.sort_values('rating', ascending=False)
    elif sort_method == 'Rating (Rendah ke Tinggi)':
        filtered_df = filtered_df.sort_values('rating', ascending=True)
    elif sort_method == 'Tahun (Terbaru)':
        filtered_df = filtered_df.sort_values('tahun', ascending=False)
    elif sort_method == 'Tahun (Terlama)':
        filtered_df = filtered_df.sort_values('tahun', ascending=True)
    elif sort_method == 'Nama (A-Z)':
        filtered_df = filtered_df.sort_values('judul', ascending=True)
    
    return filtered_df
def get_badge(tahun, rating):
    if rating >= 9.0:
        return '<span class="badge-premium">⭐ PREMIUM</span>'
    elif tahun >= 2020:
        return '<span class="badge-new">🆕 NEW</span>'
    elif tahun <= 2005:
        return '<span class="badge-classic">👑 CLASSIC</span>'
    return ''
# ============================================================
# SESSION STATE
# ============================================================
if 'searched' not in st.session_state:
    st.session_state.searched = False
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'Card'
# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔍 Cari Anime", "📚 Database", "⭐ Favorit"])
# ============================================================
# TAB 1: CARI ANIME
# ============================================================
with tab1:
    # Search Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 CARI REKOMENDASI ANIME", use_container_width=True):
            st.session_state.searched = True
            st.balloons()
    
    # View Mode Toggle
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col3:
        view_mode = st.radio("Mode:", ["Card", "List"], horizontal=True, label_visibility="collapsed")
        st.session_state.view_mode = view_mode
    
    # Tampilkan hasil
    if st.session_state.searched:
        rekomendasi = get_recommendations(
            df, genre_preference, min_rating, tahun_range, sort_by, selected_type, selected_status
        )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Results Count with Style
        st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <h2 style="color: #f093fb; font-size: 1.8rem;">
                    📺 Ditemukan <span style="color: #f5576c;">{len(rekomendasi)}</span> Anime untuk Kamu!
                </h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if len(rekomendasi) == 0:
            st.warning("😅 Tidak ada anime yang sesuai kriteria. Coba longgarkan filter!")
        else:
            # Display Results
            if st.session_state.view_mode == 'Card':
                for idx, row in rekomendasi.iterrows():
                    badge = get_badge(row['tahun'], row['rating'])
                    
                    st.markdown(f"""
                        <div class="anime-card">
                            <div style="display: flex; gap: 20px;">
                                <div style="flex-shrink: 0;">
                                    <img src="{row['image_url']}" 
                                         style="width: 180px; height: 250px; object-fit: cover; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);"
                                         onerror="this.src='https://via.placeholder.com/180x250/1e1e3f/667eea?text={row['judul'].replace(' ', '+')}'">
                                </div>
                                <div style="flex: 1;">
                                    <h3 class="anime-title">{row['judul']} {badge}</h3>
                                    <div style="margin: 10px 0;">
                                        {''.join([f'<span class="anime-genre">{g.strip()}</span>' for g in row['genre'].split(',')])}
                                    </div>
                                    <div class="anime-rating">
                                        <span class="rating-badge">⭐ {row['rating']}</span>
                                        <span style="color: #888;">({row['episodes']} eps)</span>
                                        <span style="color: #667eea;">📅 {row['tahun']}</span>
                                        <span style="color: #00ff88;">🏢 {row['studio']}</span>
                                    </div>
                                    <p style="color: #a0a0a0; margin: 15px 0; line-height: 1.6;">{row['sinopsis']}</p>
                                    <div class="custom-progress">
                                        <div class="custom-progress-bar" style="width: {int(row['rating'] * 10)}%;"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to favorites button
                    col1, col2, col3 = st.columns([6, 2, 2])
                    with col2:
                        if st.button(f"❤️ Favorit", key=f"fav_{idx}"):
                            if row['judul'] not in [f['judul'] for f in st.session_state.favorites]:
                                st.session_state.favorites.append({
                                    'judul': row['judul'],
                                    'genre': row['genre'],
                                    'rating': row['rating'],
                                    'tahun': row['tahun']
                                })
                                st.success(f"✅ {row['judul']} ditambahkan ke favorit!")
                    with col3:
                        if st.button(f"🔗 Detail", key=f"detail_{idx}"):
                            st.info(f"📺 {row['judul']} | Studio: {row['studio']} | Status: {row['status']}")
            else:
                # List View
                st.dataframe(
                    rekomendasi[['judul', 'genre', 'rating', 'tahun', 'type', 'status']],
                    use_container_width=True,
                    hide_index=True
                )
            
            # Statistik
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("""
                <h3 style="color: #f093fb; text-align: center; margin-bottom: 2rem;">
                    📊 Statistik Hasil
                </h3>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                    <div class="metric-card" style="animation-delay: 0s;">
                        <div class="metric-value">{len(rekomendasi)}</div>
                        <div class="metric-label">Total Anime</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card" style="animation-delay: 0.5s;">
                        <div class="metric-value">{rekomendasi['rating'].mean():.1f}</div>
                        <div class="metric-label">Rata-rata Rating</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class="metric-card" style="animation-delay: 1s;">
                        <div class="metric-value">{rekomendasi['rating'].max():.1f}</div>
                        <div class="metric-label">Rating Tertinggi</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class="metric-card" style="animation-delay: 1.5s;">
                        <div class="metric-value">{rekomendasi['tahun'].min()}-{rekomendasi['tahun'].max()}</div>
                        <div class="metric-label">Range Tahun</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Reset Button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Reset Pencarian", use_container_width=True):
                    st.session_state.searched = False
                    st.rerun()
    else:
        # Welcome State
        st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h2 style="color: #f093fb; font-size: 2rem; margin-bottom: 1rem;">
                    🎌 Selamat Datang di Anime Recommender Pro!
                </h2>
                <p style="color: #888; font-size: 1.1rem; line-height: 1.8; max-width: 600px; margin: 0 auto;">
                    Sistem rekomendasi anime cerdas yang membantu kamu menemukan anime 
                    terbaik sesuai genre favorit dan preferensi rating.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Features Grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                <div style="background: linear-gradient(145deg, #1e1e3f, #252550); padding: 25px; border-radius: 20px; text-align: center; border: 1px solid rgba(240, 147, 251, 0.2);">
                    <div style="font-size: 3rem; margin-bottom: 10px;">🎭</div>
                    <h3 style="color: #f093fb;">Filter Cerdas</h3>
                    <p style="color: #888;">Pilih berbagai genre dengan kombinasi rating</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div style="background: linear-gradient(145deg, #1e1e3f, #252550); padding: 25px; border-radius: 20px; text-align: center; border: 1px solid rgba(240, 147, 251, 0.2);">
                    <div style="font-size: 3rem; margin-bottom: 10px;">⭐</div>
                    <h3 style="color: #f093fb;">Rating Akurat</h3>
                    <p style="color: #888;">Database dengan rating terpercaya</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
                <div style="background: linear-gradient(145deg, #1e1e3f, #252550); padding: 25px; border-radius: 20px; text-align: center; border: 1px solid rgba(240, 147, 251, 0.2);">
                    <div style="font-size: 3rem; margin-bottom: 10px;">⚡</div>
                    <h3 style="color: #f093fb;">Cepat & Mudah</h3>
                    <p style="color: #888;">Temukan anime favorit dalam hitungan detik</p>
                </div>
            """, unsafe_allow_html=True)
# ============================================================
# TAB 2: DATABASE
# ============================================================
with tab2:
    st.markdown("""
        <h2 style="color: #f093fb; margin-bottom: 1rem;">📚 Database Anime Lengkap</h2>
    """, unsafe_allow_html=True)
    
    # Search in database
    search_query = st.text_input("🔍 Cari anime:", placeholder="Masukkan nama anime...")
    
    if search_query:
        filtered = df[df['judul'].str.contains(search_query, case=False, na=False)]
    else:
        filtered = df
    
    # Display with enhanced styling
    st.dataframe(
        filtered[['judul', 'genre', 'rating', 'tahun', 'type', 'episodes', 'studio', 'status']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "judul": st.column_config.TextColumn("Judul", width="large"),
            "genre": st.column_config.TextColumn("Genre"),
            "rating": st.column_config.ProgressColumn("Rating", min_value=0, max_value=10, format="%.1f"),
            "tahun": st.column_config.NumberColumn("Tahun"),
            "type": st.column_config.TextColumn("Tipe"),
            "episodes": st.column_config.NumberColumn("Eps"),
            "studio": st.column_config.TextColumn("Studio"),
            "status": st.column_config.TextColumn("Status")
        }
    )
    
    # Show top rated
    st.markdown("""
        <h3 style="color: #f093fb; margin-top: 2rem;">🏆 Top Rated Anime</h3>
    """, unsafe_allow_html=True)
    
    top_anime = df.nlargest(5, 'rating')
    cols = st.columns(5)
    for idx, (i, row) in enumerate(top_anime.iterrows()):
        with cols[idx]:
            st.image(row['image_url'], use_container_width=True)
            st.markdown(f"""
                <div style="text-align: center;">
                    <p style="color: white; font-weight: bold; margin: 5px 0;">{row['judul']}</p>
                    <p style="color: #f5576c; margin: 0;">⭐ {row['rating']}</p>
                </div>
            """, unsafe_allow_html=True)
# ============================================================
# TAB 3: FAVORIT
# ============================================================
with tab3:
    st.markdown("""
        <h2 style="color: #f093fb; margin-bottom: 1rem;">⭐ Anime Favorit Kamu</h2>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.favorites) == 0:
        st.info("📝 Belum ada anime favorit. Tambahkan dari tab pencarian!")
    else:
        for fav in st.session_state.favorites:
            st.markdown(f"""
                <div style="background: linear-gradient(145deg, #1e1e3f, #252550); padding: 15px; border-radius: 15px; margin: 10px 0; border: 1px solid rgba(240, 147, 251, 0.3); display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="color: white; margin: 0;">{fav['judul']}</h4>
                        <p style="color: #888; margin: 5px 0; font-size: 0.9rem;">{fav['genre']} | ⭐ {fav['rating']} | 📅 {fav['tahun']}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Hapus Semua Favorit"):
            st.session_state.favorites = []
            st.rerun()
# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer">
        <p class="footer-text">
            <b>🎌 Anime Recommender Pro - Skripsi S1 Ilmu Komputer</b><br>
            Built with <span class="footer-heart">❤️</span> using Streamlit & Python
        </p>
        <p style="color: #666; font-size: 0.8rem; margin-top: 10px;">
            © 2024 | Data dari MyAnimeList
        </p>
    </div>
""", unsafe_allow_html=True)
