import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="✨ Anime Recommender ML Pro",
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
    
    @keyframes pulse-ml {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
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
    
    /* ===== ML BADGE ===== */
    .ml-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00d9ff, #00ff88);
        color: #000;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-left: 10px;
        animation: pulse-ml 2s ease-in-out infinite;
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
    
    .similarity-badge {
        background: linear-gradient(135deg, #00d9ff, #00ff88);
        color: #000;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
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
    
    .similarity-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #00d9ff, #00ff88);
        border-radius: 10px;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
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
    
    /* ===== ML INFO BOX ===== */
    .ml-info-box {
        background: linear-gradient(145deg, #1e1e3f, #252550);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid rgba(0, 217, 255, 0.3);
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.1);
    }
    
    .ml-info-title {
        color: #00d9ff;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
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
    
    .badge-ml {
        background: linear-gradient(135deg, #00d9ff, #667eea);
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
    
    /* ===== CLUSTER COLORS ===== */
    .cluster-0 { border-left: 4px solid #667eea; }
    .cluster-1 { border-left: 4px solid #f093fb; }
    .cluster-2 { border-left: 4px solid #00d9ff; }
    .cluster-3 { border-left: 4px solid #00ff88; }
    .cluster-4 { border-left: 4px solid #ffd700; }
    
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# DATASET ANIME - ENHANCED (40 ANIME)
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
            'Steins Gate', 'Code Geass', 'Cowboy Bebop', 'Violet Evergarden',
            'Erased', 'Re:Zero', 'No Game No Life', 'Sword Art Online',
            'Tokyo Revengers', 'Chainsaw Man', 'Blue Lock', 'Haikyuu',
            'Kuroko no Basket', 'Food Wars', 'Dr Stone', 'Black Clover',
            'Fire Force', 'The Promised Neverland', 'Made in Abyss',
            'Vivy Fluorite Eye', '86 Eighty Six', 'Ousama Ranking'
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
            'Action, Adventure, Sci-Fi',
            'Drama, Fantasy, Slice of Life',
            'Mystery, Supernatural, Thriller',
            'Drama, Fantasy, Psychological, Thriller',
            'Adventure, Comedy, Fantasy, Supernatural',
            'Action, Adventure, Fantasy, Romance',
            'Action, Drama, Supernatural',
            'Action, Comedy, Fantasy, Supernatural',
            'Sports, Drama, School',
            'Comedy, Drama, Sports, School',
            'Comedy, Sports, School',
            'Ecchi, Gourmet, School, Shounen',
            'Adventure, Comedy, Sci-Fi, Shounen',
            'Action, Comedy, Fantasy, Magic, Shounen',
            'Action, Sci-Fi, Shounen, Supernatural',
            'Horror, Mystery, Psychological, Sci-Fi, Thriller',
            'Adventure, Fantasy, Mystery, Sci-Fi',
            'Action, Music, Sci-Fi, Thriller',
            'Action, Drama, Mecha, Military, Sci-Fi',
            'Action, Adventure, Fantasy'
        ],
        'rating': [
            9.0, 9.0, 8.9, 8.4, 8.7, 8.4, 9.1,
            8.6, 8.7, 8.0, 8.4, 8.2, 7.9,
            8.2, 8.1, 8.4, 8.5, 8.4, 9.0,
            9.1, 8.7, 8.8, 8.7, 8.4, 8.4,
            8.2, 7.6, 8.1, 8.7, 8.3, 8.7,
            8.1, 8.4, 8.3, 8.2, 7.8, 8.5,
            8.7, 8.5, 8.7, 8.6
        ],
        'tahun': [
            2013, 2006, 1999, 2002, 2019, 2016, 2009,
            2022, 2020, 2014, 2016, 2016, 2019,
            2021, 2008, 2018, 2015, 2016, 2011,
            2011, 2006, 1998, 2018, 2016, 2016,
            2014, 2012, 2021, 2022, 2022, 2014,
            2012, 2015, 2019, 2017, 2019, 2019,
            2017, 2021, 2021, 2021
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
            'Bounty hunters travel through space in jazz-infused noir adventures',
            'Auto Memory Doll writes letters that connect peoples hearts',
            'Man travels back in time to prevent tragedy and catch killer',
            'Boy wakes up in parallel world and discovers he can return from death',
            'Gaming genius siblings transported to world where games decide everything',
            'Players trapped in VRMMORPG must clear game to escape',
            'Delinquent travels back in time to save his girlfriend',
            'Boy makes contract with chainsaw devil and hunts devils',
            'Soccer prodigy enters special training facility to become best striker',
            'High school volleyball team aims for nationals',
            'Basketball prodigies battle in high school tournaments',
            'Boy attends elite culinary school and engages in cooking battles',
            'Scientific genius awakens in stone world and rebuilds civilization',
            'Orphan boy receives magic grimoire and aims to become Wizard King',
            'Special fire force fights spontaneous human combustion',
            'Orphans discover dark truth about their orphanage and plan escape',
            'Girl descends into abyss filled with mysteries and dangers',
            'AI songstress tries to change future and save humanity',
            'Young soldiers fight in unmanned mechs against autonomous machines',
            'Deaf prince seeks to become greatest king with help of shadow friend'
        ],
        'episodes': [
            87, 37, 1071, 220, 26, 113, 64, 25, 24, 12,
            1, 1, 1, 13, 25, 24, 24, 25, 148, 24, 50, 26,
            13, 12, 50, 12, 25, 24, 12, 24, 85, 75, 24,
            35, 24, 170, 48, 24, 13, 11, 23, 23
        ],
        'studio': [
            'Wit Studio/MAPPA', 'Madhouse', 'Toei Animation', 'Pierrot',
            'ufotable', 'Bones', 'Bones', 'Wit Studio/CloverWorks', 'MAPPA',
            'Pierrot', 'CoMix Wave', 'Kyoto Animation', 'CoMix Wave',
            'CloverWorks', 'J.C.Staff', 'A-1 Pictures', 'Madhouse', 'Bones',
            'Madhouse', 'White Fox', 'Sunrise', 'Sunrise', 'Kyoto Animation',
            'A-1 Pictures', 'White Fox', 'Madhouse', 'A-1 Pictures',
            'LIDENFILMS', 'MAPPA', '8bit', 'Production I.G', 'Production I.G',
            'J.C.Staff', 'TMS Entertainment', 'Pierrot', 'David Production',
            'CloverWorks', 'Kinema Citrus', 'Wit Studio', 'A-1 Pictures',
            'Wit Studio'
        ],
        'status': [
            'Completed', 'Completed', 'Ongoing', 'Completed', 'Completed',
            'Ongoing', 'Completed', 'Ongoing', 'Completed', 'Completed',
            'Movie', 'Movie', 'Movie', 'Completed', 'Completed', 'Completed',
            'Completed', 'Completed', 'Completed', 'Completed', 'Completed', 'Completed',
            'Completed', 'Completed', 'Ongoing', 'Completed', 'Completed',
            'Ongoing', 'Completed', 'Completed', 'Completed', 'Ongoing', 'Completed',
            'Ongoing', 'Ongoing', 'Ongoing', 'Completed', 'Ongoing', 'Completed',
            'Completed', 'Completed', 'Completed'
        ],
        'type': [
            'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV',
            'Movie', 'Movie', 'Movie', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV',
            'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV',
            'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV',
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
            'https://cdn.myanimelist.net/images/anime/4/19646.jpg',
            'https://cdn.myanimelist.net/images/anime/1795/95088.jpg',
            'https://cdn.myanimelist.net/images/anime/10/77957.jpg',
            'https://cdn.myanimelist.net/images/anime/11/79410.jpg',
            'https://cdn.myanimelist.net/images/anime/5/65187.jpg',
            'https://cdn.myanimelist.net/images/anime/11/39717.jpg',
            'https://cdn.myanimelist.net/images/anime/1179/119897.jpg',
            'https://cdn.myanimelist.net/images/anime/1802/108501.jpg',
            'https://cdn.myanimelist.net/images/anime/1339/138709.jpg',
            'https://cdn.myanimelist.net/images/anime/7/76014.jpg',
            'https://cdn.myanimelist.net/images/anime/11/79412.jpg',
            'https://cdn.myanimelist.net/images/anime/3/72943.jpg',
            'https://cdn.myanimelist.net/images/anime/1613/102576.jpg',
            'https://cdn.myanimelist.net/images/anime/2/88336.jpg',
            'https://cdn.myanimelist.net/images/anime/1498/121440.jpg',
            'https://cdn.myanimelist.net/images/anime/1125/96929.jpg',
            'https://cdn.myanimelist.net/images/anime/6/86733.jpg',
            'https://cdn.myanimelist.net/images/anime/1316/107302.jpg',
            'https://cdn.myanimelist.net/images/anime/1808/111717.jpg',
            'https://cdn.myanimelist.net/images/anime/1347/117616.jpg'
        ]
    }
    return pd.DataFrame(data)

# ============================================================
# MACHINE LEARNING FUNCTIONS
# ============================================================

@st.cache_data
def prepare_ml_features(df):
    """Prepare features for machine learning"""
    # Extract individual genres
    all_genres = set()
    for genres in df['genre']:
        for g in genres.split(','):
            all_genres.add(g.strip())
    all_genres = sorted(list(all_genres))
    
    # Create genre binary matrix
    genre_matrix = []
    for genres in df['genre']:
        genre_list = [g.strip() for g in genres.split(',')]
        genre_row = [1 if g in genre_list else 0 for g in all_genres]
        genre_matrix.append(genre_row)
    
    genre_df = pd.DataFrame(genre_matrix, columns=all_genres)
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(df[['rating', 'tahun', 'episodes']])
    
    # Combine features
    features = np.hstack([genre_df.values, numerical_features])
    
    return features, all_genres, scaler

def compute_similarity_matrix(df, features):
    """Compute cosine similarity matrix for content-based filtering"""
    # Use TF-IDF on genre + sinopsis for text similarity
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    text_features = tfidf.fit_transform(df['genre'] + ' ' + df['sinopsis'])
    
    # Combine with numerical features
    from sklearn.feature_extraction.text import TfidfTransformer
    from scipy.sparse import hstack, csr_matrix
    
    numerical_sparse = csr_matrix(features[:, -3:])  # rating, tahun, episodes
    combined_features = hstack([text_features, numerical_sparse])
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(combined_features)
    
    return similarity_matrix

def get_similar_animes(df, anime_title, similarity_matrix, n=5):
    """Get similar animes based on content similarity"""
    try:
        idx = df[df['judul'] == anime_title].index[0]
    except IndexError:
        return []
    
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # Exclude self
    
    anime_indices = [i[0] for i in sim_scores]
    similarities = [i[1] for i in sim_scores]
    
    results = []
    for i, sim in zip(anime_indices, similarities):
        results.append({
            'judul': df.iloc[i]['judul'],
            'genre': df.iloc[i]['genre'],
            'rating': df.iloc[i]['rating'],
            'tahun': df.iloc[i]['tahun'],
            'similarity': sim,
            'image_url': df.iloc[i]['image_url']
        })
    
    return results

def cluster_animes(df, features, n_clusters=5):
    """Cluster animes using K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    # Analyze clusters
    cluster_info = {}
    for i in range(n_clusters):
        cluster_animes = df_clustered[df_clustered['cluster'] == i]
        avg_rating = cluster_animes['rating'].mean()
        
        # Get most common genres
        all_genres = []
        for genres in cluster_animes['genre']:
            all_genres.extend([g.strip() for g in genres.split(',')])
        top_genres = Counter(all_genres).most_common(3)
        
        cluster_info[i] = {
            'count': len(cluster_animes),
            'avg_rating': avg_rating,
            'top_genres': [g[0] for g in top_genres],
            'animes': cluster_animes['judul'].tolist()
        }
    
    return df_clustered, cluster_info, kmeans

def get_cluster_recommendations(df_clustered, cluster_info, selected_anime, n=5):
    """Get recommendations from same cluster"""
    try:
        anime_cluster = df_clustered[df_clustered['judul'] == selected_anime]['cluster'].iloc[0]
    except IndexError:
        return []
    
    same_cluster = df_clustered[
        (df_clustered['cluster'] == anime_cluster) & 
        (df_clustered['judul'] != selected_anime)
    ].sort_values('rating', ascending=False).head(n)
    
    return same_cluster.to_dict('records')

def create_pca_visualization(features, df):
    """Create PCA visualization of anime clusters"""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    fig = px.scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        color=df['cluster'] if 'cluster' in df.columns else None,
        hover_name=df['judul'],
        hover_data={'Rating': df['rating'], 'Tahun': df['tahun']},
        title='🎯 Anime Clustering Visualization (PCA)',
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'},
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,63,0.8)',
        font_color='white',
        title_font_color='#f093fb',
        legend_font_color='white'
    )
    
    return fig

# ============================================================
# DATA LOADING
# ============================================================
df = load_data()

# Prepare ML features
features, all_genres, scaler = prepare_ml_features(df)
similarity_matrix = compute_similarity_matrix(df, features)
df_clustered, cluster_info, kmeans_model = cluster_animes(df, features, n_clusters=5)

# ============================================================
# HEADER SECTION
# ============================================================
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎌 Anime Recommender <span class="ml-badge">🤖 ML POWERED</span></h1>
        <p class="subtitle">✨ Sistem Rekomendasi Cerdas dengan Machine Learning ✨</p>
    </div>
""", unsafe_allow_html=True)

# ML Info Box
st.markdown("""
    <div class="ml-info-box">
        <div class="ml-info-title">🧠 Teknologi Machine Learning yang Digunakan:</div>
        <div style="color: #a0a0a0; line-height: 1.8;">
            • <b>TF-IDF + Cosine Similarity</b> - Untuk Content-Based Filtering berdasarkan genre & sinopsis<br>
            • <b>K-Means Clustering</b> - Mengelompokkan anime berdasarkan karakteristik serupa<br>
            • <b>PCA Visualization</b> - Visualisasi 2D dari hasil clustering<br>
            • <b>Hybrid Recommendation</b> - Kombinasi filter untuk hasil optimal
        </div>
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

# ML Mode Toggle
st.sidebar.markdown("""
    <div class="filter-section">
        <div class="filter-title">🤖 Mode Rekomendasi</div>
    </div>
""", unsafe_allow_html=True)

ml_mode = st.sidebar.radio(
    "Pilih Mode:",
    ["🎯 Content-Based ML", "📊 Cluster-Based ML", "🔍 Hybrid ML", "⚡ Traditional Filter"],
    label_visibility="collapsed"
)

# Genre Filter
st.sidebar.markdown("""
    <div class="filter-section">
        <div class="filter-title">🎭 Genre Favorit</div>
    </div>
""", unsafe_allow_html=True)

all_genres_list = ['Action', 'Adventure', 'Comedy', 'Drama', 'Romance', 
              'Fantasy', 'Sci-Fi', 'Thriller', 'Supernatural', 'Slice of Life',
              'Horror', 'Mystery', 'Sports', 'School']
genre_preference = st.sidebar.multiselect(
    "Pilih Genre:",
    options=all_genres_list,
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

# Reference Anime (for ML modes)
if "ML" in ml_mode:
    st.sidebar.markdown("""
        <div class="filter-section">
            <div class="filter-title">📺 Anime Referensi</div>
        </div>
    """, unsafe_allow_html=True)
    
    reference_anime = st.sidebar.selectbox(
        "Pilih anime yang kamu sukai:",
        options=df['judul'].tolist(),
        label_visibility="collapsed"
    )
    
    n_recommendations = st.sidebar.slider(
        "Jumlah Rekomendasi:",
        min_value=1,
        max_value=10,
        value=5
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
# SESSION STATE
# ============================================================
if 'searched' not in st.session_state:
    st.session_state.searched = False
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'Card'
if 'last_recommendations' not in st.session_state:
    st.session_state.last_recommendations = None

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Rekomendasi", "📊 ML Analytics", "📚 Database", "⭐ Favorit"])

# ============================================================
# TAB 1: REKOMENDASI
# ============================================================
with tab1:
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #00d9ff;">🤖 Mode Aktif: {ml_mode}</h2>
            <p style="color: #888;">Sistem akan menggunakan algoritma machine learning untuk memberikan rekomendasi terbaik</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Search Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 GENERATE REKOMENDASI", use_container_width=True):
            st.session_state.searched = True
            st.balloons()
    
    # Tampilkan hasil
    if st.session_state.searched:
        st.markdown("<hr>", unsafe_allow_html=True)
        
        recommendations = []
        
        # Generate recommendations based on mode
        if ml_mode == "🎯 Content-Based ML" and 'reference_anime' in locals():
            # Content-Based using Cosine Similarity
            recommendations = get_similar_animes(df, reference_anime, similarity_matrix, n_recommendations)
            st.success(f"🎯 Rekomendasi berbasis konten untuk '{reference_anime}' menggunakan Cosine Similarity")
            
        elif ml_mode == "📊 Cluster-Based ML" and 'reference_anime' in locals():
            # Cluster-Based Recommendations
            recommendations = get_cluster_recommendations(df_clustered, cluster_info, reference_anime, n_recommendations)
            anime_cluster = df_clustered[df_clustered['judul'] == reference_anime]['cluster'].iloc[0]
            st.success(f"📊 Rekomendasi dari Cluster #{anime_cluster} menggunakan K-Means")
            
        elif ml_mode == "🔍 Hybrid ML" and 'reference_anime' in locals():
            # Hybrid: Combine Content-Based and Cluster-Based
            content_recs = get_similar_animes(df, reference_anime, similarity_matrix, n_recommendations//2 + 1)
            cluster_recs = get_cluster_recommendations(df_clustered, cluster_info, reference_anime, n_recommendations//2 + 1)
            
            # Merge and remove duplicates
            seen_titles = set()
            recommendations = []
            for rec in content_recs + cluster_recs:
                if rec['judul'] not in seen_titles and rec['judul'] != reference_anime:
                    recommendations.append(rec)
                    seen_titles.add(rec['judul'])
                    if len(recommendations) >= n_recommendations:
                        break
            st.success(f"🔍 Rekomendasi Hybrid (Content + Cluster) untuk '{reference_anime}'")
            
        else:
            # Traditional Filter (fallback)
            filtered = df[
                (df['rating'] >= min_rating) & 
                (df['tahun'] >= tahun_range[0]) & 
                (df['tahun'] <= tahun_range[1])
            ].copy()
            
            if genre_preference:
                def match_genre(anime_genres, selected_genres):
                    anime_genre_list = [g.strip().lower() for g in anime_genres.split(',')]
                    return any(genre.lower() in anime_genre_list for genre in selected_genres)
                
                filtered['match'] = filtered['genre'].apply(lambda x: match_genre(x, genre_preference))
                filtered = filtered[filtered['match'] == True]
            
            recommendations = filtered.sort_values('rating', ascending=False).head(10).to_dict('records')
            for rec in recommendations:
                rec['similarity'] = rec['rating'] / 10  # Normalize for display
            st.success("⚡ Rekomendasi berbasis Filter Tradisional")
        
        st.session_state.last_recommendations = recommendations
        
        # Display Results
        st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <h2 style="color: #f093fb; font-size: 1.8rem;">
                    📺 Ditemukan <span style="color: #f5576c;">{len(recommendations)}</span> Rekomendasi!
                </h2>
            </div>
        """, unsafe_allow_html=True)
        
        if len(recommendations) == 0:
            st.warning("😅 Tidak ada rekomendasi yang cocok. Coba ubah preferensi!")
        else:
            for idx, rec in enumerate(recommendations):
                similarity_pct = int(rec.get('similarity', 0.7) * 100)
                badge_class = f"cluster-{idx % 5}"
                
                st.markdown(f"""
                    <div class="anime-card {badge_class}">
                        <div style="display: flex; gap: 20px;">
                            <div style="flex-shrink: 0;">
                                <img src="{rec['image_url']}" 
                                     style="width: 180px; height: 250px; object-fit: cover; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);"
                                     onerror="this.src='https://via.placeholder.com/180x250/1e1e3f/667eea?text=No+Image'">
                            </div>
                            <div style="flex: 1;">
                                <h3 class="anime-title">{rec['judul']}</h3>
                                <div style="margin: 10px 0;">
                                    {''.join([f'<span class="anime-genre">{g.strip()}</span>' for g in rec['genre'].split(',')])}
                                </div>
                                <div class="anime-rating">
                                    <span class="rating-badge">⭐ {rec['rating']}</span>
                                    <span class="similarity-badge">🎯 Match: {similarity_pct}%</span>
                                </div>
                                <div style="margin: 10px 0;">
                                    <span style="color: #667eea;">📅 {rec['tahun']}</span>
                                </div>
                                <div class="custom-progress">
                                    <div class="similarity-progress-bar" style="width: {similarity_pct}%;"></div>
                                </div>
                                <p style="color: #00d9ff; font-size: 0.9rem; margin-top: 10px;">
                                    💡 ML Confidence: {similarity_pct}%
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add to favorites button
                col1, col2, col3 = st.columns([6, 2, 2])
                with col2:
                    if st.button(f"❤️ Favorit", key=f"fav_{idx}"):
                        if rec['judul'] not in [f['judul'] for f in st.session_state.favorites]:
                            st.session_state.favorites.append(rec)
                            st.success(f"✅ {rec['judul']} ditambahkan ke favorit!")
            
            # Reset Button
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.searched = False
                    st.rerun()
    else:
        # Welcome State with ML Info
        st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h2 style="color: #00d9ff; font-size: 2rem; margin-bottom: 1rem;">
                    🤖 Selamat Datang di Anime Recommender ML!
                </h2>
                <p style="color: #888; font-size: 1.1rem; line-height: 1.8; max-width: 700px; margin: 0 auto;">
                    Sistem ini menggunakan <b>Machine Learning</b> untuk memberikan rekomendasi anime 
                    yang lebih akurat dan personal. Pilih mode rekomendasi di sidebar dan klik tombol di atas!
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # ML Features Grid
        col1, col2, col3, col4 = st.columns(4)
        features_ml = [
            ("🎯", "Content-Based", "TF-IDF + Cosine Similarity untuk mencocokkan genre & sinopsis"),
            ("📊", "Cluster-Based", "K-Means Clustering untuk mengelompokkan anime serupa"),
            ("🔍", "Hybrid", "Kombinasi Content + Cluster untuk hasil optimal"),
            ("⚡", "Traditional", "Filter manual untuk kontrol penuh")
        ]
        
        for col, (emoji, title, desc) in zip([col1, col2, col3, col4], features_ml):
            with col:
                st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #1e1e3f, #252550); padding: 20px; border-radius: 15px; text-align: center; border: 1px solid rgba(0, 217, 255, 0.2); height: 100%;">
                        <div style="font-size: 2.5rem; margin-bottom: 10px;">{emoji}</div>
                        <h4 style="color: #00d9ff; margin: 0;">{title}</h4>
                        <p style="color: #888; font-size: 0.8rem; margin-top: 10px;">{desc}</p>
                    </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 2: ML ANALYTICS
# ============================================================
with tab2:
    st.markdown("""
        <h2 style="color: #00d9ff; margin-bottom: 1rem;">📊 Machine Learning Analytics</h2>
    """, unsafe_allow_html=True)
    
    # Cluster Overview
    st.markdown("""
        <div class="ml-info-box">
            <div class="ml-info-title">📊 K-Means Clustering Results</div>
        </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(5)
    cluster_colors = ['#667eea', '#f093fb', '#00d9ff', '#00ff88', '#ffd700']
    
    for i, col in enumerate(cols):
        with col:
            info = cluster_info[i]
            st.markdown(f"""
                <div style="background: linear-gradient(145deg, #1e1e3f, #252550); padding: 15px; border-radius: 15px; text-align: center; border-left: 4px solid {cluster_colors[i]};">
                    <h4 style="color: {cluster_colors[i]}; margin: 0;">Cluster {i}</h4>
                    <p style="color: white; font-size: 1.5rem; font-weight: bold; margin: 10px 0;">{info['count']}</p>
                    <p style="color: #888; font-size: 0.8rem;">Anime</p>
                    <p style="color: #f5576c; margin: 5px 0;">⭐ {info['avg_rating']:.1f}</p>
                    <p style="color: #888; font-size: 0.7rem;">{', '.join(info['top_genres'][:2])}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # PCA Visualization
    st.markdown("""
        <div class="ml-info-box" style="margin-top: 2rem;">
            <div class="ml-info-title">🎯 PCA Visualization (2D Projection)</div>
            <p style="color: #888;">Visualisasi hasil clustering dalam 2 dimensi menggunakan Principal Component Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    fig = create_pca_visualization(features, df_clustered)
    st.plotly_chart(fig, use_container_width=True)
    
    # Genre Distribution
    st.markdown("""
        <div class="ml-info-box" style="margin-top: 2rem;">
            <div class="ml-info-title">📈 Genre Distribution</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Count genres
    all_genres_count = {}
    for genres in df['genre']:
        for g in genres.split(','):
            g = g.strip()
            all_genres_count[g] = all_genres_count.get(g, 0) + 1
    
    top_genres = dict(sorted(all_genres_count.items(), key=lambda x: x[1], reverse=True)[:10])
    
    fig_genre = go.Figure(data=[
        go.Bar(
            x=list(top_genres.keys()),
            y=list(top_genres.values()),
            marker_color='rgba(0, 217, 255, 0.7)'
        )
    ])
    fig_genre.update_layout(
        title='Top 10 Genre Distribution',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,63,0.8)',
        font_color='white',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_genre, use_container_width=True)

# ============================================================
# TAB 3: DATABASE
# ============================================================
with tab3:
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
                    <p style="color: white; font-weight: bold; margin: 5px 0; font-size: 0.9rem;">{row['judul']}</p>
                    <p style="color: #f5576c; margin: 0;">⭐ {row['rating']}</p>
                </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 4: FAVORIT
# ============================================================
with tab4:
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
            <b>🎌 Anime Recommender ML Pro - Skripsi S1 Ilmu Komputer</b><br>
            Built with <span class="footer-heart">❤️</span> using Streamlit + Scikit-Learn + Python
        </p>
        <p style="color: #666; font-size: 0.8rem; margin-top: 10px;">
            🤖 Powered by Machine Learning | © 2024 | Data dari MyAnimeList
        </p>
    </div>
""", unsafe_allow_html=True)
