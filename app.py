import streamlit as st
import pandas as pd
import numpy as np
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
    
    .anime-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(240, 147, 251, 0.2);
        border-color: rgba(240, 147, 251, 0.5);
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
    
    .rating-badge {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .similarity-badge {
        background: linear-gradient(135deg, #00d9ff, #00ff88);
        color: #000;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(145deg, #1e1e3f, #252550);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: float 6s ease-in-out infinite;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* ===== ML INFO BOX ===== */
    .ml-info-box {
        background: linear-gradient(145deg, #1e1e3f, #252550);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid rgba(0, 217, 255, 0.3);
    }
    
    .ml-info-title {
        color: #00d9ff;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
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
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
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
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #f093fb);
        border-radius: 10px;
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
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #f093fb, transparent);
        margin: 2rem 0;
    }
    
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# DATASET ANIME - 22 ANIME (FIXED LENGTH)
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

# ============================================================
# MACHINE LEARNING FUNCTIONS
# ============================================================

@st.cache_data
def prepare_ml_features(df):
    """Prepare features for machine learning"""
    all_genres = set()
    for genres in df['genre']:
        for g in genres.split(','):
            all_genres.add(g.strip())
    all_genres = sorted(list(all_genres))
    
    genre_matrix = []
    for genres in df['genre']:
        genre_list = [g.strip() for g in genres.split(',')]
        genre_row = [1 if g in genre_list else 0 for g in all_genres]
        genre_matrix.append(genre_row)
    
    genre_df = pd.DataFrame(genre_matrix, columns=all_genres)
    
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(df[['rating', 'tahun', 'episodes']])
    
    features = np.hstack([genre_df.values, numerical_features])
    
    return features, all_genres, scaler

def compute_similarity_matrix(df, features):
    """Compute cosine similarity matrix"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    text_features = tfidf.fit_transform(df['genre'] + ' ' + df['sinopsis'])
    
    from scipy.sparse import hstack, csr_matrix
    numerical_sparse = csr_matrix(features[:, -3:])
    combined_features = hstack([text_features, numerical_sparse])
    
    similarity_matrix = cosine_similarity(combined_features)
    
    return similarity_matrix

def get_similar_animes(df, anime_title, similarity_matrix, n=5):
    """Get similar animes"""
    try:
        idx = df[df['judul'] == anime_title].index[0]
    except IndexError:
        return []
    
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    
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
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    cluster_info = {}
    for i in range(n_clusters):
        cluster_animes = df_clustered[df_clustered['cluster'] == i]
        avg_rating = cluster_animes['rating'].mean()
        
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
    """Create PCA visualization"""
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
        title_font_color='#f093fb'
    )
    
    return fig

# ============================================================
# DATA LOADING
# ============================================================
try:
    df = load_data()
    
    # Prepare ML features
    features, all_genres, scaler = prepare_ml_features(df)
    similarity_matrix = compute_similarity_matrix(df, features)
    df_clustered, cluster_info, kmeans_model = cluster_animes(df, features, n_clusters=5)
    
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    data_loaded = False
    df = None

# ============================================================
# HEADER SECTION
# ============================================================
if data_loaded:
    st.markdown("""
        <div class="main-header">
            <h1 class="main-title">🎌 Anime Recommender <span class="ml-badge">🤖 ML POWERED</span></h1>
            <p class="subtitle">✨ Sistem Rekomendasi Cerdas dengan Machine Learning ✨</p>
        </div>
    """, unsafe_allow_html=True)

    # ML Info Box
    st.markdown("""
        <div class="ml-info-box">
            <div class="ml-info-title">🧠 Teknologi Machine Learning:</div>
            <div style="color: #a0a0a0; line-height: 1.8;">
                • <b>TF-IDF + Cosine Similarity</b> - Content-Based Filtering berdasarkan genre & sinopsis<br>
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
            <h2 style="color: #f093fb; font-size: 1.5rem;">⚙️ Preferensi</h2>
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
                  'Horror', 'Mystery']
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
            "Pilih anime:",
            options=df['judul'].tolist(),
            label_visibility="collapsed"
        )
        
        n_recommendations = st.sidebar.slider(
            "Jumlah Rekomendasi:",
            min_value=1,
            max_value=10,
            value=5
        )

    # Quick Stats
    total_anime = len(df)
    avg_rating = df['rating'].mean()

    st.sidebar.markdown("""
        <hr style="border-color: rgba(240, 147, 251, 0.3); margin: 2rem 0;">
        <div style="background: linear-gradient(145deg, #1e1e3f, #252550); padding: 15px; border-radius: 15px; text-align: center;">
            <h3 style="color: #f093fb; margin-bottom: 10px;">📈 Stats</h3>
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
                <h2 style="color: #00d9ff;">🤖 Mode: {ml_mode}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 GENERATE REKOMENDASI", use_container_width=True):
                st.session_state.searched = True
                st.balloons()
        
        if st.session_state.searched:
            st.markdown("<hr>", unsafe_allow_html=True)
            
            recommendations = []
            
            # Generate recommendations
            if ml_mode == "🎯 Content-Based ML" and 'reference_anime' in locals():
                recommendations = get_similar_animes(df, reference_anime, similarity_matrix, n_recommendations)
                st.success(f"🎯 Content-Based ML untuk '{reference_anime}'")
                
            elif ml_mode == "📊 Cluster-Based ML" and 'reference_anime' in locals():
                recommendations = get_cluster_recommendations(df_clustered, cluster_info, reference_anime, n_recommendations)
                anime_cluster = df_clustered[df_clustered['judul'] == reference_anime]['cluster'].iloc[0]
                st.success(f"📊 Cluster #{anime_cluster} ML")
                
            elif ml_mode == "🔍 Hybrid ML" and 'reference_anime' in locals():
                content_recs = get_similar_animes(df, reference_anime, similarity_matrix, n_recommendations//2 + 1)
                cluster_recs = get_cluster_recommendations(df_clustered, cluster_info, reference_anime, n_recommendations//2 + 1)
                
                seen_titles = set()
                recommendations = []
                for rec in content_recs + cluster_recs:
                    if rec['judul'] not in seen_titles and rec['judul'] != reference_anime:
                        recommendations.append(rec)
                        seen_titles.add(rec['judul'])
                        if len(recommendations) >= n_recommendations:
                            break
                st.success(f"🔍 Hybrid ML untuk '{reference_anime}'")
                
            else:
                # Traditional Filter
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
                    rec['similarity'] = rec['rating'] / 10
                st.success("⚡ Traditional Filter")
            
            # Display Results
            st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <h2 style="color: #f093fb; font-size: 1.8rem;">
                        📺 <span style="color: #f5576c;">{len(recommendations)}</span> Rekomendasi!
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            
            if len(recommendations) == 0:
                st.warning("😅 Tidak ada rekomendasi. Coba ubah preferensi!")
            else:
                for idx, rec in enumerate(recommendations):
                    similarity_pct = int(rec.get('similarity', 0.7) * 100)
                    
                    st.markdown(f"""
                        <div class="anime-card">
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
                                    <div style="display: flex; gap: 10px; margin: 10px 0;">
                                        <span class="rating-badge">⭐ {rec['rating']}</span>
                                        <span class="similarity-badge">🎯 Match: {similarity_pct}%</span>
                                    </div>
                                    <div style="margin: 10px 0;">
                                        <span style="color: #667eea;">📅 {rec['tahun']}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([6, 2, 2])
                    with col2:
                        if st.button(f"❤️ Favorit", key=f"fav_{idx}"):
                            if rec['judul'] not in [f['judul'] for f in st.session_state.favorites]:
                                st.session_state.favorites.append(rec)
                                st.success(f"✅ {rec['judul']} ditambahkan!")
                
                # Reset
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔄 Reset", use_container_width=True):
                        st.session_state.searched = False
                        st.rerun()
        else:
            # Welcome
            st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <h2 style="color: #00d9ff; font-size: 2rem; margin-bottom: 1rem;">
                        🤖 Selamat Datang!
                    </h2>
                    <p style="color: #888; font-size: 1.1rem; max-width: 700px; margin: 0 auto;">
                        Pilih mode ML di sidebar dan klik tombol Generate!
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # ============================================================
    # TAB 2: ML ANALYTICS
    # ============================================================
    with tab2:
        st.markdown("""
            <h2 style="color: #00d9ff; margin-bottom: 1rem;">📊 ML Analytics</h2>
        """, unsafe_allow_html=True)
        
        # Cluster Overview
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
                    </div>
                """, unsafe_allow_html=True)
        
        # PCA Visualization
        st.markdown("<br>", unsafe_allow_html=True)
        fig = create_pca_visualization(features, df_clustered)
        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # TAB 3: DATABASE
    # ============================================================
    with tab3:
        st.markdown("""
            <h2 style="color: #f093fb; margin-bottom: 1rem;">📚 Database</h2>
        """, unsafe_allow_html=True)
        
        search_query = st.text_input("🔍 Cari anime:", placeholder="Nama anime...")
        
        if search_query:
            filtered = df[df['judul'].str.contains(search_query, case=False, na=False)]
        else:
            filtered = df
        
        st.dataframe(
            filtered[['judul', 'genre', 'rating', 'tahun', 'type', 'episodes', 'studio', 'status']],
            use_container_width=True,
            hide_index=True
        )

    # ============================================================
    # TAB 4: FAVORIT
    # ============================================================
    with tab4:
        st.markdown("""
            <h2 style="color: #f093fb; margin-bottom: 1rem;">⭐ Favorit</h2>
        """, unsafe_allow_html=True)
        
        if len(st.session_state.favorites) == 0:
            st.info("📝 Belum ada favorit. Tambahkan dari tab pencarian!")
        else:
            for fav in st.session_state.favorites:
                st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #1e1e3f, #252550); padding: 15px; border-radius: 15px; margin: 10px 0; border: 1px solid rgba(240, 147, 251, 0.3);">
                        <h4 style="color: white; margin: 0;">{fav['judul']}</h4>
                        <p style="color: #888; margin: 5px 0; font-size: 0.9rem;">{fav['genre']} | ⭐ {fav['rating']} | 📅 {fav['tahun']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("🗑️ Hapus Semua"):
                st.session_state.favorites = []
                st.rerun()

    # ============================================================
    # FOOTER
    # ============================================================
    st.markdown("""
        <div class="footer">
            <p class="footer-text">
                <b>🎌 Anime Recommender ML Pro</b><br>
                Built with <span class="footer-heart">❤️</span> using Streamlit + Scikit-Learn
            </p>
            <p style="color: #666; font-size: 0.8rem; margin-top: 10px;">
                🤖 Powered by Machine Learning | © 2024
            </p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Failed to load data. Please check the data structure.")
