import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from collections import Counter

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CLEAN MINIMAL CSS
# ============================================================
st.markdown("""
<style>
    .main { background: #0f1419; color: #e0e0e0; }
    .block-container { padding: 2rem; max-width: 1400px; }
    
    .header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3d 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #2d3648;
    }
    .header h1 { color: #fff; font-size: 2.5rem; font-weight: 600; margin: 0; }
    .header p { color: #8892b0; font-size: 1.1rem; margin: 0.5rem 0 0 0; }
    
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #21262d;
    }
    
    .section-title {
        color: #58a6ff;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #21262d;
    }
    
    .anime-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .anime-card:hover { border-color: #30363d; background: #1c2128; }
    
    .anime-title {
        color: #f0f6fc;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .anime-meta {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    
    .meta-badge {
        background: #21262d;
        color: #8b949e;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        border: 1px solid #30363d;
    }
    .meta-badge.rating { background: #238636; color: #fff; border-color: #2ea043; }
    .meta-badge.year { background: #1f6feb; color: #fff; border-color: #388bfd; }
    .meta-badge.episodes { background: #8957e5; color: #fff; border-color: #a371f7; }
    .meta-badge.studio { background: #da3633; color: #fff; border-color: #f85149; }
    
    .genre-tags {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .genre-tag {
        background: #0d1117;
        color: #58a6ff;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        border: 1px solid #1f6feb;
    }
    
    .synopsis {
        color: #8b949e;
        line-height: 1.6;
        font-size: 0.95rem;
        margin-top: 1rem;
    }
    
    .stButton > button {
        background: #238636 !important;
        color: #fff !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        width: 100% !important;
    }
    .stButton > button:hover { background: #2ea043 !important; }
    
    button[kind="secondary"] {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        padding: 0.5rem;
        border-radius: 12px;
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #8b949e !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.25rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: #fff !important;
    }
    
    .cluster-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .cluster-number { color: #f0f6fc; font-size: 1.5rem; font-weight: 700; }
    .cluster-count { color: #8b949e; font-size: 0.875rem; margin: 0.5rem 0; }
    .cluster-rating { color: #3fb950; font-weight: 600; }
    
    .favorite-item {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .favorite-info h4 { color: #f0f6fc; margin: 0; font-size: 1rem; }
    .favorite-info p { color: #8b949e; margin: 0.25rem 0 0 0; font-size: 0.85rem; }
    
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #484f58;
        font-size: 0.875rem;
        border-top: 1px solid #21262d;
    }
    
    .similarity-score {
        background: linear-gradient(90deg, #1f6feb, #58a6ff);
        color: #fff;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .status-completed { background: #238636; color: #fff; }
    .status-ongoing { background: #1f6feb; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATASET - 30 ANIME
# ============================================================
@st.cache_data
def load_data():
    anime_list = [
        {
            'judul': 'Attack on Titan',
            'genre': 'Action, Dark Fantasy, Military',
            'rating': 9.0,
            'tahun': 2013,
            'episodes': 87,
            'studio': 'Wit Studio/MAPPA',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Humanity fights for survival against giant humanoid Titans in a post-apocalyptic world.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/10/47347.jpg'
        },
        {
            'judul': 'Death Note',
            'genre': 'Supernatural, Thriller, Psychological',
            'rating': 9.0,
            'tahun': 2006,
            'episodes': 37,
            'studio': 'Madhouse',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'A high school student discovers a supernatural notebook that grants him power over life and death.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/9/9453.jpg'
        },
        {
            'judul': 'One Piece',
            'genre': 'Action, Adventure, Comedy',
            'rating': 8.9,
            'tahun': 1999,
            'episodes': 1071,
            'studio': 'Toei Animation',
            'status': 'Ongoing',
            'type': 'TV',
            'sinopsis': 'Monkey D. Luffy and his pirate crew explore the Grand Line in search of the ultimate treasure.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/6/73245.jpg'
        },
        {
            'judul': 'Naruto',
            'genre': 'Action, Adventure, Martial Arts',
            'rating': 8.4,
            'tahun': 2002,
            'episodes': 220,
            'studio': 'Pierrot',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Young ninja seeks recognition and dreams of becoming Hokage of his village.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/13/17405.jpg'
        },
        {
            'judul': 'Demon Slayer',
            'genre': 'Action, Dark Fantasy, Historical',
            'rating': 8.7,
            'tahun': 2019,
            'episodes': 26,
            'studio': 'ufotable',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Tanjiro becomes a demon slayer to save his sister and avenge his family.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1286/99889.jpg'
        },
        {
            'judul': 'My Hero Academia',
            'genre': 'Action, Comedy, School, Superhero',
            'rating': 8.4,
            'tahun': 2016,
            'episodes': 113,
            'studio': 'Bones',
            'status': 'Ongoing',
            'type': 'TV',
            'sinopsis': 'In a world where people with superpowers are the norm, Izuku Midoriya dreams of becoming a hero.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/10/78745.jpg'
        },
        {
            'judul': 'Fullmetal Alchemist: Brotherhood',
            'genre': 'Action, Adventure, Fantasy, Military',
            'rating': 9.1,
            'tahun': 2009,
            'episodes': 64,
            'studio': 'Bones',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Two brothers search for Philosopher Stone to restore their bodies.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/5/47421.jpg'
        },
        {
            'judul': 'Spy x Family',
            'genre': 'Action, Comedy, Family, Slice of Life',
            'rating': 8.6,
            'tahun': 2022,
            'episodes': 25,
            'studio': 'Wit Studio/CloverWorks',
            'status': 'Ongoing',
            'type': 'TV',
            'sinopsis': 'A spy creates a fake family for his mission, unaware that his adopted daughter is a telepath.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1441/122795.jpg'
        },
        {
            'judul': 'Jujutsu Kaisen',
            'genre': 'Action, Dark Fantasy, School',
            'rating': 8.7,
            'tahun': 2020,
            'episodes': 24,
            'studio': 'MAPPA',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'A student fights curses in supernatural world with ancient techniques.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1171/109222.jpg'
        },
        {
            'judul': 'Tokyo Ghoul',
            'genre': 'Action, Dark Fantasy, Horror, Psychological',
            'rating': 8.0,
            'tahun': 2014,
            'episodes': 12,
            'studio': 'Pierrot',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'A college student becomes half-ghoul after an accident.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1498/107326.jpg'
        },
        {
            'judul': 'Your Name',
            'genre': 'Romance, Drama, Fantasy',
            'rating': 8.4,
            'tahun': 2016,
            'episodes': 1,
            'studio': 'CoMix Wave',
            'status': 'Completed',
            'type': 'Movie',
            'sinopsis': 'Two teenagers connected by fate across time and space.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/5/87048.jpg'
        },
        {
            'judul': 'A Silent Voice',
            'genre': 'Drama, Romance, School',
            'rating': 8.2,
            'tahun': 2016,
            'episodes': 1,
            'studio': 'Kyoto Animation',
            'status': 'Completed',
            'type': 'Movie',
            'sinopsis': 'A story about bullying, redemption, and second chances.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1122/96435.jpg'
        },
        {
            'judul': 'Horimiya',
            'genre': 'Romance, Comedy, School, Slice of Life',
            'rating': 8.2,
            'tahun': 2021,
            'episodes': 13,
            'studio': 'CloverWorks',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'High school romance between popular girl and quiet boy.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1695/111486.jpg'
        },
        {
            'judul': 'Toradora',
            'genre': 'Romance, Comedy, Drama, School',
            'rating': 8.1,
            'tahun': 2008,
            'episodes': 25,
            'studio': 'J.C.Staff',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Unlikely romance between two high school students.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/13/22128.jpg'
        },
        {
            'judul': 'Kaguya-sama: Love is War',
            'genre': 'Romance, Comedy, Psychological, School',
            'rating': 8.4,
            'tahun': 2019,
            'episodes': 24,
            'studio': 'A-1 Pictures',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Battle of wits between two student council members.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1295/106551.jpg'
        },
        {
            'judul': 'One Punch Man',
            'genre': 'Action, Comedy, Parody, Superhero',
            'rating': 8.5,
            'tahun': 2015,
            'episodes': 24,
            'studio': 'Madhouse',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'A hero who can defeat any enemy with one punch.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/12/76049.jpg'
        },
        {
            'judul': 'Mob Psycho 100',
            'genre': 'Action, Comedy, Supernatural',
            'rating': 8.4,
            'tahun': 2016,
            'episodes': 25,
            'studio': 'Bones',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'A boy with psychic powers tries to live a normal life.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/8/80356.jpg'
        },
        {
            'judul': 'Hunter x Hunter',
            'genre': 'Action, Adventure, Fantasy',
            'rating': 9.0,
            'tahun': 2011,
            'episodes': 148,
            'studio': 'Madhouse',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Young boy searches for his father and becomes a Hunter.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1337/99013.jpg'
        },
        {
            'judul': 'Steins;Gate',
            'genre': 'Sci-Fi, Thriller, Drama',
            'rating': 9.1,
            'tahun': 2011,
            'episodes': 24,
            'studio': 'White Fox',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Time travel thriller with devastating consequences.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/5/73199.jpg'
        },
        {
            'judul': 'Code Geass',
            'genre': 'Action, Drama, Mecha, Military',
            'rating': 8.7,
            'tahun': 2006,
            'episodes': 50,
            'studio': 'Sunrise',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'An exiled prince leads a rebellion using Geass power.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/5/50331.jpg'
        },
        {
            'judul': 'Cowboy Bebop',
            'genre': 'Action, Adventure, Sci-Fi',
            'rating': 8.8,
            'tahun': 1998,
            'episodes': 26,
            'studio': 'Sunrise',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Bounty hunters travel through space.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/4/19646.jpg'
        },
        {
            'judul': 'Violet Evergarden',
            'genre': 'Drama, Fantasy, Slice of Life',
            'rating': 8.7,
            'tahun': 2018,
            'episodes': 13,
            'studio': 'Kyoto Animation',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'An Auto Memory Doll writes letters connecting hearts.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1795/95088.jpg'
        },
        {
            'judul': 'Erased',
            'genre': 'Mystery, Supernatural, Thriller',
            'rating': 8.4,
            'tahun': 2016,
            'episodes': 12,
            'studio': 'A-1 Pictures',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'A man travels back in time to prevent a tragedy.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/10/77957.jpg'
        },
        {
            'judul': 'Re:Zero',
            'genre': 'Drama, Fantasy, Psychological, Thriller',
            'rating': 8.4,
            'tahun': 2016,
            'episodes': 50,
            'studio': 'White Fox',
            'status': 'Ongoing',
            'type': 'TV',
            'sinopsis': 'A boy wakes up in a parallel world and can return from death.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/11/79410.jpg'
        },
        {
            'judul': 'No Game No Life',
            'genre': 'Adventure, Comedy, Fantasy, Supernatural',
            'rating': 8.2,
            'tahun': 2014,
            'episodes': 12,
            'studio': 'Madhouse',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Gaming genius siblings in a world where games decide everything.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/5/65187.jpg'
        },
        {
            'judul': 'Sword Art Online',
            'genre': 'Action, Adventure, Fantasy, Romance',
            'rating': 7.6,
            'tahun': 2012,
            'episodes': 25,
            'studio': 'A-1 Pictures',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Players trapped in a VRMMORPG must clear the game to escape.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/11/39717.jpg'
        },
        {
            'judul': 'Tokyo Revengers',
            'genre': 'Action, Drama, Supernatural',
            'rating': 8.1,
            'tahun': 2021,
            'episodes': 24,
            'studio': 'LIDENFILMS',
            'status': 'Ongoing',
            'type': 'TV',
            'sinopsis': 'A delinquent travels back in time to save his girlfriend.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1179/119897.jpg'
        },
        {
            'judul': 'Chainsaw Man',
            'genre': 'Action, Comedy, Fantasy, Supernatural',
            'rating': 8.7,
            'tahun': 2022,
            'episodes': 12,
            'studio': 'MAPPA',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'A boy makes a contract with a chainsaw devil.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1802/108501.jpg'
        },
        {
            'judul': 'Haikyuu!!',
            'genre': 'Comedy, Drama, Sports, School',
            'rating': 8.7,
            'tahun': 2014,
            'episodes': 85,
            'studio': 'Production I.G',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'A high school volleyball team aims for nationals.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/7/76014.jpg'
        },
        {
            'judul': 'The Promised Neverland',
            'genre': 'Horror, Mystery, Psychological, Thriller',
            'rating': 8.5,
            'tahun': 2019,
            'episodes': 24,
            'studio': 'CloverWorks',
            'status': 'Completed',
            'type': 'TV',
            'sinopsis': 'Orphans discover a dark truth about their orphanage.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1125/96929.jpg'
        }
    ]
    
    return pd.DataFrame(anime_list)

# ============================================================
# ML FUNCTIONS
# ============================================================
@st.cache_data
def compute_features(df):
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
    numerical = scaler.fit_transform(df[['rating', 'tahun', 'episodes']])
    features = np.hstack([genre_df.values, numerical])
    return features

@st.cache_data
def get_similarity(df, features):
    tfidf = TfidfVectorizer(stop_words='english', max_features=50)
    text = tfidf.fit_transform(df['genre'] + ' ' + df['sinopsis'])
    from scipy.sparse import hstack, csr_matrix
    combined = hstack([text, csr_matrix(features[:, -3:])])
    return cosine_similarity(combined)

@st.cache_data
def do_clustering(df, features, n=5):
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    df_c = df.copy()
    df_c['cluster'] = clusters
    
    info = {}
    for i in range(n):
        cluster_data = df_c[df_c['cluster'] == i]
        all_genres = []
        for genres in cluster_data['genre']:
            all_genres.extend([g.strip() for g in genres.split(',')])
        top = Counter(all_genres).most_common(2)
        info[i] = {
            'count': len(cluster_data),
            'avg_rating': cluster_data['rating'].mean(),
            'top_genres': [g[0] for g in top]
        }
    return df_c, info

def content_recs(df, title, sim_matrix, n=5):
    try:
        idx = df[df['judul'] == title].index[0]
    except:
        return []
    scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    return [df.iloc[i].to_dict() | {'similarity': s} for i, s in scores]

def cluster_recs(df_c, title, n=5):
    try:
        c = df_c[df_c['judul'] == title]['cluster'].iloc[0]
    except:
        return []
    same = df_c[(df_c['cluster'] == c) & (df_c['judul'] != title)]
    return same.sort_values('rating', ascending=False).head(n).to_dict('records')

# ============================================================
# LOAD DATA
# ============================================================
df = load_data()
features = compute_features(df)
similarity_matrix = get_similarity(df, features)
df_clustered, cluster_info = do_clustering(df, features, n=5)

# ============================================================
# SESSION STATE
# ============================================================
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'results' not in st.session_state:
    st.session_state.results = None

# ============================================================
# UI
# ============================================================
st.markdown('<div class="header"><h1>🎌 Anime Recommender</h1><p>Temukan anime terbaik dengan rekomendasi cerdas</p></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<div class="section-title">⚙️ Konfigurasi</div>', unsafe_allow_html=True)
mode = st.sidebar.radio("Mode:", ["🤖 Content-Based", "📊 Cluster-Based", "🔍 Hybrid", "⚡ Manual Filter"])

if "ML" in mode or "Hybrid" in mode:
    st.sidebar.markdown('<div class="section-title">📺 Referensi</div>', unsafe_allow_html=True)
    ref = st.sidebar.selectbox("Pilih anime:", df['judul'].tolist())
    n_res = st.sidebar.slider("Jumlah:", 1, 10, 5)

if mode == "⚡ Manual Filter":
    st.sidebar.markdown('<div class="section-title">🎭 Filter</div>', unsafe_allow_html=True)
    genres_sel = st.sidebar.multiselect("Genre:", ['Action', 'Adventure', 'Comedy', 'Drama', 'Romance', 'Fantasy', 'Sci-Fi', 'Supernatural', 'Slice of Life', 'Horror', 'Mystery', 'Sports', 'Thriller'])
    min_r = st.sidebar.slider("Min Rating:", 0.0, 10.0, 7.0, 0.1)
    year_r = st.sidebar.slider("Tahun:", 1990, 2024, (2000, 2024))

st.sidebar.markdown('<div class="section-title">📈 Statistik</div>', unsafe_allow_html=True)
c1, c2 = st.sidebar.columns(2)
c1.metric("Total", len(df))
c2.metric("Avg", f"{df['rating'].mean():.1f}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Rekomendasi", "📊 Analytics", "📚 Database", "⭐ Favorit"])

# Tab 1
with tab1:
    if st.button("🔍 GENERATE REKOMENDASI", type="primary", use_container_width=True):
        if mode == "🤖 Content-Based":
            st.session_state.results = content_recs(df, ref, similarity_matrix, n_res)
            st.success(f"Content-Based: {ref}")
        elif mode == "📊 Cluster-Based":
            st.session_state.results = cluster_recs(df_clustered, ref, n_res)
            st.success(f"Cluster #{df_clustered[df_clustered['judul']==ref]['cluster'].iloc[0]}: {ref}")
        elif mode == "🔍 Hybrid":
            c = content_recs(df, ref, similarity_matrix, n_res//2+1)
            cl = cluster_recs(df_clustered, ref, n_res//2+1)
            seen = set()
            st.session_state.results = []
            for r in c + cl:
                if r['judul'] not in seen and r['judul'] != ref:
                    st.session_state.results.append(r)
                    seen.add(r['judul'])
                    if len(st.session_state.results) >= n_res:
                        break
            st.success(f"Hybrid: {ref}")
        else:
            f = df.copy()
            if genres_sel:
                f = f[f['genre'].apply(lambda x: any(g in x for g in genres_sel))]
            f = f[(f['rating']>=min_r) & (f['tahun']>=year_r[0]) & (f['tahun']<=year_r[1])]
            st.session_state.results = f.sort_values('rating', ascending=False).head(10).to_dict('records')
            for r in st.session_state.results:
                r['similarity'] = r['rating']/10
            st.success(f"Found {len(st.session_state.results)} anime")

    if st.session_state.results:
        st.markdown("---")
        for idx, a in enumerate(st.session_state.results):
            sim = int(a.get('similarity', 0.8)*100)
            col1, col2 = st.columns([1,3])
            with col1:
                st.image(a['image_url'], use_container_width=True)
            with col2:
                st.markdown(f"""
                <div class="anime-card">
                    <div class="anime-title">{a['judul']}</div>
                    <div class="anime-meta">
                        <span class="meta-badge rating">⭐ {a['rating']}</span>
                        <span class="meta-badge year">📅 {a['tahun']}</span>
                        <span class="meta-badge episodes">🎬 {a['episodes']} eps</span>
                        <span class="meta-badge studio">🏢 {a['studio']}</span>
                        <span class="meta-badge {'status-completed' if a['status']=='Completed' else 'status-ongoing'}">{a['status']}</span>
                        <span class="similarity-score">Match: {sim}%</span>
                    </div>
                    <div class="genre-tags">{''.join([f'<span class="genre-tag">{g.strip()}</span>' for g in a['genre'].split(',')])}</div>
                    <div class="synopsis">{a['sinopsis']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("❤️ Simpan", key=f"sv_{idx}"):
                    if a['judul'] not in [x['judul'] for x in st.session_state.favorites]:
                        st.session_state.favorites.append(a)
                        st.toast(f"Saved {a['judul']}!")

# Tab 2
with tab2:
    st.markdown('<h2 style="color:#f0f6fc;">📊 ML Analytics</h2>', unsafe_allow_html=True)
    cols = st.columns(5)
    colors = ['#1f6feb', '#238636', '#8957e5', '#da3633', '#d29922']
    for i, col in enumerate(cols):
        with col:
            info = cluster_info[i]
            st.markdown(f"""
            <div class="cluster-card" style="border-top:3px solid {colors[i]}">
                <div class="cluster-number" style="color:{colors[i]}">Cluster {i}</div>
                <div class="cluster-count">{info['count']} anime</div>
                <div class="cluster-rating">⭐ {info['avg_rating']:.1f}</div>
                <div style="color:#8b949e;font-size:0.8rem">{', '.join(info['top_genres'])}</div>
            </div>
            """, unsafe_allow_html=True)
    
    pca = PCA(n_components=2)
    pca_r = pca.fit_transform(features)
    fig = px.scatter(x=pca_r[:,0], y=pca_r[:,1], color=df_clustered['cluster'], 
                     hover_name=df_clustered['judul'], color_continuous_scale='viridis')
    fig.update_layout(paper_bgcolor='#0f1419', plot_bgcolor='#161b22', font_color='#e0e0e0')
    st.plotly_chart(fig, use_container_width=True)

# Tab 3 - FIXED HERE
with tab3:
    st.markdown('<h2 style="color:#f0f6fc;">📚 Database</h2>', unsafe_allow_html=True)
    search = st.text_input("🔍 Cari anime...")
    disp = df[df['judul'].str.contains(search, case=False)] if search else df
    
    # FIXED: NumberColumn not Number_column
    st.dataframe(
        disp[['judul', 'genre', 'rating', 'tahun', 'episodes', 'studio', 'status', 'type']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "judul": st.column_config.TextColumn("Judul", width="large"),
            "genre": st.column_config.TextColumn("Genre"),
            "rating": st.column_config.ProgressColumn("Rating", min_value=0, max_value=10, format="⭐ %.1f"),
            "tahun": st.column_config.NumberColumn("Tahun"),
            "episodes": st.column_config.NumberColumn("Eps"),  # FIXED!
            "studio": st.column_config.TextColumn("Studio"),
            "status": st.column_config.TextColumn("Status"),
            "type": st.column_config.TextColumn("Tipe")
        }
    )
    
    st.markdown('<h3 style="color:#f0f6fc;margin-top:2rem">🏆 Top Rated</h3>', unsafe_allow_html=True)
    top5 = df.nlargest(5, 'rating')
    cols = st.columns(5)
    for idx, (_, row) in enumerate(top5.iterrows()):
        with cols[idx]:
            st.image(row['image_url'], use_container_width=True)
            st.markdown(f"<div style='text-align:center'><div style='font-weight:600;color:#f0f6fc'>{row['judul']}</div><div style='color:#3fb950'>⭐ {row['rating']}</div></div>", unsafe_allow_html=True)

# Tab 4
with tab4:
    st.markdown('<h2 style="color:#f0f6fc;">⭐ Favorit</h2>', unsafe_allow_html=True)
    if not st.session_state.favorites:
        st.info("Belum ada favorit.")
    else:
        for f in st.session_state.favorites:
            st.markdown(f"""
            <div class="favorite-item">
                <div class="favorite-info">
                    <h4>{f['judul']}</h4>
                    <p>⭐ {f['rating']} | 📅 {f['tahun']} | {f['genre']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        if st.button("🗑️ Hapus Semua", type="secondary"):
            st.session_state.favorites = []
            st.rerun()

st.markdown('<div class="footer">🎌 Anime Recommender ML • 2024</div>', unsafe_allow_html=True)
