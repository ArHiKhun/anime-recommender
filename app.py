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
    /* Clean Base */
    .main {
        background: #0f1419;
        color: #e0e0e0;
    }
    
    .block-container {
        padding: 2rem;
        max-width: 1400px;
    }
    
    /* Clean Header */
    .header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3d 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #2d3648;
    }
    
    .header h1 {
        color: #fff;
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header p {
        color: #8892b0;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Sidebar Clean */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #21262d;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem;
    }
    
    /* Section Titles */
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
    
    /* Clean Cards */
    .anime-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .anime-card:hover {
        border-color: #30363d;
        background: #1c2128;
    }
    
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
    
    .meta-badge.rating {
        background: #238636;
        color: #fff;
        border-color: #2ea043;
    }
    
    .meta-badge.year {
        background: #1f6feb;
        color: #fff;
        border-color: #388bfd;
    }
    
    .meta-badge.episodes {
        background: #8957e5;
        color: #fff;
        border-color: #a371f7;
    }
    
    .meta-badge.studio {
        background: #da3633;
        color: #fff;
        border-color: #f85149;
    }
    
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
    
    /* Buttons */
    .stButton > button {
        background: #238636 !important;
        color: #fff !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: #2ea043 !important;
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary Button */
    button[kind="secondary"] {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }
    
    button[kind="secondary"]:hover {
        background: #30363d !important;
    }
    
    /* Input Fields */
    .stSelectbox > div > div, .stSlider > div > div {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    
    /* Tabs */
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
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: #fff !important;
    }
    
    /* DataFrame */
    .stDataFrame {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #21262d;
        margin: 2rem 0;
    }
    
    /* Stats Cards */
    .stat-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .stat-value {
        color: #58a6ff;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stat-label {
        color: #8b949e;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    /* Info Box */
    .info-box {
        background: #161b22;
        border-left: 4px solid #1f6feb;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #58a6ff;
        margin: 0 0 0.5rem 0;
    }
    
    .info-box p {
        color: #8b949e;
        margin: 0;
        font-size: 0.9rem;
    }
    
    /* Cluster Cards */
    .cluster-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .cluster-card:hover {
        border-color: #30363d;
    }
    
    .cluster-number {
        color: #f0f6fc;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .cluster-count {
        color: #8b949e;
        font-size: 0.875rem;
        margin: 0.5rem 0;
    }
    
    .cluster-rating {
        color: #3fb950;
        font-weight: 600;
    }
    
    /* Favorites */
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
    
    .favorite-info h4 {
        color: #f0f6fc;
        margin: 0;
        font-size: 1rem;
    }
    
    .favorite-info p {
        color: #8b949e;
        margin: 0.25rem 0 0 0;
        font-size: 0.85rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #484f58;
        font-size: 0.875rem;
        border-top: 1px solid #21262d;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    
    /* Image Container */
    .anime-image {
        width: 100%;
        max-width: 200px;
        height: 280px;
        object-fit: cover;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Similarity Score */
    .similarity-score {
        background: linear-gradient(90deg, #1f6feb, #58a6ff);
        color: #fff;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    /* Status Badge */
    .status-completed {
        background: #238636;
        color: #fff;
    }
    
    .status-ongoing {
        background: #1f6feb;
        color: #fff;
    }
    
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATASET - 30 ANIME (BALANCED & VERIFIED)
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
            'sinopsis': 'Humanity fights for survival against giant humanoid Titans in a post-apocalyptic world. Eren Yeoman vows to exterminate all Titans after they destroy his hometown.',
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
            'sinopsis': 'A high school student discovers a supernatural notebook that grants him power over life and death. He begins a crusade to rid the world of criminals.',
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
            'sinopsis': 'Monkey D. Luffy and his pirate crew explore the Grand Line in search of the ultimate treasure known as One Piece to become the next Pirate King.',
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
            'sinopsis': 'Young ninja seeks recognition and dreams of becoming Hokage of his village. He faces challenges and makes friends along the way.',
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
            'sinopsis': 'Tanjiro becomes a demon slayer to save his sister and avenge his family. He joins the Demon Slayer Corps to fight demons.',
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
            'sinopsis': 'In a world where people with superpowers are the norm, Izuku Midoriya dreams of becoming a hero despite being born without powers.',
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
            'sinopsis': 'Two brothers search for Philosopher Stone to restore their bodies after a failed alchemical ritual. Their journey reveals dark truths.',
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
            'sinopsis': 'A spy creates a fake family for his mission, unaware that his adopted daughter is a telepath and his wife is an assassin.',
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
            'sinopsis': 'A student fights curses in supernatural world with ancient techniques. He joins a secret organization to protect people from curses.',
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
            'sinopsis': 'A college student becomes half-ghoul after an accident and must survive in a world where ghouls hunt humans for food.',
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
            'sinopsis': 'Two teenagers connected by fate across time and space switch bodies and form a bond that transcends distance and time.',
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
            'sinopsis': 'A story about bullying, redemption, and second chances. A former bully seeks forgiveness from a deaf girl he tormented in elementary school.',
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
            'sinopsis': 'High school romance between popular girl and quiet boy who discover each other secret lives outside of school.',
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
            'sinopsis': 'Unlikely romance between two high school students with opposite personalities who help each other with their crushes.',
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
            'sinopsis': 'Battle of wits between two student council members who are too proud to confess their love for each other.',
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
            'sinopsis': 'A hero who can defeat any enemy with one punch seeks a worthy challenge. He grows bored of his overwhelming strength.',
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
            'sinopsis': 'A boy with powerful psychic powers tries to live a normal life under the mentorship of a con artist who claims to be a psychic.',
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
            'sinopsis': 'Young boy searches for his father and faces incredible challenges as he becomes a Hunter, an elite adventurer.',
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
            'sinopsis': 'Time travel thriller with devastating consequences. A scientist accidentally discovers a way to send messages to the past.',
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
            'sinopsis': 'An exiled prince leads a rebellion against a corrupt empire using the supernatural power of Geass to control people minds.',
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
            'sinopsis': 'Bounty hunters travel through space in a jazz-infused noir adventure, each running from their past while chasing criminals.',
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
            'sinopsis': 'An Auto Memory Doll writes letters that connect people hearts as she searches for the meaning behind her beloved major last words.',
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
            'sinopsis': 'A man travels back in time to prevent a tragedy from his childhood and catch a serial killer who targets children.',
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
            'sinopsis': 'A boy wakes up in a parallel world and discovers he can return from death. He tries to save his friends from a tragic fate.',
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
            'sinopsis': 'Gaming genius siblings are transported to a world where games decide everything. They aim to defeat the god of games.',
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
            'sinopsis': 'Players trapped in a VRMMORPG must clear the game to escape. Death in the game means death in real life.',
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
            'sinopsis': 'A delinquent travels back in time to save his girlfriend from being killed by a ruthless gang. He infiltrates the gang to change the future.',
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
            'sinopsis': 'A boy makes a contract with a chainsaw devil and becomes a devil hunter. He joins a special squad to hunt devils.',
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
            'sinopsis': 'A high school volleyball team aims for nationals. A short but determined player joins the team to become a great volleyball player.',
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
            'sinopsis': 'Orphans discover a dark truth about their orphanage and plan their escape. They must outsmart their caretaker to survive.',
            'image_url': 'https://cdn.myanimelist.net/images/anime/1125/96929.jpg'
        }
    ]
    
    return pd.DataFrame(anime_list)

# ============================================================
# ML FUNCTIONS
# ============================================================
@st.cache_data
def compute_ml_features(df):
    """Prepare features for ML"""
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
    return features, all_genres

@st.cache_data
def get_similarity_matrix(df, features):
    """Compute similarity"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=50)
    text_features = tfidf.fit_transform(df['genre'] + ' ' + df['sinopsis'])
    
    from scipy.sparse import hstack, csr_matrix
    numerical_sparse = csr_matrix(features[:, -3:])
    combined = hstack([text_features, numerical_sparse])
    
    return cosine_similarity(combined)

@st.cache_data
def cluster_data(df, features, n=5):
    """K-Means clustering"""
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    info = {}
    for i in range(n):
        cluster_data = df_clustered[df_clustered['cluster'] == i]
        avg_rating = cluster_data['rating'].mean()
        
        all_genres = []
        for genres in cluster_data['genre']:
            all_genres.extend([g.strip() for g in genres.split(',')])
        top_genres = Counter(all_genres).most_common(2)
        
        info[i] = {
            'count': len(cluster_data),
            'avg_rating': avg_rating,
            'top_genres': [g[0] for g in top_genres],
            'animes': cluster_data['judul'].tolist()
        }
    
    return df_clustered, info

def get_content_recommendations(df, title, sim_matrix, n=5):
    """Content-based recommendations"""
    try:
        idx = df[df['judul'] == title].index[0]
    except:
        return []
    
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    results = []
    for i, score in scores:
        anime = df.iloc[i]
        results.append({
            'judul': anime['judul'],
            'genre': anime['genre'],
            'rating': anime['rating'],
            'tahun': anime['tahun'],
            'episodes': anime['episodes'],
            'studio': anime['studio'],
            'status': anime['status'],
            'type': anime['type'],
            'sinopsis': anime['sinopsis'],
            'image_url': anime['image_url'],
            'similarity': score
        })
    return results

def get_cluster_recommendations(df_clustered, title, n=5):
    """Cluster-based recommendations"""
    try:
        cluster = df_clustered[df_clustered['judul'] == title]['cluster'].iloc[0]
    except:
        return []
    
    same_cluster = df_clustered[
        (df_clustered['cluster'] == cluster) & 
        (df_clustered['judul'] != title)
    ].sort_values('rating', ascending=False).head(n)
    
    return same_cluster.to_dict('records')

# ============================================================
# LOAD DATA
# ============================================================
df = load_data()
features, all_genres = compute_ml_features(df)
similarity_matrix = get_similarity_matrix(df, features)
df_clustered, cluster_info = cluster_data(df, features, n=5)

# ============================================================
# SESSION STATE
# ============================================================
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# ============================================================
# HEADER
# ============================================================
st.markdown("""
    <div class="header">
        <h1>🎌 Anime Recommender</h1>
        <p>Temukan anime terbaik dengan rekomendasi cerdas</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown('<div class="section-title">⚙️ Konfigurasi</div>', unsafe_allow_html=True)

# Mode
mode = st.sidebar.radio(
    "Mode Rekomendasi:",
    ["🤖 Content-Based (ML)", "📊 Cluster-Based (ML)", "🔍 Hybrid (ML)", "⚡ Manual Filter"],
    index=0
)

# Reference Anime for ML
if "ML" in mode:
    st.sidebar.markdown('<div class="section-title" style="margin-top: 1.5rem;">📺 Anime Referensi</div>', unsafe_allow_html=True)
    reference = st.sidebar.selectbox(
        "Pilih anime favorit:",
        df['judul'].tolist(),
        index=0
    )
    n_results = st.sidebar.slider("Jumlah rekomendasi:", 1, 10, 5)

# Manual Filters
if mode == "⚡ Manual Filter":
    st.sidebar.markdown('<div class="section-title" style="margin-top: 1.5rem;">🎭 Filter</div>', unsafe_allow_html=True)
    
    selected_genres = st.sidebar.multiselect(
        "Genre:",
        ['Action', 'Adventure', 'Comedy', 'Drama', 'Romance', 'Fantasy', 'Sci-Fi', 
         'Supernatural', 'Slice of Life', 'Horror', 'Mystery', 'Sports', 'Thriller'],
        default=[]
    )
    
    min_rating = st.sidebar.slider("Minimal Rating:", 0.0, 10.0, 7.0, 0.1)
    year_range = st.sidebar.slider("Tahun:", 1990, 2024, (2000, 2024))

# Stats
st.sidebar.markdown('<div class="section-title" style="margin-top: 2rem;">📈 Statistik</div>', unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)
col1.metric("Total", len(df))
col2.metric("Rating Avg", f"{df['rating'].mean():.1f}")

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Rekomendasi", "📊 Analytics", "📚 Database", "⭐ Favorit"])

# ============================================================
# TAB 1: RECOMMENDATIONS
# ============================================================
with tab1:
    # Generate Button
    if st.button("🔍 GENERATE REKOMENDASI", type="primary", use_container_width=True):
        st.session_state.search_results = None
        
        if mode == "🤖 Content-Based (ML)":
            st.session_state.search_results = get_content_recommendations(df, reference, similarity_matrix, n_results)
            st.success(f"Rekomendasi berbasis konten untuk: **{reference}**")
            
        elif mode == "📊 Cluster-Based (ML)":
            st.session_state.search_results = get_cluster_recommendations(df_clustered, reference, n_results)
            cluster_num = df_clustered[df_clustered['judul'] == reference]['cluster'].iloc[0]
            st.success(f"Rekomendasi dari Cluster #{cluster_num}: **{reference}**")
            
        elif mode == "🔍 Hybrid (ML)":
            content = get_content_recommendations(df, reference, similarity_matrix, n_results//2 + 1)
            cluster = get_cluster_recommendations(df_clustered, reference, n_results//2 + 1)
            
            seen = set()
            combined = []
            for rec in content + cluster:
                if rec['judul'] not in seen and rec['judul'] != reference:
                    combined.append(rec)
                    seen.add(rec['judul'])
                    if len(combined) >= n_results:
                        break
            
            st.session_state.search_results = combined
            st.success(f"Rekomendasi Hybrid untuk: **{reference}**")
            
        else:  # Manual Filter
            filtered = df.copy()
            if selected_genres:
                filtered = filtered[filtered['genre'].apply(
                    lambda x: any(g in x for g in selected_genres)
                )]
            filtered = filtered[
                (filtered['rating'] >= min_rating) &
                (filtered['tahun'] >= year_range[0]) &
                (filtered['tahun'] <= year_range[1])
            ]
            
            results = filtered.sort_values('rating', ascending=False).head(10).to_dict('records')
            for r in results:
                r['similarity'] = r['rating'] / 10
            
            st.session_state.search_results = results
            st.success(f"Ditemukan {len(results)} anime")
    
    # Display Results
    if st.session_state.search_results:
        st.markdown("---")
        
        for idx, anime in enumerate(st.session_state.search_results):
            similarity = int(anime.get('similarity', 0.8) * 100)
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.image(anime['image_url'], use_container_width=True)
            
            with col2:
                st.markdown(f"""
                    <div class="anime-card">
                        <div class="anime-title">{anime['judul']}</div>
                        <div class="anime-meta">
                            <span class="meta-badge rating">⭐ {anime['rating']}</span>
                            <span class="meta-badge year">📅 {anime['tahun']}</span>
                            <span class="meta-badge episodes">🎬 {anime['episodes']} eps</span>
                            <span class="meta-badge studio">🏢 {anime['studio']}</span>
                            <span class="meta-badge {'status-completed' if anime['status'] == 'Completed' else 'status-ongoing'}">{anime['status']}</span>
                            <span class="similarity-score">Match: {similarity}%</span>
                        </div>
                        <div class="genre-tags">
                            {''.join([f'<span class="genre-tag">{g.strip()}</span>' for g in anime['genre'].split(',')])}
                        </div>
                        <div class="synopsis">{anime['sinopsis']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([8, 2, 2])
                with c2:
                    if st.button("❤️ Simpan", key=f"save_{idx}"):
                        if anime['judul'] not in [f['judul'] for f in st.session_state.favorites]:
                            st.session_state.favorites.append(anime)
                            st.toast(f"✅ {anime['judul']} disimpan!")
            
            st.markdown("<br>", unsafe_allow_html=True)
    
    else:
        # Welcome
        st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #8b949e;">
                <h2 style="color: #f0f6fc; margin-bottom: 1rem;">Selamat Datang</h2>
                <p>Pilih mode rekomendasi di sidebar, lalu klik tombol "Generate Rekomendasi"</p>
                <br>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; max-width: 800px; margin: 0 auto;">
                    <div style="background: #161b22; padding: 1.5rem; border-radius: 12px; border: 1px solid #21262d;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                        <div style="font-weight: 600; color: #f0f6fc;">ML Power</div>
                        <div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.5rem;">AI-Powered recommendations</div>
                    </div>
                    <div style="background: #161b22; padding: 1.5rem; border-radius: 12px; border: 1px solid #21262d;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                        <div style="font-weight: 600; color: #f0f6fc;">Clustering</div>
                        <div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.5rem;">Smart grouping</div>
                    </div>
                    <div style="background: #161b22; padding: 1.5rem; border-radius: 12px; border: 1px solid #21262d;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                        <div style="font-weight: 600; color: #f0f6fc;">Accurate</div>
                        <div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.5rem;">High precision match</div>
                    </div>
                    <div style="background: #161b22; padding: 1.5rem; border-radius: 12px; border: 1px solid #21262d;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                        <div style="font-weight: 600; color: #f0f6fc;">Fast</div>
                        <div style="font-size: 0.85rem; color: #8b949e; margin-top: 0.5rem;">Instant results</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2: ANALYTICS
# ============================================================
with tab2:
    st.markdown('<h2 style="color: #f0f6fc; margin-bottom: 1.5rem;">📊 Machine Learning Analytics</h2>', unsafe_allow_html=True)
    
    # Info Box
    st.markdown("""
        <div class="info-box">
            <h4>Sistem ML yang Digunakan</h4>
            <p>• TF-IDF + Cosine Similarity untuk Content-Based Filtering<br>
            • K-Means Clustering untuk mengelompokkan anime berdasarkan karakteristik<br>
            • Hybrid approach menggabungkan kedua metode untuk hasil optimal</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Cluster Overview
    st.markdown('<h3 style="color: #f0f6fc; margin: 2rem 0 1rem 0;">K-Means Clusters</h3>', unsafe_allow_html=True)
    
    cols = st.columns(5)
    colors = ['#1f6feb', '#238636', '#8957e5', '#da3633', '#d29922']
    
    for i, col in enumerate(cols):
        info = cluster_info[i]
        with col:
            st.markdown(f"""
                <div class="cluster-card" style="border-top: 3px solid {colors[i]};">
                    <div class="cluster-number" style="color: {colors[i]};">Cluster {i}</div>
                    <div class="cluster-count">{info['count']} anime</div>
                    <div class="cluster-rating">⭐ {info['avg_rating']:.1f} avg</div>
                    <div style="color: #8b949e; font-size: 0.8rem; margin-top: 0.5rem;">
                        {', '.join(info['top_genres'])}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # PCA Visualization
    st.markdown('<h3 style="color: #f0f6fc; margin: 2rem 0 1rem 0;">Visualisasi PCA</h3>', unsafe_allow_html=True)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    fig = px.scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        color=df_clustered['cluster'],
        hover_name=df_clustered['judul'],
        color_continuous_scale='viridis',
        title='Distribusi Anime dalam 2D Space'
    )
    fig.update_layout(
        paper_bgcolor='#0f1419',
        plot_bgcolor='#161b22',
        font_color='#e0e0e0',
        title_font_color='#f0f6fc',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3: DATABASE
# ============================================================
with tab3:
    st.markdown('<h2 style="color: #f0f6fc; margin-bottom: 1.5rem;">📚 Database Anime</h2>', unsafe_allow_html=True)
    
    # Search
    search = st.text_input("🔍 Cari anime...", placeholder="Ketik nama anime...")
    
    if search:
        display_df = df[df['judul'].str.contains(search, case=False, na=False)]
    else:
        display_df = df
    
    # Display as styled dataframe
    st.dataframe(
        display_df[['judul', 'genre', 'rating', 'tahun', 'episodes', 'studio', 'status', 'type']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "judul": st.column_config.TextColumn("Judul", width="large"),
            "genre": st.column_config.TextColumn("Genre"),
            "rating": st.column_config.ProgressColumn("Rating", min_value=0, max_value=10, format="⭐ %.1f"),
            "tahun": st.column_config.NumberColumn("Tahun"),
            "episodes": st.column_config.NumberColumn("Eps"),
            "studio": st.column_config.TextColumn("Studio"),
            "status": st.column_config.TextColumn("Status"),
            "type": st.column_config.TextColumn("Tipe")
        }
    )
    
    # Top Rated
    st.markdown('<h3 style="color: #f0f6fc; margin-top: 2rem;">🏆 Top Rated</h3>', unsafe_allow_html=True)
    top5 = df.nlargest(5, 'rating')
    
    cols = st.columns(5)
    for idx, (_, row) in enumerate(top5.iterrows()):
        with cols[idx]:
            st.image(row['image_url'], use_container_width=True)
            st.markdown(f"""
                <div style="text-align: center; margin-top: 0.5rem;">
                    <div style="font-weight: 600; color: #f0f6fc; font-size: 0.9rem;">{row['judul']}</div>
                    <div style="color: #3fb950; font-size: 0.875rem;">⭐ {row['rating']}</div>
                </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 4: FAVORITES
# ============================================================
with tab4:
    st.markdown('<h2 style="color: #f0f6fc; margin-bottom: 1.5rem;">⭐ Anime Favorit</h2>', unsafe_allow_html=True)
    
    if not st.session_state.favorites:
        st.info("Belum ada anime favorit. Tambahkan dari tab Rekomendasi!")
    else:
        for fav in st.session_state.favorites:
            st.markdown(f"""
                <div class="favorite-item">
                    <div class="favorite-info">
                        <h4>{fav['judul']}</h4>
                        <p>⭐ {fav['rating']} | 📅 {fav['tahun']} | {fav['genre']}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Hapus Semua Favorit", type="secondary"):
            st.session_state.favorites = []
            st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer">
        <p>🎌 Anime Recommender ML • Built with Streamlit & Scikit-Learn • 2024</p>
    </div>
""", unsafe_allow_html=True)
