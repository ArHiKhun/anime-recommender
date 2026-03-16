import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi halaman
st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Judul
st.title("🎌 Anime Recommendation System")
st.markdown("### Temukan Anime Sesuai Selera Kamu!")
st.markdown("---")

# Dataset sederhana (bisa diperluas)
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
            'Action Dark Fantasy Military',
            'Supernatural Thriller Psychological',
            'Action Adventure Comedy',
            'Action Adventure Martial Arts',
            'Action Dark Fantasy Historical',
            'Action Comedy School Superhero',
            'Action Adventure Fantasy Military',
            'Action Comedy Family Slice of Life',
            'Action Dark Fantasy School',
            'Action Dark Fantasy Horror Psychological',
            'Romance Drama Fantasy',
            'Drama Romance School',
            'Romance Drama Fantasy',
            'Romance Comedy School Slice of Life',
            'Romance Comedy Drama School',
            'Romance Comedy Psychological School',
            'Action Comedy Parody Superhero',
            'Action Comedy Supernatural',
            'Action Adventure Fantasy',
            'Sci-Fi Thriller Drama',
            'Action Drama Mecha Military',
            'Action Adventure Sci-Fi'
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
            'Humanity fights for survival against giant humanoid Titans',
            'A high school student discovers a supernatural notebook',
            'Pirate adventure to find the ultimate treasure',
            'Young ninja seeks recognition and dreams of becoming Hokage',
            'Boy becomes demon slayer to save his sister',
            'Students train to become professional heroes',
            'Two brothers search for Philosopher Stone',
            'Spy creates fake family for mission',
            'Student fights curses in supernatural world',
            'College student becomes half-ghoul after accident',
            'Two teenagers connected by fate across time',
            'Story about bullying and redemption',
            'Boy meets mysterious girl who can control weather',
            'High school romance between popular and quiet students',
            'Unlikely romance between two high school students',
            'Battle of wits between two student council members',
            'Hero who can defeat any enemy with one punch',
            'Boy with psychic powers tries to live normal life',
            'Young boy searches for his father',
            'Time travel thriller with consequences',
            'Exiled prince leads rebellion with supernatural power',
            'Bounty hunters travel through space'
        ]
    }
    return pd.DataFrame(data)

# Load data
df = load_data()

# Sidebar untuk input user
st.sidebar.header("⚙️ Preferensi Kamu")

genre_preference = st.sidebar.multiselect(
    "Pilih Genre Favorit:",
    options=['Action', 'Adventure', 'Comedy', 'Drama', 'Romance', 
             'Fantasy', 'Sci-Fi', 'Thriller', 'Supernatural', 'Slice of Life'],
    default=['Action']
)

min_rating = st.sidebar.slider("Minimal Rating:", 0.0, 10.0, 7.0)

tahun_range = st.sidebar.slider("Range Tahun:", 1998, 2023, (2010, 2023))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Advanced Filter")

sort_by = st.sidebar.selectbox("Urutkan Berdasarkan:", 
                                ['Rating', 'Tahun', 'Relevansi'])

# Fungsi rekomendasi
def get_recommendations(df, genres, min_rating, year_range, sort_method):
    # Filter berdasarkan rating dan tahun
    filtered_df = df[(df['rating'] >= min_rating) & 
                     (df['tahun'] >= year_range[0]) & 
                     (df['tahun'] <= year_range[1])].copy()
    
    if not genres:
        return filtered_df
    
    # TF-IDF Vectorization untuk genre
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genre'])
    
    # Hitung similarity dengan genre preference
    user_genre = ' '.join(genres)
    user_vector = tfidf.transform([user_genre])
    cosine_sim = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    filtered_df['similarity'] = cosine_sim[filtered_df.index]
    
    # Sort berdasarkan preferensi
    if sort_method == 'Rating':
        filtered_df = filtered_df.sort_values('rating', ascending=False)
    elif sort_method == 'Tahun':
        filtered_df = filtered_df.sort_values('tahun', ascending=False)
    else:
        filtered_df = filtered_df.sort_values('similarity', ascending=False)
    
    return filtered_df

# Tombol cari rekomendasi
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button(" Cari Rekomendasi Anime", use_container_width=True):
        
        # Dapatkan rekomendasi
        rekomendasi = get_recommendations(df, genre_preference, min_rating, tahun_range, sort_by)
        
        # Tampilkan hasil
        st.markdown("---")
        st.markdown(f"### 📺 Ditemukan {len(rekomendasi)} Anime untuk Kamu!")
        st.markdown("---")
        
        # Tampilkan dalam grid
        for idx, row in rekomendasi.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Placeholder untuk gambar (bisa diganti dengan URL gambar nyata)
                    st.image(f"https://via.placeholder.com/150x200/667eea/ffffff?text={row['judul'].replace(' ', '+')}", 
                            width=150)
                
                with col2:
                    st.subheader(row['judul'])
                    st.markdown(f"**Genre:** {row['genre']}")
                    st.markdown(f"**Rating:** ⭐ {row['rating']}/10")
                    st.markdown(f"**Tahun:** {row['tahun']}")
                    st.markdown(f"**Sinopsis:** {row['sinopsis']}")
                    
                    # Progress bar untuk rating
                    st.progress(int(row['rating'] * 10))
                
                st.markdown("---")
        
        # Statistik
        st.markdown("### 📊 Statistik Hasil")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Anime", len(rekomendasi))
        col2.metric("Rata-rata Rating", f"{rekomendasi['rating'].mean():.1f}")
        col3.metric("Tertinggi", f"{rekomendasi['rating'].max():.1f}")
        col4.metric("Terendah", f"{rekomendasi['rating'].min():.1f}")

# Tampilkan semua data jika belum search
else:
    st.markdown("### 📚 Database Anime Tersedia")
    st.dataframe(df[['judul', 'genre', 'rating', 'tahun']], use_container_width=True)
    
    st.markdown("""
    ---
    ### 💡 Cara Menggunakan:
    1. Pilih genre favorit di sidebar
    2. Atur minimal rating yang diinginkan
    3. Pilih range tahun rilis
    4. Klik tombol **Cari Rekomendasi Anime**
    5. Sistem akan menampilkan anime yang sesuai preferensi kamu!
    """)

# Footer
st.markdown("---")
st.markdown("""
<center>
<b>🎌 Anime Recommendation System - Skripsi S1 Ilmu Komputer</b><br>
Built with Streamlit & Machine Learning
</center>
""", unsafe_allow_html=True)