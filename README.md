# 🎬 CineMatch — Content-Based Movie Recommender System

Ye project TMDB 5000 dataset use kar ke ek **content-based movie recommender**
banata hai (movie ke overview, genres, keywords, cast aur director ke basis
par). Frontend **Streamlit** mein hai aur ye **Streamlit Community Cloud**
par deploy karne ke liye ready hai.

Reference tutorial: [CampusX — Movie Recommender System Project](https://youtu.be/1xtrIEwY_zY)

---

## 📁 Project Structure

```
movie-recommender-system/
├── app.py                          # Streamlit frontend (main app)
├── requirements.txt                # Python dependencies
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml                 # App theme
│   └── secrets.toml.example        # TMDB API key ka template
├── preprocessing/
│   └── preprocess.py               # Dataset ko process kar ke pickle files banata hai
├── data/                           # (empty) — yahan CSV dataset daalni hai
└── artifacts/                      # (empty) — yahan generated .pkl files aayenge
```

---

## 🚀 Step-by-Step Setup (VS Code mein)

### 1️⃣ Virtual environment banao aur activate karo

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2️⃣ Dependencies install karo

```bash
pip install -r requirements.txt
```

Pehli baar NLTK ka stemmer use karne ke liye ye bhi chala lena (agar error aaye):

```bash
python -m nltk.downloader punkt
```

### 3️⃣ Dataset download karo

Kaggle se ye dataset download karo:
👉 https://www.kaggle.com/tmdb/tmdb-movie-metadata

Do files milengi:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Dono ko is project ke `data/` folder mein rakh do.

### 4️⃣ Preprocessing chalao (ye pickle files banayega)

```bash
python preprocessing/preprocess.py
```

Is se `artifacts/movie_list.pkl` aur `artifacts/similarity.pkl` ban jayengi
— ye files hi actual "model" hain jo app.py use karega.

### 5️⃣ TMDB API key lo (posters dikhane ke liye)

1. https://www.themoviedb.org par free account banao
2. Settings → API → API key (v3 auth) generate karo
3. `.streamlit/secrets.toml.example` ko copy kar ke `.streamlit/secrets.toml`
   naam do, aur apni key daal do:

```toml
TMDB_API_KEY = "yahan_apni_key_dalo"
```

> ⚠️ Agar key nahi bhi lagate, app phir bhi chalega — sirf poster ki jagah
> placeholder image dikhegi.

### 6️⃣ App run karo

```bash
streamlit run app.py
```

Browser mein `http://localhost:8501` par app khul jayega. 🎉

---

## ☁️ Streamlit Community Cloud par Deploy Karna

1. Is poore project ko GitHub par ek public repo mein push karo
   (`data/*.csv` push nahi hongi — `.gitignore` mein already excluded hai,
   lekin `artifacts/*.pkl` files **zaroor push karo**, kyunki app ko wahi
   chahiye hoti hain).

   ```bash
   git init
   git add .
   git commit -m "Initial commit - movie recommender system"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. https://share.streamlit.io par jao aur GitHub se login karo.

3. **"New app"** par click karo, apni repo select karo, aur:
   - Branch: `main`
   - Main file path: `app.py`

4. **Advanced settings → Secrets** mein ye add karo (secrets.toml waisi hi
   format mein):

   ```toml
   TMDB_API_KEY = "yahan_apni_key_dalo"
   ```

5. **Deploy** par click karo — bas! Kuch minute mein tumhari app live ho
   jayegi 🎬✨

---

## ⚠️ Important Notes

- Poori NxN similarity matrix store karne ke bajaye (jo ~4800 movies ke
  liye ~180MB ban jati hai — GitHub ki 100MB limit se zyada), ye project
  har movie ke sirf **top-20 sab se milte-julte movies ke indices** store
  karta hai (`similarity.pkl` yahan sirf ~400KB ki hai). Chahe to
  `preprocessing/preprocess.py` mein `TOP_K` value badha/ghata sakte ho.
- Ye approach chhote/medium datasets (~5000 movies) ke liye perfect hai.
  Bahut bade dataset ke liye better approach hoga approximate nearest
  neighbors (e.g. `annoy` ya `faiss`) use karna.
- Agar `nltk` install nahi hai to preprocessing warning ke sath chalti hai
  (stemming skip ho jati hai) — results phir bhi acha kaam karte hain,
  lekin best results ke liye `pip install nltk` zaroor karo.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit (custom CSS ke saath)
- **ML:** scikit-learn (CountVectorizer + Cosine Similarity), NLTK (stemming)
- **Data:** Pandas, NumPy
- **Deployment:** Streamlit Community Cloud
- **Poster API:** TMDB API
