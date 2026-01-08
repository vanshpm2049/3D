# ♻️ 3D Waste Analytics Dashboard

An interactive **Streamlit-based 3D geospatial analytics dashboard** for waste management performance analysis.
The app supports:

- 🗺️ 3D Pydeck column maps
- 🔥 Heatmaps with configurable metrics
- 📍 Rich popups (Folium)
- 📊 Cumulative & monthly analysis
- 🎛️ User-controlled bar height & heat intensity

---

## 🚀 Live Features
- Upload CSV in **Bintix format**
- Switch between **3D Pydeck** and **2D Folium**
- Toggle **heatmaps & columns**
- Download processed analytics

---

## 📂 Project Structure
```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .gitignore          # Git ignore rules
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Select `app.py` as entry point

---

## 📄 CSV Requirements
Mandatory columns:
- Latitude
- Longitude
- Community
- City
- Pincode

Metric columns format:
```
Tonnage Jan 2024
Trees Saved Feb 2024
CO2 Kgs Averted Mar 2024
```

---

Built with ❤️ by Vansh Bansal