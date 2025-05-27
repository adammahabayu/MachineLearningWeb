import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load data dan model
df = pd.read_csv('CarPrice_Assignment.csv')
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")

# Sidebar navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Visualisasi Data", "Prediksi Harga Mobil"])

if page == "Beranda":
    st.title("🚗 Aplikasi Prediksi Harga Mobil")
    st.image("https://images.unsplash.com/photo-1549924231-f129b911e442", caption="Mobil Sport", use_container_width=True)
    st.markdown("""
        Selamat datang di aplikasi prediksi harga mobil berbasis Machine Learning.  
        Anda dapat:
        - Melihat visualisasi data
        - Melakukan prediksi harga mobil baru
        - Memahami fitur-fitur yang memengaruhi harga
    """)
    st.markdown("### 📊 Preview Dataset Mobil")
    st.dataframe(df.head(), use_container_width=True)

elif page == "Visualisasi Data":
    st.title("📈 Visualisasi Data Mobil")

    # Membuat tab visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["Horsepower", "Curb Weight", "Engine Size", "Highway MPG"])
    with tab1:
        st.area_chart(df['horsepower'])
    with tab2:
        st.line_chart(df['curbweight'])
    with tab3:
        st.bar_chart(df['enginesize'])
    with tab4:
        st.line_chart(df['highwaympg'])

elif page == "Prediksi Harga Mobil":
    st.title("💰 Prediksi Harga Mobil")
    st.markdown("Silakan masukkan fitur-fitur mobil berikut untuk mendapatkan prediksi harga:")

    # Input fitur di sidebar
    horsepower = st.sidebar.slider("Horsepower", int(df['horsepower'].min()), int(df['horsepower'].max()), int(df['horsepower'].mean()))
    curbweight = st.sidebar.slider("Curb Weight", int(df['curbweight'].min()), int(df['curbweight'].max()), int(df['curbweight'].mean()))
    enginesize = st.sidebar.slider("Engine Size", int(df['enginesize'].min()), int(df['enginesize'].max()), int(df['enginesize'].mean()))
    highwaympg = st.sidebar.slider("Highway MPG", int(df['highwaympg'].min()), int(df['highwaympg'].max()), int(df['highwaympg'].mean()))

    st.write("### Fitur Input Anda:")
    st.write({
        'Horsepower': horsepower,
        'Curb Weight': curbweight,
        'Engine Size': enginesize,
        'Highway MPG': highwaympg
    })

    if st.button("🔍 Prediksi Harga Mobil"):
        input_data = np.array([[horsepower, curbweight, enginesize, highwaympg]])
        prediksi = model.predict(input_data)[0]
        st.success(f"🎯 Prediksi harga mobil: *${prediksi:,.2f}*")
        st.balloons()

        # Kategori harga
        if prediksi > 30000:
            st.info("✅ Harga mobil tergolong MAHAL.")
        elif prediksi > 15000:
            st.warning("💡 Harga mobil tergolong MENENGAH.")
        else:
            st.success("🟢 Harga mobil tergolong EKONOMIS.")

# Footer
st.markdown("---")
st.markdown("📌 Aplikasi oleh **Adam Mahabayu Muhibbulloh** | © 2025 Prediksi Harga Mobil")
