#Nomor 1
# import streamlit as st

# st.write("Hello World!")

#Nomor 2
# import streamlit as st

# st.header('st.button')

# if st.button('Say hello'):
#     st.write('Why hello there')
# else:
#     st.write('Goodbye')

#Nomor 3  
# import streamlit as st

# st.title("this is the app title")
# st.markdown("this is the markdown")
# st.header("this is the header")
# st.subheader("this is the subheader")
# st.caption("this is the caption")

# x = 2021
# st.code(f"x = {x}", language='python')

#Nomor 4
# import streamlit as st

# st.checkbox("yes")

# st.button("Click")

# st.radio("Pick your gender", ["Male", "Female"])

# st.selectbox("Pick your gender", ["Male", "Female"])

# st.selectbox("choose a planet", ["Choose an option", "Mercury", "Venus", "Earth", "Mars"])

# st.slider("Pick a mark", 0, 100, 70, format="%d", help="Bad | Good | Excellent")

# st.slider("Pick a number", 0, 50, 9)

#Nomor 5
# import streamlit as st
# import datetime

# st.number_input("Pick a number", min_value=0, step=1)

# st.text_input("Email adress")

# st.date_input("Travelling date", value=datetime.date(2022, 6, 17))

# st.time_input("School time", value=datetime.time(8, 0))

# st.text_area("Description")

# st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

# st.color_picker("Choose your favourite color", "#ff00ff")  # default warna ungu

#Nomor 6
# import numpy as np
# import altair as alt
# import pandas as pd
# import streamlit as st

# st.header('st.write')
# st.write('Hello, *World !* :sunglasses:')
# st.write(1234)

# df = pd.DataFrame({
# 'first column': [1, 2, 3, 4],
# 'second column': [10, 20, 30, 40]
# })
# st.write(df)

# st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

# df2 = pd.DataFrame(
# np.random.randn(200, 3),
# columns=['a', 'b', 'c'])
# c = alt.Chart(df2).mark_circle().encode(
# x='a', y='b', size='c' , color='c' , tooltip=['a' , 'b' , 'c' ])
# st.write(c)

#Nomor 7
# import streamlit as st
# import pandas as pd
# import numpy as np

# df= pd.DataFrame(
#     np.random.randn(10, 2),
#     columns=['x', 'y'])
# st.line_chart(df)

#Nomor 8
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# st.sidebar.title("Select Page")
# option = st.sidebar.selectbox("Menu", ["Home", "Dataset", "Chart"])

# if option == "Home":
#     st.image("37253.jpg", caption="Welcome to the Loan App!", use_container_width=True)

# elif option == "Dataset":
#     st.subheader("Dataset:")
#     df = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")  
#     st.dataframe(df.head())

# elif option == "Chart":
#     st.subheader("Applicant Income VS Loan Amount")
#     df = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")  
#     fig, ax = plt.subplots()
#     ax.bar(df.index, df['ApplicantIncome'], label="ApplicantIncome")
#     ax.bar(df.index, df['LoanAmount'], label="LoanAmount", alpha=0.7)
#     ax.legend()
#     st.pyplot(fig)

#Nomor 9
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# import pickle

# df = pd.read_csv('CarPrice_Assignment.csv')

# X = df[['horsepower', 'curbweight', 'enginesize', 'highwaympg']]  
# y = df['price']  

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model_regresi = LinearRegression()
# model_regresi.fit(X_train, y_train)

# filename = 'model_prediksi_harga_mobil.sav'
# pickle.dump(model_regresi, open(filename, 'wb'))

# print(f"Model berhasil disimpan sebagai {filename}")

#Nomor 10
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# df = pd.read_csv('CarPrice_Assignment.csv')

# model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# st.title("Prediksi Harga Mobil")

# st.subheader("Dataset")
# st.dataframe(df)

# st.subheader("Grafik Horsepower")
# st.line_chart(df['horsepower'])

# st.subheader("Grafik Curbweight")
# st.line_chart(df['curbweight'])

# st.subheader("Grafik Enginesize")
# st.line_chart(df['enginesize'])

# st.subheader("Input Fitur")

# horsepower = st.selectbox("Horsepower", sorted(df['horsepower'].unique()))
# curbweight = st.selectbox("Curb Weight", sorted(df['curbweight'].unique()))
# enginesize = st.selectbox("Engine Size", sorted(df['enginesize'].unique()))
# highwaympg = st.selectbox("Highway MPG", sorted(df['highwaympg'].unique()))

# if st.button("Prediksi"):
#     input_data = np.array([[horsepower, curbweight, enginesize, highwaympg]])
#     prediksi_harga = model.predict(input_data)
#     st.success(f"Prediksi harga mobil: ${prediksi_harga[0]:,.2f}")

#Nomor 11
import streamlit as st
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('CarPrice_Assignment.csv')
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

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
    st.markdown("### 📊 Dataset Preview")
    st.dataframe(df.head())

elif page == "Visualisasi Data":
    st.title("📈 Visualisasi Data Mobil")

    st.subheader("Distribusi Horsepower")
    st.line_chart(df['horsepower'])

    st.subheader("Distribusi Curb Weight")
    st.line_chart(df['curbweight'])

    st.subheader("Distribusi Engine Size")
    st.line_chart(df['enginesize'])

    st.subheader("Distribusi Highway MPG")
    st.line_chart(df['highwaympg'])

elif page == "Prediksi Harga Mobil":
    st.title("💰 Prediksi Harga Mobil")
    st.markdown("Silakan isi fitur-fitur di bawah untuk mendapatkan prediksi harga mobil.")

    col1, col2 = st.columns(2)

    with col1:
        horsepower = st.slider("Horsepower", int(df['horsepower'].min()), int(df['horsepower'].max()), 100)
        enginesize = st.slider("Engine Size", int(df['enginesize'].min()), int(df['enginesize'].max()), 130)

    with col2:
        curbweight = st.slider("Curb Weight", int(df['curbweight'].min()), int(df['curbweight'].max()), 2500)
        highwaympg = st.slider("Highway MPG", int(df['highwaympg'].min()), int(df['highwaympg'].max()), 30)

    if st.button("🔍 Prediksi Harga"):
        input_data = np.array([[horsepower, curbweight, enginesize, highwaympg]])
        prediksi_harga = model.predict(input_data)

        st.metric(label="💵 Harga Prediksi Mobil", value=f"${prediksi_harga[0]:,.2f}")

        st.subheader("📝 Data Input Anda:")
        st.write({
            'Horsepower': horsepower,
            'Curb Weight': curbweight,
            'Engine Size': enginesize,
            'Highway MPG': highwaympg
        })

        if prediksi_harga[0] > 30000:
            st.info("✅ Harga mobil tergolong MAHAL.")
        elif prediksi_harga[0] > 15000:
            st.warning("💡 Harga mobil tergolong MENENGAH.")
        else:
            st.success("🟢 Harga mobil tergolong EKONOMIS.")

st.markdown("---")
st.markdown("📌 Aplikasi oleh **Adam Mahabayu Muhibbulloh** | © 2025 Prediksi Harga Mobil")
