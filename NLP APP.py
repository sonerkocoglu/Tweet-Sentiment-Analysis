import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Model ve vektörleştiriciyi yükleyin
model = joblib.load('nlp_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Duygu analizi fonksiyonu
def predict_sentiment(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Streamlit arayüzü
st.set_page_config(
    page_title="ChatGPT Tweets Sentiment Analysis",
    menu_items={
        "Get help": "https://www.linkedin.com/in/berke-ilbay-05935a285/",
        "About": "For More Information\n" + "https://github.com/berkeilbay"
    }
)

# Arayüz başlığı
st.title('ChatGPT Tweets Sentiment Analysis')
st.write('This app analyzes your tweets and predicts your sentiment.')

# Kullanıcıdan tweet girişi alma
tweet = st.text_area('Enter the tweet you want to analyze:')

# Fotoğraf dosya yolları
bad_image_path = "images/bad_image.png"
good_image_path = "images/good_image.png"
neutral_image_path = "images/neutral_image.png"

# Kullanıcı tweet girdiğinde duygu tahmini yapma
if st.button('Analyze'):
    if tweet:
        sentiment = predict_sentiment(tweet)
        st.write(f'Predicted Sentiment: {sentiment}')
        
        # Tahmine göre fotoğraf gösterme
        if sentiment == 'bad':
            st.image(bad_image_path, caption='Bad Sentiment')
        elif sentiment == 'good':
            st.image(good_image_path, caption='Good Sentiment')
        elif sentiment == 'neutral':
            st.image(neutral_image_path, caption='Neutral Sentiment')
    else:
        st.write('Please enter a tweet you want to analyze..')

wordcloud_path = "images/wordcloud.png"

# Ek bilgi ve görseller
st.markdown("This app classifies tweets as **positive**, **negative** or **neutral** using a trained NLP model.")
st.image(wordcloud_path)
st.markdown("By harnessing the power of machine learning and NLP, you can improve your social media analytics and better understand users' emotional responses.")
