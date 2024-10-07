import streamlit as st
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 28px !important;
        font-weight: bold;
        color: #d0dbe5;
    }
    .stButton>button {
        background-color: #3f51b5;
        color: white;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.8);
        color: #d0dbe5;
        border: 1px solid #3f51b5;
    }
    h1, h2, h3 {
        color: #d0dbe5;
    }
    .stProgress > div > div > div > div {
        background-color: #51b53f;
    }
    p, .stAlert > div {
        color: #d0dbe5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained("babujibabuji/FineTuneBert")
    return tokenizer, model

tokenizer, model = get_model()

# Main app layout
st.title("🎭 Sentiment Analyzer")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Enter your text for sentiment analysis:")
    user_input = st.text_area("", height=150)

with col2:
    st.markdown("### Results")
    if st.button("Analyze Sentiment", key="analyze"):
        if user_input:
            with st.spinner("Analyzing..."):
                test_sample = tokenizer.encode(user_input, padding=True, truncation=True, max_length=512, return_tensors='tf')
                output = model.predict(test_sample)
                logits = output.logits[0]
                tf_prediction = tf.nn.softmax(logits)
                label = tf.argmax(tf_prediction)
                
                sentiment = "Positive" if label.numpy() == 1 else "Negative"
                confidence = tf_prediction.numpy()[label.numpy()]
                
                st.markdown(f"**Sentiment:** <p class='big-font'>{sentiment}</p>", unsafe_allow_html=True)
                st.progress(float(confidence))
                st.markdown(f"**Confidence:** {confidence:.2%}")
                st.markdown("**Raw Logits:**")
                st.write(f"Negative: {logits[0]:.4f}")
                st.write(f"Positive: {logits[1]:.4f}")
        else:
            st.warning("Please enter some text to analyze.")

st.markdown("---")

# Add some information about the model
st.markdown("### About the Model")
st.info("""
This sentiment analyzer uses a fine-tuned BERT model to classify food reviews as positive or negative.
Enter a review of a restaurant or food product to extract the overall sentiment of the text.
""")

# Footer
st.markdown("---")
st.markdown("Created by Kishan Babuji 2024")