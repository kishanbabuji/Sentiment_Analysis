 # 🍽️ Food Reviews Sentiment Analysis

This project uses a fine-tuned BERT model to analyze the sentiment of food reviews. The goal is to classify reviews as positive or negative, helping users understand the sentiment behind the text.

You can also interact with this model directly via a deployed [**Streamlit app**](https://sentimentanalysis-kishanbabuji.streamlit.app/), where you can input your own review and see the sentiment classification in real-time!

## Try it out!
- [Streamlit App](https://sentimentanalysis-kishanbabuji.streamlit.app/)

## 🎯 Features

- **Fine-tuned BERT Model**: The core of the sentiment analysis functionality, trained on Amazon food reviews to classify review sentiment.
- **Streamlit Interface**: A user-friendly web app where users can input their own review text and get a sentiment prediction instantly.
- **Real-time Testing**: Input your custom food review and get immediate feedback on whether the sentiment is positive or negative.

## 🔧 Installation & Setup to Run App Locally

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/kishanbabuji/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### 2. Create a Virtual Environment and Install Dependencies
You can set up a virtual environment to isolate dependencies and install the required packages from the requirements.txt file.

### 3. Run the Streamlit App Locally
Once you've installed the dependencies, you can run the Streamlit app to test the sentiment model locally:
```bash
streamlit run app.py
```
The app should now be accessible at http://localhost:8501 in your browser.

## 💻 Usage
Once the app is running, you can enter any food review into the text box, and the model will return a positive or negative sentiment based on the review content.

Example:
- Input: "This product is amazing! The flavor is out of this world."
- Output: Positive sentiment.

- Input: "I was really disappointed with this item. It tasted stale."
- Output: Negative sentiment.

## 📊 Dataset
The fine-tuned BERT model was trained on the Amazon Food Reviews dataset from Kaggle, which contains a large collection of product reviews from Amazon’s food category. 
- Source: [Amazon Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## 🧠 Model
The sentiment analysis model is built using BERT (Bidirectional Encoder Representations from Transformers), a transformer-based architecture that has been fine-tuned on the Amazon food reviews dataset. The fine-tuned model is [hosted on huggingface](https://huggingface.co/babujibabuji/FineTuneBert)

- Pre-trained model: BERT base uncased
- Fine-tuned on: Amazon Food Reviews dataset.
- Classification task: Binary sentiment classification (Positive/Negative).
- Achieves 90.7% accuracy on evaluation set.

