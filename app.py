import app as st
import requests
import torch
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup  # For extracting text from URLs

# Load pre-trained BERT model and tokenizer
BERT_MODEL_PATH = "bert-base-uncased"  # Replace with your fine-tuned model path if available
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)  # Updated tokenizer
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)  # Updated model

# Google Fact Check API Setup (Replace with your own API key)
GOOGLE_FACT_CHECK_API_KEY = "AIzaSyB5__sMOvpDae-Lu-KcVWuPq3tzElSLyqg"

# Load your pre-trained profile detection model
PROFILE_MODEL_PATH = "instagram_model.h5"  # Replace with the path to your trained model
try:
    profile_model = tf.keras.models.load_model(PROFILE_MODEL_PATH)
    st.success("Profile detection model loaded successfully!")
except Exception as e:
    st.error(f"Error loading profile detection model: {e}")
    profile_model = None

# Define a function to analyze text using BERT
def analyze_text_with_bert(text):
    """
    Analyze the input text using a pre-trained BERT model.
    Returns the predicted label and confidence score.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)
    confidence, predicted_label = torch.max(probabilities, dim=-1)
    return predicted_label.item(), confidence.item()

# Define a function to search for fact-checked claims
def search_fact_check(query):
    """
    Search for fact-checked claims using the Google Fact Check API.
    """
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={GOOGLE_FACT_CHECK_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Define a function to extract headline or main text from a URL
def extract_text_from_url(url):
    """
    Extract the headline or main text from a webpage.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract headline (title tag)
        headline = soup.title.string if soup.title else ""
        
        # Extract main text (first <p> tag or <article> tag)
        main_text = ""
        article = soup.find("article")
        if article:
            main_text = article.get_text(separator=" ", strip=True)
        else:
            first_paragraph = soup.find("p")
            if first_paragraph:
                main_text = first_paragraph.get_text(separator=" ", strip=True)
        
        return headline, main_text
    except Exception as e:
        return None, str(e)

# Define a function to predict if a profile is fake
def predict_fake_profile(inputs):
    """
    Predict if a profile is fake using your pre-trained model.
    """
    if profile_model is None:
        return "Error: Model not loaded."
    inputs = np.array([inputs])  # Ensure inputs are in the correct shape for the model
    prediction = profile_model.predict(inputs)
    return "Fake" if prediction[0][0] > 0.5 else "Real"

# Streamlit App
st.title("Fake Content and Profile Detection")

# Section 1: Fake Content Detection
st.header("1. Fake Content Detection")

content_input_type = st.radio("Choose Input Type", ["Text", "URL"])
if content_input_type == "Text":
    input_text = st.text_area("Paste the text here:")
    if st.button("Analyze Text"):
        if input_text:
            with st.spinner("Analyzing text..."):
                # Step 1: Analyze text with BERT
                label, confidence = analyze_text_with_bert(input_text)
                st.write(f"*BERT Prediction:* {'Fake' if label == 1 else 'Real'}")
                st.write(f"*Confidence Level:* {confidence:.2f}")

                # Step 2: Search for fact-checked claims
                fact_check_results = search_fact_check(input_text)
                if "error" in fact_check_results:
                    st.error(f"Error: {fact_check_results['error']}")
                else:
                    claims = fact_check_results.get("claims", [])
                    if claims:
                        st.success("*Fact Check Results:*")
                        for claim in claims:
                            st.write(f"- *Claim:* {claim.get('text', 'N/A')}")
                            st.write(f"  *Publisher:* {claim.get('claimReview', [{}])[0].get('publisher', {}).get('name', 'N/A')}")
                            st.write(f"  *Review:* {claim.get('claimReview', [{}])[0].get('textualRating', 'N/A')}")
                            st.write(f"  *URL:* {claim.get('claimReview', [{}])[0].get('url', 'N/A')}")
                    else:
                        st.warning("No fact-checked claims found for this text.")
        else:
            st.error("Please enter some text to analyze.")
elif content_input_type == "URL":
    input_url = st.text_input("Paste the URL here:")
    if st.button("Analyze URL"):
        if input_url:
            with st.spinner("Fetching content from URL..."):
                # Extract headline and main text from the URL
                headline, main_text = extract_text_from_url(input_url)
                if headline or main_text:
                    st.write(f"*Headline:* {headline}")
                    st.write(f"*Main Text:* {main_text[:500]}...")  # Show first 500 characters of main text

                    # Step 1: Analyze text with BERT
                    label, confidence = analyze_text_with_bert(headline or main_text)
                    st.write(f"*BERT Prediction:* {'Fake' if label == 1 else 'Real'}")
                    st.write(f"*Confidence Level:* {confidence:.2f}")

                    # Step 2: Search for fact-checked claims
                    fact_check_results = search_fact_check(headline or main_text)
                    if "error" in fact_check_results:
                        st.error(f"Error: {fact_check_results['error']}")
                    else:
                        claims = fact_check_results.get("claims", [])
                        if claims:
                            st.success("*Fact Check Results:*")
                            for claim in claims:
                                st.write(f"- *Claim:* {claim.get('text', 'N/A')}")
                                st.write(f"  *Publisher:* {claim.get('claimReview', [{}])[0].get('publisher', {}).get('name', 'N/A')}")
                                st.write(f"  *Review:* {claim.get('claimReview', [{}])[0].get('textualRating', 'N/A')}")
                                st.write(f"  *URL:* {claim.get('claimReview', [{}])[0].get('url', 'N/A')}")
                        else:
                            st.warning("No fact-checked claims found for this content.")
                else:
                    st.error("Failed to extract content from the URL.")
        else:
            st.error("Please enter a URL to analyze.")

# Section 2: Fake Profile Detection
st.header("2. Fake Profile Detection")

if profile_model is None:
    st.error("Profile detection model is not loaded. Please check the model file.")
else:
    st.write("Enter the profile details:")
    profile_pic = st.selectbox("Profile Picture", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    username_nums = st.number_input("Numeric Characters in Username Ratio", min_value=0.0, max_value=1.0, value=0.0)
    fullname_words = st.number_input("Number of Words in Full Name", min_value=0, value=0)
    fullname_nums = st.number_input("Numeric Characters in Full Name Ratio", min_value=0.0, max_value=1.0, value=0.0)
    name_username_match = st.selectbox("Username Matches Full Name", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    description_length = st.number_input("Description Length", min_value=0, value=0)
    external_url = st.selectbox("External URL", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    private = st.selectbox("Private Account", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    posts = st.number_input("Number of Posts", min_value=0, value=0)
    followers = st.number_input("Number of Followers", min_value=0, value=0)
    follows = st.number_input("Number of Follows", min_value=0, value=0)

    if st.button("Analyze Profile"):
        profile_data = [
            profile_pic,
            username_nums,
            fullname_words,
            fullname_nums,
            name_username_match,
            description_length,
            external_url,
            private,
            posts,
            followers,
            follows
        ]
        result = predict_fake_profile(profile_data)
        st.success(f"*Profile Prediction:* {result}")
