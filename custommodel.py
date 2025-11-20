import streamlit as st
import requests
import tensorflow as tf
import numpy as np

# Google Custom Search API Setup (Replace with your own API key and CX key)
GOOGLE_API_KEY = "AIzaSyB5__sMOvpDae-Lu-KcVWuPq3tzElSLyqg"
GOOGLE_CX = "144c3690a89b74fb5"

# Load pre-trained Instagram detection model
INSTAGRAM_MODEL_PATH = "my_model.h5"  # Replace with the correct path to your model
instagram_model = tf.keras.models.load_model(r"C:\Users\pmoni\Fake Profile Detection\instagram_model.h5")

# Define trusted domains
TRUSTED_DOMAINS = ["bbc.com", "reuters.com", "factcheck.org", "apnews.com"]

# Fake Content Detection Functions
def search_web(query):
    """
    Perform a Google Custom Search with the given query.
    """
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def analyze_content(input_text):
    """
    Analyze the input text to determine if it is fake or real.
    Searches the web for the content and compares results.
    """
    search_results = search_web(input_text)
    if "error" in search_results:
        return {"status": "error", "message": search_results["error"]}
    
    items = search_results.get("items", [])
    
    matches = []
    for item in items:
        domain = item["displayLink"]
        snippet = item.get("snippet", "")
        title = item.get("title", "")
        link = item.get("link", "")
        if any(trusted in domain for trusted in TRUSTED_DOMAINS):
            matches.append({"title": title, "snippet": snippet, "link": link})
    
    if matches:
        return {"status": "real", "matches": matches}
    else:
        return {"status": "fake", "matches": []}

# Fake Instagram Profile Detection Functions
def predict_instagram_fake(inputs):
    """
    Predict if an Instagram profile is fake based on the inputs.
    """
    inputs = np.array([inputs])  # Ensure inputs are in the correct shape for the model
    prediction = instagram_model.predict(inputs)
    return "Fake" if prediction[0][0] > 0.5 else "Real"

# Streamlit App
st.title("Fake Content and Instagram Profile Detection")

# Section 1: Fake Content Detection
st.header("1. Fake Content Detection")

content_input_type = st.radio("Choose Input Type", ["Text", "URL"])
if content_input_type == "Text":
    input_text = st.text_area("Paste the text here:")
    if st.button("Analyze Text"):
        if input_text:
            with st.spinner("Analyzing text..."):
                result = analyze_content(input_text)
            if result["status"] == "error":
                st.error(f"Error: {result['message']}")
            elif result["status"] == "real":
                st.success("The content seems real! Here are trusted references:")
                for match in result["matches"]:
                    st.markdown(f"- *[{match['title']}]({match['link']})*: {match['snippet']}")
            else:
                st.error("The content seems fake! No trusted references found.")
        else:
            st.error("Please enter some text to analyze.")
elif content_input_type == "URL":
    input_url = st.text_input("Paste the URL here:")
    if st.button("Analyze URL"):
        if input_url:
            with st.spinner("Fetching content from URL..."):
                # Fetch the content from the URL (using a library like BeautifulSoup if needed)
                response = requests.get(input_url)
                if response.status_code == 200:
                    page_text = response.text
                    result = analyze_content(page_text[:1000])  # Use a subset of the content for analysis
                    if result["status"] == "error":
                        st.error(f"Error: {result['message']}")
                    elif result["status"] == "real":
                        st.success("The content seems real! Here are trusted references:")
                        for match in result["matches"]:
                            st.markdown(f"- *[{match['title']}]({match['link']})*: {match['snippet']}")
                    else:
                        st.error("The content seems fake! No trusted references found.")
                else:
                    st.error("Failed to fetch content from the URL.")
        else:
            st.error("Please enter a URL to analyze.")

# Section 2: Fake Instagram Profile Detection
st.header("2. Fake Instagram Profile Detection")

username = st.text_input("Username:")
bio = st.text_area("Bio:")
posts = st.number_input("Number of Posts:", min_value=0)
followers = st.number_input("Number of Followers:", min_value=0)
following = st.number_input("Number of Following:", min_value=0)

if st.button("Analyze Instagram Profile"):
    if username and bio:
        with st.spinner("Analyzing profile..."):
            inputs = [len(username), len(bio), posts, followers, following]
            result = predict_instagram_fake(inputs)
        st.success(f"The profile is classified as: *{result}*")
    else:
        st.error("Please fill in all fields to analyze.")
