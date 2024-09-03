import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the pre-trained BERT model for sentiment analysis
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Create a sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Preprocessing function
def preprocess_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    return text

# Streamlit app
def main():
    st.title("Sentiment Analysis of Comments using BERT")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file with comments", type=["csv"])

    if uploaded_file:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

            # Ensure all comments are strings and handle NaN values
            df['Comment'] = df['Comment'].astype(str).fillna('')

            # Preprocess comments
            df['Comment'] = df['Comment'].apply(preprocess_text)

            st.write("Data Preview:")
            st.dataframe(df.head())

            def analyze_sentiment(comment):
                # Truncate the text to the first 512 tokens
                tokens = tokenizer.encode(comment, max_length=512, truncation=True)
                truncated_comment = tokenizer.decode(tokens, skip_special_tokens=True)

                # Get sentiment analysis result
                result = nlp(truncated_comment)
                sentiment = result[0]['label']

                # Mapping BERT sentiment labels to custom categories
                if sentiment in ['1 star', '2 stars']:
                    return 'negative'
                elif sentiment == '3 stars':
                    return 'neutral'
                else:
                    return 'positive'

            # Apply sentiment analysis
            df['Sentiment'] = df['Comment'].apply(analyze_sentiment)

            # Display sentiment results
            st.write("Sentiment Analysis Results:")
            st.dataframe(df[["Comment", "Sentiment"]])

            # Calculate percentages
            sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
            sentiment_counts = sentiment_counts.reset_index()
            sentiment_counts.columns = ['Sentiment Category', 'Percentage']

            # Plot the sentiment analysis results
            st.write("Sentiment Analysis Distribution:")
            plt.figure(figsize=(10, 5))
            sns.barplot(x='Sentiment Category', y='Percentage', data=sentiment_counts, palette='viridis')

            # Annotate bars with percentage values
            for index, row in sentiment_counts.iterrows():
                plt.text(row.name, row.Percentage + 1, f'{row.Percentage:.2f}%', color='black', ha='center')

            plt.title('Sentiment Analysis of Comments using BERT')
            plt.xlabel('Sentiment Category')
            plt.ylabel('Percentage')
            plt.ylim(0, 100)  # Ensure y-axis goes to 100% for clarity
            st.pyplot(plt)

        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown('<p style="text-align: center;"><img src="https://www.mygov.in/sites/all/themes/mygov/front_assets/images/logo.svg" alt="Logo" width="100"></p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;"><img src="https://user-images.githubusercontent.com/81156510/180811822-1476b05d-d389-4065-9a9d-26e366aa9f68.jpg" alt="Logo" width="300"></p>', unsafe_allow_html=True)

# Sidebar content
st.sidebar.image("https://www.mygov.in/sites/all/themes/mygov/front_assets/images/logo.svg", width=100)

st.sidebar.markdown(
    """
    <div style="background-color: #003366; padding: 9px; border-radius: 9px; text-align: center;">
        <h3 style="color: white;">🌟 " Welcome to the Portal of Sentiment Analysis \" 
        " 🌟</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### Status")
st.sidebar.markdown("<div style='font-size: 24px;'>🟢 Online</div>", unsafe_allow_html=True)

st.sidebar.title("About")
st.sidebar.info("""
    Welcome to the **Sentiment Analysis** web application.
    This tool helps in analyzing sentiments and exploring comment trends interactively.
""")

st.sidebar.header("👩‍💼 Developer Details")
st.sidebar.markdown("""
    <div style='line-height: 1.6;'>
        <strong>Developed by:</strong> <br> Analytics Team (MyGov) <br><br>
        <strong>Contact Us:</strong> <a href='mailto:analytics_team@mygov.in'>@analytics_team_mygov</a>
    </div>
""", unsafe_allow_html=True)

st.sidebar.subheader("🔗 Useful Links")
st.sidebar.markdown("""
    - [Project Documentation](https://www.example.com/documentation)
    - [Source Code](https://www.example.com/source-code)
    - [Report Issue](https://www.example.com/report-issue)
""")

st.sidebar.subheader("📅 Latest Updates")
st.sidebar.markdown("""
    - **Version 1.0**: Initial release with basic sentiment analysis features.
    - **Version 1.1**: Added data visualization tools.
    - **Version 1.2**: Improved user interface and performance.
    - **Version 1.2**: Sentiment Analysis using BERT tokenization.
""")

if __name__ == "__main__":
    main()
