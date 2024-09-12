import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained BERT model for sentiment analysis
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Create a sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, batch_size=32)

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers

    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to analyze sentiment in batches
def analyze_sentiments(comments):
    sentiments = []
    try:
        for comment in comments:
            if re.match(r'^[^A-Za-z]+$', comment) or len(comment.strip()) == 0:
                sentiments.append('neutral')  # Assign neutral to non-understandable comments
            else:
                result = nlp(comment)
                sentiment_label = result[0]['label']
                # Mapping BERT sentiment labels to custom categories
                if sentiment_label in ['1 star', '2 stars']:
                    sentiments.append('negative')
                elif sentiment_label == '3 stars':
                    sentiments.append('neutral')
                else:
                    sentiments.append('positive')
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
    return sentiments

# Streamlit app
def main():
    st.title("Enhanced Sentiment Analysis of Comments using BERT")

    # File uploader for multiple CSV files
    uploaded_files = st.file_uploader("Upload CSV files with comments", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Read the uploaded CSV file
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

                # Display the column selection option
                st.subheader(f"File: {uploaded_file.name}")
                column_name = st.selectbox(f"Select the column containing comments in {uploaded_file.name}", df.columns)

                # Input Data Preview
                st.subheader(f"Input Data Preview of {uploaded_file.name} (First 5 Rows)")
                st.write(df[[column_name]].head())

                # Ensure all comments are strings and handle NaN values
                df[column_name] = df[column_name].astype(str).fillna('')

                # Preprocess comments
                df['Processed_Comment'] = df[column_name].apply(preprocess_text)

                # Apply sentiment analysis in batches
                st.info(f"Analyzing sentiments for {uploaded_file.name}...")
                sentiments = analyze_sentiments(df['Processed_Comment'].tolist())
                df['Sentiment'] = sentiments

                # Output Data Preview with Scrollable Table
                st.subheader(f"Scrollable Output Data Preview for {uploaded_file.name} (Sentiment Analysis Results)")
                st.dataframe(df[[column_name, 'Sentiment']].head(100), height=300)

                # Display sentiment results
                st.subheader(f"Sentiment Analysis Results for {uploaded_file.name}")
                st.dataframe(df[[column_name, 'Sentiment']])  # Scrollable table for entire dataset

                # Calculate percentages
                sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
                sentiment_counts = sentiment_counts.reset_index()
                sentiment_counts.columns = ['Sentiment Category', 'Percentage']

                # Plot the sentiment analysis results
                st.write(f"Sentiment Analysis Distribution for {uploaded_file.name}")
                plt.figure(figsize=(10, 5))
                sns.barplot(x='Sentiment Category', y='Percentage', data=sentiment_counts, palette='viridis')

                # Annotate bars with percentage values
                for index, row in sentiment_counts.iterrows():
                    plt.text(index, row.Percentage + 1, f'{row.Percentage:.2f}%', color='black', ha='center')

                plt.title(f'Sentiment Analysis of {uploaded_file.name} using BERT')
                plt.xlabel('Sentiment Category')
                plt.ylabel('Percentage')
                plt.ylim(0, 100)  # Ensure y-axis goes to 100% for clarity
                st.pyplot(plt)

            except Exception as e:
                st.error(f"An error occurred while processing {uploaded_file.name}: {e}")

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
