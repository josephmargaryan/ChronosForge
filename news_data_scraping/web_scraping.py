import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


def scrape_yahoo_finance():
    url = "https://finance.yahoo.com/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract relevant financial headlines
    headlines = []
    for item in soup.find_all("h3"):
        headline = item.get_text(strip=True)
        if headline:  # Ensure the headline is not empty
            headlines.append(headline)

    # Create a DataFrame for easy manipulation
    headlines_df = pd.DataFrame(
        {
            "headline": headlines,
            "timestamp": pd.Timestamp.now(),  # Assign current timestamp for now
        }
    )

    return headlines_df


def preprocess_text(text):
    # Clean and preprocess text
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # Scrape headlines
    headlines_df = scrape_yahoo_finance()

    # Preprocess headlines
    headlines_df["cleaned_headline"] = headlines_df["headline"].apply(preprocess_text)

    # Display scraped and cleaned data
    print("Scraped Headlines:")
    print(headlines_df.head())


if __name__ == "__main__":
    main()
