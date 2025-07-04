# Twitter-Sentiment-Analysis

**Google Colab link:** https://colab.research.google.com/drive/1hADA9byUK2AXHSouetkDIX6VStdotMha

** Dataset Link:** https://www.kaggle.com/datasets/kazanova/sentiment140

This notebook performs sentiment analysis on a dataset of tweets. The goal is to classify tweets as either positive, negative, or neutral.

## Data

The dataset used is `training.1600000.processed.noemoticon.csv`, which contains 1.6 million tweets. The original data includes the following columns:

- `label`: The sentiment of the tweet (0: Negative, 2: Neutral, 4: Positive)
- `id`: The tweet ID
- `date`: The date and time of the tweet
- `query`: The query used to collect the tweets (not relevant for sentiment analysis)
- `user`: The Twitter user who posted the tweet
- `tweet`: The text of the tweet

For this analysis, the `id`, `date`, `query`, and `user` columns are dropped, keeping only the `label` and `tweet`. The labels are also converted to more descriptive strings: 'Negative', 'Neutral', and 'Positive'.

## Notebook Steps

The notebook follows these main steps:

1.  **Data Loading and Reduction**: Loads the dataset and removes irrelevant columns.
2.  **Label Conversion**: Converts the numerical sentiment labels to descriptive strings.
3.  **Data Distribution Analysis**: Visualizes the distribution of sentiment labels in the dataset.
4.  **Preprocessing**: Cleans the tweet text by removing mentions, URLs, and punctuation, converting to lowercase, and removing stop words. Optional stemming is also included but commented out.
5.  **Tweet Length Analysis**: Analyzes the distribution of tweet lengths in characters and words after preprocessing.
6.  **Letter Frequency Analysis**: Compares the frequency of letters in the tweets to the expected frequency of letters in English text using a Chi-square test and Pearson correlation.
7.  **Most Common Words**: Identifies and visualizes the most common words in the entire dataset, as well as separately for positive and negative tweets. Word clouds are also generated for positive and negative tweets.
8.  **Training and Test Data Splitting**: Splits the data into training and testing sets for model development (although no model training is performed in this notebook).
9.  **Tokenization**: Converts the tweet text into numerical sequences using Keras Tokenizer.
10. **GLOVE Embedding**: Loads pre-trained GLOVE word embeddings and creates an embedding matrix for the vocabulary.

## How to Run the Notebook

1.  **Clone the repository**: If this notebook is part of a repository, clone it to your local machine or Google Drive.
2.  **Upload data**: Ensure the `training.1600000.processed.noemoticon.csv` file is in a directory named `data` in the same location as the notebook. You will also need the `letter_frequency_en_US.csv` file in the same `data` directory and the `glove.6B.300d.txt` file in a directory named `glove`.
3.  **Open in Google Colab**: Open the notebook in Google Colab.
4.  **Run all cells**: Execute all the code cells in the notebook sequentially. You can do this by clicking "Runtime" -> "Run all" in the Colab menu.
5.  **Review the outputs**: Examine the outputs of the cells, including the data visualizations and statistical analysis results.

## Dependencies

The notebook requires the following libraries:

-   `numpy`
-   `re`
-   `pandas`
-   `matplotlib`
-   `sklearn`
-   `nltk`
-   `collections`
-   `wordcloud`
-   `tensorflow` (with Keras)
-   `scipy`

These libraries are generally available in Google Colab environments. If running locally, ensure you have them installed (`pip install ...`). You will also need to download the NLTK stopwords (`nltk.download('stopwords')`).
