import numpy as np
import pandas as pd
import re
import nltk
from collections import Counter
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from Emoticons import Emoticons
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
import time

pd.options.mode.chained_assignment = None # NOTE: What is that?


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')



class Preprocessing:
  
  def __init__(self):
    self.cnt = Counter()
    self.stemmer = PorterStemmer()
    self.lemmatizer = WordNetLemmatizer()
    self.spell = SpellChecker()
    self.n_rare_words = 10
    self.chat_words_map_dict = {}
    self.chat_words_list = []
    self.PUNCT_TO_REMOVE = string.punctuation
    self.wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

  def remove_punctuation(self, text):
    return text.translate(str.maketrans('', '', self.PUNCT_TO_REMOVE))

  def remove_stopwords(self, text):
    self.STOPWORDS = set(stopwords.words('english'))
    words = text.split()
    return " ".join([word for word in words if word not in self.STOPWORDS])
  
  def count_word_frequencies(self, df):
    all_words = ' '.join(df['text_wo_stop']).split()
    self.cnt = Counter(all_words)

  def remove_freqwords(self, text):
    if not self.cnt:
        print("ERROR: Counter is empty before removing frequent words!")
    
    self.FREQWORDS = set(word for word, count in self.cnt.most_common(10))
    words = text.split()
    return " ".join([word for word in words if word not in self.FREQWORDS])

  def remove_rarewords(self, text):
    # self.RAREWORDS = set(self.cnt[self.cnt <= self.n_rare_words].index) 
    self.RAREWORDS = set(word for word, count in self.cnt.items() if count <= self.n_rare_words)
    words = text.split()
    return " ".join([word for word in words if word not in self.RAREWORDS])

  def stem_words(self, text):
    return ' '.join([self.stemmer.stem(word) for word in text.split()])

  def lemmatize_words(self, text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([self.lemmatizer.lemmatize(word, self.wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

  # Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
  def remove_emoji(self, string):
      emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
      return emoji_pattern.sub(r'', string)

  def remove_emoticons(self, text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in Emoticons.EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

  def convert_emoticons(self, text):
    emoticon_dict = Emoticons.EMOTICONS  # Sözlüğü al
    emoticon_pattern = re.compile("|".join(re.escape(k) for k in emoticon_dict.keys()))  # Tek seferde regex hazırla

    return emoticon_pattern.sub(lambda m: "_".join(emoticon_dict[m.group()].replace(",", "").split()), text)

  def convert_emojis(self, text):
    emoji_dict = Emoticons.UNICODE_EMO  # Sözlüğü al
    emoji_pattern = re.compile("|".join(re.escape(k) for k in emoji_dict.keys()))  # Tek seferde regex hazırla
    
    return emoji_pattern.sub(lambda m: "_".join(emoji_dict[m.group()].replace(",", "").replace(":", "").split()), text)

  def remove_urls(self, text):
      url_pattern = re.compile(r'https?://\S+|www\.\S+')
      return url_pattern.sub(r'', text)

  def remove_html(self, text):
      html_pattern = re.compile('<.*?>')
      return html_pattern.sub(r'', text)

  def remove_html_with_bs4(self, text):
    return BeautifulSoup(text, 'lxml').text

  def chat_words_pre(self):
    for line in Emoticons.chat_words_str.split('\n'):
      line = line.strip()
      if line and '=' in line:
        splitted_line = line.split('=', 1)
        cw = splitted_line[0].strip()
        cw_expanded = splitted_line[1].strip()
        self.chat_words_list.append(cw)
        self.chat_words_map_dict[cw] = cw_expanded

    self.chat_words_list = set(self.chat_words_list)
    
  def chat_words_conversion(self, text):
    new_text = []
    for w in text.split():
      if w.upper() in self.chat_words_list:
        new_text.append(self.chat_words_map_dict[w.upper()])
      else:
        new_text.append(w)
    return ' '.join(new_text)
  
  
  # For email
  def count_word_frequencies_for_email(self, email):
    all_words = ' '.join(email).split()
    self.cnt = Counter(all_words)

  # def correct_spellings(self, text):
  #   corrected_text = []
  #   words = text.split()
  #   misspelled_words = spell.unknown(words)
    
  #   # Process only misspelled words
  #   for word in words:
  #       corrected_word = spell.correction(word) if word in misspelled_words else word
  #       # Ensure it's not None before appending
  #       corrected_text.append(str(corrected_word) if corrected_word is not None else word)
    
  #   return " ".join(corrected_text)


class Timer:
  def __init__(self, name="Operation"):
    self.name = name

  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.end = time.time()
    self.interval = self.end - self.start
    print(f"{self.name} took {self.interval:.2f} seconds")


def main():
  full_df = pd.read_csv(r'c:\Users\Victus\Desktop\AI Email Assistant\data\twcs\low_count_sample_twcs.csv')

  # full_df = pd.read_csv('/content/email_response_assistant/model_and_nlp_staff/low_count_twcs.csv')
  
  df = full_df[['text']]
  df['text'] = df['text'].astype(str)
  
  preprocessor = Preprocessing()
  
  with Timer("Removing punctuation"):
    df['text_wo_punct'] = df['text'].apply(lambda text: preprocessor.remove_punctuation(text))
  
  with Timer("Removing stopwords"):
    df['text_wo_stop'] = df['text_wo_punct'].apply(lambda text: preprocessor.remove_stopwords(text))
  
  with Timer("Counting word frequencies"):
    preprocessor.count_word_frequencies(df)

  with Timer("Removing frequent words"):
    df['text_wo_stopfreq'] = df['text_wo_stop'].apply(lambda text: preprocessor.remove_freqwords(text))
  
  with Timer("Removing rare words"):
    df['text_wo_stopfreqrare'] = df['text_wo_stopfreq'].apply(lambda text: preprocessor.remove_rarewords(text))
  
  df.drop(['text_wo_punct', 'text_wo_stop', 'text_wo_stopfreq'], axis=1, inplace=True)
  
  with Timer("Stemming words"):
    df['text_wo_stemmed'] = df['text_wo_stopfreqrare'].apply(lambda text: preprocessor.stem_words(text))
  
  with Timer("Lemmatizing words"):
    df['text_wo_lemmatized'] = df['text_wo_stemmed'].apply(lambda text: preprocessor.lemmatize_words(text))
  
  with Timer("Removing emojis"):
    df['text_wo_lemmatized_remove_emoji'] = df['text_wo_lemmatized'].apply(lambda text: preprocessor.remove_emoji(text))
  
  with Timer("Removing emoticons"):
    df['text_wo_lemmatized_remove_emoticons'] = df['text_wo_lemmatized_remove_emoji'].apply(lambda text: preprocessor.remove_emoticons(text))
  
  with Timer("Converting emojis"):
    df['text_wo_lemmatized_convert_emojis'] = df['text_wo_lemmatized_remove_emoticons'].apply(lambda text: preprocessor.convert_emojis(text))
  
  with Timer("Converting emoticons"):
    df['text_wo_lemmatized_convert_emoticons'] = df['text_wo_lemmatized_convert_emojis'].apply(lambda text: preprocessor.convert_emoticons(text))
  
  with Timer("Removing URLs"):
    df['text_wo_urls'] = df['text_wo_lemmatized_convert_emoticons'].apply(lambda text: preprocessor.remove_urls(text))
  
  with Timer("Removing HTML"):
    df['text_wo_html'] = df['text_wo_urls'].apply(lambda text: preprocessor.remove_html(text))
  
  with Timer("Preparing chat words"):
    preprocessor.chat_words_pre()
  
  with Timer("Converting chat words"):
    df['text_wo_chat_words'] = df['text_wo_html'].apply(lambda text: preprocessor.chat_words_conversion(text))
  
  full_df['cleaned_text'] = df['text_wo_chat_words']

  df.drop(['text_wo_stopfreqrare',
           'text_wo_stemmed',
           'text_wo_chat_words',
           'text_wo_lemmatized', 
           'text_wo_lemmatized_remove_emoji', 
           'text_wo_lemmatized_remove_emoticons', 
           'text_wo_lemmatized_convert_emojis', 
           'text_wo_lemmatized_convert_emoticons', 
           'text_wo_urls', 
           'text_wo_html'], axis=1, inplace=True)
  
  
  print(full_df.head()['cleaned_text'])
  
  
  full_df.to_csv(r'c:\Users\Victus\Desktop\AI Email Assistant\data\twcs\cleaned_twcs.csv', index=False)





## Email Preprocessing
import joblib
vectorizer = joblib.load(r'C:\Users\Victus\Desktop\AI Email Assistant\models\tfidf_vectorizer.pkl')


# Tokenization
def tokenize(text):
    return nltk.word_tokenize(text)

# Preprocess email
def preprocess_email(email):
  preprocessor = Preprocessing()
  
  cleaned_email = preprocessor.remove_punctuation(email)
  cleaned_email = preprocessor.remove_stopwords(cleaned_email)
  preprocessor.count_word_frequencies_for_email(cleaned_email)
  cleaned_email = preprocessor.remove_freqwords(cleaned_email)
  cleaned_email = preprocessor.remove_rarewords(cleaned_email)
  cleaned_email = preprocessor.stem_words(cleaned_email)
  cleaned_email = preprocessor.lemmatize_words(cleaned_email)
  cleaned_email = preprocessor.remove_emoji(cleaned_email)
  cleaned_email = preprocessor.remove_emoticons(cleaned_email)
  cleaned_email = preprocessor.convert_emojis(cleaned_email)
  cleaned_email = preprocessor.convert_emoticons(cleaned_email)
  cleaned_email = preprocessor.remove_urls(cleaned_email)
  cleaned_email = preprocessor.remove_html(cleaned_email)
  cleaned_email = preprocessor.chat_words_conversion(cleaned_email)
  cleaned_email_tokenized = " ".join(tokenize(cleaned_email))
  cleaned_email_vectorized = vectorizer.transform([cleaned_email_tokenized])
  
  return cleaned_email_vectorized




if __name__ == "__main__":
  main()
