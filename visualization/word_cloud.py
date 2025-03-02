from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from pathlib import Path
from collections import Counter
import re


# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

# root = Path("./data")

# text_list = []

# for case in root.iterdir():
#     prompt_file = case.joinpath("prompts.txt")
#     with open(prompt_file.as_posix(), "r") as file:
#         prompt = file.read().strip()

#     eval_file = case.joinpath("eval_prompts.txt")
#     with open(eval_file.as_posix(), "r") as file:
#         prompts = [prompt.strip() for prompt in file.readlines()]

#     text_list.append(prompt)
#     text_list.extend(prompts)

# all_text = ' '.join(text_list)

# def clean_text(text):
#     text = re.sub(r'[^\w\s]', '', text)
#     text = text.lower()
#     return text

# all_text = clean_text(all_text)

# def extract_verbs(text):
#     words = nltk.word_tokenize(text)
#     pos_tags = nltk.pos_tag(words)
    
#     verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
#     return verbs

# verbs = extract_verbs(all_text)

# verb_freq = Counter(verbs)

# freq_dict = dict(verb_freq)

# del freq_dict["is"]
# del freq_dict["are"]
# del freq_dict["be"]

freq_dict = {
  'walk': 90,
  'hold': 84,
  'throw': 6,
  'stand': 55,
  'participate': 2,
  'drive': 51,
  'compete': 2,
  'transport': 1,
  'race': 8,
  'talk': 7,
  'interact': 6,
  'mix': 1,
  'wear': 27,
  'work': 1,
  'read': 1,
  'grade': 1,
  'drink': 1,
  'take': 8,
  'give': 1,
  'stretch': 1,
  'play': 12,
  'hit': 6,
  'bend': 6,
  'adjust': 6,
  'fix': 6,
  'dance': 36,
  'run': 13,
  'engage': 1,
  'scale': 1,
  'navigate': 15,
  'secure': 1,
  'descend': 1,
  'perform': 20,
  'ride': 109,
  'wind': 2,
  'paraglide': 6,
  'soar': 6,
  'come': 3,
  'rise': 3,
  'swirl': 2,
  'trail': 2,
  'flow': 1,
  'skateboard': 5,
  'push': 3,
  'guide': 1,
  'support': 1,
  'assist': 1,
  'carry': 7,
  'windsurf': 6,
  'box': 1,
  'observe': 6,
  'fence': 1,
  'spar': 1,
  'billow': 1,
  'drift': 1,
  'cycle': 1,
  'negotiate': 1,
  'maneuver': 1,
  'handle': 1,
  'tackle': 1,
  'rest': 1,
  'shade': 1,
  'travel': 6,
  'line': 1,
  'paint': 1,
  'adorn': 1
}

extra_stopwords = {'be', 'is', 'are', 'am', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did'}
stop_words = set(stopwords.words('english')).union(extra_stopwords)

wordcloud = WordCloud(
    background_color='white',
    stopwords=stop_words,
    width=800,
    height=600,
    max_words=100
).generate_from_frequencies(freq_dict)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig("test.jpg")
plt.show()