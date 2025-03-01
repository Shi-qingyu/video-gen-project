from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from pathlib import Path
from collections import Counter
import re

# 下载必要的NLTK数据
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# 定义数据目录
root = Path("./data")

# 初始化文本列表
text_list = []

# 遍历目录并读取文件
for case in root.iterdir():
    prompt_file = case.joinpath("prompts.txt")
    with open(prompt_file.as_posix(), "r") as file:
        prompt = file.read().strip()

    eval_file = case.joinpath("eval_prompts.txt")
    with open(eval_file.as_posix(), "r") as file:
        prompts = [prompt.strip() for prompt in file.readlines()]

    text_list.append(prompt)
    text_list.extend(prompts)

# 将所有文本合并成一个字符串
all_text = ' '.join(text_list)

# 清理文本
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = text.lower()  # 转换为小写
    return text

all_text = clean_text(all_text)

# 提取动词
def extract_verbs(text):
    # 分词并标注词性
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    
    # 提取动词（VB, VBD, VBG, VBN, VBP, VBZ）
    verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
    return verbs

# 提取所有动词
verbs = extract_verbs(all_text)

# 统计词频
verb_freq = Counter(verbs)

# 将词频转换为字典格式，用于WordCloud
freq_dict = dict(verb_freq)

del freq_dict["is"]

# 扩展停用词列表
extra_stopwords = {'be', 'is', 'are', 'am', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did'}
stop_words = set(stopwords.words('english')).union(extra_stopwords)

# 生成词云
wordcloud = WordCloud(
    background_color='white',
    stopwords=stop_words,
    width=800,
    height=400,
    max_words=100
).generate_from_frequencies(freq_dict)  # 使用词频生成词云

# 显示并保存词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 不显示横纵坐标
plt.savefig("test.jpg")  # 保存词云为图片
plt.show()