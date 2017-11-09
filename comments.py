# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import jieba
from wordcloud import WordCloud, ImageColorGenerator
import PIL.Image as Image
# load data
column_names = ['Votes', 'Useful', 'User', 'Watched', 'Score', 'Date', 'Comment']
data = pd.read_csv('./data/comments_clean.csv', header=None, names=column_names, skipinitialspace = True, quotechar = '`')
# set value as string
data['Votes'] = data['Votes'].astype(str)
data['Useful'] = data['Useful'].astype(str)
data['User'] = data['User'].astype(str)
data['Watched'] = data['Watched'].astype(str)
data['Score'] = data['Score'].astype(str)
data['Date'] = data['Date'].astype(str)
data['Comment'] = data['Comment'].astype(str)
# clean up the data with error format
data = data[data['Score'].map(len) == 6]
data = data[data['Score'] != '看过']
data = data[data['Date'].map(len) == 19]
print('rows:', data.shape[0], ', columns: ', data.shape[1]) # count rows of total dataset
# out: ('rows:', 176875, ', columns: ', 7)
print(data['Score'].value_counts())

index = np.arange(5)
score_counts = data['Score'].value_counts()
values = (score_counts[0], score_counts[1], score_counts[2], score_counts[4], score_counts[3])
bar_width = 0.35
plt.figure(figsize=(20, 10))
plt.bar(index, values, bar_width, alpha=0.6, color='rgbym')
plt.xlabel('Score', fontsize=16)  
plt.ylabel('Counts', fontsize=16)
plt.title('Comments Level', fontsize=18)  
plt.xticks(index, ('5-star', '4-star', '3-star', '2-star', '1-star'), fontsize=14, rotation=20)
plt.ylim(0, 90000)
plt.grid()
for idx, value in zip(index, values):
    plt.text(idx, value + 0.1, '%d' % value, ha='center', va='bottom', fontsize=14, color='black')
plt.show()

def segment_words(stars):
    comments = None
    if stars == 'all':
        comments = data['Comment']
    else:
        comments = data[data['Score'] == stars]['Comment']
    comments_list = []
    for comment in comments:
        comment = str(comment).strip().replace('span', '').replace('class', '').replace('emoji', '')
        comment = re.compile('1f\d+\w*|[<>/=]').sub('', comment)
        if (len(comment) > 0):
            comments_list.append(comment)
    text = ''.join(comments_list)
    word_list = jieba.cut(text, cut_all=True)
    words = ' '.join(word_list)
    return words

def plot_word_cloud(words):
    coloring = np.array(Image.open('./data/chinese.jpg'))
    wc = WordCloud(background_color='white', max_words=2000, mask=coloring, max_font_size=60, random_state=42, 
                   font_path='./data/DroidSansFallbackFull.ttf', scale=2).generate(words)
    image_color = ImageColorGenerator(coloring)
    plt.figure(figsize=(32, 16))
    plt.imshow(wc.recolor(color_func=image_color))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    
all_words = segment_words('all')
plot_word_cloud(all_words)
five_start_words = segment_words('力荐')
plot_word_cloud(five_start_words)
four_start_words = segment_words('推荐')
plot_word_cloud(four_start_words)
two_start_words = segment_words('较差')
plot_word_cloud(two_start_words)
one_start_words = segment_words('很差')
plot_word_cloud(one_start_words)
