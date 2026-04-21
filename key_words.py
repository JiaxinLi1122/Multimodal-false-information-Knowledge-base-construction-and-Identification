import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import sqlite3
import json
import nltk
nltk.download('wordnet')

df = pd.read_csv('./data/twitter/test_posts.csv')
text_column = df['post_text']
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
all_keywords = []

for text in text_column:
    # 分词和词形还原
    words = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text) 
             if word.isalpha() and word.lower() not in stop_words]
    
    # 计算词频并提取关键词
    fdist = FreqDist(words)
    keywords = [word for word, freq in fdist.most_common(10)]
    all_keywords.append(keywords)

for i in range(10):
    print(all_keywords[i])

conn=sqlite3.connect('./data/test.db')
cur=conn.cursor()
def get_pop_kw(cnt):
    sqltxt='SELECT kwtext,COUNT(*) as cnt from news_kw group by kwtext ORDER by cnt DESC limit %d;'%(cnt,)
    cur.execute(sqltxt)
    return cur.fetchall()

def make_res(cnt):
    res=get_pop_kw(cnt)
    ret={
        'count':cnt,
        'res':[]
    }
    for i in res:
        ret['res'].append({'kw':i[0],'times':i[1]})
    print(ret)
    jsonres=json.dumps(ret).encode()
    open('./data/kw_res.json','wb').write(jsonres)

make_res(40)