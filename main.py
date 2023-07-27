import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
from flask import Flask , request , render_template ,json, redirect,url_for
import numpy as np
import nltk
import spacy
from sense2vec import Sense2Vec
import re
from nltk.corpus import stopwords
from keybert import KeyBERT
from gensim.models import Word2Vec
import gensim.downloader as api 
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

app = Flask(__name__,template_folder='templates')
nlp = spacy.load('en_core_web_lg')
#nlp = spacy.load(r'C:\Users\vishal.jha\Anaconda3\Lib\site-packages\en_core_web_lg\en_core_web_lg-3.5.0')

sw = stopwords.words('english')
s2v = Sense2Vec().from_disk('s2v_old')

# Download the word2vec model 
def google_word2vec():
    model_name = "word2vec-google-news-300" 
    model = api.load(model_name) 
    return model

kw = KeyBERT()
def pre_process(file_text):
    
    file_text = file_text.replace("-",'')
    file_text = file_text.replace(",",'')
    file_text = re.sub('[^a-zA-Z0-9.]',' ',file_text)
    file_text = " ".join([word for word in file_text.split()])
    
    return file_text

def pre_process_vocab(file_text):
    file_text = file_text.replace("-",'')
    file_text = file_text.replace(",",'')
    file_text = re.sub('[^a-zA-Z0-9.]',' ',file_text)
    file_text = " ".join([word for word in file_text.split() if word not in sw])
    
    return file_text

def capitalize_word(arr):
    n_arr = []
    for i in range(len(arr)):
        v = []
        for j in range(len(arr[i])):
            ae = ''
            doc = nlp(arr[i][j])
            s = arr[i][j].split(' ')
            for k in doc.ents:
                k = str(k).split()
                for l in k:
                    for m in range(len(s)):
                        if l==s[m]:
                            s[m] = s[m].capitalize()
            
            ae+=" ".join(s) 
            v.append(ae)
        n_arr.append(v)
    return n_arr

def sense2vec_dis(word,s2v):
    distractors = []
    #if(type(word)==str):
    word = word.lower()
    word = word.replace(" ","_")

    sense = s2v.get_best_sense(word)
    if sense is None:
        return ["None of these"]
        
    most_similar = s2v.most_similar(sense , n=10)

    if(most_similar is None):
        return "Not found any matching words!"
    
    for i in most_similar:
        w = i[0].split("|")[0].replace("_"," ")
        if w.lower() not in word and w.lower()!=word:
            distractors.append(w.title())
    return list(set(distractors))


@app.route('/question-generator-api')
def home():
    return render_template('upload.html')


@app.route('/question-generator-api', methods=['POST','GET'])
def upload_file():
    global text
    global pre_text
    global num
    text = ''
    pre_text = ''
    if request.method=='POST':
        tex = ''
        n = request.form.get("num")
        ques_type = request.form["answer"]
        num = int(n)
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
          
            with open(uploaded_file.filename,encoding="utf8") as f:
                r = f.readlines()

            for i in r:
                tex+="".join(i.replace("\n",''))  
            text = pre_process(tex) 
            pre_text = pre_process_vocab(tex)
          
            if(ques_type == "long"):
                return redirect(url_for('genQues'))   
            elif(ques_type == "bool"):
                return redirect(url_for('genBoolQues')) 
            elif(ques_type=="mcq"):
                return redirect(url_for('genMCQQues')) 
    else:
        return redirect(url_for('home'))

def break_sentence(file_text,arr): #o_text
    s = ''
    if len(file_text.split()) <= 500:
#         print(text)
        arr.append(file_text)
        #print(o_text)
    
    else:
        s+=" ".join(file_text.split()[:500])
        last_fullstop = s.rfind('.')

        if last_fullstop == -1:  # If no full stop found, break at 300th character
            last_fullstop = len(file_text.split())
        arr.append(s[:last_fullstop])
        break_sentence(file_text[last_fullstop+1:],arr)
    return arr




summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def postprocesstext (content):
    final=""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final +" "+sent
    return final


def summarizer(file_text,model,tokenizer):
    file_text = file_text.strip().replace("\n"," ")
    file_text = "summarize: "+file_text
  # print (text)
    max_len = 512
    encoding = tokenizer.encode_plus(file_text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=500)


    dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary= summary.strip()

    return summary


def Text_with_stopwords(file_text):
    o_text = []
    t = file_text
    o_text = break_sentence(t,o_text)
    return o_text


def Text_without_stopwords(file_text):
    p_text = []
    #print(pre_text)
    t = file_text
    p_text = break_sentence(t,p_text)
    return p_text

def create_token(file_text):
    token = []
    text = Text_without_stopwords(file_text)
    for j in text:
        tok = []
        t = j.split()
        for k in t:
            tok.append(k)
        token.append(tok)
    return token


def get_summarized_text(file_text):
    summarized_text=[]
    o_text = Text_with_stopwords(file_text)   
    for i in o_text:
        summarized_text.append(summarizer(i,summary_model,summary_tokenizer))

    return summarized_text

def get_keywords(file_text):
    keys_words = []
    summarized_text = get_summarized_text(file_text)
    for i in summarized_text:
        word = []
        keywords = kw.extract_keywords(i,top_n = 5,use_mmr=True)
        for j in keywords:
            word.append(j[0].capitalize())
        keys_words.append(word)

    return keys_words

 
def word2vec_google(model,word):
    words = []
    try:
        dis = model.most_similar(word, topn=3)     
        for i in dis:
            i = list(i)
            i[0] = " ".join([word for word in i[0].split('_')])
            words.append(i[0].title()) 
        
    except Exception:
        words.append("None")

    return words

def create_distractors(model,w2v_model,file_text):
    
    option = []
    difficulty = []
    keys_words =get_keywords(file_text)
    for i in range(len(keys_words)):
        ans = []
        diffi = []
        for j in keys_words[i]:
            op = sense2vec_dis(j,s2v)
            if(len(op)>=3):
                ans.append(op[:3])
                diffi.append("Hard")
            else:
                op = word2vec_google(model,j)
                if(op!=['None']):
                    ans.append(op)
                    diffi.append("Moderate")
                else:
                    try:
                        r = []
                        op = w2v_model.wv.most_similar(j.lower(),topn = 3)
                        for i in op:
                            i = list(i)
                            i[0] = " ".join([word for word in i[0].split('_')])
                            r.append(i[0].title())
                        ans.append(r)
                        diffi.append("Easy")
                    except Exception:
                        ans.append(['None'])
        option.append(ans)
        difficulty.append(diffi)
    return option , difficulty

def long_answer():
   
    question_model = T5ForConditionalGeneration.from_pretrained('vishal2014/updated_t5_squad_long_vam')
    question_tokenizer = T5Tokenizer.from_pretrained('vishal2014/updated_t5_squad_long_vam')
    question_model = question_model.to(device)
    return question_model , question_tokenizer

def bool_ans():
 
    question_model = T5ForConditionalGeneration.from_pretrained('vishal2014/t5_boolean_gen')
    question_tokenizer = T5Tokenizer.from_pretrained('vishal2014/t5_boolean_gen')
    question_model = question_model.to(device)
    return question_model , question_tokenizer

def bool_ques_ans():
    ans_model = T5ForConditionalGeneration.from_pretrained('vishal2014/bool_ans_vam')
    ans_tokenizer = T5Tokenizer.from_pretrained('vishal2014/bool_ans_vam')
    ans_model = ans_model.to(device)
    return ans_model , ans_tokenizer

def mcq_ans():
  
    question_model = T5ForConditionalGeneration.from_pretrained('vishal2014/updated_t5_squad_mcq_vam')
    question_tokenizer = T5Tokenizer.from_pretrained('vishal2014/updated_t5_squad_mcq_vam') 
    question_model = question_model.to(device)
    return question_model , question_tokenizer

def get_question(context,answer,model,tokenizer,n):

    text = "context: {} answer: {}".format(context,answer)
    encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=30,
                                  num_return_sequences=n,
                                  no_repeat_ngram_size=2,
                                  max_length=72)


    dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
    #return dec
    c = 0
    l = []    
    for i in range(len(dec)):
        c+=1
        Question = dec[i].replace('question: ','')
        l.append(Question.capitalize())
        Question = dec[i].replace("question: ","Question: "+str(c)+" ")
        Question= Question.strip()

    return l

def get_answer(context,question,model,tokenizer):
    text = "context: {} question: {}".format(context,question)
    encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=3,
                                  no_repeat_ngram_size=2,
                                  max_length=72)


    dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
    
    c = 0
    l = []
    for i in range(len(dec)):
        c+=1
        Answer = dec[i].replace('answer: ','')
        l.append(Answer)
        Answer = dec[i].replace("answer: ","Answer: "+str(c)+" ")
        Answer= Answer.strip()
        
    return l
        

@app.route("/question-generator-api/genQuestions",methods = ["GET", "POST"])
def genQues():
    question_model , question_tokenizer = long_answer()
    n = num
    t = text
    o_text = Text_with_stopwords(t)
    summarized_text=get_summarized_text(t)
    arr = []
    for i in range(len(o_text)):
        a = get_question(o_text[i],summarized_text[i],question_model,question_tokenizer,n)
        arr.append(a)
    
    new_ques = capitalize_word(arr)

    if request.method == 'POST':
        return new_ques
    else:
        questions = new_ques
        return render_template('LongQA.html', questions=questions)


@app.route("/question-generator-api/genBoolQuestions",methods = ["GET", "POST"])
def genBoolQues():
    n = num
    t = text
    question_model , question_tokenizer = bool_ans()
    ans_model , ans_tokenizer = bool_ques_ans()
    o_text = Text_with_stopwords(t)
    
    ans = ''
   
    arr = []
    for i in range(len(o_text)):
        a = get_question(o_text[i],ans,question_model,question_tokenizer,n)
        arr.append(a)

    new_ques = capitalize_word(arr)
    answer= []
    for i in range(len(arr)):
        ar = []
        for j in range(len(arr[i])):
            s = ''
            a = get_answer(o_text[i],arr[i][j],ans_model,ans_tokenizer)
        
            s = "".join(new_ques[i][j]) 
            s+=","
            s+=a[j]
            ar.append(s)

        answer.append(ar)
    data = []
    for i in range(len(answer)):

        questionitem = answer[i]

        for j in range(len(questionitem)):
            question = questionitem[j].split(',')

            dicti = {}
    
            dicti['question'] = question[0]
            question[1] = question[1].capitalize()
            dicti['ans']=question[1]

            data.append(dicti)
    print(data)

    if request.method == 'POST':
        return data
    else:
        return render_template('boolean.html', data=data)
    

@app.route("/question-generator-api/genMCQQuestions",methods = ["GET", "POST"])
def genMCQQues():
    question_model , question_tokenizer = mcq_ans()
    n = num
    t = text
    summarized_text=get_summarized_text(t)

    keys_words = get_keywords(t)
    vocab = create_token(t)
    model = google_word2vec()
    w2v_model = Word2Vec(vocab,min_count=1)
    distractors , difficulty = create_distractors(model,w2v_model,t)

    ques = []
    for i in range(len(keys_words)):
        arr = []
        for j in range(len(keys_words[i])):
            f = get_question(summarized_text[i],keys_words[i][j],question_model,question_tokenizer,n)
            arr.append(f)
        ques.append(arr)
    
    data = []
    for i in range(len(keys_words)):
        answeritem=keys_words[i]
        questionitem = ques[i]
        distractoritem=distractors[i]
        difficultyitem = difficulty[i]
        for j in range(len(answeritem)):
            dicti = {}
            if((len(distractoritem[j])<3) and (len(distractoritem[j])!=3)):
                continue
            else:
                dicti['question'] = questionitem[j]
                dicti['answer']=str(answeritem[j])
                dicti['options']=distractoritem[j]
                dicti['difficulty'] = difficultyitem[j]
                data.append(dicti)
 
    for i in data: 

        i['options'].append(str(i['answer']))
        i['options'] = random.sample(i['options'],len(i['options']))

    if request.method == 'POST':
        return ques
    else:
        return render_template('MCQ.html', data=data)
    
@app.route('/question-generator-api/generateQuestion',methods = ['POST'])   
def genQuestion():
    id = 0
    response = {}
    m_ques = []
    context = json.loads(request.data)['context']
    summary = summarizer(context,summary_model,summary_tokenizer)
    question_model , question_tokenizer = long_answer()
    ques = get_question(context,summary,question_model,question_tokenizer,2)

    for i in ques:
        m_ques.append(i)
    #m_ques = [m_ques]
    data = []
    for i in m_ques:
        d = {}
        d['id'] = id+1
        d['question'] = i
        d['answer'] = summary
        id+= 1
        data.append(d)
        
    response['data'] = data

    return json.dumps(response)

@app.route('/question-generator-api/generateBooleanQuestion',methods = ['POST'])   
def genBooleanQuestion():
    response = {}
    context = json.loads(request.data)['context']
    question_model , question_tokenizer = bool_ans()
    ans_model , ans_tokenizer = bool_ques_ans()
    o_text = Text_with_stopwords(context)
    id = 0
    ans = ''
    arr = []
    for i in range(len(o_text)):
        a = get_question(o_text[i],ans,question_model,question_tokenizer,2)
        arr.append(a)
    new_ques = capitalize_word(arr)
    answer= []
    for i in range(len(arr)):
        ar = []
        for j in range(len(arr[i])):
            s = ''
            a = get_answer(o_text[i],arr[i][j],ans_model,ans_tokenizer)
        
            s = "".join(new_ques[i][j]) 
            s+=","
            s+=a[j]
            ar.append(s)

        answer.append(ar)
    data = []
    for i in range(len(answer)):
        questionitem = answer[i]
        dicti = {}
        for j in range(len(questionitem)):
            question = questionitem[j].split(',')
            dicti['id'] = id+1
            dicti['question'] = question[0]
            question[1] = question[1].capitalize()
            dicti['ans']=question[1]
            data.append(dicti)
            id+=1
    response['data'] = data
    return json.dumps(response)

@app.route('/question-generator-api/generateMCQQuestion',methods = ['POST'])   
def genMcqQuestion():
    response = {}
    id = 0
    context = json.loads(request.data)['context']
    question_model , question_tokenizer = mcq_ans()
   
    summarized_text=get_summarized_text(context)
    keys_words = get_keywords(context)
    vocab = create_token(context)
    model = google_word2vec()
    w2v_model = Word2Vec(vocab,min_count=1)
    distractors , difficulty = create_distractors(model,w2v_model,context)

    ques = []
    for i in range(len(keys_words)):
        arr = []
        for j in range(len(keys_words[i])):
            f = get_question(summarized_text[i],keys_words[i][j],question_model,question_tokenizer,1)
            arr.append(f)
        ques.append(arr)
    
    data = []
    for i in range(len(keys_words)):
        answeritem=keys_words[i]
        questionitem = ques[i]
        distractoritem=distractors[i]
        difficultyitem = difficulty[i]

        for j in range(len(answeritem)):
            dicti = {}
            if((len(distractoritem[j])<3) and (len(distractoritem[j])!=3)):
                continue
            else:
                dicti['id'] = id+1
                dicti['question'] = questionitem[j][0]
                dicti['answer']=str(answeritem[j])
                dicti['options']=distractoritem[j]
                dicti['difficulty'] = difficultyitem[j]
                id+=1
                data.append(dicti)
    for i in data: 

        i['options'].append(str(i['answer']))
        i['options'] = random.sample(i['options'],len(i['options']))
    response['Data'] = data

    return json.dumps(response)

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port=5002,debug=True)