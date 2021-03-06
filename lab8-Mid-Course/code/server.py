# SJTU EE208
from flask import Flask, redirect, render_template, request, url_for
import SearchFiles_zhCN as _search
import SearchFiles_img as _imgsearch
from bs4 import BeautifulSoup
app = Flask(__name__)
Directory = ""
imgDirectory = ""
Searcher = None
imgSearcher = None
Analyzer = None
imgAnalyzer = None


@app.route('/', methods=['POST', 'GET'])
def index():
    return redirect(url_for('search'))

@app.route('/index', methods=['POST', 'GET'])
def _index():
    return redirect(url_for('search'))
# 都重定向到/search

@app.route('/search', methods=['POST', 'GET'])
def search():
    if request.method == "POST":
        command = request.form['command']
        search_count = request.form['search_count']
        return redirect(url_for('search_results', command=command, search_count=search_count))
    return render_template("search.html")


@app.route('/imgsearch', methods=['POST', 'GET'])
def imgsearch():
    if request.method == "POST":
        command = request.form['command']
        search_count = request.form['search_count']
        return redirect(url_for('imgsearch_results', command=command, search_count=search_count))
    return render_template("imgsearch.html")


@app.route('/search_results', methods=['POST','GET'])
def search_results():
    
    global Searcher,Analyzer
    if request.method == "POST":
        command = request.form['command']
        search_count = request.form['search_count']
        return redirect(url_for('search_results', command=command, search_count=search_count))
    
    command = request.args.get('command')
    search_count = int(request.args.get('search_count'))
    docs, keyword = _search.get_search_res(command=command,search_count=search_count,searcher = Searcher,analyzer = Analyzer)

    if ' ' in keyword:
        keyword = keyword[:keyword.find(' ')]   # 目前使用keyword的第一个单词来显示摘要

    try:
        abstracts = get_abstracts(keyword,docs)  # 将关键词上下文信息添加到docs
    except Exception as e:
        print(e)
        abstracts = {}
    #print(docs)
    print("Search Done!")
    return render_template("search_results.html", command = command,docs = docs,doc_count = len(docs),abstracts = abstracts)


@app.route('/imgsearch_results', methods=['POST','GET'])
def imgsearch_results():
    
    global imgSearcher,imgAnalyzer
    if request.method == "POST":
        command = request.form['command']
        search_count = request.form['search_count']
        return redirect(url_for('imgsearch_results', command=command, search_count=search_count))
        
    
    command = request.args.get('command')
    search_count = int(request.args.get('search_count'))
    docs, keyword = _imgsearch.get_search_res(command=command,search_count=search_count,searcher = imgSearcher,analyzer = imgAnalyzer)

    print("Image Search Done!")
    for doc in docs:
        print(doc['img_url'])
    return render_template("imgsearch_results.html", command = command,docs = docs,doc_count = len(docs))




@app.before_first_request  # 启动时初始化JVM和索引，搜索时只调用搜索函数
def loadIndex():
    global Directory, Searcher, Analyzer, imgDirectory, imgSearcher, imgAnalyzer

    Directory, Searcher, Analyzer = _search.init_search()
    print("Loaded word search Index!")
    imgDirectory, imgSearcher, imgAnalyzer = _imgsearch.init_search()
    print("Loaded img search Index!")


def get_abstracts(keyword,docs):#通过doc中存储的文件地址访问html文件，从中搜索关键词从而获得摘要
    #print("kw:",keyword)
    CONTEXT_RANGE = [40,40]
    HTML_FOLDER_PATH = ""
    abstracts = {}
    for doc in docs:
        try:
            #print(doc['path'])
            with open(HTML_FOLDER_PATH + doc['path'],'r') as file:
                filetext = clean_html(file.read())
                pos = filetext.find(keyword)
                if pos != -1:
                    abstract = filetext[pos - CONTEXT_RANGE[0] : pos + CONTEXT_RANGE[1]].strip()
                    if len(abstract) > 10:
                        abstract = "..." + abstract + "..."

                    abstracts[doc['path']] = abstract
                    #print(abstract)
                    #print('---------------')
                else:
                    abstracts[doc['path']] = ""
        except:
            abstracts[doc['path']] = ""
            

    return abstracts
           
def clean_html(content):
    soup = BeautifulSoup(content,features="html.parser")
    for script in soup(["script", "style"]):   # 去除javascript https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python
        script.extract()
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # drop blank lines
    text = '\n'.join(line for line in lines if line)
    return text         

if __name__ == '__main__':
    print("running flask app")
    app.run(debug=True, port=8080)
    
