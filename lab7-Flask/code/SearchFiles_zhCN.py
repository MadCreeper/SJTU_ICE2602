# SJTU EE208

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import BooleanClause
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.util import Version
import jieba
import paddle
paddle.enable_static()
"""
This script is loosely based on the Lucene (java implementation) demo class 
org.apache.lucene.demo.SearchFiles.  It will prompt for a search query, then it
will search the Lucene index in the current directory called 'index' for the
search query entered against the 'contents' field.  It will then display the
'path' and 'name' fields for each of the hits it finds in the index.  Note that
search.close() is currently commented out because it causes a stack overflow in
some cases.
"""
def parseCommand(command,options_dict): #将指令中的每个冒号指令提取出来,返回dict
    '''
    input: C title:T author:A language:L
    output: {'contents':C, 'title':T, 'author':A, 'language':L}

    Sample:
    input:'contenance title:henri language:french author:william shakespeare'
    output:{'author': ' william shakespeare',
                   'language': ' french',
                   'contents': ' contenance',
                   'title': ' henri'}
    '''
    allowed_opt = options_dict.keys()
    command_dict = {}
    opt = 'contents'
    for i in command.split(' '):
        if ':' in i:
            opt, value = i.split(':')[:2]
            opt = opt.lower()
            if opt in allowed_opt and value != '':
                command_dict[opt] = (command_dict.get(opt, '') + ' ' + value).strip()
        else:
            command_dict[opt] = (command_dict.get(opt, '') + ' ' + i).strip()
    return command_dict
def printScoreDocs(scoreDocs):
    print("%s total matching documents." % len(scoreDocs))


    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
    
        print ('path:', doc.get("path"), '\nfilename:', doc.get("name"), '\nscore:', scoreDoc.score)
        print("URL:",doc.get("url"),"\ndomain:",doc.get("site"),"\ntitle:",doc.get("title").strip())
        print("-"*50)
    

def create_query_combined(command,analyzer,options_dict,tokenized = True):  #  将多种query通过options_dict的布尔要求创建组合的query
    command_dict = parseCommand(command,options_dict)
    if 'contents' in command_dict and tokenized:
        command_dict['contents'] = ' '.join(jieba.cut_for_search(command_dict['contents']))
    print(command_dict,options_dict)
    querys = BooleanQuery.Builder()
    for k,v in command_dict.items():    # k: field  v: text input
        query = QueryParser(k, analyzer).parse(v)
        querys.add(query, options_dict[k] if k in options_dict else BooleanClause.Occur.MUST)  # 允许对不同的query指定不同的option(MUST/SHOULD/...)
    return querys

def run(searcher, analyzer,command = ""):
    # while True:
    if not command:
        print()
        print ("Hit enter with no input to quit.")
        command = input("Query:")

    if command == '':
        return
    #command = " ".join(jieba.cut_for_search(command))   
    print()
    
    MUST = BooleanClause.Occur.MUST
    SHOULD = BooleanClause.Occur.SHOULD
    querys = create_query_combined(command,analyzer,options_dict = {'site':MUST})
    
    """
    command_dict = parseCommand(command)
    print(command_dict)
    querys = BooleanQuery.Builder()
    for k,v in command_dict.items():
        query = QueryParser(k, analyzer).parse(v)
        querys.add(query, BooleanClause.Occur.MUST)
    """

    scoreDocs = searcher.search(querys.build(), 50).scoreDocs
    rawdocs = [searcher.doc(scoreDoc.doc) for scoreDoc in scoreDocs]
    return rawdocs
    
    
            # print 'explain:', searcher.explain(query, scoreDoc.doc)
def init_search():
    STORE_DIR = "index_zhCN"
    try:
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        print ('lucene', lucene.VERSION)
    except ValueError:
        print("Unable to start JVM, or JVM already running")
    
    directory = SimpleFSDirectory(File(STORE_DIR).toPath())
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = WhitespaceAnalyzer()#Version.LUCENE_CURRENT)
    print("done initializing searcher")
    return directory,searcher,analyzer

def get_search_res(command,searcher,analyzer):
    result = run(searcher = searcher,analyzer = analyzer,command=command)
    return result

if __name__ == '__main__':
    STORE_DIR = "index_zhCN"
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print ('lucene', lucene.VERSION)
    #base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    directory = SimpleFSDirectory(File(STORE_DIR).toPath())
    searcher = IndexSearcher(DirectoryReader.open(directory))
    analyzer = WhitespaceAnalyzer()#Version.LUCENE_CURRENT)
    run(searcher, analyzer)
    del searcher
