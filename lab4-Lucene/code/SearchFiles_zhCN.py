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
def run(searcher, analyzer):
    # while True:
    print()
    print ("Hit enter with no input to quit.")
    command = input("Query:")
    # command = unicode(command, 'GBK')
    # command = 'london author:shakespeare' 
    if command == '':
        return
    command = " ".join(jieba.cut_for_search(command))
    print()
    print ("Searching for:", command)
    query = QueryParser("contents", analyzer).parse(command)
    scoreDocs = searcher.search(query, 50).scoreDocs
    print ("%s total matching documents." % len(scoreDocs))

    for i, scoreDoc in enumerate(scoreDocs):
        doc = searcher.doc(scoreDoc.doc)
        print ('path:', doc.get("path"), '  filename:', doc.get("name"), '  score:', scoreDoc.score)
        print("URL:",doc.get("url"),"  title:",doc.get("title").strip())
        print("-"*50)
            # print 'explain:', searcher.explain(query, scoreDoc.doc)


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
