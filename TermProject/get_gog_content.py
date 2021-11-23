# SJTU EE208

import sys, os, lucene, threading, time
from datetime import datetime

# from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.document import Document, Field, FieldType, StringField, TextField
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
import jieba
import paddle
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
#pip install paddlepaddle,lxml
paddle.enable_static()

"""
This class is loosely based on the Lucene (java implementation) demo class 
org.apache.lucene.demo.IndexFiles.  It will take a directory as an argument
and will index all of the files in that directory and downward recursively.
It will index on the file path, the file name and the file contents.  The
resulting Lucene index will be placed in the current directory and called
'index'.
"""

class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)


def get_self_url(content):
    prefix = "<!- SELF_URL_TAG:"
    suffix = " -->"
    off1 = len(prefix)
    off2 = - len(suffix)
    pattern = re.compile(prefix + ".*?" + suffix)
    res = re.search(pattern=pattern,string = content)
    st,ed = res.span()[0],res.span()[1]
    return content[st + off1:ed + off2]

def get_domain(url):
    return urlparse(url).netloc

def get_title(content):
    soup = BeautifulSoup(content,features="html.parser")
    title_ele = soup.find("title")
    title = title_ele.string
    return title
    
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

class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, root, storeDir):

        if not os.path.exists(storeDir):
            os.mkdir(storeDir)

        # store = SimpleFSDirectory(File(storeDir).toPath())
        store = SimpleFSDirectory(Paths.get(storeDir))
        analyzer = WhitespaceAnalyzer()
        analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        config = IndexWriterConfig(analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(store, config)

        self.indexDocs(root, writer)
        ticker = Ticker()
        print('commit index')
        threading.Thread(target=ticker.run).start()
        writer.commit()
        writer.close()
        ticker.tick = False
        print('done')

    def indexDocs(self, root, writer):

        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(False)
        t1.setIndexOptions(IndexOptions.NONE)  # Not Indexed
        
        t2 = FieldType()
        t2.setStored(False)
        t2.setTokenized(True)
        t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)  # Indexes documents, frequencies and positions.
        
        for root, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith('.html'):
                    continue
                print("adding", filename)
                try:
                    path = os.path.join(root, filename)
                    file = open(path, encoding='utf-8')

                    contents = file.read()
                    #print(contents[:100])
                    page_url = get_self_url(contents)
                    page_domain = get_domain(page_url)
                    page_title = get_title(contents)
                    
                    contents = clean_html(contents)
                    
                    cut_words = jieba.cut_for_search(contents)    # requires paddlepaddle. -> pip install paddlepaddle
                    contents = " ".join(cut_words)
                    #print(contents)
                    file.close()
                    
                    doc = Document()
                    doc.add(Field("name", filename, t1))
                    doc.add(Field("path", path, t1))
                    doc.add(Field("url", page_url, t1))
                    doc.add(TextField("site",page_domain,Field.Store.YES))       # 不能用t1，要想搜索site必须把site设为indexed！
                    doc.add(Field("title", page_title, t1))

                    print(filename,path,page_url,page_domain,page_title)

                    if len(contents) > 0:
                        doc.add(Field("contents", contents, t2))
                    else:
                        print("warning: no content in %s" % filename)
                    writer.addDocument(doc)
                except Exception as e:
                    print("Failed in indexDocs:", e)

if __name__ == '__main__':
    lucene.initVM()#vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    # import ipdb; ipdb.set_trace()
    start = datetime.now()
    try:
        IndexFiles('../../html', "index_zhCN")
        end = datetime.now()
        print(end - start)
    except Exception as e:
        print("Failed: ", e)
        raise e
