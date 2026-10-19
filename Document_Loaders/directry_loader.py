from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls = PyPDFLoader
)

docs=loader.load()

# print(len(docs))

# print(docs[0].page_content)

# print(docs[0].metadata)


#If there are lots of documents and to load them 
#into memory is not possible thus we use lazy load
docs2 = loader.lazy_load()

for document in docs:
    print(document.metadata)