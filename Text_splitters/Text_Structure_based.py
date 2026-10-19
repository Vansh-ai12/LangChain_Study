# first para based splitting
#then line based splitting 
#then whitespace based splitting
#then character based splitting


#Also merging is always there to see if under or equal to chunk size


from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
Artificial intelligence has become one of the most important areas of computer science, changing the way people interact with computers and software. Modern AI systems are capable of understanding natural language, recognizing images, generating text, writing code, analyzing large amounts of information, and assisting people with complex tasks. One of the major technologies behind these systems is machine learning, where computers learn patterns from data instead of being explicitly programmed for every possible situation. Deep learning takes this idea further by using neural networks with many layers to learn increasingly complex representations of information.

Large language models are a particularly interesting application of deep learning because they can process and generate human language. These models are trained on enormous collections of text and learn statistical relationships between words, sentences, and concepts. When building applications with language models, developers often need to provide the model with external information so that it can answer questions about specific documents, websites, databases, or private company information. This is where retrieval augmented generation becomes useful.

In a typical retrieval augmented generation system, documents are first collected from different sources and then divided into smaller pieces called chunks. These chunks are converted into numerical representations called embeddings and stored inside a vector database. When a user asks a question, the question is also converted into an embedding, and the system searches for chunks that are semantically similar to the question. The most relevant chunks are then provided to the language model as context, allowing it to generate an answer based on the retrieved information.

Splitting documents into appropriate chunks is an important part of this process because extremely large chunks may contain unnecessary information, while extremely small chunks may lose important context. Text splitters therefore provide different strategies for dividing documents. A simple character based splitter can divide text after a fixed number of characters, while a recursive character splitter attempts to split text using different separators such as paragraphs, new lines, spaces, and individual characters.

The goal is to create chunks that are small enough for efficient processing while still preserving meaningful relationships between pieces of information. Chunk overlap can also be used so that some text from one chunk is repeated in the following chunk. This helps preserve context when an important sentence or idea appears near the boundary between two chunks. Choosing the correct chunk size and overlap depends on the application, the type of documents being processed, and the capabilities of the language model.

After documents have been split, additional processing steps can be performed, such as cleaning unnecessary characters, adding metadata, generating embeddings, and storing the resulting documents in a vector store. Metadata can contain useful information such as the original filename, page number, document title, source URL, or section name. During retrieval, this metadata can help developers understand where a particular piece of information came from and can also be used to filter search results.

Frameworks such as LangChain provide components that make it easier to connect these different stages into a single pipeline. A developer can load a PDF or webpage, split its contents into chunks, create embeddings for those chunks, store them in a vector database, retrieve relevant information when a user asks a question, and finally pass that information to a language model to generate a response. Understanding how the data moves through this pipeline is important for developing modern AI applications.

By experimenting with different documents, chunk sizes, overlap values, retrieval methods, and prompts, developers can understand how each component affects the final quality of an AI application. Text splitting may appear to be a simple step, but it has a significant impact on how effectively a retrieval system can find and provide relevant information to a language model.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(len(chunks))

print(chunks)