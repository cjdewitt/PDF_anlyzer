from flask import Flask, request, render_template
import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import tempfile

app = Flask(__name__)
app.debug = True

openai_api_key = ""


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/qa', methods=['POST'])
def qa():
    file = request.files['file']
    api_key = request.form['apiKey']
    question = request.form['question']

    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(suffix='.pdf') as temp_file:
        file.save(temp_file.name)
        result = run_qa(temp_file.name, api_key, question)

    return render_template('result.html', question=question, result=result['result'])

def run_qa(file_path, api_key, question):
    os.environ["OPENAI_API_KEY"] = api_key

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)

    retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': 2})

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type='map_rerank', retriever=retriever, return_source_documents=True
    )

    result = qa({'query': question})
    return result

if __name__ == '__main__':
    app.run(debug=True)
