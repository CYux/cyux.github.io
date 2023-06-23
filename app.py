from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pickle


def main():
    load_dotenv()
    st.set_page_config(page_title='GE semantic search')
    st.header('GE semantic search 💬')
    
    # File uploader widget
    uploaded_file = st.file_uploader("Upload your text file.", type=["txt"])

    # Read the contents of the uploaded file
    if uploaded_file:
        file_contents = uploaded_file.read().decode("utf-8")
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(file_contents)

        # create embeddings
        # embeddings = OpenAIEmbeddings()
        # knowledge_base = FAISS.from_texts(chunks, embeddings)
        # with open("embeddings.pkl", "wb") as file:
        #     pickle.dump(embeddings, file) 
        # with open("knowledge_base.pkl", "wb") as file:
        #     pickle.dump(knowledge_base, file)
        # Load embeddings
        with open("embeddings.pkl", "rb") as file:
            embeddings = pickle.load(file)

        # Load knowledge base
        with open("knowledge_base.pkl", "rb") as file:
            knowledge_base = pickle.load(file)

        # show user input
        user_question = st.text_input('Ask a question about your PDF:')
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            cnt = 0
            for doc in docs:
                st.write('➡️', f'No. {cnt}', '✅', doc.page_content[201:])
                cnt += 1
            # llm = OpenAI()
            # chain = load_qa_chain(llm, chain_type='stuff') 
            # with get_openai_callback() as cb: 
            #     response = chain.run(input_documents=docs, question=user_question)
            #     st.write(cb)
            
            # st.write(response)
       
        
if __name__ == '__main__':
    main()