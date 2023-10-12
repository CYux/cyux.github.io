from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pickle
import os

def main():
    load_dotenv()
    path = os.getcwd()
    st.set_page_config(page_title='GE semantic search')
    st.header('GE semantic search 💬')
    
    # File uploader widget
    selected = option_menu("Books", ["Romola", 'Others'])
    # File uploader widget

    # Read the contents of the uploaded file
    if selected:
        # file_contents = uploaded_file.read().decode("utf-8")
        
        # # split into chunks
        # text_splitter = CharacterTextSplitter(
        #     separator='\n',
        #     chunk_size=1500,
        #     chunk_overlap=200,
        #     length_function=len
        # )
        # chunks = text_splitter.split_text(file_contents)

        # # create embeddings
        # embeddings = OpenAIEmbeddings()
        # knowledge_base = FAISS.from_texts(chunks, embeddings)
        # with open("embeddings.pkl", "wb") as file:
        #     pickle.dump(embeddings, file)

        # with open("embeddings.pkl", "rb") as file:
        #     st.download_button(
        #         label="Download embeddings",
        #         data=file,
        #         file_name="embeddings.pkl",
        #         mime="text/plain"
        #     )

        # with open("knowledge_base.pkl", "wb") as file:
        #     pickle.dump(knowledge_base, file)

        # with open("knowledge_base.pkl", "rb") as file:
        #     st.download_button(
        #         label="Download knowledge_base",
        #         data=file,
        #         file_name="knowledge_base.pkl",
        #         mime="text/plain"
        #     )
        # Load embeddings
        with open(f"{path}/embeddings.pkl", "rb") as file:
            embeddings = pickle.load(file)

        # Load knowledge base
        with open(f"{path}/knowledge_base.pkl", "rb") as file:
            knowledge_base = pickle.load(file)

        # show user input
        user_question = st.text_input('Keywords:')
        k = st.number_input("Most similarity:", 
                            min_value=1, max_value=100, step=1)

        if user_question:
            docs = knowledge_base.similarity_search(user_question, k)
            cnt = 1
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
