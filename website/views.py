from flask import Blueprint, render_template, request, url_for, redirect
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
import pandas as pd
from .models import History
from . import db

views = Blueprint('views', __name__)


def chat(chat_history, user_input):
    bot_response = qa_chain({"query": user_input})
    bot_response = bot_response['result']
    response = "".join(bot_response)
    chat_history.append((user_input, response))
    return chat_history


dataset_path = 'hi_DOLA_List_of_Advocate_Tamilnadu_2013_to_june2014.csv'
advocate_data = pd.read_csv(dataset_path, encoding='unicode_escape')
checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# # Load the model with offload_folder argument
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     offload_folder="ipc_vector_data/92ca7e32-064c-43f8-973e-1060c76a995e",  # Replace with the actual path
#     device_map="auto",
#     torch_dtype=torch.float32
# )
#
# embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
#
# db = Chroma(persist_directory="ipc_vector_data", embedding_function=embeddings)
#
# pipe = pipeline(
#     'text2text-generation',
#     model=base_model,
#     tokenizer=tokenizer,
#     max_length=10000,
#     do_sample=True,
#     temperature=1,
#     top_p=0.95
# )
# local_llm = HuggingFacePipeline(pipeline=pipe)
#
# qa_chain = RetrievalQA.from_chain_type(
#     llm=local_llm,
#     chain_type='stuff',
#     retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
#     return_source_documents=True,
# )

# Example usage
# chat_history = []
# user_input = input("Enter your initial input: ")
# chat_history = chat(chat_history, user_input)

# Display the conversation history
# for prompt, response in chat_history:
#     print(f"User: {prompt}")
#     print(f"Saul: {response}")


@views.route('/')
def home():
    return render_template("home.html")


@views.route('input', methods=['POST', 'GET'])
def Input():
    if request.method == 'POST':
        global qa_chain
        chat_history = []
        # Load the model with offload_folder argument
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            offload_folder="ipc_vector_data/92ca7e32-064c-43f8-973e-1060c76a995e",  # Replace with the actual path
            device_map="auto",
            torch_dtype=torch.float32
        )

        embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

        dbb = Chroma(persist_directory="ipc_vector_data", embedding_function=embeddings)

        pipe = pipeline(
            'text2text-generation',
            model=base_model,
            tokenizer=tokenizer,
            max_length=10000,
            do_sample=True,
            temperature=1,
            top_p=0.95
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)

        qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            chain_type='stuff',
            retriever=dbb.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
            return_source_documents=True,
        )
        inp = request.form.get('input')
        chat_history = chat(chat_history, inp)
        # for prompt, response in chat_history:
        #     print(f"User: {prompt}")
        #     print(f"Saul: {response}")
        # return "<h1>Finished</h1>"
        if chat_history:
            for prompt, response in chat_history:
                hist = History(user=prompt, bot=response)
                db.session.add(hist)
                db.session.commit()
        return render_template("output.html", chat_history=chat_history)
    return render_template("input.html")


@views.route('search', methods=['POST'])
def search():
    search_term = request.form.get('address')

    # Filter the dataset based on the entered address
    filtered_data = advocate_data[advocate_data['ADDRESS'].str.contains(search_term, case=False, na=False)]
    columns_to_keep_indices = [0, 1, 4]

    # Create a new DataFrame with only the specified columns
    df_filtered = advocate_data.iloc[:, columns_to_keep_indices]

    # Save the new DataFrame to a new CSV file
    df_filtered.to_csv('new_file.csv', index=False)
    return render_template('search_result.html', data=filtered_data)


@views.route('history')
def history():
    items = History.query.all()
    return render_template("history.html", items=items)
