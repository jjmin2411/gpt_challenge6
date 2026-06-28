import os
import streamlit as st
import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import BaseOutputParser

class JsonOutputParser(BaseOutputParser):

    def parse(self,text:str):
        text = text.replace("```","").replace("json","")
        return json.loads(text)

output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a helpful assistant that is role playing as a teacher.
    
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    
    Use (o) to signal the correct answer.
    
    Question examples:
    
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
    
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
    
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
    
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
    Your turn!
    
    Context: {context}
    """,
            )
        ]
    )


formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a powerful formatting algorithm.

You format exam questions into JSON format.
Answers with (o) are the correct ones.

Example Input:
Question: What is the color of the ocean?
Answers: Red|Yellow|Green|Blue(o)

Question: What is the capital or Georgia?
Answers: Baku|Tbilisi(o)|Manila|Beirut


Example Output:

```json
{{ "questions": [
{{
"question": "What is the color of the ocean?",
"answers": [
{{
"answer": "Red",
"correct": false
}},
{{
"answer": "Yellow",
"correct": false
}},
{{
"answer": "Green",
"correct": false
}},
{{
"answer": "Blue",
"correct": true
}}
]
}},
{{
"question": "What is the capital or Georgia?",
"answers": [
{{
"answer": "Baku",
"correct": false
}},
{{
"answer": "Tbilisi",
"correct": true
}},
{{
"answer": "Manila",
"correct": false
}},
{{
"answer": "Beirut",
"correct": false
}}
]
}}
]
}}
```
Your turn!
Questions: {context}
""",
)
]
)



@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, api_key):
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        openai_api_key=api_key,
    )

    questions_chain = {"context": format_docs} | questions_prompt | llm
    formatting_chain = formatting_prompt | llm

    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

with st.sidebar:
    st.link_button(
    "GitHub Repository",
    "https://github.com/jjmin2411/gpt_challenge6",
)
    api_key = st.text_input("OpenAI API Key", type="password")
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
          docs = wiki_search(topic)

if not api_key:
    st.warning("Please provide an OpenAI API key on the sidebar.")
    st.stop()

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    openai_api_key = api_key,
)

questions_chain = {"context": format_docs} | questions_prompt | llm
formatting_chain = formatting_prompt | llm

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    with st.spinner("Generating quiz..."):
        st_callback = StreamlitCallbackHandler(st.container())
        response = run_quiz_chain(docs, topic if topic else file.name, api_key)

    with st.form("questions_form"):
        user_answers = []

        for i, question in enumerate(response["questions"]):
            st.write(question["question"])

            value = st.radio(
                "Select a option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                key=f"question_{i}",
            )

            user_answers.append(value)

        button = st.form_submit_button("Submit")

    if button:
        score = 0
        for i, question in enumerate(response["questions"]):
            value = user_answers[i]

            if {"answer": value, "correct": True} in question["answers"]:
                st.success(f"Question {i + 1}: Correct!")
                score += 1
            elif value is not None:
                st.error(f"Question {i + 1}: Wrong!")
            else:
                st.warning(f"Question {i + 1}: No answer selected.")    
            if score == len(response["questions"]):
                st.balloons()