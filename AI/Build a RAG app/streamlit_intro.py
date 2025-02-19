import streamlit as st


st.sidebar.title("Demo RAG app")

st.sidebar.markdown("This is *first* demo **app** by Zarina B")

#create a text area widget
user_question = st.text_area("Ask a question", key="user_question")
st.write(f"User question: {user_question}")

#containers and dropdowns

col1, col2 = st.columns(2)

n_results = col1.number_input("Number of results", min_value=1,
max_value=10, value=1)
col1.write(f"Number of results: {n_results}")

model_choice = col2.selectbox("Choose a model", ["gpt-3", "gpt-4"])
col2.write(f"Selected model: {model_choice}")

#view outputs button
if st.button("Get answers!"):
     st.write(f"Model: {model_choice}")
     st.write(f"Question: {user_question}")
     st.write(f"Number of Results: {n_results}")
     st.write("RAG output goes here...")