import streamlit as st
import LLm  # Ensure LLm.py is in the same directory

st.title("Definition GPT")

# Create two columns
col1, col2 = st.columns(2)

# User Input for Query
ques = col1.text_input("Enter the Query:")

# Select Model from Dropdown
llm_select = col2.selectbox(
    "Choose Language Model",
    (
        "Select",
        "Mixtral-8x7B",
        "Mistral-7B",
        "gemma-1.1",
        "deepseek",
        "TinyLlama",
    ),
)

# Submit Button
res = st.button("Submit")

if res:
    if llm_select == "Select":
        st.warning("Please select a valid language model.")
    else:
        match llm_select:
            case "Mixtral-8x7B":
                data = LLm.Mixtra8(ques)
            case "Mistral-7B":
                data = LLm.Mixtra7(ques)
            case "gemma-1.1":
                data = LLm.gemma(ques)
            case "deepseek":
                data = LLm.deepseek(ques)
            case "TinyLlama":
                data = LLm.TinyLlama(ques)
            case _:
                data = "Invalid selection."

        st.write(data)
