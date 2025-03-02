from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain import HuggingFaceHub
import os

# Set Hugging Face API Token
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Your api key"

# Common function to create an LLM chain
def generate_response(ques, repo_id):
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.4,
            "repetition_penalty": 1.03,
        },
    )
    
    prompt = PromptTemplate(
        input_variables=["question"],
        template="{question} in 100 words"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    res = chain.invoke({"question": ques})
    
    return res.get("text", "No response generated")


def Mixtra8(ques):
    return generate_response(ques, "mistralai/Mixtral-8x7B-Instruct-v0.1")

def Mixtra7(ques):
    return generate_response(ques, "mistralai/Mistral-7B-Instruct-v0.2")

def gemma(ques):
    return generate_response(ques, "google/gemma-1.1-2b-it")

def TinyLlama(ques):
    return generate_response(ques, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def deepseek(ques):
    return generate_response(ques, "deepseek-ai/deepseek-coder-1.3b-instruct")

def phi(ques):
    return generate_response(ques, "core42/jais-13b")

