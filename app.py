import openai
import os
import streamlit as st
from text_extractor.functions import summarize_using_gpt, summarize_using_palm

# GPT
openai.api_key = os.getenv('OPENAI_KEY')

# Google
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'experimenting-297418-9266c3a6d9ee.json'

# initialize state variable 
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "json" not in st.session_state:
    st.session_state["json"] = ""

st.title("Colonoscopy Screening Interval Recommendation System")

input_colon_text = st.text_area(label='Enter colonoscopy impression:', value="", height=250)

input_path_text = st.text_area(label='Enter pathology impression:', value="", height=250)

st.button(
    "Submit",
    on_click=summarize_using_palm,
    kwargs={"prompt": 'Colonoscopy: ------- ' + input_colon_text + ' ' + 'Pathology Findings: ' + input_path_text},
    )

# configure text area to populate with current state of summary
output_json = st.text_area(label='Polyp Summary:', value=st.session_state["json"], height=250)
output_text = st.text_area(label='Recommended Screening Colonoscopy Interval:', value=st.session_state["summary"], height=250)
