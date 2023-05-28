import openai
import os
import streamlit as st
from text_extractor.functions import *

# GPT
openai.api_key = os.getenv('OPENAI_KEY')

# Google
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'experimenting-297418-9266c3a6d9ee.json'

# initialize state variable 
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "polyps_table" not in st.session_state:
    st.session_state["polyps_table"] = pd.DataFrame()
    

st.title("Colonoscopy Screening Interval Recommendation System")

input_colon_text = st.text_area(label='Enter colonoscopy impression:', value="", height=250)

input_path_text = st.text_area(label='Enter pathology impression:', value="", height=250)

st.button(
    "Submit",
    on_click=summarize_using_gpt_two_prompt,
    kwargs={"prompt": 'Colonoscopy: ' + input_colon_text + ' ' + 'Pathology Findings: ' + input_path_text},
    )

# configure text area to populate with current state of summary
st.subheader('Polyp Summary')
st.dataframe(st.session_state["polyps_table"])
output_text = st.text_area(label='Recommended Screening Colonoscopy Interval:', value=st.session_state["summary"], height=250)

# Download as CSV
csv = st.session_state["polyps_table"].to_csv(index=False).encode('utf-8')

st.download_button(
   "Download Polyp Findings as CSV file",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)
