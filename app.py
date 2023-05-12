import openai
import os
import streamlit as st
from text_extractor.functions import summarize

try:
    openai.api_key = os.getenv('OPENAI_KEY')

    # initialize state variable 
    if "summary" not in st.session_state:
        st.session_state["summary"] = ""

    st.title("Colonoscopy Screening Interval Recommendation System")

    input_colon_text = st.text_area(label='Enter colonoscopy impression:', value="", height=250)

    input_path_text = st.text_area(label='Enter pathology impression:', value="", height=250)

    st.button(
        "Submit",
        on_click=summarize,
        kwargs={"prompt": 'Colonoscopy: ' + input_colon_text + ' ' + 'Pathology Findings: ' + input_path_text},
        )

    # configure text area to populate with current state of summary
    output_text = st.text_area(label='Recommended Screening Colonoscopy Interval:', value=st.session_state["summary"], height=250)
except:
    st.write('There was an error =)')
