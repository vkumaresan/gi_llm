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
    
# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.title("Colonoscopy Screening Interval Recommendation System")


st.markdown("This tool allows you to enter findings from colonoscopy and pathology reports and receive the guideline-recommended screening interval for that patient.")

with st.sidebar:
    st.markdown("This is an experimental tool that is still in development, and as such, results should be taken with caution as adjustments and improvements are still being made.")
    st.markdown("IMPORTANT: At this moment, we ask that you do not enter actual patient health information in this tool. Instead, below are examples of synthetic data that you can modify and test in this tool.")

    example_input_text = pd.DataFrame([['A 3mm polyp in the ascending colon.', 'Tubular adenoma with low-grade dysplasia.'],
                                    ['Two 6mm polyps in the transverse colon. A 9mm polyp in the rectum. ', 'Transverse colon: 1st polyp: Sessile serrated polyp. 2nd polyp: Sessile serrated polyp. Rectum: Sessile serrated polyp.'],
                                    ['A 10mm polyp in the sigmoid colon. A 9mm polyp in the rectum. A 7mm polyp in the ascending colon. A 6mm polyp in the transverse colon. A 5mm polyp in the descending colon.', 'Sigmoid colon - Sessile serrated polyp. Rectum - Sessile serrated polyp. Ascending colon - Sessile serrated polyp. Transverse colon - Sessile serrated polyp. Descending colon - Sessile serrated polyp.']], 
                                    columns=['Example Colonoscopy Text', 'Example Pathology Text'])
    st.table(example_input_text)

with st.expander("Evidence"):
    st.write("""
        TBD
    """)

with st.expander("Methods"):
    st.markdown("""
        This tool leverages Large Language Models (LLM) to parse the given colonoscopy and pathology text, 
        match the findings, and use this output alongside the colonoscopy screening guidelines to identify 
        the appropriate colonoscopy screening interval. 

        LLMs are still actively being researched in the AI community, and there are a variety of parameters that can change the 
        output accuracy and depth, so we will continue to experiment and test different methods in order to determine the best approach
        for each use case. 
        
        If you'd like to learn more or collaborate with us for your use case, please contact us (V² Labs)
        at the email below.
        
    """)

st.divider()
col1, col2 = st.columns(2)

with col1:
    input_colon_text = st.text_area(label='Enter colonoscopy impression:', value="", height=250)
with col2:
    input_path_text = st.text_area(label='Enter pathology impression:', value="", height=250)

st.button(
    "Submit",
    on_click=summarize_using_gpt_two_prompt,
    kwargs={"prompt": 'Colonoscopy: ' + input_colon_text + ' ' + 'Pathology Findings: ' + input_path_text},
    )

# configure text area to populate with current state of summary
st.markdown('Polyp Summary')

# Display output table
st.table(st.session_state["polyps_table"])

# Display output recommendation
output_text = st.text_area(label='Recommended Screening Colonoscopy Interval', value=st.session_state["summary"], height=250)

# Button to allow user to download output table as CSV
csv = st.session_state["polyps_table"].to_csv(index=False).encode('utf-8')

st.download_button(
   "Download Polyp Findings as CSV file",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)

st.markdown("For any questions/feedback/collaboration inquiries, please contact V² Labs at <thev2labs@gmail.com>.")