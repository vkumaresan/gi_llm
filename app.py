import openai
import os
import streamlit as st
from text_extractor.functions import *
import gspread
from PIL import Image
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import pandas as pd

# GPT
openai.api_key = os.getenv('OPENAI_KEY')

# Google
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'experimenting-297418-9266c3a6d9ee.json'

# Google Sheets for export
credentials = {
  "type": "service_account",
  "project_id": "experimenting-297418",
  "private_key_id": "fe9386899f891306fb6748470e1c31aa539c477b",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDiVEF3nYoZUIfH\nVC+PimtyaYU2V3v6Mcnss0lyG+22qURVSIQjTWDPgMb9qCWfZUz5NFsGMHFn0com\nAjYCBBg4VGHLf0ktS0g9aPAxehB+UCDX2PAs7QZmPt6yvnAsrEb1ed+7ZNqL4CMr\nRe/4Z3Lm89x7lz4B/8EfagcM0KCzxOGujKhU/19Q9zGDcKFfTOWgX/OCR2nocWKs\nL1MOS+De2ptQHrZSN2aEDfbawTH5LPlNAP7WZjXLcag+W7rtjc8hMMDszqwuur1q\ndYGyUKoxkaUnpLWtVDvn/e2VrVOhia0PXcKDgiudCxRkgpsnoljJvS2qDI94Oi6W\nOPqRyXl1AgMBAAECggEAaAyWIFiTsXmdQl0IlHPtW6b5L/deLrKPAzuVS1ldmnkM\nyixRWy1qkVrBoMGZskLO9U5Ffn3s1O7UgU7I53pcbCEW6If261TNvDWvHv/f70IF\nJ1Y7bFv3ci/7D6+PQGpfIOFLowoFkwKTCITZgpiEcXqw5TytrBuY/EkxPon3J01B\nVjKWPz5vKh7on6PHpY7sjHUOFl07B8CnFUzBKpjvQucdB8qhHmqkSNAmULvY5B04\nRuhubfREc4Fz/C7Ap/Q+iNLOg0a+x9lbc9N0N7cIwfOst4IctcwZG8rvFgieYHUO\n1IsBiPZGwBN7Gbe4l+0E6/CK+/7HtAYT/31cUPWcjQKBgQD9v51dWg2qjdSWdpF1\nAAdAyIHbVh8m3z/OMXuONo5y3VvQjR5UdDCuyBM6HWg0HLOsxYGvrqPO1jLj6iN8\n2SzG+Gt3aBF8Hrb8LmqC/Pkc3MuppB7fxyAbkIU/Ssq6qrDBr3bVs46oJwmIJX8s\nKey1e+vubIjz0OddmFC0LVVMpwKBgQDkVlvAL77336zgT78jUfg+5TpW6vM9DedR\nw0qvYHg1HwS6RS8OU45UQig8jKjbMdK9zIt07cV1JB1Es6Ho+D0NYlvqYRE4eCZ2\nMHRAvnlSpNXDiza+2SMd1gflN1HM0/s1xqWO+4HllDf8PaFEdYaxxBA3qkITJU6v\nzpgXo+rAgwKBgQCpHTGl68S77LbIaNFcpt4uoPNa2TT91UBTDcuI5ndduoXcopCa\nPK3Nbu7RhpPSV2awORnLmpr12PAl0gBAzwT2vs3w0N0GWfoebFj0X+EvCUB7GTSy\nc6XEeTc1DYW7jtMq4uRSXM8w5oOFx2fQaUinU6XUS2WjCZGJYWA4FRaKdQKBgQCt\nauE4L4sMWvbDTg3O13yA2Dvcs4iVQDAFKxtX4x6oyawfhFfeu5sHZ0+D3RiJkWeK\n+wSXg9ZJx2nrObqoY5CKz78bXSllB+u+K8K/QWqHV+V6JAsqG2POTzWj4sXfmMfb\n6cjntSDMqitzCaOniNMJw+zFOiwAun7uiyt8GOQ73QKBgHdeaZjHlvV6gB3DCmEz\nZDm6BXlgrBx2TB4KQA0sJaY1tv0MG0sl/eLuIdQgE/BmUts78FVz0n9m5rlhrhrF\nJTorYXOHDAELuxGzSIRLmZA2tpW3BOOU4aFVDNqJex+viG5Vl+GrfElUWt19JC97\ni3/6mrXy1rZ0GglH+jCuMOHa\n-----END PRIVATE KEY-----\n",
  "client_email": "gi-llm-app@experimenting-297418.iam.gserviceaccount.com",
  "client_id": "114400237465926164627",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/gi-llm-app%40experimenting-297418.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
gc = gspread.service_account_from_dict(credentials)

st.set_page_config(layout='wide')

# initialize state variables
if "summary" not in st.session_state:
    st.session_state["summary"] = ""
if "json" not in st.session_state:
    st.session_state["json"] = ""
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
#Remove whitespace from the top of the page and sidebar
st.markdown("""
    <style>
      section[data-testid="stSidebar"][aria-expanded="true"]{
        height: 80% 
        width: 50% !important;
      }
      section[data-testid="stSidebar"][aria-expanded="false"]{
        height: 80% 
        width: 50% !important;
      }
    </style>""", unsafe_allow_html=True)


st.title("Colonoscopy Screening Interval Calculator")


st.markdown("This tool allows you to enter findings from colonoscopy and pathology reports and receive the guideline-recommended screening interval for that patient.")

with st.sidebar:
    st.sidebar.markdown("This is an experimental tool that is still in development, and as such, results should be taken with caution as adjustments and improvements are still being made.")
    st.sidebar.markdown("IMPORTANT: At this moment, we ask that you do not enter actual patient health information in this tool. Instead, below are examples of synthetic data that you can modify and test.")

    example_input_text = pd.DataFrame([['A solitary 5 mm polyp was excised using a hot biopsy forceps from the cecum.', 
                                        'Cecum: Tubular adenoma.'],
                                    ['One 11 mm polyp was removed with a cold snare from the ascending colon. One 7 mm polyp was removed from the sigmoid colon. One 9 mm polyp was removed from the rectum.', 
                                     'Ascending Colon: Tubular adenoma. Sigmoid Colon: Tubular adenoma. Rectum: Tubular adenoma.'],
                                    ['Two polyps were removed; one 5 mm polyp from the cecum, and a 9 mm polyp from the rectum.', 
                                     'Cecum: tubular adenoma. Rectum: hyperplastic polyp.']], 
                                    columns=['Example Colonoscopy Text', 'Example Pathology Text'])
    st.table(example_input_text)

with st.expander("Evidence"):
    st.markdown("""
        Colonoscopies are routinely performed for colorectal cancer (CRC) screening between the ages of 45-75 as recommended by the United States Preventative Services Task Force (USPSTF)¹. Postprocedure, colonoscopists are expected to provide follow-up recommendations based on the number, size, and type of polyp(s). 
For average risk patients, these risk-stratified repeat colonoscopy intervals after biopsy of non-cancerous polyps follow an algorithmic approach as outlined by Gupta et al.² """)
    st.markdown("""
<style>
.small-font {
    font-size:12px;
}
</style>
""", unsafe_allow_html=True)

    st.markdown("""<p class="small-font">¹https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/colorectal-cancer-screening</p>""", unsafe_allow_html=True)
    st.markdown("""<p class="small-font">²Gupta S, Lieberman D, Anderson JC, Burke CA, Dominitz JA, Kaltenbach T, Robertson DJ, Shaukat A, Syngal S, Rex DK. Recommendations for Follow-Up After Colonoscopy and Polypectomy: A Consensus Update by the US Multi-Society Task Force on Colorectal Cancer. Am J Gastroenterol. 2020 Mar;115(3):415-434. doi: 10.14309/ajg.0000000000000544. PMID: 32039982; PMCID: PMC7393611.
    </p>""", unsafe_allow_html=True)

    image = Image.open('gupta_figure.jpeg')
    image = image.resize((600, 400))

    st.image(image)
with st.expander("Clinical Assumptions"):
    st.markdown("""
    1. No personal history/increased risk of colorectal cancer 
    2. No family history/increased risk of colorectal cancer 
    3. High Quality Colonoscopy: 
        * Complete to cecum 
        * Adequate Bowel prep to detect polyps > 5 mm 
        * Adequate colonoscopist adenoma detection rate 
        * Complete polyp resection 
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
    input_colon_text = st.text_area(label='Enter colonoscopy impression:', value="", height=150)
with col2:
    input_path_text = st.text_area(label='Enter pathology impression:', value="", height=150)

st.button(
    "Submit",
    on_click=summarize_using_gpt_JSON,
    kwargs={"prompt": 'Colonoscopy: ' + input_colon_text + ' ' + 'Pathology Findings: ' + input_path_text},
    )

# configure text area to populate with current state of summary
st.markdown('Polyp Summary')

# Display output table
st.table(st.session_state["polyps_table"])

# Display output recommendation
output_text = st.text_area(label='Recommended Screening Colonoscopy Interval', value=st.session_state["summary"], height=50)

# Export data to Google Sheet
if output_text != '':
    sh = gc.open_by_key('19KBn8TMXqluLF1f8QSpc_wWqSXrv1W3tga4qitmQJiU')
    worksheet = sh.get_worksheet(0)
    # Get current values from Google Sheet and append on output table
    df_sheet = pd.DataFrame(worksheet.get_all_records())
    data = {'Colonoscopy Text': [input_colon_text], 
            'Pathology Text': [input_path_text],
            'JSON': [st.session_state["json"]], 
            'Recommended Interval': [output_text]}
    df_app = pd.DataFrame(data=data)
    df_combined = pd.concat([df_sheet, df_app])
    df_combined = df_combined.fillna('')
    worksheet.update([df_combined.columns.values.tolist()] + df_combined.values.tolist())

    # Button to allow user to download output table as CSV
    if "polyps_table" not in st.session_state:
        st.session_state["polyps_table"] = pd.DataFrame()
    polyp_summary = st.session_state["polyps_table"]

    def data_exporter(polyp_summary, screening_information):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            polyp_summary.to_excel(writer, sheet_name="Polyp Summary", index=False)
            df_app.to_excel(writer, sheet_name="Screening Information", index=False)
            writer.close()
            processed_data = output.getvalue()
        return processed_data
    
    df_xlsx = data_exporter(polyp_summary, df_app)
    st.download_button(label='📥 Download findings and screening information',
                                data=df_xlsx ,
                                file_name= 'gi_calc_output.xlsx')

st.markdown("For any questions/feedback/collaboration inquiries, please contact V² Labs at <thev2labs@gmail.com>.")
# Logo
image = Image.open('v2labs.png')
image = image.resize((200, 200))
st.image(image)