import openai
import streamlit as st

def summarize(prompt):
    augmented_prompt = f"""Your task is to provide the recommended screening colonoscopy interval \
from the colonoscopy and pathology findings. 

Below are the rules, delimited by single backticks, to determine the screening colonoscopy interval.  
There are several features of interest including: Number of polyps, type of polyp, size of polyps, histology of polyp. 
The recommended screening interval is the smallest interval recommended that can be applied to that specific situation. 
If multiple rules apply, the recommended screening interval is the smallest interval recommended that can be applied to that specific situation. 
If no rules apply, output "ERROR."

Rules:
`Tubular Adenomas
Rule 1: If there are more than 10 tubular adenomas of any size, repeat colonoscopy in 1 year. 
Rule 2: If there are 1-2 tubular adenomas and each are less than 10 mm in size, repeat colonoscopy in 7-10 years. 
Rule 3: If there are 3-4 tubular adenomas, and each are less than 10 mm in size, repeat colonoscopy in 3-5 years. 
Rule 4: If there are 5-10 tubular adenomas of any size, repeat colonoscopy in 3 years. 
Rule 5: If there are any number of tubular adenomas greater than or equal to 10 mm in size, repeat colonoscopy in 3 years. 
Rule 6: If there are any number of tubular adenomas of any size with villous histology, repeat colonoscopy in 3 years. 
Rule 7: If there are any number of tubular adenomas of any size with tubulovillous histology, repeat colonoscopy in 3 years. 
Rule 8: If there are any number of tubular adenomas of any size with high grade dysplasia hiotology, repeat colonoscopy in 3 years. 

Sessile Serrated Polyps: 
Rule 9: If there are 1-2 sessile serrated polyps, and each are less than 10 mm in size, repeat colonscopy in 5-10 years. 
Rule 10: If there are 3-4 sessile serrated polyps, and each are less than 10 mm in size, repeat colonoscopy in 3-5 years. 
Rule 11: If there are 5-10 sessile serrated polyps of any size, repeat colonoscopy in 3 years. 
Rule 12: If there are any number of sessile serrated polyps greater than or equal to 10 mm, repeat colonoscopy in 3 years. 

Hyperplastic Polyps: 
Rule 13: If there are less than or equal to 20 hyperplastic polyps, and each are less than 10 mm in size, repeat colonoscopy in 10 years. 
Rule 14: If there are any number of hyperplastic polyps greater than or equal to 10 mm in size, repeat colonoscopy in 3-5 years.
`

Provide the recommended colonoscopy screening interval for the input text below, delimited by triple 
backticks, in at most 30 words. Also print the exact wording of the rule that was used to calculate this recommendation, without the rule number and name.

Input text: ```{prompt}```"""
    try:
        messages = [{"role": "user", "content": augmented_prompt}]
        st.session_state["summary"] = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        ).choices[0].message["content"]
    except:
        st.write('There was an error =(')