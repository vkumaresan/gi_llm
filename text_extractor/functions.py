import openai
import streamlit as st
import vertexai
from vertexai.preview.language_models import TextGenerationModel
import json
import re
import pandas as pd


##### GPT
def summarize_using_gpt_two_prompt(prompt):
    # Execute first LLM call to generate JSON for the input prompt and store it
    json_prompt = f"""Your task is to extract size, location, and type for all polyps, using the colonoscopy and pathology text. 
    Use the location of the polyp as the link between the colonoscopy and pathology text. 

The input text is listed below, delimited by triple backticks.

Format the output as a JSON.

Input text: ```{prompt}```
    """
    try:
        messages = [{"role": "user", "content": json_prompt}]
        output_json = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            ).choices[0].message["content"]
    except:
        st.write('There was an error')

    # Execute second LLM call to take in the JSON and return the recommendation+explanation
    final_prompt = f"""Your task is to provide the recommended screening colonoscopy interval \
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

  Provide the recommended colonoscopy screening interval for the input JSON below, delimited by triple 
  backticks, in at most 30 words. 

  Input text: ```{output_json}```"""
    try:
        messages = [{"role": "user", "content": final_prompt}]
        st.session_state["summary"] = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        ).choices[0].message["content"]
        # Convert JSON to pandas dataframe
        matches = [m.group(1) for m in re.finditer("```([\w\W]*?)```", output_json)]
        df = pd.read_json(matches[0].strip('\n'))
        final_polyps_output = pd.DataFrame()
        for i,r in df.iterrows():
            polyp_df = pd.DataFrame.from_dict(df['polyps'][i], orient='index').T
            final_polyps_output = pd.concat([final_polyps_output, polyp_df])
        st.session_state["polyps_table"] = final_polyps_output
    except:
        st.write('There was an error')

def summarize_using_gpt_one_prompt(prompt):
    # Combine JSON creation and recommendation into one prompt
    final_prompt = f"""
  Your task is to provide the recommended screening colonoscopy interval \
  from the colonoscopy and pathology findings. 

  First, extract size, location, and type for all polyps, using the colonoscopy and pathology text. Use the location of the polyp as the
  link between the colonoscopy and pathology text. 

  The input text is listed below, delimited by triple backticks.
  Input text: ```{prompt}```

  Format the output as a JSON.

  Next, use the following rules, delimited by single backticks, to determine the screening colonoscopy interval.  
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

  Provide the recommended colonoscopy screening interval for the input text in at most 30 words. 
"""
    try:
        messages = [{"role": "user", "content": final_prompt}]
        st.session_state["summary"] = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        ).choices[0].message["content"]
    except:
        st.write('There was an error')

##### PaLM
def predict_large_language_model_sample(
    project_id: str,
    model_name: str,
    temperature: float,
    max_decode_steps: int,
    top_p: float,
    top_k: int,
    content: str,
    location: str = "us-central1",
    tuned_model_name: str = "",
    ) :
    """Predict using a Large Language Model."""
    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    if tuned_model_name:
      model = model.get_tuned_model(tuned_model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
        top_k=top_k,
        top_p=top_p,)
    return response.text

def summarize_using_palm_two_prompt(prompt):
    # Execute first LLM call to generate JSON for the input prompt and store it
    output_json = predict_large_language_model_sample("experimenting-297418", "text-bison@001", 0, 256, 0.8, 40, 
                                            f"""Your task is to extract size, location, and type for all polyps, using the colonoscopy and pathology text. 
        Use the location of the polyp as the link between the colonoscopy and pathology text. 

    The input text is listed below, delimited by triple backticks.

    Format the output as a JSON.

    Input text: ```{prompt}```""", "us-central1")
    
    # Execute second LLM call to take in the JSON and return the recommendation+explanation
    final_prompt = f"""Your task is to provide the recommended screening colonoscopy interval \
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

  Provide the recommended colonoscopy screening interval for the input JSON below, delimited by triple 
  backticks, in at most 30 words. Also provide an explanation for the recommended interval. 

  Input text: ```{output_json}```"""
    try:
        st.session_state["summary"] = predict_large_language_model_sample("experimenting-297418", "text-bison@001", 0, 256, 0.8, 40, 
                                            final_prompt, "us-central1")
        st.session_state["json"] = output_json
    except:
        st.write('There was an error')

##### Other processing functions
def export_polyp_findings(polyps_table):
    polyps_table.to_csv(index=False)