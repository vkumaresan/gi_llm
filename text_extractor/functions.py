import openai
import streamlit as st
import vertexai
from vertexai.preview.language_models import TextGenerationModel
import json
import re
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from thefuzz import fuzz

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
                model="gpt-4",
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
            model="gpt-4",
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

# Use LLM to create JSON
size_schema = ResponseSchema(name="size",
                             description="What was the size of the polyp?\
                             Extract the numerical value and the unit.")
location_schema = ResponseSchema(name="location",
                                      description="Where was the polyp found?\
                                      Output the value as a string.")
histology_schema = ResponseSchema(name="histology",
                                    description="What was the histology of the polyp?\
                                    Output as a string.")
dysplasia_schema = ResponseSchema(name="dysplasia",
                                    description="What was the dysplasia of the polyp?\
                                    Histology will either be high-grade dysplasia, \
                                    low-grade dysplasia, dysplasia, villous, tubulovillous, or not applicable." \
                                    )
response_schemas = [size_schema, 
                    location_schema,
                    histology_schema,
                    dysplasia_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

@st.cache_resource(show_spinner=False)
def summarize_using_gpt_JSON(prompt):
    prompt_template = """\
For the following text, extract the following information for each polyp:

size: Extract the size of the polyp \
and output the number and units.

location: Extract the location of the polyp\
and output as a string.

histology: Extract the histology of the polyp\
and output as a string.

dysplasia: Extract the grade of the dysplasia (if present) \
and output as a string. If not present, output 'No dysplasia found'

text: {text}

{format_instructions}

If multiple polyps are found in a location, output a json for each polyp.
"""

    final_prompt = ChatPromptTemplate.from_template(template=prompt_template)

    final_prompt = final_prompt.format_messages(text=prompt, 
                                format_instructions=format_instructions)
    try:
        messages = [{"role": "user", "content": final_prompt[0].content}]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        ).choices[0].message["content"]
        # Save JSON output
        st.session_state["json"] = response
        # Get clinical recommendation from JSON using helper function
        #st.session_state["summary"] = clinical_rec_calc(response)
        # Get polyp table

    except:
        st.write(clinical_rec_calc(response)) 
        st.write('There was an error')

@st.cache_resource(show_spinner=False)
def summarize_using_gpt_one_prompt(prompt):
    # Combine JSON creation and recommendation into one prompt
    final_prompt = f"""
  You are a gastroenterologists determining the risk-stratified screening colonoscopy interval. The rules to follow are below. Remember these rules and recall them by the term "CRC Screening Rules"

CRC Screening Rules

10 Years:
Normal Colonoscopy with no polyps biopsied
There are less than or equal to 20 total hyperplastic polyps AND each polyp is less than 10 mm in size
7-10 years:
There are 1-2 tubular adenomas AND each polyp is less than 10 mm in size
5-10 years:
There are 1-2 sessile serrated polyps AND each polyp is less than 10 mm in size
3-5 years:
There are 3-4 tubular adenomas AND each polyp is less than 10 mm in size
There are 3-4 sessile serrated polyps AND each polyp is less than 10 mm in size
There are any number of hyperplastic polyps that are greater than of equal to 10 mm in size
3 years:
There are 5-10 tubular adenomas
There are 5-10 sessile serrated polyps
There are any tubular adenomas greater than or equal to 10 mm
There are any sessile serrated polyps greater than or equal to 10 mm
There are any tubular adenomas with villous histology
There are any tubulovillous adenomas
There are any tubular adenomas with high grade dysplasia
There are any sessile serrated polyps with dysplasia
There are any traditional serrated adenoma
1 year:
There are more than 10 tubular adenomas

Using CRC Screening Rules, what is the interval in this example?

{prompt}

Work through your answer step by step.

Step 1: Identify the types and sizes of polyps found in the colonoscopy.

Step 2: Cross-reference each of these findings with the CRC Screening Rules.

Step 3: Determine the shortest interval among the categories the patient falls into.

Output your final results as one of the following options, without any additional text:

  '1 year'
  '3 years'
  '3-5 years'
  '5-10 years'
  '7-10 years'
  '10 years'
"""
    try:
        messages = [{"role": "user", "content": final_prompt}]
        st.session_state["explanation"] = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        ).choices[0].message["content"]
        # Use regex to parse out the final recommendation
        matches = [m.group(1) for m in re.finditer("'(.*]*)'", st.session_state["explanation"])]
        if len(matches) != 0:
            st.session_state["summary"] = matches[0]
        else:
            st.session_state["summary"] = st.session_state["explanation"]
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

def clinical_rec_calc(json_response):
    # Parse JSON and use rules to determine final interval
    # Initialize variables
    adenoma_count = 0
    hyperplastic_count = 0
    sessile_count = 0
    traditional_count = 0
    greater_than_10mm_count_adenoma = 0
    greater_than_10mm_count_hyperplastic = 0
    greater_than_10mm_count_sessile = 0
    adenoma_dysplasia_count = 0
    sessile_dysplasia_count = 0

    dysplasia_types_adenoma = ['high-grade', 'villous', 'tubulovillous']
    dysplasia_types_sessile = ['high-grade', 'low-grade', 'dysplasia', 'villous', 'tubulovillous']
    # Get JSONs
    matches = re.findall('\{(?:[^{}])*\}', json_response.replace('\n', '').replace('\t', ''))
    for match in matches:
        #print(match)
        polyp_dict = output_parser.parse(match)

        ### Extract variables and add to list
        polyp_size = polyp_dict.get('size')
        polyp_location = polyp_dict.get('location').lower()
        polyp_histology = polyp_dict.get('histology').lower()
        polyp_dysplasia = polyp_dict.get('dysplasia').lower()

        # Adenoma
        if fuzz.partial_token_sort_ratio("adenoma", polyp_histology) > 50:
            # check for traditional
            if fuzz.partial_token_sort_ratio("traditional", polyp_histology) > 50:
                traditional_count += 1
            adenoma_count += 1
            # >= 10mm 
            sizes = [int(i) for i in polyp_size.split() if i.isdigit()]
            for size in sizes:
                if size >= 10:
                    greater_than_10mm_count_adenoma += 1
                # cm
            if 'cm' in polyp_size:
                greater_than_10mm_count_adenoma += 1  
            # histology
            for t in dysplasia_types_adenoma:
                if (fuzz.partial_token_sort_ratio(t, polyp_dysplasia) > 75) or (fuzz.partial_token_sort_ratio(t, polyp_dysplasia) > 75):
                    adenoma_dysplasia_count += 1

        # Hyperplastic
        if fuzz.partial_token_sort_ratio("hyperplastic", polyp_histology) > 50:
            hyperplastic_count += 1
            # >= 10mm 
            sizes = [int(i) for i in polyp_size.split() if i.isdigit()]
            for size in sizes:
                if size >= 10:
                    greater_than_10mm_count_hyperplastic += 1
                # cm
            if 'cm' in polyp_size:
                greater_than_10mm_count_hyperplastic += 1
        # Sessile
        if fuzz.partial_token_sort_ratio("sessile", polyp_histology) > 50:
            sessile_count += 1
            # >= 10mm 
            sizes = [int(i) for i in polyp_size.split() if i.isdigit()]
            for size in sizes:
                if size >= 10:
                    greater_than_10mm_count_sessile += 1
                # cm
            if 'cm' in polyp_size:
                greater_than_10mm_count_sessile += 1
            for t in dysplasia_types_sessile:
                if (fuzz.partial_token_sort_ratio(t, polyp_dysplasia) > 50) or (fuzz.partial_token_sort_ratio(t, polyp_dysplasia) > 50):
                    # add if any dysplasia
                    sessile_dysplasia_count += 1
        
    # Calculate clinical recommendation
    clinical_recommendation = []
    # Adenoma Loop
    if adenoma_count >= 1:
        if greater_than_10mm_count_adenoma >= 1:
            clinical_recommendation.append('3 years')
        elif adenoma_count > 10:
            clinical_recommendation.append('1 year')
        else:
            if adenoma_count >= 5:
                clinical_recommendation.append('3 years')
            elif adenoma_count >= 3:
                clinical_recommendation.append('3-5 years')
            else:
                clinical_recommendation.append('7-10 years')
        if adenoma_dysplasia_count >= 1:
            clinical_recommendation.append('3 years')
    # Hyperplastic Loop
    if hyperplastic_count >= 1:
        if greater_than_10mm_count_hyperplastic >= 1:
            clinical_recommendation.append('3-5 years')
        else:
            if hyperplastic_count < 20:
                clinical_recommendation.append('10 years')

    # SSP Loop
    if sessile_count >= 1:
        if greater_than_10mm_count_sessile >= 1:
            clinical_recommendation.append('3 years')
        else:
            if sessile_count >= 5:
                clinical_recommendation.append('3 years')
            elif sessile_count >= 3:
                clinical_recommendation.append('3-5 years') 
            else:
                clinical_recommendation.append('5-10 years')
        if sessile_dysplasia_count >= 1:
            clinical_recommendation.append('3 years')
    # Traditional Loop
    if traditional_count >= 1:
        clinical_recommendation.append('3 years')
    # Choose most conservative option
    final_rec = ''
    if len(clinical_recommendation) > 1:
        for rec in clinical_recommendation:
            if rec == '1 year':
                final_rec = '1 year'
            elif rec == '3 years':
                if final_rec == '1 year':
                    continue
                else:
                    final_rec = '3 years'
            elif rec == '3-5 years':
                if (final_rec == '1 year') | (final_rec == '3 years'):
                    continue
                else:
                    final_rec = '3-5 years'
            elif rec == '5-10 years':
                if (final_rec == '1 year') | (final_rec == '3 years') | (final_rec == '3-5 years'):
                    continue
                else:
                    final_rec = '5-10 years'
            elif rec == '7-10 years':
                if (final_rec == '1 year') | (final_rec == '3 years') | (final_rec == '3-5 years') | (final_rec == '5-10 years'):
                    continue
                else:
                    final_rec = '7-10 years'
            elif rec == '10 years':
                if (final_rec == '1 year') | (final_rec == '3 years') | (final_rec == '3-5 years') | (final_rec == '5-10 years') | (final_rec == '7-10 years'):
                    continue
                else:
                    final_rec = '10 years'
    else:
        if len(clinical_recommendation) == 0:
            # Use LLM to get output
            final_prompt = f"""Your task is to use the following rules, delimited by single backticks, to determine the screening colonoscopy interval.  
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
  Rule 8: If there are any number of tubular adenomas of any size with high grade dysplasia hisotology, repeat colonoscopy in 3 years. 

  Sessile Serrated Polyps: 
  Rule 9: If there are 1-2 sessile serrated polyps, and each are less than 10 mm in size, repeat colonscopy in 5-10 years. 
  Rule 10: If there are 3-4 sessile serrated polyps, and each are less than 10 mm in size, repeat colonoscopy in 3-5 years. 
  Rule 11: If there are 5-10 sessile serrated polyps of any size, repeat colonoscopy in 3 years. 
  Rule 12: If there are any number of sessile serrated polyps greater than or equal to 10 mm, repeat colonoscopy in 3 years.
  Rule 13: If there are any number of sessile serrated polyps with dysplasia, repeat colonoscopy in 3 years. 

  Traditional Serrated Polyps:
  Rule 14: If there are any number of traditional serrated adenomas, repeat colonoscopy in 3 years.

  Hyperplastic Polyps: 
  Rule 15: If there are less than or equal to 20 hyperplastic polyps, and each are less than 10 mm in size, repeat colonoscopy in 10 years. 
  Rule 16: If there are any number of hyperplastic polyps greater than or equal to 10 mm in size, repeat colonoscopy in 3-5 years.
  
  Rule 17: If there are no polyps found, repeat colonoscopy in 10 years
  `
    Provide the recommended colonoscopy screening interval for the input JSON below, delimited by triple backticks. 
        Input text: ```{json_response}``

  Provide the recommended colonoscopy screening interval as one of the following options:
  '1 year'
  '3 years'
  '3-5 years'
  '5-10 years'
  '7-10 years'
  '10 years'

        `"""
            messages = [{"role": "user", "content": final_prompt}]
            final_rec = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            ).choices[0].message["content"]
        else:
            final_rec = clinical_recommendation[0]
    return final_rec

def polyp_table_formatter(json_response):
    final_polyps_output = pd.DataFrame()
    matches = re.findall('\{(?:[^{}])*\}', json_response.replace('\n', '').replace('\t', ''))
    #print(matches)
    i=0
    polyp_count = 0
    polyp_size = ''
    polyp_location = ''
    polyp_histology = ''
    polyp_dysplasia = ''

    for match in matches:
        polyp_dict = output_parser.parse(match)

        # check against previous polyp to see if we need to append a new row or add to previous count
        if ((polyp_dict.get('size') == polyp_size) & (polyp_dict.get('location') == polyp_location) & (polyp_dict.get('histology') == polyp_histology) & (polyp_dict.get('dysplasia') == polyp_dysplasia)):
            polyp_count += 1
            final_polyps_output.at[i, 'Polyp Count'] = str(int(polyp_count))
        else:
            i+=1
            polyp_count = 1
            final_polyps_output.at[i, 'Polyp Count'] = str(int(polyp_count))
            final_polyps_output.at[i, 'Polyp Size'] = polyp_dict.get('size')
            final_polyps_output.at[i, 'Polyp Location'] = polyp_dict.get('location').lower()
            final_polyps_output.at[i, 'Polyp Histology'] = polyp_dict.get('histology').lower()
            final_polyps_output.at[i, 'Polyp Dysplasia'] = polyp_dict.get('dysplasia').lower()
        

        ### Save variables
        polyp_size = polyp_dict.get('size')
        polyp_location = polyp_dict.get('location').lower()
        polyp_histology = polyp_dict.get('histology').lower()
        polyp_dysplasia = polyp_dict.get('dysplasia').lower()
    return final_polyps_output

        