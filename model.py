import openai
import json
from tqdm import tqdm
import re
import os

# API配置 - 使用原生OpenAI格式的中转API
openai.api_key = ""
openai.api_base = ""  # 根据你的API文档，使用/v1路径


def get_completion(prompt, model="claude-3-5-sonnet-20241022"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def save_API_file(file_name, code_str):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding='utf-8') as f_java:
        f_java.write(code_str)


def save_json_file(file_name, data):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', encoding='utf-8') as f_json:
        json.dump(data, f_json, indent=2)


def post_process(response, language, wo_cot=False):
    Chatgpt_class = None
    if language == 'java':
        if re.search(r"(Complete code:\s*```(?:java)?\s*)([\s\S]*?)(?=```)",
                     response):
            Chatgpt_class = re.search(r"(Complete code:\s*```(?:java)?\s*)([\s\S]*?)(?=```)",
                                      response).group(2)
        elif re.search(r"(Complete code:\s*```(?:java)?\s*)([\s\S]*)",
                       response):
            Chatgpt_class = re.search(r"(Complete code:\s*```(?:java)?\s*)([\s\S]*)",
                                      response).group(2)
        elif re.search(r"(Complete code:\s*)([\s\S]*)",
                       response):
            Chatgpt_class = re.search(r"(Complete code:\s*)([\s\S]*)",
                                      response).group(2)

        if wo_cot and Chatgpt_class is None:
            if re.search(r"(```(?:java)?\s*)([\s\S]*?)(?=```)",
                         response):
                Chatgpt_class = re.search(r"(```(?:java)?\s*)([\s\S]*?)(?=```)",
                                          response).group(2)
            elif re.search(r"(```(?:java)?\s*)([\s\S]*)",
                           response):
                Chatgpt_class = re.search(r"(```(?:java)?\s*)([\s\S]*)",
                                          response).group(2)

    elif language == 'python':
        if re.search(r"(Complete code:\s*```(?:python)?\s*)([\s\S]*?)(?=```)",
                     response):
            Chatgpt_class = re.search(r"(Complete code:\s*```(?:python)?\s*)([\s\S]*?)(?=```)",
                                      response).group(2)
        elif re.search(r"(Complete code:\s*```(?:python)?\s*)([\s\S]*)",
                       response):
            Chatgpt_class = re.search(r"(Complete code:\s*```(?:python)?\s*)([\s\S]*)",
                                      response).group(2)
        elif re.search(r"(Complete code:\s*)([\s\S]*)",
                       response):
            Chatgpt_class = re.search(r"(Complete code:\s*)([\s\S]*)",
                                      response).group(2)
    return Chatgpt_class


if __name__ == '__main__':
    language = 'python'
    # language = 'java'

    if language == 'java':
        exam_json_file = 'few_shot_learning/java_examples.json'
        eval_snippets = 'eval_snippets/eval_snippets_java.json'

        save_API_dir = 'eval_API/java/'

    elif language == 'python':
        exam_json_file = 'few_shot_learning/python_examples.json'
        eval_snippets = 'eval_snippets/eval_snippets_python.json'

        save_API_dir = 'eval_API/python/'

    with open(exam_json_file, 'r', encoding='utf-8') as f1:
        examples = json.load(f1)

    with open(eval_snippets, 'r') as f:
        data_list = json.load(f)
        for data in tqdm(data_list[:]):
            question_title = data['question_title']
            question_post = data['question_post']
            answer_post = data['answer_post']
            code_snippets = data['code_snippets']
            answer_id = data['answer_id']
            code_snippet = code_snippets[0]
            i = 0
            
            # 断点续传: 检查文件是否已存在,跳过已生成的API
            output_file = save_API_dir + f'Code2API{answer_id}.' + ('java' if language == 'java' else 'py')
            if os.path.exists(output_file):
                continue

            if answer_id == '573468':
                # The answer post contains two code snippets,
                # and the code snippet that needs to be converted is the second one
                code_snippet = code_snippets[1]
                i = 1

            prompt_java_wo_fs = f"""
Give you a context including a question title, a question post and an answer post, 
your task is to transform the java code snippet in the answer post into a java method based on the context. 

To solve the problem, do the following:
step 1 - Recover import statements based on the code snippet. If necessary, it can be none.
step 2 - Define a public class Chatgpt that will be used to wrap the method.
step 3 - Create "public static" modifier for the method.
step 4 - Create the method name based on the context or the code snippet itself.
step 5 - Infer parameter list based on the code snippet. If necessary, it can be none.
step 6 - Infer return statements based on the code snippet. If necessary, it can be none.
step 7 - Infer throws statements based on the code snippet. If necessary, it can be none.
step 8 - Output the complete code based on above results.

When performing step 5, you can consider the following principle:
Any variable or literal that can be rewritten as input parameters must be a parameter.
When performing step 6, you can consider the following principle:
According to the intent of the code snippet or context, necessary calculation results must be used as return values.

Now, give you the following context:
Question title: <{question_title}>
Qustion post: <{question_post}>
Answer post: <{answer_post}>
Code snippet in the answer post: <{code_snippet}>

Please output the results in the following format:
Specific steps: <the results of step 1-7>
Complete code: <the result of step 8>
"""

            wo_cot_example_output = """Complete code: \n```java\npublic class Chatgpt {\n    public static String getBinaryOrResult(String binary1, String binary2) {\n        int x1 = Integer.parseInt(binary1, 2);\n        int x2 = Integer.parseInt(binary2, 2);\n        int val = x1 | x2;\n        return Integer.toBinaryString(val);\n    }\n}\n```"""
            prompt_java_wo_cot = f"""
Give you a context including a question title, a question post and an answer post, 
your task is to transform the java code snippet in the answer post into a java method based on the context. 

Here are some examples:
Example one:
Question title: <{examples[0]['question_title']}>
Qustion post: <{examples[0]['question_post']}>
Answer post: <{examples[0]['answer_post']}>
Code snippet in the answer post: <{examples[0]['code_snippets'][0]}>
{wo_cot_example_output}

Now, give you the following context:
Question title: <{question_title}>
Qustion post: <{question_post}>
Answer post: <{answer_post}>
Code snippet in the answer post: <{code_snippet}>

Please output the results in the following format:
Complete code: <the complete code here>
"""

            prompt_java_wo_both = f"""
Give you a context including a question title, a question post and an answer post, 
your task is to transform the java code snippet in the answer post into a java method based on the context. 

Now, give you the following context:
Question title: <{question_title}>
Qustion post: <{question_post}>
Answer post: <{answer_post}>
Code snippet in the answer post: <{code_snippet}>

Please output the results in the following format:
Specific steps: <your steps to solve this task>
Complete code: <the complete code here>
"""
            prompt_java = f"""
Give you a context including a question title, a question post and an answer post, 
your task is to transform the java code snippet in the answer post into a java method based on the context. 

To solve the problem, do the following:
step 1 - Recover import statements based on the code snippet. If necessary, it can be none.
step 2 - Define a public class Chatgpt that will be used to wrap the method.
step 3 - Create "public static" modifier for the method.
step 4 - Create the method name based on the context or the code snippet itself.
step 5 - Infer parameter list based on the code snippet. If necessary, it can be none.
step 6 - Infer return statements based on the code snippet. If necessary, it can be none.
step 7 - Infer throws statements based on the code snippet. If necessary, it can be none.
step 8 - Output the complete code based on above results.

When performing step 5, you can consider the following principle:
Any variable or literal that can be rewritten as input parameters must be a parameter.
When performing step 6, you can consider the following principle:
According to the intent of the code snippet or context, necessary calculation results must be used as return values.

Here are some examples:
Example one:
Question title: <{examples[0]['question_title']}>
Qustion post: <{examples[0]['question_post']}>
Answer post: <{examples[0]['answer_post']}>
Code snippet in the answer post: <{examples[0]['code_snippets'][0]}>
{examples[0]['response']}

Now, give you the following context:
Question title: <{question_title}>
Qustion post: <{question_post}>
Answer post: <{answer_post}>
Code snippet in the answer post: <{code_snippet}>

Please output the results in the following format:
Specific steps: <the results of step 1-7>
Complete code: <the result of step 8>
"""
            prompt_python = f"""
Give you a context including a question title, a question post and an answer post, 
your task is to transform the python code snippet in the answer post into a python function based on the context. 

To solve the problem, do the following:
step 1 - Recover import statements based on the code snippet. If necessary, it can be none.
step 2 - Create the method name based on the context or the code snippet itself.
step 3 - Infer parameter list based on the code snippet. If necessary, it can be none.
step 4 - Infer return statements based on the code snippet. If necessary, it can be none.
step 5 - Output the complete code based on above results.

When performing step 3, you can consider the following principle:
Any variable or literal that can be rewritten as input parameters must be a parameter.
When performing step 4, you can consider the following principle:
According to the intent of the code snippet or context, necessary calculation results must be used as return values.

Here are some examples:
Example one:
Question title: <{examples[0]['question_title']}>
Qustion post: <{examples[0]['question_post']}>
Answer post: <{examples[0]['answer_post']}>
Code snippet in the answer post: <{examples[0]['code_snippets'][0]}>
{examples[0]['response']}

Now, give you the following context:
Question title: <{question_title}>
Qustion post: <{question_post}>
Answer post: <{answer_post}>
Code snippet in the answer post: <{code_snippet}>

Please output the results in the following format:
Specific steps: <the results of step 1-4>
Complete code: <the result of step 5>
"""
            if language == 'java':
                prompt = prompt_java
            elif language == 'python':
                prompt = prompt_python

            Long_input = False
            response = ''
            while not response:
                try:
                    response = get_completion(prompt)
                except (openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.Timeout,
                        openai.error.APIConnectionError):
                    continue
                except (openai.error.InvalidRequestError, openai.error.RateLimitError):
                    print(f'answer_id:{answer_id}, input is too long')
                    Long_input = True
                    break

            if Long_input:
                continue

            # data['response'] = response
            # print(response)

            Chatgpt_class = post_process(response, language, wo_cot=False)
            if not Chatgpt_class:
                print(f'answer_id:{answer_id}, the output format is wrong.')
                continue

            Chatgpt_api = re.sub('Chatgpt', f'Code2API{answer_id}', Chatgpt_class)
            if language == 'java':
                Chatgpt_api = 'package com.code2api.api;\n' + Chatgpt_api
                save_API_file(save_API_dir + f'Code2API{answer_id}.java', Chatgpt_api)
            elif language == 'python':
                save_API_file(save_API_dir + f'Code2API{answer_id}.py', Chatgpt_api)

