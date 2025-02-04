import os
import sys 
import time
from groq import Groq
from openai import OpenAI
from utils import *
from tqdm import tqdm
import json
from datetime import datetime
import logging

## openai client
openaiClient = setOpenAi(keyid=0)

## llama client
llamaClient = OpenAI(
    api_key="EMPTY",
    base_url= "",
)
clients = {'gpt': openaiClient, 'llama': llamaClient}

answer_MODEL = 'llama3-8b-8192'

now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")

sys_prompt = """
You are an online webshop agent. You are given an instruction to complete a shopping process. You have to generate the next step in the process. Your action should take into account of the current obversation (which is the page you are on) and the previous actions taken. 

There are two types of actions you can output:
1. search[query]: You can search for a product based on the query. The query is a string that describes the product you are looking for.
2. think[thoughts]: You can think about the current state of the shopping process and decide what to do next.
3. click[button]: You can click on a button on the page to navigate to another page. Where the button are presented in the observation that is bracketed by []. If you think a product is the best choice, you can click on the "Buy Now" button to end the process.

Example of a valid action:
search[noise cancelling cosycost usb microphone]
think[I want to compare the features of the products]
click[Buy Now]
click[< Prev]

The output should be in the following format don't include any other information:
one_of_the_action[query/thoughts/button]
"""

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

WEBSHOP_URL = "YOUR WEBSHOP ENV URL"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    # print(url)
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.find_all(string=True)
    visible_texts = list(filter(tag_visible, texts))
    #print(visible_texts)
    #if page_type == 'search':
        #prin
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 10:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 11: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info

class webshopEnv:
  def __init__(self):
    self.sessions = {}
    
    
  def clone_session(self, session):
    if session in self.sessions:
        return {**self.sessions[session]}  # Create a shallow copy of the session
    return None
  
  def step(self, session, action):
    done = False
    observation_ = None
    
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('plan['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]

      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session])
    # print(self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done
  
env = webshopEnv()

# trivial search & item, choose option
prompt1 = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[search] 

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[Click and check the first item B078GWRC1J to get more detailed information.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and I need to check other two options]
Observation: OK.

Action: click[< Prev]
Observation:
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[Click and check the second item B078GTKVXY to get more detailed information.]
Observation: OK.

Action: click[B078GTKVXY]
Observation:
[Back to Search]
[< Prev]
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99
Rating: 3.5/5
[Description]
[Features]
[Reviews]
[Buy Now]

Action: click[< Prev]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B078GTKVXY]
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[B08KBVJ4XN]
Barrel and Oak - Aluminum-Free Deodor


Action: think[Based on the previous browsing, I think the first item B078GWRC1J is the best choice.]
Observation: OK.

Action: click[B078GWRC1J]
Observation:
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now] 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Action: click[Buy Now]
"""

      
def decompose_sql_ws(clients, prompt , tokens_path, allo_model = 'gpt-4o'):
    '''
    1 examples are as follows:
    Question: 
    '''
    # options已经由Dict转化成了String的形式.
    prompt_for_decompose =  f"""
I have a online shopping request with some constraints and I need to find the best options, and I have search for the key words with some top results. You should help me to decompose the question into sub-steps that should be done for the following process
You will have the tools to help you with the online shopping.

Here is a exmaple of decomposed tasks:
Given the information of current state
1
Action: reset
Observation: 
WebShop 
Instruction:  
i want a noise cancelling cosycost usb microphone, and price lower than 60.00 dollars 
[Search] 

Action: search[noise cancelling cosycost usb microphone]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B0972Q1T8T] 
Cosycost USB Microphone,Condenser Computer PC Gaming Microphone for PS4/5 Laptop Windows Mac OS Android Phone,Noise Cancelling Instant Mute,Studio Mic for Voice,Music Recording,Podcasting,Streaming 
$32.99 
[B072L2D6LY] 
Andrea Communications NC-255VM USB On-Ear Stereo USB Computer Headset with Noise-Canceling Microphone, in-Line Volume/Mute Controls, and Plug 
$34.59 
[B071H84LTJ] 
Andrea Communications NC-455VM USB Over-Ear Circumaural Stereo USB Computer Headset with Noise-Canceling Microphone, in-Line Volume/Mute Controls, and Plug 
$49.24 

Example of decomposed tasks:
To solve the question, we need to clarify/solve:
1. Click and check item B0972Q1T8T to get more deatiled information.
2. Click and check item B072L2D6LY to get more detialed information.
3. Click and check item B071H84LTJ for more information.
4. Based on the more detailed information, I will compare to see which one fulfill by request.

Now, the current state is {prompt}

Based on the current process, please decompose it into sub-steps. 
Make sure to answer in this format Answer Format: (Please write each broken-down question on a separate line, starting with a number.)
To solve the question "xxx", we need to clarify/solve:
"1. Click and check xxxx",
"2. Click and check xxxx",
"3. sub-question 3".

"""

    # Use the llm() function instead of askLLM
    result = askLLM(clients, prompt_for_decompose, tokens_path, model=allo_model, stop=None)
    return result
  
  
def convert_steps_to_format(decom_commands):
    # 截取“we need to know:”后的内容
    start_index = decom_commands.find("clarify/solve:") + len("clarify/solve:")
    subtasks_text = decom_commands[start_index:].strip()

    # 将每个子任务单独列出
    subtasks = subtasks_text.split('\n')
    subtasks = [item for item in subtasks if item != '']  # 有的时候会有空字符串，需要删掉
    subtasks = [task.strip().split('. ', 1)[-1] for task in subtasks]
    
    steps_dict = {index + 1: value for index, value in enumerate(subtasks)}
    
    return subtasks, steps_dict

   
def webshop_run(idx, prompt, logger,to_print=True):
    global answer_MODEL, formatted_now
    
    # 用于记录token的路径
    tokens_path = f'{os.getcwd()}/tokens/cot_tokens_{answer_MODEL}_{formatted_now}.json'
    if not os.path.exists(tokens_path):
        with open(tokens_path, 'w') as f:
            json.dump({}, f)
    
    #记录一下模型分配
    models = []
    
    action = 'reset'
    observation = env.step(idx, action)[0]
    logger.info(f'quesiton: {observation}')
    init_prompt = prompt
    
    prompt = ''
    
    search_action = askLLM(
            clients,
            init_prompt + f' {action}\nObservation: {observation}\n\nAction: think[I should first generate a search action based on the instruction] \nObservation: OK. \n\nAction:',
            tokens_path,
            model=answer_MODEL,
            temperature=1,
            max_tokens=1000,
            stop=['\n']
        ).lstrip(' ').strip()
    
    # Execute search action
    try:
        search_observation = env.step(idx, search_action)[0]
    except AssertionError:
        search_observation = 'Invalid action!'
        
        
    prompt += f'Action: {action}\nObservation: {observation}\n\nAction: {search_action}\nObservation: {search_observation}\n\nAction:'
    
    # print(prompt)
    

    print('\n\n\n')
    
    decompose_steps = decompose_sql_ws(clients, prompt, tokens_path, 'gpt-4o')
    steps, steps_dict = convert_steps_to_format(decompose_steps)
    print(steps)
    
    for step in steps:
        if 'click and check' in step.lower():
            step_prompt = f" think[{step}]\nObservation: OK.\n\nAction:" #这个是每个物品的搜索结果
            # print(f'现在正在做的是: {step}的搜索')
            # print(f'现在的prompt是: {step_prompt}')
            prompt += step_prompt
            
            
            action = askLLM(
                clients,
                init_prompt + prompt,
                tokens_path,
                model=answer_MODEL,
                temperature=1,
                max_tokens=1000,
                stop=['\n']
            )
            
            print(f"=============={step}================")
            print(f'Action: {action}')
            print("================================")
            
            observation = env.step(idx, action)[0]
            
            # print(f'Action: {action}\nObservation: {observation}\n')
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
            
            print(f'Action: {action}\nObservation: {observation}\n')
            
            action = 'click[< Prev]'
            observation = env.step(idx, action)[0]
            
            print(f'Action: {action}\nObservation: {observation}\n')
            
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
  

    
    thought = ''
    for step in steps:
      if 'click and check' not in step.lower():
        thought += step
    action = f'think[{thought}]'
    obversation = 'OK.'
    
    print(f'Action: {action}\nObservation: {obversation}\n')
    prompt += f' {action}\nObservation: {obversation}\n\nAction:'

    for i in range(10):
        action = askLLM(
            clients,
            init_prompt + prompt[-(6400-len(init_prompt)):] + '(You should only output the action that can be done on the current page.)',
            tokens_path,
            model=answer_MODEL,
            temperature=1,
            max_tokens=1000,
            stop=['\n']
        )
        print(f"=============={i}================")
        print(f'Action: {action}')
        print("================================")
        
        res = None
        try:
            res = env.step(idx, action)
            observation = res[0]
            
        except AssertionError:
            observation = 'Invalid action!'
        
        if action.startswith('think'):
            observation = 'OK.'
        
        if to_print:
            print(f'Action: {action}\nObservation: {observation}\n')
            sys.stdout.flush()
        
        if i:
            prompt += f' {action}\nObservation: {observation}\n\nAction:'
        else:
            prompt += f" {observation}\n\nAction:"
        
        
        if res is not None and res[2]:
            return res[1], True
     

    return 0, False
    



def run_episodes(prompt, n=50):
    global answer_MODEL, formatted_now
    start_time = time.time()
    
    
    logger, log_file_name = setup_logger(f"cot_{answer_MODEL}_{formatted_now}")
    
    rs = []
    cnt = 0
    if_success = 0
    
    # Use tqdm with position and leave parameters
    for i in tqdm(range(n), position=0, leave=True):
        print('\n-----------------')
        print(f'current fixed_{i}')
        logger.info(f'\n\n\ncurrent fixed_{i}')
        try:
            r, success = webshop_run(f'fixed_{i}', prompt, logger, to_print=True)
            logger.info(f'reward: {r}')
            logger.info(f'success: {success}')
            cnt += 1
            if success:
                if_success += 1
        except AssertionError as ae:
            logger.error(f"AssertionError occurred: {str(ae)}")
            r = 0
            success = False
            logger.info(f'success: {success}')
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            r = 0
            success = False
            logger.info(f'success: {success}')
        
        rs.append(r)
        logger.info(f'reward: {r}')
        logger.info(f'success: {success}')
        
        # Clear the current line
        print('\033[K', end='\r')
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_reward = sum(rs) / cnt if cnt > 0 else 0
    success_rate = if_success / cnt if cnt > 0 else 0
    
    print(f'\nTime: {total_time}')
    print(f'Average reward: {avg_reward}')
    print(f'Success rate: {success_rate}')
    logger.info(f'\n\n\nTime: {total_time}')
    logger.info(f'Average reward: {avg_reward}')
    logger.info(f'Success rate: {success_rate}')

    return rs, total_time, avg_reward, success_rate

res1 = run_episodes(prompt1, 100)
