import os
import time
import json
import pickle
import gettext
import gradio as gr
from newspaper import Article
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, pipeline
import torch
import tiktoken
import re
import requests
import base64
from pathlib import Path
import locale

# システムのロケール設定を取得
default_locale = locale.getdefaultlocale()[0]
if default_locale == 'ja_JP':
    lang = gettext.translation('chatswitch',localedir='locale',languages=[default_locale])
else:
    lang = gettext.NullTranslations()
    
lang.install()
_ = lang.gettext

model_limit_token = {"stabilityai/japanese-stablelm-base-alpha-7b": 2048}

llm_model_list = [ ## "matsuo-lab/weblab-10b-instruction-sft"
    "stabilityai/japanese-stablelm-base-alpha-7b",
    "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2",
    "cyberagent/open-calm-7b",
    "line-corporation/japanese-large-lm-1.7b-instruction-sft"
    ]

talk_paramaters = {
    'system_role_name': 'system',
    'user_role_name': 'user',
    'assistant_role_name': 'assistant',
    'model_txt': '',
    'temperature': 0.7,
    'top_p': 0.37,
    'top_k': 100,
    'typical_p': 1.1,
    'repetition_penalty': 1.23,
    'encoder_repetition_penalty': 1,
    'no_repeat_ngram_size': 0,
    'epsilon_cutoff': 3,
    'eta_cutoff': 3,
    'min_length': 0,
    }

sd_subject = ""

def count_tokens(text, encoding: tiktoken.Encoding):
    return len(encoding.encode(text))

def get_url(text):
    match = re.search(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', text)
    return match.group()

def get_content(url):
    article = Article(url)
    article.download()
    article.parse()
    content = article.text
    return content

def limit_message(messages, messages_limit, encoding: tiktoken.Encoding):
    while True:
        contents = ""
        for message in messages:
            content = message['content']
            contents += content
        token_count = count_tokens(contents, encoding)
        if token_count >= messages_limit:
            if len(messages) < 4:
                return ''
            messages.pop(1)
            messages.pop(1)
            continue
        else:
            break
    return messages

def split_string_with_limit(text: str, limit: int , encoding: tiktoken.Encoding):
    parts = []
    current_part = []
    current_count = 0
    tokens = encoding.encode(text)

    for token in tokens:
        current_part.append(token)
        current_count += 1

        if current_count >= limit:
            parts.append(current_part)
            break

    if current_part:
        parts.append(current_part)

    return encoding.decode(parts[0])

def init_chat():
    global messages
    global messages_limit
    global ai_token_limit
    global encoding
    messages = []
    messages_limit = 1024 * 2
    ai_token_limit = 200
    messages_limit -= ai_token_limit + 30
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def load_conversation(data: gr.SelectData, list_history, chatbot) -> int:
    global conversations
    global current_conversation
    global messages
    select_index = data.index[0]
    select_col = data.index[1]
    select_id = list_history[select_index][2]
    if select_id == "":
        return list_history, chatbot
    if select_col == 1:
        del memory["conversations"][select_id]
        del list_history[select_index]
        if "id" in current_conversation:
            if select_id == current_conversation["id"]:
                new_chat()
        if list_history == []:
            list_history = [['','','']]
    else:
        memory["last_conversation"] = select_id
        current_conversation = memory["conversations"][select_id]
        messages = current_conversation["messages"]
    save_memory()
    chatbot = load_last_conversation()
    return list_history, chatbot

def load_last_conversation():
    temp = []
    if "messages" in current_conversation:
        temp2 = current_conversation["messages"]
        for i in range(1, len(temp2), 2):
            user_content = temp2[i]["content"]
            assistant_content = temp2[i + 1]["content"]
            temp.append((user_content, assistant_content))
    return temp

def add_history(list_history):
    match = False
    global current_conversation
    if "id" not in current_conversation:
        return list_history
    current_id = current_conversation["id"]
    for history_id in list_history:
        if current_id == history_id[2]:
            match = True
            break
    if not match:
        if list_history == [['','','']]:
            list_history = []
        list_history += [[current_conversation["title"],"🗑️",current_conversation["id"]],]
    return list_history

def load_history():
    if "conversations" in memory:
        tmp = []
        for conversation in memory["conversations"]:
            tmp.append([memory["conversations"][conversation]["title"],"🗑️",memory["conversations"][conversation]["id"]])
            return tmp
    return [['','','']]

def new_chat():
    global current_conversation
    global messages
    messages = []
    memory["last_conversation"] = ""
    current_conversation = {}
    save_memory()

def save_memory():
    # データを保存
    with open('memory.pkl', 'wb') as file:
        pickle.dump(memory, file)

def load_conversations():
    global conversations
    global current_conversation
    if "conversations" in memory:
        conversations = memory["conversations"]
        if "last_conversation" in memory and memory["last_conversation"] != "" and memory["last_conversation"] in conversations:
            current_conversation = conversations[memory["last_conversation"]]

# データを読み込み
memory = {}
memory_path = 'memory.pkl'

if os.path.exists(memory_path):
    with open(memory_path, 'rb') as file:
        memory = pickle.load(file)
    with open('memory.json', 'w', encoding='utf-8') as file:
        json.dump(memory, file, ensure_ascii=False, indent=4)

init_chat()
conversations = {}
current_conversation = {}
if "last_language" in memory:
    if memory["last_language"] == '日本語':
        default_locale = 'ja_JP'
    else:
        default_locale = 'en_US'
else:
    default_locale = locale.getdefaultlocale()[0]
if default_locale == 'ja_JP':
    lang = gettext.translation('chatswitch',localedir='locale',languages=[default_locale])
else:
    lang = gettext.NullTranslations()
lang.install()
_ = lang.gettext

load_conversations()

def add_text(history, text, system_txt):
    history = history + [(text, None)]
    current_conversation["system_iput"] = system_txt
    return history, gr.update(value="", interactive=False), system_txt

def chat(history,model_txt,ai_temperature,ai_top_p,ai_top_k,ai_typical_p,ai_repetition_penalty,ai_encoder_repetition_penalty,ai_no_repeat_ngram_size,ai_min_length):
    global messages
    global talk_paramaters

    talk_paramaters["temperature"] = float(ai_temperature)
    talk_paramaters["top_p"] = float(ai_top_p)
    talk_paramaters["top_k"] = int(ai_top_k)
    talk_paramaters["typical_p"] = float(ai_typical_p)
    talk_paramaters["repetition_penalty"] = float(ai_repetition_penalty)
    talk_paramaters["encoder_repetition_penalty"] = float(ai_encoder_repetition_penalty)
    talk_paramaters["no_repeat_ngram_size"] = int(ai_no_repeat_ngram_size)
    talk_paramaters["min_length"] = int(ai_min_length)

    system_input =  current_conversation["system_iput"]
    system_message = {}
    if len(messages) < 1:
        count = count_tokens(system_input, encoding)
        if count > messages_limit:
            system_input = split_string_with_limit(system_input, messages_limit, encoding)
        system_message =  {"role": talk_paramaters["assistant_role_name"], "content": system_input}
        messages.append(system_message)
    else :
        messages[0]["content"] = system_input

    user_input = history[-1][0]

    count = count_tokens(user_input, encoding)
    if count > messages_limit:
        user_input = split_string_with_limit(user_input, messages_limit, encoding)

    user_message = {"role": talk_paramaters["user_role_name"], "content": user_input}
    messages.append(user_message)

    talk_paramaters["model_txt"] = model_txt
    output = genarate_talk(messages)
    
    history[-1][1] = output

    ai_response = {"role": talk_paramaters['assistant_role_name'], "content": history[-1][1]}
    messages.append(ai_response)
    if "title" not in current_conversation:
        current_conversation["title"] = user_input[:10]
        current_conversation["create_time"] = time.time()
        current_conversation["id"] = str(time.time()) + user_input[:10]
        current_conversation["messages"] = []
        current_conversation["messages"].append(system_message)
        current_conversation["update_time"] = time.time()
    current_conversation["update_time"] = time.time()
    current_conversation["messages"].append(user_message)
    current_conversation["messages"].append(ai_response)
    conversations[current_conversation["id"]] = current_conversation
    memory["conversations"] = conversations
    memory["last_conversation"] = current_conversation["id"]

    save_memory()
    return history

def get_memory_variable(var_name):
    if var_name in memory:
        return memory[var_name]
    return ""

def set_memory_variable(var_name, value):
    global memory
    memory[var_name] = value
    save_memory()

def system_prompt_init():
    if "last_system_prompt" in memory:
        return memory["last_system_prompt"]
    return ""
    
def sd_host_init():
    if "last_sd_host" in memory:
        return memory["last_sd_host"]
    return ""
    
def sd_chekpoint_init():
    if "last_sd_chekpoint" in memory:
        return memory["last_sd_chekpoint"]
    return ""
    
def sd_prompt_init():
    if "last_sd_prompt" in memory:
        return memory["last_sd_prompt"]
    return ""
    
def sd_negative_init():
    if "last_sd_negative" in memory:
        return memory["last_sd_negative"]
    return ""
    
def get_SD_pictures(history,sd_enable,sd_host,sd_prompt,sd_negative,sd_chekpoint):
    if not sd_enable:
        return history
    params = {
        'address': sd_host,
        'save_img': True,
        'prompt_prefix': sd_prompt,
        'negative_prompt': sd_negative,
        'width': 512,
        'height': 512,
        'denoising_strength': 0.7,
        'restore_faces': False,
        'enable_hr': False,
        'hr_upscaler': 'ESRGAN_4x',
        'hr_scale': '1.0',
        'seed': -1,
        'sampler_name': 'Euler a',
        'steps': 20,
        'cfg_scale': 7,
        'sd_checkpoint': sd_chekpoint,
        'checkpoint_list': [" "]
    }
    payload = {
        "sd_model_checkpoint": params["sd_checkpoint"]
    }
    try:
        requests.post(url=f'{params["address"]}/sdapi/v1/options', json=payload)
    except:
        pass
    payload = {
        "prompt": params['prompt_prefix'],
        "seed": params['seed'],
        "sampler_name": params['sampler_name'],
        "enable_hr": params['enable_hr'],
        "hr_scale": params['hr_scale'],
        "hr_upscaler": params['hr_upscaler'],
        "denoising_strength": params['denoising_strength'],
        "steps": params['steps'],
        "cfg_scale": params['cfg_scale'],
        "width": params['width'],
        "height": params['height'],
        "restore_faces": params['restore_faces'],
        "override_settings_restore_afterwards": True,
        "negative_prompt": params['negative_prompt']
    }
    response = requests.post(url=f'{params["address"]}/sdapi/v1/txt2img', json=payload)
    response.raise_for_status()
    r = response.json()
    for img_str in r['images']:
        if params['save_img']:
            img_data = base64.b64decode(img_str)
            save_path = f'{date.today().strftime("%Y_%m_%d")}/{int(time.time())}'
            output_file = Path(f'outputs/{save_path}.png')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file.as_posix(), 'wb') as f:
                f.write(img_data)
    history = history + [((f"outputs/{save_path}.png",), None)]

    return history

def language_change(language):
    if language == "日本語":
        lang = gettext.translation('chatswitch',localedir='locale',languages=["ja_JP"])
    else:
        lang = gettext.NullTranslations()
    lang.install()
def language_init():
    if "last_language" in memory:
        return memory["last_language"]
    return ""
def temperature_init():
    if "temperature" in memory:
        return memory["temperature"]
    return talk_paramaters['temperature']
def top_p_init():
    if "top_p" in memory:
        return memory["top_p"]
    return talk_paramaters['top_p']
def top_k_init():
    if "top_k" in memory:
        return memory["top_k"]
    return talk_paramaters['top_k']
def typical_p_init():
    if "typical_p" in memory:
        return memory["typical_p"]
    return talk_paramaters['typical_p']
def repetition_penalty_init():
    if "repetition_penalty" in memory:
        return memory["repetition_penalty"]
    return talk_paramaters['repetition_penalty']
def encoder_repetition_penalty_init():
    if "encoder_repetition_penalty" in memory:
        return memory["encoder_repetition_penalty"]
    return talk_paramaters['encoder_repetition_penalty']
def no_repeat_ngram_size_init():
    if "no_repeat_ngram_size" in memory:
        return memory["no_repeat_ngram_size"]
    return talk_paramaters['no_repeat_ngram_size']
def min_length_init():
    if "min_length" in memory:
        return memory["min_length"]
    return talk_paramaters['min_length']
def model_init():
    if "last_model" in memory:
        return memory["last_model"]
    return ""

def model_save(model_txt):
    memory["last_model"] = model_txt
    save_memory()

model_loaded = False
def model_load(model_txt):
    global model
    global tokenizer
    global messages_limit
    
    if model_txt == "stabilityai/japanese-stablelm-base-alpha-7b":
        tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'])
        model = AutoModelForCausalLM.from_pretrained(model_txt,trust_remote_code=True,)
        messages_limit = model_limit_token["stabilityai/japanese-stablelm-base-alpha-7b"]
        model.half()
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
    elif model_txt == "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2":
        tokenizer = AutoTokenizer.from_pretrained(model_txt, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_txt)
        if torch.cuda.is_available():
            model = model.to("cuda")
    elif model_txt == "cyberagent/open-calm-7b":
        tokenizer = AutoTokenizer.from_pretrained(model_txt)
        model = AutoModelForCausalLM.from_pretrained(model_txt, device_map="auto", torch_dtype=torch.float16, trust_remote_code=False )
    elif model_txt == "line-corporation/japanese-large-lm-1.7b-instruction-sft":
        tokenizer = AutoTokenizer.from_pretrained(model_txt, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_txt)

    return model_txt

def genarate_talk(messages):
    global model
    global tokenizer
    global talk_paramaters

    model_txt = talk_paramaters['model_txt']

    talk = ""
    if model_txt == "stabilityai/japanese-stablelm-base-alpha-7b":
        for message in messages:
            talk += f"{message['content']}"
    else:
        talk = messages[0]['content'] + '\n'
        for message in messages[1:]:
            talk += f"{message['role']}: {message['content']}\n"
        talk += f"\n{talk_paramaters['assistant_role_name']}: "

    if 'tokenizer' not in globals():
        model_load(talk_paramaters['model_txt'])
    if model_txt == "stabilityai/japanese-stablelm-base-alpha-7b":
        input_ids = tokenizer.encode(talk,add_special_tokens=False,return_tensors="pt")
        torch.manual_seed(23)
        tokens = model.generate(
            input_ids.to(device=model.device),
            max_new_tokens=ai_token_limit,
            temperature=talk_paramaters["temperature"],
            top_p=talk_paramaters["top_p"],
            repetition_penalty=talk_paramaters["repetition_penalty"],
            encoder_repetition_penalty=talk_paramaters["encoder_repetition_penalty"],
            no_repeat_ngram_size=talk_paramaters["no_repeat_ngram_size"],
            do_sample=True,
        )
        output = tokenizer.decode(tokens[0], skip_special_tokens=True)
        output = output.replace(talk, '')
    elif  model_txt == "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2":
        talk = re.sub(talk_paramaters["user_role_name"], 'ユーザー', talk)
        talk = re.sub(talk_paramaters["assistant_role_name"], 'システム', talk)
        talk = talk.replace("\n","<NL>")
        token_ids = tokenizer.encode(talk, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                do_sample=True,
                max_new_tokens=ai_token_limit,
                temperature=talk_paramaters["temperature"],
                repetition_penalty=talk_paramaters["repetition_penalty"],
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
        output = output.replace("<NL>", "\n")
    elif model_txt == "cyberagent/open-calm-7b": ## "matsuo-lab/weblab-10b-instruction-sft" 
        inputs = tokenizer(talk, return_tensors="pt").to(model.device)
        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=ai_token_limit,
                temperature=talk_paramaters["temperature"],
                top_p=talk_paramaters["top_k"],
                top_k=talk_paramaters["top_k"],
                typical_p=talk_paramaters["typical_p"],
                epsilon_cutoff=talk_paramaters["epsilon_cutoff"],
                eta_cutoff=talk_paramaters["eta_cutoff"],
                repetition_penalty=talk_paramaters["repetition_penalty"],
                encoder_repetition_penalty=talk_paramaters["encoder_repetition_penalty"],
                no_repeat_ngram_size=talk_paramaters["no_repeat_ngram_size"],
                min_length=talk_paramaters["min_length"],
                pad_token_id=tokenizer.pad_token_id,
            ) 
        output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    elif model_txt == "line-corporation/japanese-large-lm-1.7b-instruction-sft":
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        talk = re.sub(talk_paramaters["user_role_name"], 'ユーザー', talk)
        talk = re.sub(talk_paramaters["assistant_role_name"], 'システム', talk)
        output = generator(
            talk,
            max_length = ai_token_limit,
            do_sample = True,
            temperature=talk_paramaters["temperature"],
            top_p=talk_paramaters["top_k"],
            top_k=talk_paramaters["top_k"],
            repetition_penalty=talk_paramaters["repetition_penalty"],
            num_beams = 1,
            pad_token_id = tokenizer.pad_token_id,
            num_return_sequences = 1,
        )[0]['generated_text']
        output = re.sub('システム', talk_paramaters["assistant_role_name"], output)
    output = re.sub('(.|\n)*' + talk_paramaters["assistant_role_name"] + ': (.*?)', '\\2', output)

    return output

with gr.Blocks(title="ChatSwitch") as demo:
    with gr.Row():
        with gr.Column(scale=5, min_width=600):
            chatbot = gr.Chatbot(load_last_conversation, elem_id="chatbot",height=800)
            user_prompt = gr.Textbox(placeholder=_("new line:Shift+Enter Send:Enter"),show_label=False,container=True,lines=1,autofocus=True)
        with gr.Column(scale=2):
            new_button = gr.ClearButton(value=_("new chat"),components=[chatbot])
            model_txt = gr.Dropdown(label=_("model"),value=model_init,container=True,choices=llm_model_list)
            system_txt = gr.Textbox(label=_("system prompt"),placeholder=_("system: user: assistant: save:Enter"),value=get_memory_variable("last_system_prompt"),container=True,lines=1)
            prompt_memo = gr.Textbox(label=_("prompt memo"),placeholder=_("memo holder save:Enter"),value=get_memory_variable("prompt_memo"),container=True,lines=1)
            with gr.Tab(_("Chat history")):
                list_history = gr.List(load_history,headers=[_("title"),"DEL","id"],datatype="str",col_count=(3, "fixed"),max_rows=10,interactive=False)
            with gr.Tab("AI"):
                with gr.Row():
                    ai_temperature = gr.Textbox(label="temperature",value=temperature_init(),scale=1)
                    ai_top_p = gr.Textbox(label="top_p",value=top_p_init(),scale=1)
                    ai_top_k = gr.Textbox(label="top_k",value=top_k_init(),scale=1)
                    ai_typical_p = gr.Textbox(label="typical_p",value=typical_p_init(),scale=1)
                    ai_repetition_penalty = gr.Textbox(label="repetition_penalty",value=repetition_penalty_init(),scale=1)
                    ai_encoder_repetition_penalty = gr.Textbox(label="encoder_repetition_penalty",value=encoder_repetition_penalty_init(),scale=1)
                    ai_no_repeat_ngram_size = gr.Textbox(label="no_repeat_ngram_size",value=no_repeat_ngram_size_init(),scale=1)
                    ai_min_length = gr.Textbox(label="min_length",value=min_length_init(),scale=1)
            with gr.Tab("Stable Diffusion"):
                sd_enable = gr.Checkbox(label=_("enable"),value=False)
                sd_host = gr.Textbox(label="Host",value=sd_host_init())
                sd_chekpoint = gr.Textbox(label="checkpoint",value=sd_chekpoint_init())
                sd_prompt = gr.Textbox(label="prompt",value=sd_prompt_init(),lines=3)
                sd_negative = gr.Textbox(label="negative prompt",value=sd_negative_init(),lines=2)
            with gr.Tab("option"):
                language = gr.Dropdown(label="language *need python reboot",value=language_init,container=True,choices=["Englich","日本語"])

    txt_msg = user_prompt.submit(add_text, [chatbot, user_prompt, system_txt], [chatbot, user_prompt, system_txt]).then(chat, inputs=[chatbot,model_txt,ai_temperature,ai_top_p,ai_top_k,ai_typical_p,ai_repetition_penalty,ai_encoder_repetition_penalty,ai_no_repeat_ngram_size,ai_min_length], outputs=chatbot).then(add_history, list_history, list_history).then(get_SD_pictures, inputs=[chatbot,sd_enable,sd_host,sd_prompt,sd_negative,sd_chekpoint], outputs=chatbot)
    txt_msg.then(lambda: gr.update(interactive=True), None, [user_prompt])
    new_button.click(new_chat).then(model_load,model_txt)
    model_txt.change(lambda x: memory.update({"last_model": x}),inputs=model_txt).then(save_memory).then(model_load,model_txt,model_txt)
    prompt_memo.submit(lambda x: memory.update({"prompt_memo": x}),prompt_memo).then(save_memory)
    list_history.select(fn=load_conversation,inputs=[list_history,chatbot],outputs=[list_history,chatbot])
    system_txt.submit(lambda x: memory.update({"last_system_prompt": x}),system_txt).then(save_memory)
    sd_enable.change(lambda x: memory.update({"last_sd_enable": x}),sd_enable).then(save_memory)
    sd_host.submit(lambda x: memory.update({"last_sd_host": x}),sd_host).then(save_memory)
    sd_chekpoint.submit(lambda x: memory.update({"last_sd_chekpoint": x}),sd_chekpoint).then(save_memory)
    sd_prompt.submit(lambda x: memory.update({"last_sd_prompt": x}),sd_prompt).then(save_memory)
    sd_negative.submit(lambda x: memory.update({"last_sd_negative": x}),sd_negative).then(save_memory)
    ai_temperature.submit(lambda x: memory.update({"last_ai_temperature": x}),ai_temperature).then(save_memory)
    ai_top_p.submit(lambda x: memory.update({"last_ai_top_p": x}),ai_top_p).then(save_memory)
    ai_top_k.submit(lambda x: memory.update({"last_ai_top_k": x}),ai_top_k).then(save_memory)
    ai_typical_p.submit(lambda x: memory.update({"last_ai_typical_p": x}),ai_typical_p).then(save_memory)
    ai_repetition_penalty.submit(lambda x: memory.update({"last_ai_repetition_penalty": x}),ai_repetition_penalty).then(save_memory)
    ai_encoder_repetition_penalty.submit(lambda x: memory.update({"last_ai_encoder_repetition_penalty": x}),ai_encoder_repetition_penalty).then(save_memory)
    ai_no_repeat_ngram_size.submit(lambda x: memory.update({"last_ai_no_repeat_ngram_size": x}),ai_no_repeat_ngram_size).then(save_memory)
    ai_min_length.submit(lambda x: memory.update({"last_ai_min_length": x}),ai_min_length).then(save_memory)
    language.change(lambda x: memory.update({"last_language": x}),inputs=language).then(save_memory).then(language_change,language)

if __name__ == "__main__":
    config_json = {}
    script_dir = os.getcwd()
    user_config_path = os.path.join(script_dir, "user_config.json")
    if os.path.exists(user_config_path):
        with open(user_config_path, 'r', encoding='utf-8') as file:
            config_json = json.load(file)

    if "server_port" in config_json:
        config_server_port = config_json["server_port"]
    else:
        config_server_port = None
    if "server_name" in config_json:
        config_server_name = config_json["server_name"]
    else:
        config_server_name = None
    if "inbrowser" in config_json:
        config_inbrowser = config_json["inbrowser"]
    else:
        config_inbrowser = False
    demo.launch(server_port=config_server_port,server_name=config_server_name, inbrowser=config_inbrowser)
