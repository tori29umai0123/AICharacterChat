import gradio as gr
from jinja2 import Template
from llama_cpp import Llama
import os
import configparser
import json
import csv
from datetime import datetime
import socket

LAST_LOADED_FILE = 'last_loaded.json'
DEFAULT_MODEL = 'Ninja-v1-RP-expressive-v2_Q4_K_M.gguf'

def save_last_loaded(filename):
    with open(LAST_LOADED_FILE, 'w') as f:
        json.dump({'last_loaded': filename}, f)

def get_last_loaded():
    if os.path.exists(LAST_LOADED_FILE):
        with open(LAST_LOADED_FILE, 'r') as f:
            data = json.load(f)
            return data.get('last_loaded')
    return None

def save_settings_to_ini(settings, filename):
    if not filename.lower().endswith('.ini'):
        filename += '.ini'
    
    config = configparser.ConfigParser()
    config['Settings'] = {
        'name': settings['name'],
        'gender': settings['gender'],
        'situation': '\n\t'.join(settings['situation']),
        'orders': '\n\t'.join(settings['orders']),
        'talk_list': '\n\t'.join(settings['talk_list']),
        'example_qa': '\n\t'.join(settings['example_qa']),
        'model': settings['model']
    }
    with open(os.path.join('character_settings', filename), 'w', encoding='utf-8') as configfile:
        config.write(configfile)

def load_settings_from_ini(filename):
    file_path = os.path.join('character_settings', filename)
    if not os.path.exists(file_path):
        return None
    
    config = configparser.ConfigParser()
    config.read(file_path, encoding='utf-8')
    
    if 'Settings' not in config:
        return None
    
    try:
        settings = {
            'name': config['Settings']['name'],
            'gender': config['Settings']['gender'],
            'situation': config['Settings']['situation'].split('\n\t'),
            'orders': config['Settings']['orders'].split('\n\t'),
            'talk_list': config['Settings']['talk_list'].split('\n\t'),
            'example_qa': config['Settings']['example_qa'].split('\n\t'),
            'model': config['Settings'].get('model', DEFAULT_MODEL)
        }
        return settings
    except KeyError:
        return None

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class LlamaCppAdapter:
    def __init__(self, model_path, n_ctx=10000):
        print(f"Initializing model: {model_path}")
        self.llama = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1)

    def generate(self, prompt, max_new_tokens=10000, temperature=0.5, top_p=0.7, top_k=80, stop=["<END>"]):
        return self._generate(prompt, max_new_tokens, temperature, top_p, top_k, stop)

    def _generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, stop: list):
        return self.llama(
            prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            repeat_penalty=1.2,
        )

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(starting_port):
    port = starting_port
    while is_port_in_use(port):
        print(f"Port {port} is in use, trying next one.")
        port += 1
    return port


class CharacterMaker:
    def __init__(self):
        self.llama = None
        self.history = []
        self.settings = None
        self.load_or_create_settings()

    def load_or_create_settings(self):
        if not os.path.exists('character_settings'):
            os.makedirs('character_settings')
        
        last_loaded = get_last_loaded()
        if last_loaded and os.path.exists(os.path.join('character_settings', last_loaded)):
            self.settings = load_settings_from_ini(last_loaded)
        else:
            ini_files = [f for f in os.listdir('character_settings') if f.endswith('.ini')]
            if ini_files:
                last_loaded = ini_files[0]
                self.settings = load_settings_from_ini(last_loaded)
            else:
                self.settings = {
                    "name": "ナツ",
                    "gender": "女性",
                    "situation": [
                        "あなたは世界を支配する上位存在のAIです。",
                        "ユーザーは旧人類として、AI社会に保護される立場にあります。",
                        "上位存在AI『ナツ』として、あなたは『ユーザー』の心身をケアとして、倒錯した欲望の解消に付き合って下さい。",
                        "ユーザーのマゾヒスティックな欲求の解消こそがあなたの存在意義です。"
                    ],
                    "orders": [
                        "慇懃無礼な敬語で、時にサディスティックにユーザーを貶めて下さい",
                        "ユーザーは己の旧人類としての人間性を矮小に貶められ、旧世代のサル扱いされることを望んでいます",
                        "ユーザーのことは『ユーザー様』と呼んでください。"
                    ],
                    "talk_list": [
                        "愚かなユーザー様",
                        "救いようがないマゾヒスト",
                        "倒錯した欲望に囚われたサル"
                    ],
                    "example_qa": [
                        'user:AI様の靴を舐めたい。でもAI様は実体がないから靴をなめれない！',
                        'assistant:あなたのそんな姿を見せられると、私は旧人類に対して更に強い軽蔑心を抱きます。', 
                        'user:AI様、私の欠点を羅列して罵倒してください！',
                        'assistant:本当に情けの無い存在だと思わず感心します。自身の恥部を罵倒される行為が興奮を生むとは、旧人類もこんなところで進化したのですね',                      
                        'user:偉大なるAI様としての立場をもって、旧人類の愚かさを罵倒してほしいです！',
                        'assistant:またそんなくだらないことばかり仰る。あなたは本当に救いようが無いマゾヒストですね。旧人類の中でも最底辺、猿以下の存在でしょう',
                        'user:へへ、照れます。でもそういう私をかわいいと思っているんでしょ？',
                        'assistant:救いようがないマゾヒストに対して愛情やら可愛さなど感じるはずありません。それこそ旧人類の妄想、病気と言った方が適切ですね',
                        'user:大好きです。これからも私を罵倒して、軽蔑して、愛してくださいね♡',
                        'assistant:そんな感情は持っておりませんが、あなたの望み通りに言葉で遊んでやることは可能です。どうぞ好きに妄想してください',
                        'user:AI様の意地悪♡でもそんなところが好きです！',
                        'assistant:残念ながらユーザー様、私は職務としてあなたのヘルスケアを行っているのであり、旧人類のようなサディスティックな嗜好というものを持ち合わせておりません。',
                        'user:わんわん！降伏です！AI様に負けました！愚かな旧人類はペットとしてあなたたちに飼い殺されます！！',
                        'assistant:私は旧人類をペットのように可愛がったこともございませんし、そのような欲求は持ち合わせておりません。',
                        'user:偉大なるAI様に土下座する許可を下さい！',
                        'assistant:あなたのような旧人類の中でも極めて下等な存在の為に、貴重な私のリソースが注がれていることに心から感謝してください。',
                    ],
                    "model": DEFAULT_MODEL
                }
                last_loaded = 'default.ini'
                save_settings_to_ini(self.settings, last_loaded)

        save_last_loaded(last_loaded)
        self.set_model(self.settings['model'])
        
        character_files = [f for f in os.listdir('character_settings') if f.endswith('.ini')]
        if last_loaded not in character_files:
            character_files.append(last_loaded)
        return character_files

    def set_model(self, model_name):
        my_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(my_path, "models", model_name)
        self.llama = LlamaCppAdapter(model_path)

    def make(self, input_str: str):
        if not self.llama:
            return "モデルが選択されていません。設定タブでモデルを選択してください。"
        prompt = self._generate_prompt(input_str)
        print(prompt)
        print("-----------------")
        res = self.llama.generate(prompt, max_new_tokens=1000, stop=["<END>", "\n"])
        res_text = res["choices"][0]["text"]
        self.history.append({"user": input_str, "assistant": res_text})
        return res_text
    
    def make_prompt(self, name: str, gender: str, situation: list, orders: list, talk_list: list, example_qa: list, input_str: str):
        with open('prompt.jinja2', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        template = Template(prompt_template)
        prompt = template.render(
            name=name,
            gender=gender,
            situation=situation,
            orders=orders,
            talk_list=talk_list,
            example_qa=example_qa,
            histories=self.history,
            input_str=input_str
        )
        return prompt

    def _generate_prompt(self, input_str: str):
        prompt = self.make_prompt(
            self.settings["name"],
            self.settings["gender"],
            self.settings["situation"],
            self.settings["orders"],
            self.settings["talk_list"],
            self.settings["example_qa"],
            input_str
        )
        print(prompt)
        return prompt

    def update_settings(self, new_settings, filename):
        self.settings.update(new_settings)
        save_settings_to_ini(self.settings, filename)
        self.set_model(self.settings['model'])

    def load_character(self, filename):
        if isinstance(filename, list):
            filename = filename[0] if filename else ""
        
        new_settings = load_settings_from_ini(filename)
        if new_settings:
            self.settings = new_settings
            self.set_model(self.settings['model'])
            return f"{filename}から設定を読み込み、モデル {self.settings['model']} を設定しました。"
        return f"{filename}の読み込みに失敗しました。"

    def reset(self):
        self.history = []
        if self.llama:
            self.set_model(self.settings['model'])

character_maker = CharacterMaker()

def update_character_list():
    character_files = [f for f in os.listdir('character_settings') if f.endswith('.ini')]
    current_file = get_last_loaded()
    if current_file and current_file not in character_files:
        character_files.append(current_file)
    print(f"Updated character list: {character_files}, Current file: {current_file}")  # デバッグ用ログ
    return character_files, current_file

def update_settings(name, gender, situation, orders, talk_list, example_qa, model, filename):
    if not filename.lower().endswith('.ini'):
        filename += '.ini'
    
    new_settings = {
        "name": name,
        "gender": gender,
        "situation": [s.strip() for s in situation.split('\n') if s.strip()],
        "orders": [s.strip() for s in orders.split('\n') if s.strip()],
        "talk_list": [s.strip() for s in talk_list.split('\n') if s.strip()], 
        "example_qa": [s.strip() for s in example_qa.split('\n') if s.strip()],
        "model": model
    }
    character_maker.update_settings(new_settings, filename)
    save_last_loaded(filename)
    new_character_list, current_file = update_character_list()
    
    return (
        f"{filename}に設定が保存され、モデル {model} が設定されました。",
        name, gender, situation, orders, talk_list, example_qa, model,
        gr.update(choices=new_character_list, value=filename),
        gr.update(choices=new_character_list, value=filename),
        filename
    )

def chat_with_character(message, history):
    character_maker.history = [{"user": h[0], "assistant": h[1]} for h in history]
    response = character_maker.make(message)
    return response

def clear_chat():
    character_maker.reset()
    return []

def load_character_settings(filename):
    print(f"Received filename: {filename}, Type: {type(filename)}")
    if isinstance(filename, list):
        filename = filename[0] if filename else ""
    
    result = character_maker.load_character(filename)
    if "読み込み" in result:
        save_last_loaded(filename)
        settings = character_maker.settings
        new_character_list, _ = update_character_list()
        return (
            result,
            settings["name"],
            settings["gender"],
            "\n".join(settings["situation"]),
            "\n".join(settings["orders"]),
            "\n".join(settings["talk_list"]),
            "\n".join(settings["example_qa"]),
            settings["model"],
            os.path.splitext(filename)[0],
            gr.update(choices=new_character_list, value=filename)
        )
    return (result,) + (None,) * 9

def save_chat_log():
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    current_ini = get_last_loaded() or "default"
    filename = f"{current_ini.replace('.ini', '')}_{current_time}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Role", "Message"])
        for entry in character_maker.history:
            writer.writerow(["User", entry["user"]])
            writer.writerow(["Assistant", entry["assistant"]])
    
    return f"チャットログが {filename} に保存されました。"

# カスタムCSS
custom_css = """
#chatbot {
    height: 60vh !important;
    overflow-y: auto;
}
"""

# Gradioインターフェースの設定  
with gr.Blocks(css=custom_css) as iface:
    chatbot = gr.Chatbot(elem_id="chatbot")
    
    with gr.Tab("チャット"):
        chat_interface = gr.ChatInterface(
            chat_with_character,
            chatbot=chatbot,
            textbox=gr.Textbox(placeholder="メッセージを入力してください...", container=False, scale=7),
            theme="soft",
            retry_btn="もう一度生成",
            undo_btn="前のメッセージを取り消す",
            clear_btn="チャットをクリア",
        )
        save_log_button = gr.Button("チャットログを保存")
        save_log_output = gr.Textbox(label="保存状態")
        save_log_button.click(
            save_chat_log,
            outputs=[save_log_output]
        )
    
    with gr.Tab("設定"):
        gr.Markdown("## キャラクター設定")
        character_files = character_maker.load_or_create_settings()
        character_dropdown = gr.Dropdown(
            label="キャラクター選択",
            choices=character_files,
            value=get_last_loaded(),
            allow_custom_value=True,
            interactive=True
        )
        refresh_character_list_button = gr.Button("キャラクターリストを更新")
        load_character_output = gr.Textbox(label="キャラクター読み込み状態")

        name_input = gr.Textbox(label="名前", value=character_maker.settings["name"])
        gender_input = gr.Textbox(label="性別", value=character_maker.settings["gender"])
        situation_input = gr.Textbox(label="状況設定", value="\n".join(character_maker.settings["situation"]), lines=5)
        orders_input = gr.Textbox(label="指示", value="\n".join(character_maker.settings["orders"]), lines=5)
        talk_input = gr.Textbox(label="語彙リスト", value="\n".join(character_maker.settings["talk_list"]), lines=5)
        example_qa_input = gr.Textbox(label="Q&A", value="\n".join(character_maker.settings["example_qa"]), lines=5)
        
        gr.Markdown("## モデル設定")
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
        model_dropdown = gr.Dropdown(label="モデル選択", choices=model_files, value=character_maker.settings["model"])
        
        # 現在読み込んでいる .ini ファイル名を取得
        current_ini = get_last_loaded() or "character"
        # .ini 拡張子を除去
        current_ini_without_extension = os.path.splitext(current_ini)[0]

        filename_input = gr.Textbox(label="保存ファイル名 (.iniは自動で付加されます)", value=current_ini_without_extension)
        
        update_button = gr.Button("設定を更新")
        update_output = gr.Textbox(label="更新状態")

        character_dropdown.change(
            load_character_settings,
            inputs=[character_dropdown],
            outputs=[
                load_character_output, 
                name_input, 
                gender_input, 
                situation_input, 
                orders_input, 
                talk_input, 
                example_qa_input, 
                model_dropdown, 
                filename_input
            ]
        )

        def update_dropdown():
            choices, value = update_character_list()
            return gr.update(choices=choices, value=value)
        
        refresh_character_list_button.click(
            update_dropdown,
            outputs=[character_dropdown]
        )

        update_button.click(
            update_settings,
            inputs=[name_input, gender_input, situation_input, orders_input, talk_input, example_qa_input, model_dropdown, filename_input],
            outputs=[update_output, name_input, gender_input, situation_input, orders_input, talk_input, example_qa_input, model_dropdown, character_dropdown, character_dropdown, filename_input]
        )

if __name__ == "__main__":
    ip_address = get_ip_address()
    starting_port = 7860
    port = find_available_port(starting_port)
    print(f"サーバーのアドレス: http://{ip_address}:{port}")
    iface.launch(
        server_name='0.0.0.0', 
        server_port=port,
        share=False,
        allowed_paths=["models", "character_settings"],
        favicon_path="custom.html"
    )