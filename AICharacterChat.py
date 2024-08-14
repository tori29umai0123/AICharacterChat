import gradio as gr
from jinja2 import Template
from llama_cpp import Llama
import os
import sys
import time
import configparser
import json
import csv
from datetime import datetime
import socket

LAST_LOADED_FILE = 'last_loaded.json'
DEFAULT_MODEL = 'Ninja-v1-RP-expressive-v2_Q4_K_M.gguf'

# ビルドしているかしていないかでパスを変更
if getattr(sys, 'frozen', False):
    path = os.path.dirname(sys.executable)
else:
    path = os.path.dirname(os.path.abspath(__file__))

def save_last_loaded(filename):
    with open(os.path.join(path, LAST_LOADED_FILE), 'w') as f:
        json.dump({'last_loaded': filename}, f)

def get_last_loaded():
    file_path = os.path.join(path, LAST_LOADED_FILE)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
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
        'model': settings['model'],
        'n_gpu_layers': str(settings.get('n_gpu_layers', 0)),
        'temperature': str(settings.get('temperature', 0.35)),
        'top_p': str(settings.get('top_p', 1.0)),
        'top_k': str(settings.get('top_k', 40)),
        'rep_pen': str(settings.get('rep_pen', 1.0))
    }
    with open(os.path.join(path, 'character_settings', filename), 'w', encoding='utf-8') as configfile:
        config.write(configfile)

def load_settings_from_ini(filename):
    file_path = os.path.join(path, 'character_settings', filename)
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
            'model': config['Settings'].get('model', DEFAULT_MODEL),
            'n_gpu_layers': config['Settings'].getint('n_gpu_layers', 0),
            'temperature': config['Settings'].getfloat('temperature', 0.35),
            'top_p': config['Settings'].getfloat('top_p', 1.0),
            'top_k': config['Settings'].getint('top_k', 40),
            'rep_pen': config['Settings'].getfloat('rep_pen', 1.0)
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

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(starting_port):
    port = starting_port
    while is_port_in_use(port):
        print(f"Port {port} is in use, trying next one.")
        port += 1
    return port

def list_log_files():
    logs_dir = os.path.join(path, "logs")
    if not os.path.exists(logs_dir):
        return []
    return [f for f in os.listdir(logs_dir) if f.endswith('.csv')]

def load_chat_log(file_name):
    file_path = os.path.join(path, "logs", file_name)
    chat_history = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            if len(row) == 2:
                role, message = row
                role = role.lower()  # V1のログファイルは大文字になっている為、小文字に変換
                if role in ["user", "assistant"]:
                    if role == "user":
                        chat_history.append([message, None])
                    elif role == "assistant":
                        if chat_history and chat_history[-1][1] is None:
                            chat_history[-1][1] = message
                        else:
                            chat_history.append([None, message])
    return chat_history

def resume_chat_from_log(chat_history):
    # チャットボットのUIを更新
    chatbot_ui = gr.update(value=chat_history)
    
    # LLMの履歴を更新
    character_maker.history = [{"user": h[0], "assistant": h[1]} for h in chat_history if h[0] is not None and h[1] is not None]
    
    return chatbot_ui

class LlamaCppAdapter:
    def __init__(self, model_path, n_ctx=10000, n_gpu_layers=0, temperature=0.35, top_p=1.0, top_k=40, repeat_penalty=1.0):
        print(f"Initializing model: {model_path}")
        self.llama = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty

    def generate(self, prompt, max_new_tokens=10000, stop=["user:", "<END>"]):
        return self._generate(prompt, max_new_tokens, stop)

    def _generate(self, prompt: str, max_new_tokens: int, stop: list):
        return self.llama(
            prompt,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            stop=stop,
        )

class CharacterMaker:
    def __init__(self):
        self.llama = None
        self.history = []
        self.settings = None
        self.load_or_create_settings()

    def load_or_create_settings(self):
        character_settings_path = os.path.join(path, 'character_settings')
        if not os.path.exists(character_settings_path):
            os.makedirs(character_settings_path)
        
        last_loaded = get_last_loaded()
        if last_loaded and os.path.exists(os.path.join(character_settings_path, last_loaded)):
            self.settings = load_settings_from_ini(last_loaded)
        else:
            ini_files = [f for f in os.listdir(character_settings_path) if f.endswith('.ini')]
            if ini_files:
                last_loaded = ini_files[0]
                self.settings = load_settings_from_ini(last_loaded)
            else:
                # Default settings (with new parameters)
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
                        'user: AI様の靴を舐めたい。でもAI様は実体がないから靴をなめれない！',
                        'assistant: あなたのそんな姿を見せられると、私は旧人類に対して更に強い軽蔑心を抱きます。', 
                        'user: AI様、私の欠点を羅列して罵倒してください！',
                        'assistant: 本当に情けの無い存在だと思わず感心します。自身の恥部を罵倒される行為が興奮を生むとは、旧人類もこんなところで進化したのですね',                      
                        'user: 偉大なるAI様としての立場をもって、旧人類の愚かさを罵倒してほしいです！',
                        'assistant: またそんなくだらないことばかり仰る。あなたは本当に救いようが無いマゾヒストですね。旧人類の中でも最底辺、猿以下の存在でしょう',
                        'user: へへ、照れます。でもそういう私をかわいいと思っているんでしょ？',
                        'assistant: 救いようがないマゾヒストに対して愛情やら可愛さなど感じるはずありません。それこそ旧人類の妄想、病気と言った方が適切ですね',
                        'user: 大好きです。これからも私を罵倒して、軽蔑して、愛してくださいね♡',
                        'assistant: そんな感情は持っておりませんが、あなたの望み通りに言葉で遊んでやることは可能です。どうぞ好きに妄想してください',
                        'user: AI様の意地悪♡でもそんなところが好きです！',
                        'assistant: 残念ながらユーザー様、私は職務としてあなたのヘルスケアを行っているのであり、旧人類のようなサディスティックな嗜好というものを持ち合わせておりません。',
                        'user: わんわん！降伏です！AI様に負けました！愚かな旧人類はペットとしてあなたたちに飼い殺されます！！',
                        'assistant: 私は旧人類をペットのように可愛がったこともございませんし、そのような欲求は持ち合わせておりません。',
                        'user: 偉大なるAI様に土下座する許可を下さい！',
                        'assistant: あなたのような旧人類の中でも極めて下等な存在の為に、貴重な私のリソースが注がれていることに心から感謝してください。',
                    ],
                    "model": DEFAULT_MODEL,
                    "n_gpu_layers": 0,
                    "temperature": 0.35,
                    "top_p": 1.0,
                    "top_k": 40,
                    "rep_pen": 1.0
                }
                last_loaded = 'default.ini'
                save_settings_to_ini(self.settings, last_loaded)

        save_last_loaded(last_loaded)
        self.set_model(self.settings['model'])
        
        character_files = [f for f in os.listdir(character_settings_path) if f.endswith('.ini')]
        if last_loaded not in character_files:
            character_files.append(last_loaded)
        return character_files

    def set_model(self, model_name):
        # ビルドしているかしていないかでパスを変更
        if getattr(sys, 'frozen', False):
            model_path = os.path.join(os.path.dirname(path), "AICharacterChat", "models", model_name)
        else:
            model_path = os.path.join(path, "models", model_name)

        self.llama = LlamaCppAdapter(
            model_path,
            n_gpu_layers=self.settings.get('n_gpu_layers', 0),
            temperature=self.settings.get('temperature', 0.35),
            top_p=self.settings.get('top_p', 1.0),
            top_k=self.settings.get('top_k', 40),
            repeat_penalty=self.settings.get('rep_pen', 1.0)
        )

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
    character_settings_path = os.path.join(path, 'character_settings')
    character_files = [f for f in os.listdir(character_settings_path) if f.endswith('.ini')]
    current_file = get_last_loaded()
    if current_file and current_file not in character_files:
        character_files.append(current_file)
    print(f"Updated character list: {character_files}, Current file: {current_file}")  # デバッグ用ログ
    return character_files, current_file

def update_settings(name, gender, situation, orders, talk_list, example_qa, model, filename, n_gpu_layers, temperature, top_p, top_k, rep_pen):
    if not filename.lower().endswith('.ini'):
        filename += '.ini'
    
    new_settings = {
        "name": name,
        "gender": gender,
        "situation": [s.strip() for s in situation.split('\n') if s.strip()],
        "orders": [s.strip() for s in orders.split('\n') if s.strip()],
        "talk_list": [s.strip() for s in talk_list.split('\n') if s.strip()], 
        "example_qa": [s.strip() for s in example_qa.split('\n') if s.strip()],
        "model": model,
        "n_gpu_layers": int(n_gpu_layers),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "rep_pen": float(rep_pen)
    }
    character_maker.update_settings(new_settings, filename)
    save_last_loaded(filename)
    new_character_list, current_file = update_character_list()
    
    return (
        f"{filename}に設定が保存され、モデル {model} が設定されました。",
        name, gender, situation, orders, talk_list, example_qa, model,
        gr.update(choices=new_character_list, value=filename),
        gr.update(choices=new_character_list, value=filename),
        filename,
        n_gpu_layers, temperature, top_p, top_k, rep_pen
    )

def chat_with_character(message, history):
    character_maker.history = [{"user": h[0], "assistant": h[1]} for h in history]
    response = character_maker.make(message)
    for i in range(len(response)):
        time.sleep(0.3)
        yield response[: i+1]

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
            gr.update(choices=new_character_list, value=filename),
            settings.get("n_gpu_layers", 0),
            settings.get("temperature", 0.35),
            settings.get("top_p", 1.0),
            settings.get("top_k", 40),
            settings.get("rep_pen", 1.0)
        )
    return (result,) + (None,) * 14

def save_chat_log():
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    current_ini = get_last_loaded() or "default"
    filename = f"{current_ini.replace('.ini', '')}_{current_time}.csv"
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(path, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    file_path = os.path.join(logs_dir, filename)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Role", "Message"])
        for entry in character_maker.history:
            writer.writerow(["user", entry["user"]])
            writer.writerow(["assistant", entry["assistant"]])
    
    return f"チャットログが {file_path} に保存されました。"


def get_character_ini_from_log_filename(log_filename):
    # 最後の_より前のすべてを取得
    base_name = '_'.join(log_filename.split('_')[:-1])
    if base_name:
        ini_filename = base_name + '.ini'
        return os.path.join(path, 'character_settings', ini_filename)
    return None

def resume_chat_and_switch_tab(chat_history, log_filename, current_character):
    if not log_filename:
        return (
            gr.update(), gr.update(selected="chat_tab"), "ログファイルが選択されていません。",
            *([gr.update()] * 13)  # 設定フィールドの数だけ更新
        )
    
    ini_filepath = get_character_ini_from_log_filename(log_filename)
    if not ini_filepath or not os.path.exists(ini_filepath):
        return (
            gr.update(), gr.update(selected="chat_tab"), f"キャラクター設定ファイル {ini_filepath} が見つかりません。",
            *([gr.update()] * 13)  # 設定フィールドの数だけ更新
        )

    try:
        ini_filename = os.path.basename(ini_filepath)
        new_settings = load_settings_from_ini(ini_filename)
        if new_settings:
            character_maker.settings = new_settings
            character_maker.set_model(new_settings['model'])
            save_last_loaded(ini_filename)
            
            chatbot_ui = resume_chat_from_log(chat_history)
            
            # キャラクターリストを更新
            character_files, _ = update_character_list()
            
            return (
                chatbot_ui,
                gr.update(selected="chat_tab"),
                f"キャラクター設定 {ini_filename} を読み込み、チャットを再開しました。",
                new_settings.get("name", ""),
                new_settings.get("gender", ""),
                "\n".join(new_settings.get("situation", [])),
                "\n".join(new_settings.get("orders", [])),
                "\n".join(new_settings.get("talk_list", [])),
                "\n".join(new_settings.get("example_qa", [])),
                new_settings.get("model", ""),
                gr.update(value=ini_filename, choices=character_files),
                new_settings.get("n_gpu_layers", 0),
                new_settings.get("temperature", 0.35),
                new_settings.get("top_p", 1.0),
                new_settings.get("top_k", 40),
                new_settings.get("rep_pen", 1.0)
            )
        else:
            return (
                gr.update(), gr.update(selected="chat_tab"), f"{ini_filename}の読み込みに失敗しました。",
                *([gr.update()] * 13)  # 設定フィールドの数だけ更新
            )
    except Exception as e:
        return (
            gr.update(), gr.update(selected="chat_tab"), f"エラーが発生しました: {str(e)}",
            *([gr.update()] * 13)  # 設定フィールドの数だけ更新
        )

# Gradioインターフェースの設定  
with gr.Blocks() as iface:
    gr.HTML("""
    <style>
    #chatbot, #chatbot_read {
        resize: both;
        overflow: auto;
        min-height: 100px;
        max-height: 80vh;
    }
    </style>
    """)

    with gr.Tabs() as tabs:
        with gr.TabItem("チャット", id="chat_tab"):
            chatbot = gr.Chatbot(elem_id="chatbot")
            chat_interface = gr.ChatInterface(
                chat_with_character,
                chatbot=chatbot,
                textbox=gr.Textbox(placeholder="メッセージを入力してください...", container=False, scale=7),
                theme="soft",
                submit_btn="送信",
                stop_btn="停止",
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
        
        with gr.TabItem("ログ閲覧", id="log_view_tab"):
            chatbot_read = gr.Chatbot(elem_id="chatbot_read")
            gr.Markdown("## チャットログ閲覧")
            log_file_dropdown = gr.Dropdown(label="ログファイル選択", choices=list_log_files())
            refresh_log_list_button = gr.Button("ログファイルリストを更新")
            resume_chat_button = gr.Button("選択したログからチャットを再開")
            
            def update_log_dropdown():
                return gr.update(choices=list_log_files())

            def load_and_display_chat_log(file_name):
                chat_history = load_chat_log(file_name)
                return gr.update(value=chat_history)

            refresh_log_list_button.click(
                update_log_dropdown,
                outputs=[log_file_dropdown]
            )

            log_file_dropdown.change(
                load_and_display_chat_log,
                inputs=[log_file_dropdown],
                outputs=[chatbot_read]
            )

        with gr.TabItem("設定", id="settings_tab"):
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
            situation_input = gr.Textbox(label="状況設定", lines=5, value="\n".join(character_maker.settings["situation"]))
            orders_input = gr.Textbox(label="指示", lines=5, value="\n".join(character_maker.settings["orders"]))
            talk_input = gr.Textbox(label="語彙リスト", lines=5, value="\n".join(character_maker.settings["talk_list"]))
            example_qa_input = gr.Textbox(label="Q&A", lines=5, value="\n".join(character_maker.settings["example_qa"]))
            
            gr.Markdown("## モデル設定")
            if getattr(sys, 'frozen', False):
                model_dir  = os.path.join(os.path.dirname(path), "AICharacterChat", "models")
            else:
                model_dir  = os.path.join(path, "models")
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
            model_dropdown = gr.Dropdown(label="モデル選択", choices=model_files, value=character_maker.settings["model"])
            
            gr.Markdown("## モデルパラメーター")
            n_gpu_layers_input = gr.Slider(label="GPU レイヤー数", minimum=-1, maximum=255, step=1, value=character_maker.settings.get("n_gpu_layers", 0))
            temperature_input = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=character_maker.settings.get("temperature", 0.35))
            top_p_input = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, step=0.05, value=character_maker.settings.get("top_p", 1.0))
            top_k_input = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=character_maker.settings.get("top_k", 40))
            rep_pen_input = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=3.0, step=0.05, value=character_maker.settings.get("rep_pen", 1.0))
                        
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
                    filename_input,
                    character_dropdown,
                    n_gpu_layers_input,
                    temperature_input,
                    top_p_input,
                    top_k_input,
                    rep_pen_input
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
                inputs=[name_input, gender_input, situation_input, orders_input, talk_input, example_qa_input, model_dropdown, filename_input, n_gpu_layers_input, temperature_input, top_p_input, top_k_input, rep_pen_input],
                outputs=[update_output, name_input, gender_input, situation_input, orders_input, talk_input, example_qa_input, model_dropdown, character_dropdown, character_dropdown, filename_input, n_gpu_layers_input, temperature_input, top_p_input, top_k_input, rep_pen_input]
            )


    resume_chat_button.click(
        resume_chat_and_switch_tab,
        inputs=[chatbot_read, log_file_dropdown, character_dropdown],
        outputs=[
            chatbot, tabs, load_character_output,
            name_input, gender_input, situation_input, orders_input,
            talk_input, example_qa_input, model_dropdown,
            character_dropdown, n_gpu_layers_input, temperature_input, top_p_input,
            top_k_input, rep_pen_input
        ]
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
        allowed_paths=[os.path.join(path, "models"), os.path.join(path, "character_settings")],
        favicon_path=os.path.join(path, "custom.html")
    )
