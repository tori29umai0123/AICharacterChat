# AICharacterChat
このアプリケーションは、AIキャラクターとチャットができる対話型アプリです。ユーザーはカスタマイズ可能なAIキャラクターと会話を楽しむことができます。

## ビルド方法
```
pyinstaller "AICharacterChat.spec"
copy AICharacterChat_model_DL.cmd dist\AICharacterChat\AICharacterChat_model_DL.cmd
copy custom.html dist\AICharacterChat\custom.html
copy prompt.jinja2 dist\AICharacterChat\prompt.jinja2
copy AICharacterChat_ReadMe.txt dist\AICharacterChat\AICharacterChat_ReadMe.txt
```
