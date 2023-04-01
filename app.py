import gradio as gr
import os
import sys
import argparse
from utils import *
from chat_func import *
   
my_api_key = os.environ.get('my_api_key')

if my_api_key == "empty":
    print("Please give a api key!")
    sys.exit(1)
    
gr.Chatbot.postprocess = postprocess


with open("css_new.css", "r", encoding="utf-8") as f:
    css = f.read()

with gr.Blocks(css=css,theme='gradio/soft') as demo:
    history = gr.State([])
    token_count = gr.State([])
    promptTemplates = gr.State(load_template('myprompts.json', mode=2))
    user_api_key = gr.State(my_api_key)
    TRUECOMSTANT = gr.State(True)
    FALSECONSTANT = gr.State(False)
    gr.Markdown(title)
    
    with gr.Row().style(equal_height=True):
        with gr.Column(scale=12):
            with gr.Accordion("Build by [45度科研人](WeChat Public Accounts)", open=False):
                gr.Markdown(description)
        with gr.Column(scale=1):
            with gr.Box():
                toggle_dark = gr.Button(value="Toggle Dark").style(full_width=True)
   
    with gr.Row(scale=1).style(equal_height=True):
        with gr.Column(scale=5):
            with gr.Column():
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(show_label=False, placeholder="Enter text and press submit", visible=True).style(container=False)
                submitBtn = gr.Button("Submit",variant="primary").style(container=False)
                emptyBtn = gr.Button("Restart Conversation",variant="secondary")
                status_display = gr.Markdown("")

        with gr.Column():
            with gr.Column(min_width=50):
                with gr.Tab(label="ChatGPT"):
                    with gr.Column():
                        with gr.Row():
                            keyTxt = gr.Textbox(show_label=False, placeholder=f"You can input your own openAI API-key",value=hide_middle_chars(my_api_key),visible=True, type="password",  label="API-Key")
                            systemPromptTxt = gr.Textbox(show_label=True,placeholder=f"Set a custom insruction for the chatbot: You are a helpful assistant.",label="Custom prompt",value=initial_prompt,lines=10,)

                        with gr.Row():
                            templateSelectDropdown = gr.Dropdown(label="load from template",choices=load_template('myprompts.json', mode=1),
                                multiselect=False,value=load_template('myprompts.json', mode=1)[0],).style(container=False)                
                
                with gr.Tab(label="Settings"):
                    with gr.Column():
                        with gr.Row():
                            with gr.Column(scale=3):
                                saveFileName = gr.Textbox(show_label=True, placeholder=f"output file name...",label='Save conversation history', value="")
                            with gr.Column(scale=1):
                                exportMarkdownBtn = gr.Button("Save")
                        with gr.Row():
                            with gr.Column(scale=1):
                                downloadFile = gr.File(interactive=False)
    gr.Markdown("""
    ###  <div align=center>you can follow the WeChat public account [45度科研人] and leave me a message!  </div>
    <div style="display:flex; justify-content:center;">
        <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/wechat-simple.png" style="margin-right:25px;width:200px;height:200px;">
        <div style="width:25px;"></div>
        <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/shoukuanma222.png" style="margin-left:25px;width:170px;height:190px;">
    </div>
    """)
    
    toggle_dark.click(None,_js="""
        () => {
            document.body.classList.toggle('dark');
            document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)'
        }""",)
    
    keyTxt.submit(submit_key, keyTxt, [user_api_key, status_display])
    keyTxt.change(submit_key, keyTxt, [user_api_key, status_display])
    # Chatbot
    user_input.submit(predict,[user_api_key,systemPromptTxt,history,user_input,chatbot,token_count,],[chatbot, history, status_display, token_count],show_progress=True)
    user_input.submit(reset_textbox, [], [user_input])

    submitBtn.click(predict,[user_api_key,systemPromptTxt,history,user_input,chatbot,token_count,],[chatbot, history, status_display, token_count],show_progress=True)
    submitBtn.click(reset_textbox, [], [user_input])

    emptyBtn.click(reset_state,outputs=[chatbot, history, token_count, status_display],show_progress=True,)

    templateSelectDropdown.change(get_template_content,[promptTemplates, templateSelectDropdown, systemPromptTxt],[systemPromptTxt],show_progress=True,)
    exportMarkdownBtn.click(export_markdown,[saveFileName, systemPromptTxt, history, chatbot],downloadFile,show_progress=True,)
    downloadFile.change(load_chat_history,[downloadFile, systemPromptTxt, history, chatbot],[saveFileName, systemPromptTxt, history, chatbot],)


if __name__ == "__main__":
    demo.queue().launch(debug=False,show_api=False) 
