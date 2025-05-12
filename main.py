import gradio as gr
import subprocess
import shlex
import signal
import requests

# 用于保存当前运行的子进程
current_process = None

def run_command(command):
    """执行命令并实时返回输出"""
    global current_process
    try:
        args = shlex.split(command)
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        current_process = process  # 保存当前进程对象

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                yield output.strip()

        remaining = process.stdout.read()
        if remaining:
            yield remaining.strip()

    except Exception as e:
        yield f"错误: {str(e)}"
    finally:
        current_process = None  # 清除当前进程引用

def stop_process():
    """终止当前运行的子进程"""
    global current_process
    if current_process is not None:
        # 先尝试优雅终止
        current_process.send_signal(signal.SIGTERM)
        try:
            current_process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            # 若仍未关闭，强制杀死
            current_process.kill()
        return "进程已终止"
    else:
        return "当前没有运行的进程"


#生成大模型部署命令
def generate_command(model_path, deploy_type):
    """根据模型路径和部署类型生成完整命令"""
    common_params = "--host 0.0.0.0 --port 30000"
    if deploy_type == "sglang":
        return f"python3 -m sglang.launch_server --model-path {model_path} {common_params}"
    elif deploy_type == "vllm serve":
        return f"vllm serve {model_path} {common_params}"
    else:
        return "错误：未知的部署类型"
#生成模型下载命令
def generate_download_command(git_path, savepath):
    """
    生成模型下载命令
    :param git_path: 模型的 Git 仓库地址，例如 "https://huggingface.co/Qwen/Qwen2-7B "
    :param savepath: 本地保存路径，例如 "./Qwen2-7B"
    :return: 下载命令字符串
    """
    return f"git clone {git_path} {savepath}"

def generate_quantize_command(model_path, savepath):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = model_path
    quant_path = savepath
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    return(f'Model is quantized and saved at "{quant_path}"')

# 检查服务是否运行
def is_server_running():
    try:
        res = requests.get("http://localhost:30000/health")
        return res.status_code == 200
    except:
        return False

# 获取当前服务状态
def get_status():
    return "🟢 运行中" if is_server_running() else "🔴 未运行"

def model_predict(message, chat_history):
    try:
        response = requests.post("http://localhost:30000/generate", json={
            "text": message,
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 256
            }
        })
        response.raise_for_status()
        data = response.json()

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": data["text"]})

        return "", chat_history

    except requests.exceptions.ConnectionError:
        return "", [{"role": "system", "content": "❌ 无法连接到服务，请先启动模型服务器"}]
    except Exception as e:
        return "", [{"role": "system", "content": f"❌ 错误: {str(e)}"}]


# 创建 Gradio 界面
with gr.Blocks(title="极简大模型部署脚本") as demo:
    with gr.Tab("🏗︎ 部署操作"):
        gr.Markdown("# 📋 命令行执行器\n在下方输入命令，按回车执行")

        model_path_input = gr.Textbox(
            label="模型路径",
            placeholder="例如：Qwen/Qwen3-0.6B",
            value="Qwen/Qwen3-0.6B"
        )

        with gr.Row():
            deploy_type = gr.Radio(
                choices=["sglang", "vllm serve"],
                label="部署类型",
                value="sglang",
                scale=3
            )
            deploy_button = gr.Button("确认部署", scale=1)

        command_input = gr.Textbox(
            label="输入命令",
            placeholder="例如：ping 8.8.8.8 或 ls -l",
            value="echo Hello World"
        )

        output_box = gr.Textbox(
            label="日志输出",
            lines=20,
            max_lines=50,
            interactive=False
        )

        with gr.Row():
            stop_button = gr.Button("终止进程", variant="stop")
            # 命令生成
        deploy_button.click(
            fn=generate_command,
            inputs=[model_path_input, deploy_type],
            outputs=command_input
        )

        # 命令执行
        command_input.submit(fn=run_command, inputs=command_input, outputs=output_box, scroll_to_output=True)

        # 终止命令
        stop_button.click(fn=stop_process, inputs=[], outputs=output_box)


    with gr.Tab("服务监控与聊天测试"):
        gr.Markdown("# 🤖 大模型服务监控")

        status_box = gr.Textbox(
            label="📡 服务状态",
            value=get_status,
            every=3,
            interactive=False
        )

        with gr.Row():
            chatbot = gr.Chatbot(
                label="对话历史",
                bubble_full_width=False,
                type="messages"
            )
            msg = gr.Textbox(
                label="输入问题",
                lines=5,
                placeholder="输入你的问题...",
            )

        with gr.Row():
            clear = gr.ClearButton([msg, chatbot])
            send_button = gr.Button("发送", variant="primary")

        msg.submit(fn=model_predict, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=True)
        send_button.click(fn=model_predict, inputs=[msg, chatbot], outputs=[msg, chatbot])

    with gr.Tab("模型下载"):
        gr.Markdown("# ⬇️ 模型下载")
        with gr.Row():
            model_path_input = gr.Textbox(
                label="模型路径",
                placeholder="https://huggingface.co/Qwen/Qwen3-0.6B",
                value="https://huggingface.co/Qwen/Qwen3-0.6B"
            )
            savepath_input = gr.Textbox(
                label="保存路径",
                placeholder="./models",
                value="./models"
            )
            dowload_button = gr.Button("确认下载", scale=1)
        command_input = gr.Textbox(
            label="输入命令",
            placeholder="例如：ping 8.8.8.8 或 ls -l",
            value="echo Hello World"
        )

        output_box = gr.Textbox(
            label="日志输出",
            lines=20,
            max_lines=50,
            interactive=False
        )


        with gr.Row():
            stop_button = gr.Button("终止进程", variant="stop")
            # 命令生成
        dowload_button.click(
            fn=generate_download_command,
            inputs=[model_path_input, savepath_input],
            outputs=command_input
        )
        # 命令执行
        command_input.submit(fn=run_command, inputs=command_input, outputs=output_box, scroll_to_output=True)

        # 终止命令
        stop_button.click(fn=stop_process, inputs=[], outputs=output_box)
    with gr.Tab("模型量化"):
        gr.Markdown("# 🔹模型量化")
        with gr.Row():
            model_path_input = gr.Textbox(
                label="模型路径",
                placeholder="https://huggingface.co/Qwen/Qwen3-0.6B",
                value="https://huggingface.co/Qwen/Qwen3-0.6B"
            )
            savepath_input = gr.Textbox(
                label="保存路径",
                placeholder="./models",
                value="./models"
            )
            dowload_button = gr.Button("确认量化", scale=1)
        command_input = gr.Textbox(
            label="量化结果",
        )

        output_box = gr.Textbox(
            label="日志输出",
            lines=20,
            max_lines=50,
            interactive=False
        )


        with gr.Row():
            stop_button = gr.Button("终止进程", variant="stop")
            # 命令生成
        dowload_button.click(
            fn=generate_quantize_command,
            inputs=[model_path_input, savepath_input],
            outputs=command_input
        )
        # 命令执行
        command_input.submit(fn=run_command, inputs=command_input, outputs=output_box, scroll_to_output=True)


if __name__ == "__main__":
    demo.launch(share=True)
