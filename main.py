from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modelo import CNN, device
from database import init_db, salvar_resultado, carregar_estatisticas
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
import io
from llama_cpp import Llama 
import pandas as pd
import matplotlib.pyplot as plt
import base64

# Inicializações
app = FastAPI()
init_db()

# Jinja2 e arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Modelos
model_cnn = CNN().to(device)
model_cnn.load_state_dict(torch.load("models/cnn_model.pth", map_location=device))
model_cnn.eval()

# LLM via llama-cpp-python (modelo em português)
llm = Llama(model_path="models/gguf/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048, n_threads=4)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

labels = ['Doente', 'Saudável']

# Função de IA com Llama
def chat_with_ia(user_input: str) -> str:
    prompt = (
        "Você é um assistente veterinário inteligente e gentil. "
        "Se a pergunta for simples ou informal, responda de forma amigável e acessível, como em uma conversa cotidiana. "
        "Se a pergunta for técnica, responda com linguagem acadêmica e precisão, adequada a estudantes universitários. "
        "Todas as respostas devem estar em português do Brasil, com clareza e empatia.\n\n"
        f"Pergunta: {user_input}"
    )
    
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Você é um assistente veterinário que responde em português do Brasil."},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=300,
        stop=None
    )
    return output['choices'][0]['message']['content'].strip()


# Página inicial
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Rota de mensagem de texto
@app.post("/message")
async def process_message(request: Request):
    data = await request.json()
    user_msg = data.get("message", "")
    ia_response = chat_with_ia(user_msg)
    return JSONResponse({"response": ia_response})

# Rota de upload de imagem
@app.post("/upload")
async def process_image(image: UploadFile = File(...)):
    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_cnn(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    result = labels[predicted.item()]
    confidence = confidence.item() * 100
    salvar_resultado(result)

    resposta = f"Resultado: {result}\nConfiança: {confidence:.2f}%"
    return JSONResponse({"response": resposta})

# Rota de geração do grafico de estatisticas
@app.get("/grafico")
async def gerar_grafico():
    try:
        df = carregar_estatisticas()  
        if "resultado" not in df.columns:
            return JSONResponse(content={"grafico": None})

        contagem = df["resultado"].value_counts()

        fig, ax = plt.subplots()
        contagem.plot(kind="bar", ax=ax, color=["salmon", "mediumseagreen"])
        ax.set_title("Diagnósticos")
        ax.set_xlabel("Classe")
        ax.set_ylabel("Quantidade")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return JSONResponse(content={"grafico": img_base64})
    except Exception as e:
        print("Erro ao gerar gráfico:", e)
        return JSONResponse(content={"grafico": None})