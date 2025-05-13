import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from modelo import CNN, device
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
from huggingface_hub import login
import pandas as pd
import matplotlib.pyplot as plt
from database import init_db, salvar_resultado, carregar_estatisticas
import torch.nn.functional as F
import os  # Adicionado para ler variável de ambiente

# Configurações iniciais
hf_logging.set_verbosity_error()
st.set_page_config(page_title="DogTor Chat", layout="centered")

# Carregamento do CSS
try:
    with open("styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    st.error("Erro ao carregar CSS")
    st.write(e)

st.title("🧠 DogTor Chat")
st.caption("Converse com a IA e envie imagens para detectar doenças!")

# Inicializa o banco de dados
init_db()

# Carrega modelo CNN
device_to_use = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cnn = None
try:
    model_cnn = CNN().to(device_to_use)
    model_cnn.load_state_dict(torch.load('models/cnn_model.pth', map_location=device_to_use))
    model_cnn.eval()
except Exception as e:
    st.error(f"Erro ao carregar modelo CNN: {e}")

labels = ['Doente', 'Saudável']
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemma_model" not in st.session_state:
    st.session_state.gemma_model = None
    st.session_state.gemma_tokenizer = None

# Função para carregar modelo Gemma
def carregar_modelo_gemma():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        st.error("Token do Hugging Face não encontrado nas variáveis de ambiente.")
        st.stop()
    login(token=hf_token)
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", revision="float16", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b-it", torch_dtype=torch.float16, device_map="auto"
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it").to("cpu")
    return tokenizer, model

# Carregamento do modelo Gemma
if st.session_state.gemma_model is None or st.session_state.gemma_tokenizer is None:
    with st.spinner("🔄 Carregando modelo de linguagem... pode levar até 2 minutos!"):
        try:
            tokenizer_gemma, model_gemma = carregar_modelo_gemma()
            st.session_state.gemma_model = model_gemma
            st.session_state.gemma_tokenizer = tokenizer_gemma
        except Exception as e:
            st.error(f"Erro ao carregar modelo Gemma: {e}")

# Função de conversa com a IA
def chat_with_gemma(user_input):
    tokenizer = st.session_state.gemma_tokenizer
    model = st.session_state.gemma_model
    if not tokenizer or not model:
        return "❌ O modelo ainda não foi carregado corretamente. Tente novamente mais tarde."

    prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1].strip()
    else:
        response = response.strip()
    return response if response else "Desculpe, não consegui entender. Pode repetir?"

# Exibição invertida do histórico da conversa
for msg in reversed(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], caption="Imagem enviada", width=250)
        else:
            st.markdown(msg["content"])

# Entrada de texto e imagem integradas
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.chat_input("Digite uma mensagem...")
with col2:
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Processamento do texto
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            if any(word in user_input.lower() for word in ["imagem", "foto"]):
                reply = "📌 Para enviar uma imagem, use o botão à direita da barra de mensagens."
            else:
                reply = chat_with_gemma(user_input)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# Processamento da imagem
if uploaded_file and model_cnn:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device_to_use)

    with torch.no_grad():
        output = model_cnn(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence = probs.max().item() * 100
        _, predicted = torch.max(output, 1)
        result = labels[predicted.item()]

    salvar_resultado(result)

    # Exibe imagem e resultado no chat
    st.session_state.messages.append({"role": "user", "type": "image", "content": image})
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"🧬 Resultado da imagem: **{result}**\n🎯 Confiança: **{confidence:.2f}%**"
    })

    # Geração de resposta automática da IA comentando o resultado
    with st.spinner("🧠 Interpretando o resultado..."):
        ia_reply = chat_with_gemma(f"A imagem foi classificada como '{result}' com {confidence:.2f}% de confiança. O que isso significa?")
    st.session_state.messages.append({"role": "assistant", "content": ia_reply})

elif uploaded_file and not model_cnn:
    st.warning("Modelo não carregado. Não é possível processar a imagem no momento.")

# Estatísticas
st.subheader("📊 Estatísticas de Diagnósticos")
df = carregar_estatisticas()
if not df.empty:
    contagem = df["resultado"].value_counts()
    st.bar_chart(contagem)
    st.caption("Distribuição dos resultados das imagens analisadas")
