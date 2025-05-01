import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from modelo import CNN, device
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
from database import init_db, salvar_resultado, carregar_estatisticas

# Configuração da página
st.set_page_config(page_title="DogTor Chat", layout="centered")

# Estilo customizado
try:
    with open("styles.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    st.error("Erro ao carregar CSS")

st.title("🧠 DogTor Chat")
st.caption("Converse com a IA e envie imagens para detectar doenças!")

# Inicializa o banco
init_db()

# Carrega modelo de detecção de imagem
try:
    model_cnn = CNN().to(device)
    model_cnn.load_state_dict(torch.load('models/cnn_model.pth', map_location=device))
    model_cnn.eval()
except Exception as e:
    st.error(f"Erro ao carregar modelo CNN: {e}")

labels = ['Doente', 'Saudável']
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Sessão de conversa
if "messages" not in st.session_state:
    st.session_state.messages = []

# Carrega modelo de linguagem com autenticação Hugging Face
@st.cache_resource(show_spinner=True)
def load_gemma_model():
    import os
    from huggingface_hub import login

    # Login com token secreto da Hugging Face
    login(token=st.secrets["huggingface_token"])

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", revision="float16", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it", torch_dtype=torch.float16, device_map="auto"
    )
    return tokenizer, model

tokenizer_gemma, model_gemma = load_gemma_model()

# Função de conversa com IA
def chat_with_gemma(user_input):
    prompt = f"<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer_gemma(prompt, return_tensors="pt").to(model_gemma.device)
    with torch.no_grad():
        outputs = model_gemma.generate(
            **inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9
        )
    response = tokenizer_gemma.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<start_of_turn>model")[-1].strip()
    return response if response else "Desculpe, não consegui entender. Pode repetir?"

# Renderizar mensagens anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Campo de entrada do usuário
user_input = st.chat_input("Digite uma mensagem ou envie uma imagem...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            if any(word in user_input.lower() for word in ["imagem", "foto"]):
                reply = "📌 Para enviar uma imagem, use o botão abaixo."
            else:
                reply = chat_with_gemma(user_input)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# Upload de imagem para diagnóstico
uploaded_file = st.file_uploader("📷 Enviar imagem para análise", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model_cnn(input_tensor)
        _, predicted = torch.max(output, 1)
        result = labels[predicted.item()]

    salvar_resultado(result)

    with st.chat_message("user"):
        st.image(image, caption="Imagem enviada", use_container_width=True)

    reply = f"🧬 Resultado da imagem: **{result}**"
    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# Estatísticas dos diagnósticos
df = carregar_estatisticas()
if not df.empty:
    contagem = df["resultado"].value_counts()
    st.subheader("📊 Estatísticas de Diagnósticos")
    st.bar_chart(contagem)
    st.caption("Distribuição dos resultados das imagens analisadas")
