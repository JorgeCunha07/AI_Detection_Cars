# scr_final/main.py

import sys
import random

# IMPORTS dos módulos
from scr.validacao.validador_respostas import validar_resposta
from scr.pesquisa.similar_semantic import pesquisar_artigo
from scr.chat.inferencia import gerar_resposta_chat

def mostrar_menu():
    print("=" * 64)
    print(" Bem-vindo ao ChatBot do Código da Estrada Português!")
    print(" Digite 'conversa'  para iniciar uma conversa normal")
    print(" Digite 'quiz'      para iniciar modo de perguntas e respostas")
    print(" Digite 'pesquisa'  para consultar informação")
    print(" Digite 'ajuda'     para ver opções disponíveis")
    print(" Digite 'sair'      para terminar o chat")
    print("=" * 64)

def modo_conversa():
    print("🧠 Modo conversa iniciado! Escreve algo ou 'sair' para terminar.")
    
    while True:
        user_input = input("🗣️  Tu: ").strip()

        if user_input.lower() in ["sair", "exit", "q", "quit"]:
            print("👋 A encerrar conversa. Vai com Deus e com o Código da Estrada.")
            break

        resposta = gerar_resposta_chat(user_input)
        print(f"🤖 Assistente: {resposta}")


def modo_quiz():
    print("\n🎯 Modo Quiz iniciado! Responde às perguntas do Código da Estrada.")
    print("🧠 Escreve 'dica' para ver as respostas possíveis (sem penalização).")
    print("❌ Escreve 'sair' para terminar o quiz.\n")

    from pathlib import Path
    import json
    path_questoes = Path(__file__).resolve().parent / "validacao" / "questoes.json"

    with open(path_questoes, encoding="utf-8") as f:
        questoes = json.load(f)

    while True:
        pergunta_data = random.choice(questoes)
        pergunta = pergunta_data["pergunta"]
        corretas = pergunta_data.get("respostas_corretas", [])
        print(f"❓ Pergunta: {pergunta}")

        while True:
            resposta = input("💬 A tua resposta: ").strip()

            if resposta.lower() in ["sair", "exit", "quit", "q"]:
                print("👋 A sair do modo quiz. Vai estudar... ou não.")
                return

            if resposta.lower() == "dica":
                print("💡 Dica: Algumas respostas aceitáveis são:")
                for r in corretas:
                    print(f"   - {r}")
                continue  # Pergunta outra vez sem sair

            resultado = validar_resposta(pergunta, resposta)

            if resultado.get("correto"):
                print(f"✅ Correto! Método: {resultado.get('metodo', 'desconhecido')}\n")
            else:
                sugestao = random.choice(corretas) if corretas else "Sem sugestão disponível"
                print(f"❌ Errado. Uma possível resposta correta seria: {sugestao}\n")
            break  # Avança para a próxima pergunta

def modo_pesquisa():
    print("\n📚 MODO PESQUISA ATIVADO (digite 'voltar' para regressar ao menu)\n")
    while True:
        pergunta = input("🔎 O que deseja consultar? ")
        if pergunta.lower() == "sair" or pergunta.lower() == "voltar" or pergunta.lower() == "nao" or pergunta.lower() == "não":
            break
        resultado = pesquisar_artigo(pergunta)
        print(f"\n📘 Artigo encontrado: {resultado['artigo_encontrado']}")
        print(f"📄 Conteúdo: {resultado['conteudo']}")
        print(f"📈 Similaridade: {resultado['score_similaridade']:.2f}\n")

def main():
    while True:
        mostrar_menu()
        comando = input("\nDigite uma opção: ").strip().lower()
        

        if comando == "conversa":
            modo_conversa()
        elif comando == "quiz":
            modo_quiz()
        elif comando == "pesquisa":
            modo_pesquisa()
        elif comando == "ajuda":
            mostrar_menu()
        elif comando == "sair":
            print("👋 Até logo, condutor responsável!")
            sys.exit()
        else:
            print("❌ Comando não reconhecido. Digite 'ajuda' para opções.")

if __name__ == "__main__":
    main()
