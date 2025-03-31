# Código para gerar um notebook válido
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Adicionar células Markdown e de código
cells = [
    nbf.v4.new_markdown_cell("# Relatório Técnico: ChatBot do Código da Estrada Português\n\n..."),
    nbf.v4.new_markdown_cell("## Índice\n\n1. [Introdução e Visão Geral](#1-introdução-e-visão-geral)\n..."),
    # mais células aqui
]

nb['cells'] = cells

# Escrever para arquivo
with open('ChatBot_Codigo_Estrada_Relatorio_Corrigido.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)