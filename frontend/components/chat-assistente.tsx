"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, Loader2, ImageIcon } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

// Tipos para as mensagens
type Message = {
  id: string
  content: string
  role: "user" | "assistant"
}

interface ChatAssistenteProps {
  initialMessages?: Message[]
}

export function ChatAssistente({ initialMessages = [] }: ChatAssistenteProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [selectedChatModel, setSelectedChatModel] = useState<string | undefined>()

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!input.trim()) return

    // Adiciona mensagem do usuário
    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: "user",
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      // Envia o histórico das últimas mensagens para manter o contexto
      const recentMessages = messages.slice(-5) // Mantém as últimas 5 mensagens
      const messageHistory = recentMessages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }))

      // Em um ambiente real, enviaríamos para o backend
      // Aqui estamos simulando uma resposta
      setTimeout(() => {
        // Respostas simuladas específicas para contexto de condução
        const simulatedResponses = [
          "Nesta situação de trânsito, é importante observar o semáforo e respeitar a sinalização. O condutor deve parar completamente quando o sinal estiver vermelho e só avançar quando estiver verde, após verificar se o cruzamento está livre.",
          "A faixa de pedestres deve ser sempre respeitada. Mesmo com o sinal verde para os veículos, se houver pedestres atravessando, o condutor deve aguardar que completem a travessia antes de prosseguir.",
          "Ao se aproximar de um cruzamento, é fundamental reduzir a velocidade e estar atento à sinalização e aos outros veículos. Lembre-se que a preferência nem sempre é de quem está na via principal.",
          "Nesta situação, o condutor demonstra boa prática ao manter distância segura do veículo à frente. A regra dos 2 segundos é uma boa referência para manter uma distância segura em condições normais.",
          "É importante estar sempre atento aos retrovisores para ter consciência do que acontece ao redor do veículo. Antes de mudar de faixa, sinalize com antecedência e verifique os pontos cegos.",
        ]

        const randomIndex = Math.floor(Math.random() * simulatedResponses.length)
        const aiResponse: Message = {
          id: (Date.now() + 1).toString(),
          content: simulatedResponses[randomIndex],
          role: "assistant",
        }

        setMessages((prev) => [...prev, aiResponse])
        setIsLoading(false)
      }, 1500)
    } catch (error) {
      console.error("Erro:", error)
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: "Desculpe, ocorreu um erro ao processar sua mensagem. Tente novamente.",
        role: "assistant",
      }
      setMessages((prev) => [...prev, errorResponse])
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-2xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500">
          Tire suas dúvidas sobre esta situação:
        </h3>
        <Select value={selectedChatModel} onValueChange={setSelectedChatModel}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Selecione o modelo" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Modelo</SelectLabel>
              <SelectItem value="instrutor">Instrutor Virtual</SelectItem>
              <SelectItem value="examinador">Examinador Virtual</SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>
      <div className="h-80 overflow-y-auto p-4 rounded-lg bg-gray-50 dark:bg-gray-900 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <ImageIcon className="mx-auto h-12 w-12 mb-3 text-gray-400" />
            <p>Faça perguntas sobre a situação de trânsito mostrada na imagem.</p>
            <p className="text-sm mt-2">Exemplo: "O que devo fazer nesta situação?" ou "Quais são os riscos aqui?"</p>
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                message.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground"
              }`}
            >
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] p-3 rounded-lg bg-secondary text-secondary-foreground">
              <Loader2 className="w-4 h-4 animate-spin" />
            </div>
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <Textarea
          value={input}
          onChange={handleInputChange}
          placeholder="Faça uma pergunta sobre esta situação de trânsito..."
          className="flex-1 resize-none"
          rows={2}
        />
        <Button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="self-end bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 text-white"
        >
          <Send className="w-4 h-4" />
        </Button>
      </form>
    </div>
  )
}

