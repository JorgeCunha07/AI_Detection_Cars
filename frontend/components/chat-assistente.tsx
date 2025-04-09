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
  type?: 'conversa' | 'quiz' | 'pesquisa'
}

export function ChatAssistente({ initialMessages = [], type = 'conversa' }: ChatAssistenteProps) {
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

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: "user",
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch(`/api/chatbot/${type}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          history: messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        }),
      })

      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const data = await response.json()

      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        role: "assistant",
      }

      setMessages((prev) => [...prev, aiResponse])
    } catch (error) {
      console.error("Erro:", error)
      // Adiciona mensagem de erro ao chat
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "Desculpe, ocorreu um erro ao processar sua mensagem.",
        role: "assistant",
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
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

