"use client"

import type React from "react"

import { useState } from "react"
import { sendChatMessage, type Message } from "@/lib/api-service"

type ChatMessage = {
  id: string
  content: string
  role: "user" | "assistant"
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!input.trim()) return

    // Adiciona mensagem do usuário
    const userMessage: ChatMessage = {
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
      const messageHistory: Message[] = recentMessages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }))

      const response = await sendChatMessage(input, messageHistory)

      const aiResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        content: response,
        role: "assistant",
      }

      setMessages((prev) => [...prev, aiResponse])
    } catch (error) {
      console.error("Erro:", error)
      const errorResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        content: "Desculpe, ocorreu um erro ao processar sua mensagem. Tente novamente.",
        role: "assistant",
      }
      setMessages((prev) => [...prev, errorResponse])
    } finally {
      setIsLoading(false)
    }
  }

  return {
    messages,
    input,
    isLoading,
    handleInputChange,
    handleSubmit,
    setMessages,
  }
}

