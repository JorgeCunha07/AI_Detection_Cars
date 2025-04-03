"use client"

import { useChat } from "@/hooks/use-chat"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Loader2, Send } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useState } from "react"

interface ImageChatProps {
  imageDescription: string
}

export function ImageChat({ imageDescription }: ImageChatProps) {
  const { messages, input, isLoading, handleInputChange, handleSubmit } = useChat()
  const [selectedModel, setSelectedModel] = useState<string | undefined>()

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-2xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500">
          Chat com IA sobre esta imagem:
        </h3>
        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Selecione o modelo" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Modelo</SelectLabel>
              <SelectItem value="nosso">Nosso</SelectItem>
              <SelectItem value="prefeito">Pré-feito</SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>
      <div className="h-80 overflow-y-auto p-4 rounded-lg bg-gray-50 dark:bg-gray-900 space-y-4">
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
          placeholder="Pergunte sobre a imagem..."
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

