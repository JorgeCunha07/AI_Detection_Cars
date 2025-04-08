"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft } from "lucide-react"
import Link from "next/link"
import { UploadImagem } from "@/components/upload-imagem"
import { ChatAssistente } from "@/components/chat-assistente"
import Tabs from "@/components/Tabs"

export default function FerramentaPage() {
  const tabs = [
    { id: "conversa", label: "Conversa" },
    { id: "quiz", label: "Quiz" },
    { id: "pesquisa", label: "Pesquisa" },
  ]

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-6">
        <Link href="/" className="flex items-center text-blue-600 hover:text-blue-800">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Voltar para a página inicial
        </Link>
      </div>

      <Card className="w-full max-w-full mx-auto">
        <CardHeader>
          <CardTitle className="text-4xl text-center text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 font-bold">
            Análise de Situações de Trânsito
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Componente de Upload de Imagem */}
          <UploadImagem />

          {/* Instruções */}
          <div className="mt-8 p-6 bg-blue-50 rounded-lg">
            <h3 className="text-xl font-semibold text-blue-700 mb-4">Como utilizar esta ferramenta:</h3>
            <ol className="list-decimal pl-5 space-y-2 text-gray-700">
              <li>Faça upload de uma imagem de uma situação de trânsito</li>
              <li>Selecione o modelo de análise desejado</li>
              <li>Clique em "Analisar Situação" para processar a imagem</li>
              <li>Receba uma análise detalhada da situação</li>
              <li>Converse com nosso assistente virtual para esclarecer dúvidas</li>
            </ol>
            <p className="mt-4 text-blue-600">
              Esta ferramenta é ideal para estudantes que desejam praticar a identificação de situações de trânsito
              antes do exame prático.
            </p>
          </div>

          <h3 className="text-2xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-blue-500 to-purple-500 mb-4">
            Tire suas dúvidas sobre esta situação:
          </h3>
          <Tabs tabs={tabs}>
            <ChatAssistente />
            <ChatAssistente />
            <ChatAssistente />
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}

