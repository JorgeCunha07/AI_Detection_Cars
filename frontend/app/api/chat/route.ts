import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const { message, history } = await request.json()

    // Em um ambiente real, enviaríamos para um serviço de IA
    // Aqui estamos simulando uma resposta
    const simulatedResponses = [
      "This appears to be a beautiful landscape photo. The colors are vibrant and the composition is excellent.",
      "I notice there are several interesting elements in this image. The lighting creates a nice atmosphere.",
      "Based on what I can see, this image has great detail. The subject is well-focused and stands out clearly.",
      "The perspective in this image is quite unique. It offers an interesting viewpoint that draws the viewer in.",
      "I can see why you'd be interested in this image. It has a compelling visual story that captures attention.",
    ]

    const randomIndex = Math.floor(Math.random() * simulatedResponses.length)

    // Simula um pequeno atraso para parecer mais realista
    await new Promise((resolve) => setTimeout(resolve, 500))

    return NextResponse.json({ response: simulatedResponses[randomIndex] })
  } catch (error) {
    console.error("Erro:", error)
    return NextResponse.json({ error: "Erro ao processar mensagem" }, { status: 500 })
  }
}

