import { NextResponse } from "next/server"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export async function POST(
  request: Request,
  { params }: { params: { type: string } }
) {
  try {
    const body = await request.json()
    const { type } = params

    const response = await fetch(`${API_URL}/api/chatbot/${type}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      throw new Error('Backend response was not ok')
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Erro:", error)
    return NextResponse.json(
      { error: "Erro ao processar mensagem" }, 
      { status: 500 }
    )
  }
}